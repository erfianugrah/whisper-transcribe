"""
Discord TL;DW Bot — watches for YouTube links, transcribes via local
WhisperX service, summarizes via local LLM, posts result as embed.

Posts three embeds per video:
  - Brief: one-paragraph TL;DW
  - Key Points: bullet-point breakdown
  - Chapters: chronological section-by-section summary
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present (no external dependency)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import aiohttp
import discord
from discord.ext import commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("tldw")

# ─── Config ───────────────────────────────────────────────────────────────────

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
WHISPER_API = os.environ.get("WHISPER_API_URL", "http://localhost:7860")
LLM_API = os.environ.get("LLM_API_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-4B-Q8_0")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "turbo")
MAX_DURATION = int(os.environ.get("MAX_DURATION", "14400"))  # 4 hours
ALLOWED_CHANNELS: set[int] | None = None

# Transcript cache directory and TTL (default 24 hours)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(Path(__file__).parent / "cache")))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "86400"))
CACHE_DIR.mkdir(exist_ok=True)

if raw := os.environ.get("ALLOWED_CHANNELS"):
    ALLOWED_CHANNELS = {int(c.strip()) for c in raw.split(",") if c.strip()}

YT_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)"
    r"([\w-]{11})"
)

# ─── Prompts ──────────────────────────────────────────────────────────────────

PROMPT_BRIEF = """\
Summarize this video transcript in a single concise paragraph (3-5 sentences). \
Capture the main thesis, key argument, and conclusion. No bullet points. \
No timestamps. Plain language.

Transcript:
{transcript}"""

PROMPT_KEY_POINTS = """\
Summarize this video transcript as a structured list of key points.

Format:
- One-sentence overview at the top
- 5-12 bullet points covering the most important ideas, arguments, and conclusions
- Note any calls-to-action or recommendations made
- Keep each bullet to 1-2 sentences
- No timestamps
- Use plain language; preserve technical terms only when necessary

Transcript:
{transcript}"""

PROMPT_CHAPTERS = """\
Summarize this video transcript by dividing it into logical sections/chapters.

Format:
- Identify 4-8 major topic shifts or sections in the video
- Give each section a short descriptive heading
- Under each heading, write 2-3 sentences summarizing that section
- Sections should be in chronological order
- No timestamps
- Use plain language

Transcript:
{transcript}"""


# ─── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    url: str
    video_id: str
    message: discord.Message
    channel: discord.TextChannel


# ─── Bot ──────────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
queue: asyncio.Queue[Job] = asyncio.Queue()
http: aiohttp.ClientSession | None = None


@bot.event
async def on_ready():
    global http
    http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900))
    bot.loop.create_task(worker())
    bot.loop.create_task(cache_cleanup_loop())
    log.info("Bot ready as %s — processing queue", bot.user)


async def cache_cleanup_loop():
    """Periodically remove cached transcripts older than CACHE_TTL."""
    while True:
        await asyncio.sleep(3600)  # check every hour
        now = time.time()
        removed = 0
        for f in CACHE_DIR.glob("*.txt"):
            if now - f.stat().st_mtime > CACHE_TTL:
                f.unlink()
                removed += 1
        if removed:
            log.info("Cache cleanup: removed %d expired transcripts", removed)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if ALLOWED_CHANNELS and message.channel.id not in ALLOWED_CHANNELS:
        return

    matches = YT_PATTERN.findall(message.content)
    if not matches:
        await bot.process_commands(message)
        return

    for video_id in dict.fromkeys(matches):  # dedupe, preserve order
        url = f"https://www.youtube.com/watch?v={video_id}"
        job = Job(url=url, video_id=video_id, message=message, channel=message.channel)
        await queue.put(job)
        await message.add_reaction("\u23f3")  # ⏳
        log.info("Queued %s from %s", video_id, message.author)


# ─── Worker ───────────────────────────────────────────────────────────────────


async def worker():
    """Sequential worker — one transcription at a time (GPU bound)."""
    while True:
        job = await queue.get()
        try:
            await process(job)
        except Exception as e:
            log.exception("Job failed: %s", job.video_id)
            await safe_react(job.message, "\u274c")  # ❌
            await job.channel.send(
                f"Failed to process `{job.video_id}`: {type(e).__name__}: {e}",
                reference=job.message,
            )
        finally:
            queue.task_done()


async def process(job: Job):
    assert http

    # 1. Check whisper service status
    async with http.get(f"{WHISPER_API}/api/status") as resp:
        if resp.status != 200:
            raise RuntimeError("Whisper service unavailable")

    # 2. Download audio
    log.info("[%s] Downloading...", job.video_id)
    async with http.post(
        f"{WHISPER_API}/api/yt-download",
        json={"url": job.url},
    ) as resp:
        if resp.status != 200:
            body = await resp.json()
            raise RuntimeError(f"Download failed: {body.get('error', resp.status)}")
        dl = await resp.json()

    title = dl.get("title", job.video_id)
    duration = dl.get("duration", 0)
    file_path = dl["filename"]

    if duration > MAX_DURATION:
        raise RuntimeError(
            f"Video too long ({duration}s > {MAX_DURATION}s limit)"
        )

    # 3. Transcribe
    log.info("[%s] Transcribing '%s' (%ds)...", job.video_id, title, duration)
    await safe_react(job.message, "\U0001f3a7")  # 🎧

    async with http.post(
        f"{WHISPER_API}/api/transcribe",
        json={
            "file_path": file_path,
            "model": WHISPER_MODEL,
            "cleanup": True,
        },
    ) as resp:
        if resp.status == 409:
            raise RuntimeError("Whisper busy — another transcription running")
        if resp.status != 200:
            body = await resp.json()
            raise RuntimeError(f"Transcription failed: {body.get('error', resp.status)}")
        result = await resp.json()

    transcript = result["transcript"]
    status = result.get("status", "")
    log.info("[%s] Transcribed: %s", job.video_id, status)

    # Cache transcript to disk
    cache_file = CACHE_DIR / f"{job.video_id}.txt"
    cache_file.write_text(f"# {title}\n# {status}\n\n{transcript}")

    # 4. Summarize in multiple styles (concurrent — model handles full context)
    log.info("[%s] Summarizing (%d chars)...", job.video_id, len(transcript))
    await safe_react(job.message, "\U0001f9e0")  # 🧠

    brief, key_points, chapters = await asyncio.gather(
        summarize(transcript, PROMPT_BRIEF, 1024),
        summarize(transcript, PROMPT_KEY_POINTS, 2048),
        summarize(transcript, PROMPT_CHAPTERS, 2048),
    )

    # 5. Post results as embeds
    embed = discord.Embed(
        title=f"TL;DW: {truncate(title, 240)}",
        url=job.url,
        description=truncate(brief, 4000),
        color=0xFF0000,
    )
    embed.set_footer(text=f"{format_duration(duration)} | {status}")
    await job.channel.send(embed=embed, reference=job.message)

    kp_embed = discord.Embed(
        title="Key Points",
        description=truncate(key_points, 4000),
        color=0xFF6600,
    )
    await job.channel.send(embed=kp_embed)

    ch_embed = discord.Embed(
        title="Chapters",
        description=truncate(chapters, 4000),
        color=0xFFAA00,
    )
    await job.channel.send(embed=ch_embed)

    # Clean up reactions
    await safe_remove_react(job.message, "\u23f3")
    await safe_remove_react(job.message, "\U0001f3a7")
    await safe_remove_react(job.message, "\U0001f9e0")
    await safe_react(job.message, "\u2705")  # ✅

    log.info("[%s] Done — posted 3 embeds", job.video_id)


async def summarize(transcript: str, prompt_template: str, max_tokens: int) -> str:
    assert http

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_template.format(transcript=transcript),
            }
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    async with http.post(f"{LLM_API}/chat/completions", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"LLM failed ({resp.status}): {body[:200]}")
        data = await resp.json()

    return data["choices"][0]["message"]["content"]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "\u2026"


def format_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


async def safe_react(message: discord.Message, emoji: str):
    try:
        await message.add_reaction(emoji)
    except discord.HTTPException:
        pass


async def safe_remove_react(message: discord.Message, emoji: str):
    try:
        await message.remove_reaction(emoji, bot.user)
    except discord.HTTPException:
        pass


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)
