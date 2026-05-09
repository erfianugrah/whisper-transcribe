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

# Optional: channel ID for detailed summaries (key points + chapters)
# If unset, all embeds go to the original channel
SUMMARY_CHANNEL: int | None = None
if raw := os.environ.get("SUMMARY_CHANNEL"):
    SUMMARY_CHANNEL = int(raw.strip())

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
- 5-10 bullet points covering the most important ideas, arguments, and conclusions
- Note any calls-to-action or recommendations made
- Keep each bullet to 1 sentence
- No timestamps
- Use plain language; preserve technical terms only when necessary
- IMPORTANT: Keep total output under 3500 characters

Transcript:
{transcript}"""

PROMPT_CHAPTERS = """\
Summarize this video transcript by dividing it into logical sections/chapters.

The transcript has timestamps in [MM:SS] or [H:MM:SS] format at the start of lines.

Format:
- Identify 4-8 major topic shifts or sections in the video
- For each section, include the approximate start timestamp from the transcript
- Give each section a short descriptive heading
- Under each heading, write 1-2 sentences summarizing that section
- Format: **[H:MM:SS] Section Title** followed by summary
- Use plain language
- IMPORTANT: Keep total output under 3500 characters

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

    brief, key_points, chapters_raw = await asyncio.gather(
        summarize(transcript, PROMPT_BRIEF, 1024),
        summarize(transcript, PROMPT_KEY_POINTS, 2048),
        summarize(transcript, PROMPT_CHAPTERS, 2048),
    )
    chapters = linkify_timestamps(chapters_raw, job.video_id)

    # 5. Post results as embeds
    # Determine where detailed summaries go
    detail_channel = job.channel
    if SUMMARY_CHANNEL:
        detail_channel = bot.get_channel(SUMMARY_CHANNEL) or job.channel

    use_split = detail_channel.id != job.channel.id

    # Post detailed summaries first (so we can link to them)
    detail_msg = None
    if use_split:
        header = discord.Embed(
            title=f"{truncate(title, 240)}",
            url=job.url,
            description=f"Requested by {job.message.author.mention} in <#{job.channel.id}>",
            color=0xFF0000,
        )
        header.set_footer(text=format_duration(duration))
        detail_msg = await detail_channel.send(embed=header)

    await send_long_embed(detail_channel, "Key Points", key_points, 0xFF6600)
    await send_long_embed(detail_channel, "Chapters", chapters, 0xFFAA00)

    # Post brief TL;DW in original channel
    embed = discord.Embed(
        title=f"TL;DW: {truncate(title, 240)}",
        url=job.url,
        description=truncate(brief, 4000),
        color=0xFF0000,
    )
    embed.set_footer(text=f"{format_duration(duration)} | {status}")

    if use_split and detail_msg:
        jump_url = detail_msg.jump_url
        embed.add_field(
            name="",
            value=f"[Full breakdown →]({jump_url})",
            inline=False,
        )

    await job.channel.send(embed=embed, reference=job.message)

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


async def send_long_embed(channel, title: str, content: str, color: int):
    """Send embed, splitting into continuation embeds if >4000 chars."""
    chunks = split_content(content, 4000)
    for i, chunk in enumerate(chunks):
        t = title if i == 0 else f"{title} (cont.)"
        embed = discord.Embed(title=t, description=chunk, color=color)
        await channel.send(embed=embed)


# Matches [H:MM:SS], [MM:SS], or bare H:MM:SS / MM:SS at start of line or after **
TIMESTAMP_RE = re.compile(
    r"\[(\d{1,2}):(\d{2}):(\d{2})\]"       # [H:MM:SS]
    r"|\[(\d{1,2}):(\d{2})\]"               # [MM:SS]
    r"|(?:^|\*\*)(\d{1,2}):(\d{2}):(\d{2})" # bare H:MM:SS (start of line or after **)
    r"|(?:^|\*\*)(\d{1,2}):(\d{2})(?=\s)",   # bare MM:SS followed by space
    re.MULTILINE
)


def linkify_timestamps(text: str, video_id: str) -> str:
    """Replace timestamps with clickable YouTube timestamp links."""
    def replace(match):
        groups = match.groups()
        if groups[0] is not None:  # [H:MM:SS]
            h, m, s = int(groups[0]), int(groups[1]), int(groups[2])
        elif groups[3] is not None:  # [MM:SS]
            h, m, s = 0, int(groups[3]), int(groups[4])
        elif groups[5] is not None:  # bare H:MM:SS
            h, m, s = int(groups[5]), int(groups[6]), int(groups[7])
        else:  # bare MM:SS
            h, m, s = 0, int(groups[8]), int(groups[9])
        total_seconds = h * 3600 + m * 60 + s
        display = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        url = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}"
        link = f"[{display}]({url})"
        # Preserve ** prefix if present
        full = match.group(0)
        if full.startswith("**"):
            return f"**{link}"
        return link
    return TIMESTAMP_RE.sub(replace, text)


def split_content(text: str, max_len: int) -> list[str]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find last double newline within limit
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


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
