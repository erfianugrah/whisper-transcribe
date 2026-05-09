"""
Discord TL;DW Bot — watches for YouTube links, transcribes via local
WhisperX service, summarizes via local LLM, posts result as embed.

Posts three embeds per video:
  - Brief: one-paragraph TL;DW
  - Key Points: bullet-point breakdown
  - Chapters: chronological section-by-section summary
"""

import asyncio
import html as html_mod
import json as json_mod
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
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
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

# YouTube URL → extract video ID for timestamp linking
YT_PATTERN = re.compile(
    r"https?://(?:(?:[\w-]+\.)?youtube\.com/(?:watch\?v=|shorts/|live/)|youtu\.be/)"
    r"([\w-]{11})"
)

# Broader pattern: any URL from known video/audio platforms (yt-dlp handles the download)
VIDEO_DOMAINS = {
    "youtube.com", "youtu.be", "m.youtube.com", "music.youtube.com",
    "twitch.tv", "clips.twitch.tv",
    "vimeo.com", "player.vimeo.com",
    "dailymotion.com", "dai.ly",
    "tiktok.com",
    "twitter.com", "x.com",
    "instagram.com",
    "reddit.com", "v.redd.it",
    "rumble.com",
    "odysee.com",
    "kick.com",
    "bilibili.com", "b23.tv",
    "soundcloud.com",
    "podcasts.apple.com",
    "spotify.com",  # yt-dlp may not support, fails gracefully
}

VIDEO_URL_PATTERN = re.compile(
    r"(https?://(?:[\w-]+\.)*(" + "|".join(re.escape(d) for d in VIDEO_DOMAINS) + r")/\S+)"
)

# ─── Prompts ──────────────────────────────────────────────────────────────────



PROMPT_BRIEF = """\
Video title: {title}

{reference_block}\
Summarize this video transcript in a single concise paragraph (3-5 sentences). \
Capture the main thesis, key argument, and conclusion. No bullet points. \
No timestamps. Plain language. Use correct spellings from the reference material \
when available.

Transcript:
{transcript}"""

PROMPT_KEY_POINTS = """\
Video title: {title}

{reference_block}\
Summarize this video transcript as a structured list of key points.

Format:
- One-sentence overview at the top
- 5-10 bullet points covering the most important ideas, arguments, and conclusions
- Note any calls-to-action or recommendations made
- Keep each bullet to 1 sentence
- No timestamps
- Use correct spellings from the reference material when available
- IMPORTANT: Keep total output under 3500 characters

Transcript:
{transcript}"""

PROMPT_CHAPTERS = """\
Video title: {title}

{reference_block}\
Summarize this video transcript by dividing it into logical sections/chapters.

The transcript has timestamps in [MM:SS] or [H:MM:SS] format at the start of lines.

Format:
- Identify 4-8 major topic shifts or sections in the video
- For each section, include the approximate start timestamp from the transcript
- Give each section a short descriptive heading
- Under each heading, write 1-2 sentences summarizing that section
- Format: **[H:MM:SS] Section Title** followed by summary
- Use correct spellings from the reference material when available
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

    # Collect video URLs — YouTube gets special handling (ID extraction for timestamps)
    jobs_to_queue = []
    seen = set()

    # YouTube URLs → extract video ID for timestamp linking
    for video_id in YT_PATTERN.findall(message.content):
        if video_id in seen:
            continue
        seen.add(video_id)
        url = f"https://www.youtube.com/watch?v={video_id}"
        jobs_to_queue.append(Job(url=url, video_id=video_id, message=message, channel=message.channel))

    # Other video platform URLs
    for url_match in VIDEO_URL_PATTERN.finditer(message.content):
        url = url_match.group(1)
        # Skip if already handled as YouTube
        if any(d in url for d in ("youtube.com", "youtu.be")):
            continue
        # Use URL hash as ID for non-YouTube
        vid = re.sub(r"[^\w-]", "", url.split("/")[-1])[:20] or url[-15:]
        if vid in seen:
            continue
        seen.add(vid)
        jobs_to_queue.append(Job(url=url, video_id=vid, message=message, channel=message.channel))

    if not jobs_to_queue:
        await bot.process_commands(message)
        return

    for job in jobs_to_queue:
        await queue.put(job)
        await message.add_reaction("\u23f3")  # ⏳
        log.info("Queued %s from %s", job.video_id, message.author)


# ─── Worker ───────────────────────────────────────────────────────────────────


MAX_RETRIES = 3
RETRY_BACKOFF = [10, 30, 90]  # seconds


async def worker():
    """Sequential worker — one transcription at a time (GPU bound)."""
    while True:
        job = await queue.get()
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                    log.info("[%s] Retry %d/%d in %ds...", job.video_id, attempt, MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                await process(job)
                last_error = None
                break
            except Exception as e:
                last_error = e
                log.warning("[%s] Attempt %d failed: %s", job.video_id, attempt + 1, e)

        if last_error:
            log.error("[%s] All %d attempts failed", job.video_id, MAX_RETRIES + 1)
            await safe_react(job.message, "\u274c")  # ❌
            await job.channel.send(
                f"Failed to process `{job.video_id}` after {MAX_RETRIES + 1} attempts: "
                f"{type(last_error).__name__}: {last_error}",
                reference=job.message,
            )
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

    # 3. Gather context for terminology accuracy
    description, web_context = await asyncio.gather(
        fetch_video_description(job.video_id),
        search_topic_context(title),
    )
    if description:
        log.info("[%s] Got video description (%d chars)", job.video_id, len(description))
    if web_context:
        log.info("[%s] Got web context (%d chars)", job.video_id, len(web_context))

    # 4. Extract hotwords directly from reference material (no LLM needed)
    all_context = f"{title}\n{description}\n{web_context}"
    hotwords = extract_hotwords_from_context(all_context)
    if hotwords:
        log.info("[%s] Hotwords (%d chars): %s...", job.video_id, len(hotwords), hotwords[:100])



    # 5. Transcribe with hotwords + initial_prompt (single pass)
    initial_prompt = build_initial_prompt(title, web_context)
    if initial_prompt:
        log.info("[%s] Initial prompt: %s...", job.video_id, initial_prompt[:80])

    log.info("[%s] Transcribing '%s' (%ds)...", job.video_id, title, duration)
    await safe_react(job.message, "\U0001f3a7")  # 🎧

    transcribe_payload = {
        "file_path": file_path,
        "model": WHISPER_MODEL,
        "cleanup": True,
    }
    if hotwords:
        transcribe_payload["hotwords"] = hotwords
    if initial_prompt:
        transcribe_payload["initial_prompt"] = initial_prompt

    async with http.post(
        f"{WHISPER_API}/api/transcribe",
        json=transcribe_payload,
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

    # Build reference block for summary prompts (gives LLM correct terminology)
    ref_block = ""
    if web_context:
        ref_block = (
            "Reference material (use correct spellings from this):\n"
            f"{web_context[:2000]}\n\n"
        )

    brief, key_points, chapters_raw = await asyncio.gather(
        summarize(transcript, PROMPT_BRIEF, 1024, title=title, reference_block=ref_block),
        summarize(transcript, PROMPT_KEY_POINTS, 2048, title=title, reference_block=ref_block),
        summarize(transcript, PROMPT_CHAPTERS, 2048, title=title, reference_block=ref_block),
    )
    # Only linkify timestamps for YouTube videos (other platforms don't support ?t=)
    if "youtube.com" in job.url or "youtu.be" in job.url:
        chapters = linkify_timestamps(chapters_raw, job.video_id)
    else:
        chapters = chapters_raw

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


async def summarize(transcript: str, prompt_template: str, max_tokens: int, **kwargs) -> str:
    assert http

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_template.format(transcript=transcript, **kwargs),
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


# ─── Video Context ────────────────────────────────────────────────────────────


async def fetch_video_description(video_id: str) -> str:
    """Fetch full YouTube video description from page JSON data."""
    assert http
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en"}
        async with http.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return ""
            page = await resp.text()

        # Try to extract full description from ytInitialPlayerResponse JSON
        match = re.search(r"var ytInitialPlayerResponse\s*=\s*(\{.+?\});", page)
        if match:
            try:
                data = json_mod.loads(match.group(1))
                desc = data.get("videoDetails", {}).get("shortDescription", "")
                if desc:
                    return desc
            except (json_mod.JSONDecodeError, KeyError):
                pass

        # Fallback: try ytInitialData for description
        match = re.search(r"var ytInitialData\s*=\s*(\{.+?\});", page)
        if match:
            try:
                data = json_mod.loads(match.group(1))
                # Navigate to description in structured data
                contents = (
                    data.get("engagementPanels", [{}])[0]
                    .get("engagementPanelSectionListRenderer", {})
                    .get("content", {})
                )
                # This path varies — fallback to meta tag
            except Exception:
                pass

        # Final fallback: meta description (short, ~160 chars)
        match = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', page)
        if match:
            return html_mod.unescape(match.group(1))
    except Exception as e:
        log.warning("Failed to fetch video description: %s", e)
    return ""


async def search_topic_context(title: str, description: str = "") -> str:
    """Use Exa to find authoritative sources with correct terminology."""
    assert http
    if not EXA_API_KEY:
        log.debug("No EXA_API_KEY — skipping web research")
        return ""

    # Build a focused query using title + description context
    query = title
    if description:
        # Add first sentence of description for topic specificity
        first_sentence = description.split(".")[0] if "." in description else description[:100]
        query = f"{title} — {first_sentence}"

    payload = {
        "query": query,
        "type": "auto",
        "numResults": 5,
        "contents": {
            "highlights": True,
            "text": {"maxCharacters": 5000},
        },
        "excludeDomains": ["youtube.com", "reddit.com"],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": EXA_API_KEY,
    }

    try:
        async with http.post(
            "https://api.exa.ai/search",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning("Exa search failed (%d): %s", resp.status, body[:200])
                return ""
            data = await resp.json()
    except Exception as e:
        log.warning("Exa search error: %s", e)
        return ""

    results = data.get("results", [])
    if not results:
        log.info("Exa: no results for '%s'", title[:60])
        return ""

    # Combine highlights and text from top results
    context_parts = []
    for r in results[:3]:
        parts = []
        url = r.get("url", "")
        r_title = r.get("title", "")
        if r_title:
            parts.append(f"Source: {r_title} ({url})")

        # Highlights are query-relevant excerpts
        highlights = r.get("highlights", [])
        if highlights:
            parts.extend(highlights)

        # Full text if available
        text = r.get("text", "")
        if text and not highlights:
            parts.append(text[:2000])

        if parts:
            context_parts.append("\n".join(parts))

    context = "\n---\n".join(context_parts)
    if context:
        log.info("Exa: got %d results, %d chars context", len(results), len(context))
    return context





def build_initial_prompt(title: str, web_context: str) -> str:
    """Build a compact natural-language sentence with key proper nouns for Whisper's
    initial_prompt. Limited to ~50 words (≤224 tokens). Places highest-value terms
    near the end for maximum decoder influence."""
    if not web_context:
        return ""

    # Extract unique capitalized terms from web context (likely proper nouns)
    terms = set()
    terms.update(re.findall(r"\b[A-Z][a-zA-Z''-]{2,}(?:\s[A-Z][a-zA-Z''-]{2,})*\b", web_context))
    terms.update(re.findall(r'"([^"]{2,30})"', web_context))

    # Filter to unique, non-trivial terms (skip generic words)
    generic = {"The", "This", "That", "New", "Each", "Some", "More", "Also",
               "However", "Instead", "Players", "Content", "Update", "System"}
    terms = sorted(t for t in terms if t not in generic and len(t) >= 3)

    if not terms:
        return ""

    # Build compact sentence — whisper uses last ≤224 tokens, so keep it tight
    # Use max ~40 terms to stay under token limit
    selected = terms[:40]
    prompt = f"This video is about {title}. Key terms: {', '.join(selected)}."

    # Hard cap at ~200 words (rough token proxy)
    words = prompt.split()
    if len(words) > 200:
        prompt = " ".join(words[:200])

    return prompt


def extract_hotwords_from_context(text: str) -> str:
    """Extract unique terms from reference text to use as whisper hotwords.
    No language assumptions — just finds words that look distinctive."""
    if not text:
        return ""
    # Extract all words 3+ chars that contain uppercase (proper nouns in any latin script)
    # Plus any quoted terms, hyphenated terms, or terms with apostrophes
    terms = set()
    # Capitalized words/phrases
    terms.update(re.findall(r"\b[A-Z][a-zA-Z''-]{2,}(?:\s[A-Z][a-zA-Z''-]{2,})*\b", text))
    # Quoted terms
    terms.update(re.findall(r'"([^"]{2,30})"', text))
    # Terms with special characters (apostrophes, hyphens) likely proper nouns
    terms.update(re.findall(r"\b[A-Za-z]+['''-][A-Za-z]+\b", text))
    # Filter to unique, non-trivial terms
    return ", ".join(sorted(t for t in terms if len(t) >= 3)[:150])


# ─── Helpers ──────────────────────────────────────────────────────────────────


async def send_long_embed(channel, title: str, content: str, color: int):
    """Send embed, splitting into continuation embeds if >4000 chars."""
    chunks = split_content(content, 4000)
    for i, chunk in enumerate(chunks):
        t = title if i == 0 else f"{title} (cont.)"
        embed = discord.Embed(title=t, description=chunk, color=color)
        await channel.send(embed=embed)


# Matches timestamps in various formats the LLM might output
TIMESTAMP_RE = re.compile(
    r"\[(\d{1,3}):(\d{2}):(\d{2})\]"        # [H:MM:SS] or [MMM:SS:??]
    r"|\[(\d{1,3}):(\d{2})\]"               # [MM:SS] or [MMM:SS]
    r"|(?:^|\*\*)(\d{1,3}):(\d{2}):(\d{2})" # bare H:MM:SS
    r"|(?:^|\*\*)(\d{1,3}):(\d{2})(?=\s)",   # bare MM:SS followed by space
    re.MULTILINE
)


def linkify_timestamps(text: str, video_id: str) -> str:
    """Replace timestamps with clickable YouTube timestamp links."""
    def replace(match):
        groups = match.groups()
        if groups[0] is not None:  # [H:MM:SS]
            h, m, s = int(groups[0]), int(groups[1]), int(groups[2])
        elif groups[3] is not None:  # [MM:SS] — might be >59 min
            h, m, s = 0, int(groups[3]), int(groups[4])
        elif groups[5] is not None:  # bare H:MM:SS
            h, m, s = int(groups[5]), int(groups[6]), int(groups[7])
        else:  # bare MM:SS
            h, m, s = 0, int(groups[8]), int(groups[9])

        # Normalize: if minutes > 59, convert to hours
        if m >= 60:
            h += m // 60
            m = m % 60

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
