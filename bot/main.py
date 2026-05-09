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
ALLOWED_CHANNELS: set[int] | None = None

# Transcript cache directory and TTL (default 24 hours)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(Path(__file__).parent / "cache")))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "86400"))
CACHE_DIR.mkdir(exist_ok=True)

# Persistent hotwords dictionary — learns correct terms over time
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
DATA_DIR.mkdir(exist_ok=True)
HOTWORDS_DB = DATA_DIR / "hotwords.json"

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

PROMPT_HOTWORDS = """\
Given this video title and description, list proper nouns, technical terms, \
jargon, and names that a speech-to-text model might mishear. Include character \
names, place names, game/product terminology, brand names, and specialized vocabulary. \
Output ONLY a comma-separated list, nothing else. No explanations.

Title: {title}
Description: {description}"""

PROMPT_CORRECTIONS = """\
Video title: {title}

Below is a transcript from speech recognition. Some proper nouns, names, and \
technical terms may be misspelled or misheard.

I have fetched reference material from the web about this topic. Use it as \
ground truth for correct spellings of names, places, mechanics, and terminology. \
Compare the transcript against the reference and fix any misheard terms.

Output ONLY a JSON object mapping wrong → correct. If nothing needs fixing, output {{}}.
Example: {{"Kalgoorin": "Kalguuran", "Eziomite": "Ezomyte", "Pharoah": "Farrow"}}

Reference material:
{reference}

Transcript excerpt (first 5000 chars):
{excerpt}"""

PROMPT_BRIEF = """\
Video title: {title}

Summarize this video transcript in a single concise paragraph (3-5 sentences). \
Capture the main thesis, key argument, and conclusion. No bullet points. \
No timestamps. Plain language.

Transcript:
{transcript}"""

PROMPT_KEY_POINTS = """\
Video title: {title}

Summarize this video transcript as a structured list of key points.

Format:
- One-sentence overview at the top
- 5-10 bullet points covering the most important ideas, arguments, and conclusions
- Note any calls-to-action or recommendations made
- Keep each bullet to 1 sentence
- No timestamps
- Preserve all proper nouns, names, and terminology exactly as they appear — do not guess spellings
- IMPORTANT: Keep total output under 3500 characters

Transcript:
{transcript}"""

PROMPT_CHAPTERS = """\
Video title: {title}

Summarize this video transcript by dividing it into logical sections/chapters.

The transcript has timestamps in [MM:SS] or [H:MM:SS] format at the start of lines.

Format:
- Identify 4-8 major topic shifts or sections in the video
- For each section, include the approximate start timestamp from the transcript
- Give each section a short descriptive heading
- Under each heading, write 1-2 sentences summarizing that section
- Format: **[H:MM:SS] Section Title** followed by summary
- Preserve all proper nouns, names, and terminology exactly as they appear — do not guess spellings
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

    # 3. Gather context for proper noun accuracy
    description, web_context = await asyncio.gather(
        fetch_video_description(job.video_id),
        search_topic_context(title),
    )
    if description:
        log.info("[%s] Got video description (%d chars)", job.video_id, len(description))
    if web_context:
        log.info("[%s] Got web context (%d chars)", job.video_id, len(web_context))

    # 4. Generate hotwords from title + description + web context + accumulated dictionary
    generated = await generate_hotwords(title, description, web_context)
    accumulated = get_accumulated_hotwords()
    hotwords = ", ".join(filter(None, [generated, accumulated]))
    if hotwords:
        log.info("[%s] Hotwords (%d chars): %s...", job.video_id, len(hotwords), hotwords[:100])

    # 5. Transcribe
    log.info("[%s] Transcribing '%s' (%ds)...", job.video_id, title, duration)
    await safe_react(job.message, "\U0001f3a7")  # 🎧

    transcribe_payload = {
        "file_path": file_path,
        "model": WHISPER_MODEL,
        "cleanup": True,
    }
    if hotwords:
        transcribe_payload["hotwords"] = hotwords

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

    # 6. Post-transcription correction — fix remaining misheard terms
    transcript = await correct_transcript(transcript, title, web_context)

    # Cache transcript to disk
    cache_file = CACHE_DIR / f"{job.video_id}.txt"
    cache_file.write_text(f"# {title}\n# {status}\n\n{transcript}")

    # 4. Summarize in multiple styles (concurrent — model handles full context)
    log.info("[%s] Summarizing (%d chars)...", job.video_id, len(transcript))
    await safe_react(job.message, "\U0001f9e0")  # 🧠

    brief, key_points, chapters_raw = await asyncio.gather(
        summarize(transcript, PROMPT_BRIEF, 1024, title=title),
        summarize(transcript, PROMPT_KEY_POINTS, 2048, title=title),
        summarize(transcript, PROMPT_CHAPTERS, 2048, title=title),
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


# ─── Hotwords Dictionary ──────────────────────────────────────────────────────


def load_hotwords_db() -> dict[str, int]:
    """Load accumulated hotwords with frequency counts."""
    if HOTWORDS_DB.exists():
        try:
            return json_mod.loads(HOTWORDS_DB.read_text())
        except (json_mod.JSONDecodeError, OSError):
            pass
    return {}


def save_hotwords_db(db: dict[str, int]):
    """Save hotwords dictionary, pruning terms seen only once if >500 entries."""
    if len(db) > 500:
        db = {k: v for k, v in db.items() if v > 1}
    HOTWORDS_DB.write_text(json_mod.dumps(db, indent=2))


def learn_hotwords(corrections: dict[str, str]):
    """Add corrected terms to the persistent dictionary."""
    if not corrections:
        return
    db = load_hotwords_db()
    for correct in corrections.values():
        # Store each word in the correction as a known term
        for word in correct.split():
            if len(word) > 2 and not word.isdigit():
                db[word] = db.get(word, 0) + 1
    save_hotwords_db(db)
    log.info("Learned %d terms, dictionary size: %d", len(corrections), len(db))


def get_accumulated_hotwords(limit: int = 100) -> str:
    """Get top accumulated hotwords as comma-separated string."""
    db = load_hotwords_db()
    if not db:
        return ""
    # Sort by frequency, take top N
    sorted_terms = sorted(db.items(), key=lambda x: x[1], reverse=True)[:limit]
    return ", ".join(term for term, _ in sorted_terms)


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


async def search_topic_context(title: str) -> str:
    """Web search for the video title, fetch top result for correct terminology."""
    assert http
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TLDWBot/1.0)"}

    # Step 1: Search DuckDuckGo for the video title
    query = title.replace(" ", "+")
    search_url = f"https://html.duckduckgo.com/html/?q={query}+summary+guide+wiki"
    try:
        async with http.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return ""
            page = await resp.text()
    except Exception as e:
        log.warning("Web search failed: %s", e)
        return ""

    # Step 2: Extract result URLs — prefer wikis, guides, official sources
    urls = re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)"', page)
    if not urls:
        urls = re.findall(r'href="(https?://[^"]+)"', page)

    # Rank URLs: prefer wiki/guide/official over random
    preferred = ["wiki", "guide", "fandom", "mobalytics", "ign.com", "gamespot",
                 "polygon", "eurogamer", "pcgamer", "official"]
    ranked = sorted(urls[:10], key=lambda u: -sum(p in u.lower() for p in preferred))

    # Step 3: Fetch the top 1-2 result pages and extract text
    context_parts = []
    for url in ranked[:2]:
        try:
            async with http.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    continue
                content_type = resp.headers.get("content-type", "")
                if "html" not in content_type:
                    continue
                page_html = await resp.text()

            # Extract visible text from paragraphs and headings
            text_chunks = re.findall(r"<(?:p|h[1-6]|li|td)[^>]*>(.*?)</(?:p|h[1-6]|li|td)>", page_html, re.DOTALL)
            clean = [re.sub(r"<[^>]+>", "", chunk).strip() for chunk in text_chunks]
            clean = [c for c in clean if len(c) > 20]  # skip tiny fragments
            page_text = "\n".join(clean[:50])  # first 50 paragraphs

            if page_text:
                context_parts.append(page_text[:3000])
                log.info("Fetched context from %s (%d chars)", url[:60], len(page_text))

        except Exception as e:
            log.debug("Failed to fetch %s: %s", url[:60], e)
            continue

    return "\n---\n".join(context_parts)


async def generate_hotwords(title: str, description: str, web_context: str = "") -> str:
    """Ask LLM to generate hotwords from video title + description + web context."""
    assert http
    if not title:
        return ""

    context_parts = []
    if description:
        context_parts.append(f"Description: {description[:2000]}")
    if web_context:
        context_parts.append(f"Web search results: {web_context[:2000]}")
    context = "\n".join(context_parts) or "(no additional context)"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_HOTWORDS.format(title=title, description=context),
            }
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }

    try:
        async with http.post(f"{LLM_API}/chat/completions", json=payload) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("Hotword generation failed: %s", e)
        return ""


async def correct_transcript(transcript: str, title: str, web_context: str = "") -> str:
    """Use LLM to identify and fix misheard proper nouns in transcript."""
    assert http

    excerpt = transcript[:5000]
    reference = web_context[:3000] if web_context else "(no reference material available)"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_CORRECTIONS.format(
                    title=title, excerpt=excerpt, reference=reference
                ),
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    try:
        async with http.post(f"{LLM_API}/chat/completions", json=payload) as resp:
            if resp.status != 200:
                return transcript
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()

        # Parse JSON corrections
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        corrections = json_mod.loads(content)

        if not corrections:
            return transcript

        log.info("Applying %d corrections: %s", len(corrections), corrections)
        for wrong, correct in corrections.items():
            transcript = transcript.replace(wrong, correct)

        # Learn correct terms for future transcriptions
        learn_hotwords(corrections)

        return transcript
    except (json_mod.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("Correction parse failed: %s", e)
        return transcript
    except Exception as e:
        log.warning("Correction pass failed: %s", e)
        return transcript


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
