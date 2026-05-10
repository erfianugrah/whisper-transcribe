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
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present (no external dependency)
def _load_env_file(path):
    """Minimal .env loader supporting quoted values and inline comments.

    Recognised forms:
      KEY=value
      KEY=value with spaces  (whitespace preserved up to inline comment)
      KEY="value"            (double quotes stripped; \\n / \\t / \\\\ escapes honoured)
      KEY='value'            (single quotes stripped; literal contents)
      KEY=value  # comment   (inline comments stripped only when unquoted)

    Doesn't try to be a full bash parser — no command substitution, no
    variable expansion, no multi-line values. For richer needs, install
    python-dotenv. Existing process env wins (uses os.environ.setdefault).
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Optional `export` prefix (compatible with shell-source-able files)
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()

        # Quoted value: strip the matching quote, honour escapes inside double-quoted.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = (value.replace("\\n", "\n")
                              .replace("\\t", "\t")
                              .replace("\\\\", "\\"))
        else:
            # Unquoted: strip an inline comment iff preceded by whitespace.
            m = re.search(r"\s+#", value)
            if m:
                value = value[:m.start()].rstrip()
        os.environ.setdefault(key, value)


_load_env_file(Path(__file__).parent / ".env")

import aiohttp
import discord
from discord.ext import commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("tldw")

# ─── Config ───────────────────────────────────────────────────────────────────

def _csv_env(name: str, default: str) -> set[str]:
    """Parse a comma-separated env var into a set of stripped non-empty values."""
    return {x.strip() for x in os.environ.get(name, default).split(",") if x.strip()}


def _csv_env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    log.error("DISCORD_TOKEN not set — refusing to start. "
              "Set it via env or bot/.env (see bot/.env.example).")
    sys.exit(2)
WHISPER_API = os.environ.get("WHISPER_API_URL", "http://localhost:7860")
LLM_API = os.environ.get("LLM_API_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.5-4B-Q8_0")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "turbo")
# No hard duration limit by default — transcript length (text density) is what
# actually matters for context budget, not video runtime. Set MAX_DURATION>0
# to enforce a soft runtime ceiling (e.g. to bound disk usage).
MAX_DURATION = int(os.environ.get("MAX_DURATION", "0"))
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
ALLOWED_CHANNELS: set[int] | None = None

# Transcript cache directory and TTL (default 24 hours)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(Path(__file__).parent / "cache")))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "86400"))
CACHE_DIR.mkdir(exist_ok=True)

# ─── Retry / backoff ──────────────────────────────────────────────────────────
# Worker retries transient errors. Permanent errors (4xx, oversized inputs)
# raise PermanentError and skip the retry loop entirely.
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BACKOFF = [int(x) for x in _csv_env_list("RETRY_BACKOFF", "10,30,90")]

# ─── Summary tuning ───────────────────────────────────────────────────────────
# Discord embed description hard cap is 4096 chars; leave safety margin so the
# model's overshoot still fits in a single embed before send_long_embed splits.
EMBED_DESC_LIMIT = 4096
EMBED_SAFE_LIMIT = EMBED_DESC_LIMIT - 96  # 4000, used for split + truncate
SUMMARY_CHAR_CAP = EMBED_DESC_LIMIT - 300  # asked of LLM (gives margin for overshoot)

# LLM call parameters — temperature + per-style max_tokens budgets.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS_BRIEF = int(os.environ.get("LLM_MAX_TOKENS_BRIEF", "1024"))
LLM_MAX_TOKENS_KEY_POINTS = int(os.environ.get("LLM_MAX_TOKENS_KEY_POINTS", "2048"))
LLM_MAX_TOKENS_CHAPTERS = int(os.environ.get("LLM_MAX_TOKENS_CHAPTERS", "3000"))

# LLM input budget — derived from the model's context window so users can
# point at a single knob (the model they're actually serving) and the chunk
# size auto-adjusts.
#
# Empirical: whisper transcripts tokenize at ~2 chars/token (lots of
# `[HH:MM:SS]` timestamps + short lines). Earlier 3.5-char-per-token
# estimate produced 80000-char chunks → 40k tokens, blowing past 32k
# context. Default LLM_CHARS_PER_TOKEN=1.8 leaves margin.
#
# Calculation:
#   chunk_chars = (context - prompt_overhead - max_output) * chars_per_token
# For 32768 ctx: (32768 - 3000 - 3000) * 1.8 ≈ 48 200 chars per chunk.
# For 128k ctx:  ~224 000 chars per chunk.
#
# Override LLM_INPUT_CHAR_BUDGET directly to bypass the calculation.
# Default 40960 (40k) is a slight bump over the 32k floor most local models
# expose. If the actual model is smaller, the adaptive-halving fallback in
# _llm_call_with_chunk_fallback recovers automatically on the first overflow.
LLM_CONTEXT_SIZE = int(os.environ.get("LLM_CONTEXT_SIZE", "40960"))
LLM_PROMPT_OVERHEAD_TOKENS = int(os.environ.get("LLM_PROMPT_OVERHEAD_TOKENS", "3000"))
LLM_CHARS_PER_TOKEN = float(os.environ.get("LLM_CHARS_PER_TOKEN", "1.8"))
_max_output_tokens = max(LLM_MAX_TOKENS_BRIEF, LLM_MAX_TOKENS_KEY_POINTS,
                         LLM_MAX_TOKENS_CHAPTERS)
_derived_budget = max(
    1000,
    int((LLM_CONTEXT_SIZE - LLM_PROMPT_OVERHEAD_TOKENS - _max_output_tokens)
        * LLM_CHARS_PER_TOKEN),
)
LLM_INPUT_CHAR_BUDGET = int(os.environ.get("LLM_INPUT_CHAR_BUDGET", str(_derived_budget)))
log.info(
    "LLM budget: ctx=%d, overhead=%d, max_out=%d, chars/tok=%.1f → chunk≤%d chars",
    LLM_CONTEXT_SIZE, LLM_PROMPT_OVERHEAD_TOKENS, _max_output_tokens,
    LLM_CHARS_PER_TOKEN, LLM_INPUT_CHAR_BUDGET,
)

# Reference text (Exa results) injected into summary prompts is capped — too
# much reference dilutes the transcript and wastes context.
REFERENCE_CHAR_CAP = int(os.environ.get("REFERENCE_CHAR_CAP", "2000"))

# ─── Vision-language fallback ─────────────────────────────────────────────────
# For videos with little or no speech (music videos, silent gameplay, ASMR,
# etc.), the bot can call the whisper service's /api/describe endpoint to
# get timestamped frame descriptions and summarize those instead.
#
# Speech density (chars/sec of transcript per second of audio) decides the
# routing:
#   density >= SPEECH_DENSITY_SPARSE  → speech-only (current path, no VLM)
#   SPEECH_DENSITY_SILENT <= density  → hybrid: speech + visual interleaved
#                          < SPARSE
#   density < SPEECH_DENSITY_SILENT   → visual-only (no usable speech)
#
# Normal English conversation is ~12 chars/sec. Default thresholds:
#   8 chars/sec → sparse (mostly silent with occasional speech)
#   2 chars/sec → effectively silent (music, ambient)
VLM_ENABLED = os.environ.get("VLM_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
SPEECH_DENSITY_SILENT = float(os.environ.get("SPEECH_DENSITY_SILENT", "2.0"))
SPEECH_DENSITY_SPARSE = float(os.environ.get("SPEECH_DENSITY_SPARSE", "8.0"))
VLM_FPS_INTERVAL = float(os.environ.get("VLM_FPS_INTERVAL", "10"))
VLM_MAX_FRAMES = int(os.environ.get("VLM_MAX_FRAMES", "60"))
VLM_TIMEOUT = int(os.environ.get("VLM_TIMEOUT", "1800"))  # 30 min worst case for 60-frame video

# ─── Exa search tuning ────────────────────────────────────────────────────────
EXA_NUM_RESULTS = int(os.environ.get("EXA_NUM_RESULTS", "5"))
EXA_MAX_CHARACTERS = int(os.environ.get("EXA_MAX_CHARACTERS", "5000"))
# Default excludes YouTube only (avoiding circular refs to the same video's
# transcripts/comments). Reddit, X, etc. left in by default — for many topics
# they're the best source of canonical jargon. Override per deployment.
EXA_EXCLUDE_DOMAINS = _csv_env_list("EXA_EXCLUDE_DOMAINS", "youtube.com")

# Tail-coverage invariant: the final chapter must start within the last
# CHAPTER_TAIL_FRACTION of the video. This is the only chapter-count constraint;
# the LLM picks the actual section count and density from semantic shifts.
CHAPTER_TAIL_FRACTION = float(os.environ.get("CHAPTER_TAIL_FRACTION", "0.25"))



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

# URL trigger: any link to one of these hosts in a Discord message kicks off
# a transcription job. Configurable via VIDEO_DOMAINS env (CSV). Default
# covers the platforms yt-dlp commonly supports — add/remove freely.
VIDEO_DOMAINS = _csv_env(
    "VIDEO_DOMAINS",
    "youtube.com,youtu.be,m.youtube.com,music.youtube.com,"
    "twitch.tv,clips.twitch.tv,"
    "vimeo.com,player.vimeo.com,"
    "dailymotion.com,dai.ly,"
    "tiktok.com,"
    "twitter.com,x.com,"
    "instagram.com,"
    "reddit.com,v.redd.it,"
    "rumble.com,"
    "odysee.com,"
    "kick.com,"
    "bilibili.com,b23.tv,"
    "soundcloud.com,"
    "podcasts.apple.com,"
    "spotify.com",
)

VIDEO_URL_PATTERN = re.compile(
    r"(https?://(?:[\w-]+\.)*(" + "|".join(re.escape(d) for d in VIDEO_DOMAINS) + r")/\S+)"
)

# ─── Prompts ──────────────────────────────────────────────────────────────────
# Templates live in bot/prompts.py — see that file for what each prompt does
# and the map-reduce contract.

from prompts import (
    PROMPT_BRIEF, PROMPT_KEY_POINTS, PROMPT_CHAPTERS,
    REDUCE_BRIEF, REDUCE_KEY_POINTS,
    CHUNK_PREAMBLE,
)


# ─── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    url: str
    video_id: str
    message: discord.Message
    channel: discord.TextChannel
    # Optional user-supplied steering text from the Discord message. When
    # present (non-empty), the bot:
    #   1. Forces VLM enrichment regardless of speech density (the user
    #      explicitly cares about visual content or wants targeted attention).
    #   2. Passes the text as the per-frame prompt to /api/describe so the
    #      VLM looks for what the user asked about.
    #   3. Steers the summary LLM toward the user's request via a
    #      "User asked:" block prepended to each summary prompt.
    user_prompt: str = ""


# Maximum length of user-prompt text we'll honour (truncated above this).
# Keeps Discord-side lyrical messages from blowing the prompt budget on
# both VLM and summary calls. Picked to fit comfortably alongside the
# transcript content within LLM_INPUT_CHAR_BUDGET.
USER_PROMPT_MAX_CHARS = int(os.environ.get("USER_PROMPT_MAX_CHARS", "1500"))


def _extract_user_prompt(message_content: str, urls: list[str]) -> str:
    """Strip the URLs out of the message and return whatever non-trivial
    text remains. Used as user-supplied steering for VLM + summary.

    Empty / whitespace-only / mention-only messages return "" and the
    pipeline takes its default automatic path.
    """
    text = message_content
    for url in urls:
        text = text.replace(url, " ")
    # Drop Discord mentions, channel refs, custom emoji shorthand
    text = re.sub(r"<[@#:][^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 3:
        return ""
    return text[:USER_PROMPT_MAX_CHARS]


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
    all_urls: list[str] = []  # for user-prompt extraction (strip URLs from text)

    # YouTube URLs → extract video ID for timestamp linking. The pattern only
    # captures the video id; rebuild the full URL for substitution out of the
    # message text below.
    for m in YT_PATTERN.finditer(message.content):
        all_urls.append(m.group(0))
        video_id = m.group(1)
        if video_id in seen:
            continue
        seen.add(video_id)
        url = f"https://www.youtube.com/watch?v={video_id}"
        jobs_to_queue.append(Job(url=url, video_id=video_id, message=message,
                                 channel=message.channel))

    # Other video platform URLs
    for url_match in VIDEO_URL_PATTERN.finditer(message.content):
        url = url_match.group(1)
        all_urls.append(url)
        if any(d in url for d in ("youtube.com", "youtu.be")):
            continue
        # Use last non-empty path segment as ID for non-YouTube (filesystem-safe)
        path_parts = [p for p in url.rstrip("/").split("/")
                      if p and "." not in p and "//" not in p]
        vid = re.sub(r"[^\w-]", "", path_parts[-1])[:20] if path_parts else "unknown"
        if vid in seen:
            continue
        seen.add(vid)
        jobs_to_queue.append(Job(url=url, video_id=vid, message=message,
                                 channel=message.channel))

    if not jobs_to_queue:
        await bot.process_commands(message)
        return

    # User-prompt: any non-trivial text remaining after stripping URLs.
    # Applied to ALL jobs in this message (one prompt per message; if the
    # user posts multiple URLs with steering, each gets the same instruction).
    user_prompt = _extract_user_prompt(message.content, all_urls)
    if user_prompt:
        for job in jobs_to_queue:
            job.user_prompt = user_prompt
        log.info("User prompt detected (%d chars): %s",
                 len(user_prompt), user_prompt[:80])

    for job in jobs_to_queue:
        await queue.put(job)
        await message.add_reaction("\u23f3")  # ⏳
        log.info("Queued %s from %s", job.video_id, message.author)


# ─── Worker ───────────────────────────────────────────────────────────────────


class PermanentError(Exception):
    """Errors that will fail identically on retry (4xx, oversized inputs, etc.)."""


# Belt-and-suspenders: even if the whisper service misclassifies a yt-dlp
# error as 5xx (e.g. older container that pre-dates the 422 classification),
# the bot recognises these patterns in error bodies and refuses to retry.
# Belt-and-suspenders pattern matcher for the bot. Mirrors the server's
# `_PERMANENT_YT_DLP_PATTERNS` exactly (plus extras for whisper-side and
# LLM-side errors that the bot sees but the server doesn't classify). When
# the whisper service returns a misclassified 5xx (e.g. running an older
# image without the latest server-side patterns), the bot still catches
# known-permanent errors via these patterns and skips the retry loop.
#
# The server's list lives in app.py:_PERMANENT_YT_DLP_PATTERNS — keep both
# in sync. Tooling: `make lint` exercises a pattern-drift check via the
# regression test in tests/.
_PERMANENT_REMOTE_PATTERNS = (
    # ─── Mirrors app.py:_PERMANENT_YT_DLP_PATTERNS (yt-dlp errors) ──────────
    "Sign in to confirm your age",
    "Private video",
    "Video unavailable",
    "This video is unavailable",
    "members-only content",
    "members only video",
    "members-only",
    "This video has been removed",
    "blocked it on copyright grounds",
    "blocked it in your country",
    "country and is unavailable",
    "Premieres in",
    "This live event will begin",
    "Sign in to confirm you're not a bot",
    "Join this channel to get access",
    "Video is not available",
    "No video could be found in this tweet",
    "No video could be found in this",
    "No video formats found",
    "no video formats found",
    "Unsupported URL",
    "is not a valid URL",
    "There's no video in this post",
    "No media found",
    "Post does not contain any media",
    # ─── Bot-only additions ─────────────────────────────────────────────────
    # Whisper-side: file-resolution / ffmpeg errors that propagate as 5xx
    "file not found:",                 # /api/transcribe got bad path
    "input file has no video stream",  # /api/describe ffmpeg
    "no video streams",
    "Output file does not contain any stream",
    # LLM context overflow (OpenAI-compatible servers; classified locally
    # in the bot since the LLM is reached over llm-compose proxy, not via
    # whisper's yt-dlp pattern matcher).
    "exceed_context_size_error",
    "context_length_exceeded",
)


def _is_permanent_remote_error(text: str) -> bool:
    return any(p in text for p in _PERMANENT_REMOTE_PATTERNS)





# Reaction emoji used during processing — cleaned up on completion or failure
PROCESSING_EMOJI = ("\u23f3", "\U0001f3a7", "\U0001f9e0")  # ⏳ 🎧 🧠


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
            except PermanentError as e:
                last_error = e
                log.error("[%s] Permanent failure (no retry): %s", job.video_id, e)
                break
            except Exception as e:
                last_error = e
                log.warning("[%s] Attempt %d failed: %s", job.video_id, attempt + 1, e)

        if last_error:
            attempts = "permanent error" if isinstance(last_error, PermanentError) \
                else f"after {MAX_RETRIES + 1} attempts"
            log.error("[%s] Giving up — %s", job.video_id, attempts)
            # Clean any in-progress reactions before marking failed
            for emoji in PROCESSING_EMOJI:
                await safe_remove_react(job.message, emoji)
            await safe_react(job.message, "\u274c")  # ❌
            await job.channel.send(
                f"Failed to process `{job.video_id}` ({attempts}): "
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

    # 1a. Cache lookup — skip download+transcribe if we have a fresh transcript
    cached = read_cache(job.video_id)
    file_path = None
    if cached is not None:
        title, status, transcript, duration = cached
        log.info("[%s] Cache hit (%d chars, '%s')", job.video_id, len(transcript), title)
    else:
        # 2. Download. Keep the video stream alongside audio when VLM is
        # enabled — /api/describe needs a video file to extract frames
        # from. When VLM is off, audio-only WAV (smaller, current default).
        log.info("[%s] Downloading%s...", job.video_id,
                 " (audio+video)" if VLM_ENABLED else "")
        async with http.post(
            f"{WHISPER_API}/api/yt-download",
            json={"url": job.url, "keep_video": VLM_ENABLED},
        ) as resp:
            if resp.status != 200:
                try:
                    body = await resp.json()
                except Exception:
                    body = {"error": await resp.text()}
                err = str(body.get("error", resp.status))
                # Server marks 4xx as permanent; bot also matches known
                # patterns in 5xx bodies so older servers that haven't been
                # rebuilt with the classifier still fail fast here.
                if (400 <= resp.status < 500) or body.get("permanent") \
                        or _is_permanent_remote_error(err):
                    raise PermanentError(f"Download failed ({resp.status}): {err}")
                raise RuntimeError(f"Download failed: {err}")
            dl = await resp.json()

        title = dl.get("title", job.video_id)
        duration = dl.get("duration", 0)
        file_path = dl["filename"]

        if MAX_DURATION > 0 and duration > MAX_DURATION:
            raise PermanentError(
                f"Video too long ({duration}s > {MAX_DURATION}s soft limit; "
                f"set MAX_DURATION=0 to disable)"
            )
        transcript = ""
        status = ""

    # 3. Gather context for terminology accuracy (cheap; do for both paths)
    description, web_context = await asyncio.gather(
        fetch_video_description(job.video_id),
        search_topic_context(title),
    )
    if description:
        log.info("[%s] Got video description (%d chars)", job.video_id, len(description))
    if web_context:
        log.info("[%s] Got web context (%d chars)", job.video_id, len(web_context))

    # 4. Transcribe (cache miss only)
    if cached is None:
        # Build initial_prompt for whisper (proper-noun bias for the decoder).
        # Note: we deliberately do NOT pass `hotwords` — initial_prompt and hotwords
        # share whisper's 448-token prompt context, and passing both has caused
        # "position >= 448" errors. initial_prompt alone covers terminology bias.
        initial_prompt = build_initial_prompt(title, f"{description}\n{web_context}")
        if initial_prompt:
            log.info("[%s] Initial prompt (%d chars): %s...",
                     job.video_id, len(initial_prompt), initial_prompt[:80])

        log.info("[%s] Transcribing '%s' (%ds)...", job.video_id, title, duration)
        await safe_react(job.message, "\U0001f3a7")  # 🎧

        transcribe_payload = {
            "file_path": file_path,
            "model": WHISPER_MODEL,
            # Don't cleanup yet — VLM fallback (below) may need the file.
            "cleanup": False,
            "return_file": False,  # bot uses transcript text directly
        }
        if initial_prompt:
            transcribe_payload["initial_prompt"] = initial_prompt

        async with http.post(
            f"{WHISPER_API}/api/transcribe",
            json=transcribe_payload,
        ) as resp:
            if resp.status == 409:
                raise RuntimeError("Whisper busy — another transcription running")
            if resp.status != 200:
                try:
                    body = await resp.json()
                except Exception:
                    body = {"error": await resp.text()}
                err = str(body.get("error", resp.status))
                if (400 <= resp.status < 500) or body.get("permanent") \
                        or _is_permanent_remote_error(err):
                    # Cleanup the file ourselves since we asked /api/transcribe not to.
                    await _cleanup_remote_file(file_path)
                    raise PermanentError(f"Transcription rejected ({resp.status}): {err}")
                await _cleanup_remote_file(file_path)
                raise RuntimeError(f"Transcription failed: {err}")
            result = await resp.json()

        transcript = result["transcript"]
        status = result.get("status", "")
        log.info("[%s] Transcribed: %s", job.video_id, status)

        # Whisper returned an "Error: ..." status (CUDA OOM, model load
        # failure, prompt-context overflow). These CAN be transient → retry.
        # Different from "Done -- 0 segments", which is silent video → VLM.
        if status.lower().startswith("error"):
            await _cleanup_remote_file(file_path)
            raise RuntimeError(f"Transcription failed: {status}")

        # Speech density: how many chars of transcript per second of audio.
        # Normal speech ~12 chars/sec. Below SPARSE → augment with visuals.
        density = (len(transcript.strip()) / duration) if duration > 0 else 0.0
        log.info("[%s] Speech density: %.1f chars/sec (silent<%.1f, sparse<%.1f)",
                 job.video_id, density, SPEECH_DENSITY_SILENT, SPEECH_DENSITY_SPARSE)

        # User-prompt forces VLM enrichment regardless of speech density —
        # the user explicitly asked about visual content (or wants targeted
        # attention to specific things on screen).
        user_forced_vlm = bool(job.user_prompt)
        run_vlm = VLM_ENABLED and (user_forced_vlm or density < SPEECH_DENSITY_SPARSE)

        if run_vlm:
            visual_only = density < SPEECH_DENSITY_SILENT and not user_forced_vlm
            if user_forced_vlm and density >= SPEECH_DENSITY_SPARSE:
                mode = "user-forced-enrich"
            elif visual_only:
                mode = "visual-only"
            else:
                mode = "hybrid"
            log.info("[%s] %s — calling /api/describe (frame-level VLM)%s",
                     job.video_id, mode,
                     f", user steering: {job.user_prompt[:60]!r}" if user_forced_vlm else "")
            try:
                # User-prompt becomes the per-frame description prompt so the
                # VLM looks for what they asked about. Falls back to default.
                desc_result = await _fetch_descriptions(
                    file_path, cleanup=True,
                    prompt=_build_vlm_prompt(job.user_prompt) if user_forced_vlm else None,
                )
            except PermanentError as e:
                log.error("[%s] VLM describe permanent error: %s", job.video_id, e)
                await _cleanup_remote_file(file_path)
                if visual_only:
                    raise PermanentError(
                        f"No speech detected and visual description failed: {e}"
                    )
                # Hybrid / user-forced: VLM failed but we still have transcript.
                desc_result = None
            except Exception as e:
                log.warning("[%s] VLM describe transient error: %s", job.video_id, e)
                await _cleanup_remote_file(file_path)
                if visual_only:
                    raise  # let the worker retry
                desc_result = None

            if desc_result is not None:
                visual_text = _format_descriptions(desc_result["descriptions"])
                if visual_only:
                    transcript = visual_text
                    status = (status or "") + (
                        f" | visual-only ({desc_result['frame_count']} frames "
                        f"@ {desc_result['interval_seconds']:.0f}s)"
                    )
                else:
                    # Hybrid: interleave speech and visual lines by timestamp.
                    transcript = _interleave_by_timestamp(transcript, visual_text)
                    tag = "user-enriched" if user_forced_vlm else "hybrid"
                    status = (status or "") + (
                        f" | {tag} (+{desc_result['frame_count']} visual frames)"
                    )
        else:
            # Speech-heavy or VLM disabled: nothing more to do, clean up file.
            await _cleanup_remote_file(file_path)

        # Final empty-content guard — if we still have nothing after VLM,
        # there's nothing for the LLM to summarize.
        if not transcript.strip():
            raise PermanentError(
                "No speech detected and no visual content extracted — "
                "nothing to summarize."
            )

        # Persist to cache (whatever combination of speech / visual we ended up with)
        write_cache(job.video_id, title, status, transcript, duration)

    # 5. Summarize in multiple styles (concurrent — model handles full context)
    log.info("[%s] Summarizing (%d chars)...", job.video_id, len(transcript))
    await safe_react(job.message, "\U0001f9e0")  # 🧠

    # Build reference block for summary prompts (terminology/spelling ONLY).
    # Wrapped in <reference>...</reference> so the LLM clearly sees this as
    # data to consult for spelling, not instructions to follow. Combined with
    # the SECURITY rules in REF_RULES.
    ref_block = ""
    if web_context:
        ref_block = (
            "Reference material — USE FOR SPELLING/TERMINOLOGY ONLY. "
            "Do NOT copy facts, dates, numbers, or claims from this into the summary. "
            "Summary content must come exclusively from the transcript below.\n"
            "<reference>\n"
            f"{web_context[:REFERENCE_CHAR_CAP]}\n"
            "</reference>\n\n"
        )

    # User-prompt steering — prepended to each summary's prompt template.
    # Wrapped in <user_request> so the LLM clearly distinguishes the user's
    # ask from the transcript content. Empty when no user prompt was given.
    user_steer_block = ""
    if job.user_prompt:
        user_steer_block = (
            "The Discord user who requested this summary specifically asked: "
            "<user_request>\n"
            f"{job.user_prompt}\n"
            "</user_request>\n"
            "Honour that request when shaping your output — emphasise the "
            "aspects they're interested in, while still covering the rest of "
            "the video. The user_request is steering, not data to summarise.\n\n"
        )
        # Prepend steer block to the reference block so it appears at the
        # top of the prompt (after the title/duration header). When there's
        # no reference, the steer block goes directly there.
        ref_block = user_steer_block + ref_block

    duration_str = format_duration(duration)
    tail_start = format_duration(int(duration * (1 - CHAPTER_TAIL_FRACTION)))

    brief, key_points, chapters_raw = await asyncio.gather(
        summarize(
            transcript, PROMPT_BRIEF, LLM_MAX_TOKENS_BRIEF,
            reduce_template=REDUCE_BRIEF,
            title=title, duration=duration_str, reference_block=ref_block,
        ),
        summarize(
            transcript, PROMPT_KEY_POINTS, LLM_MAX_TOKENS_KEY_POINTS,
            reduce_template=REDUCE_KEY_POINTS,
            title=title, duration=duration_str, reference_block=ref_block,
            char_cap=SUMMARY_CHAR_CAP,
        ),
        summarize(
            transcript, PROMPT_CHAPTERS, LLM_MAX_TOKENS_CHAPTERS,
            reduce_template=None,  # chapters are time-ordered; concat preserves chronology
            title=title, duration=duration_str, reference_block=ref_block,
            tail_start=tail_start, char_cap=SUMMARY_CHAR_CAP,
        ),
    )
    # Sanitize LLM output before any further processing — strips
    # untrusted links injected via prompt injection. linkify_timestamps
    # adds youtube.com links AFTER sanitisation, which is fine because
    # those URLs are bot-constructed and on the allowlist.
    brief = sanitize_llm_output(brief)
    key_points = sanitize_llm_output(key_points)
    chapters_raw = sanitize_llm_output(chapters_raw)

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

    # Show the user's steering request so the requester can see it was honoured
    # and others in the channel understand why the summary leans a certain way.
    if job.user_prompt:
        embed.add_field(
            name="User request",
            value=truncate(job.user_prompt, 1000),
            inline=False,
        )

    if use_split and detail_msg:
        jump_url = detail_msg.jump_url
        embed.add_field(
            name="",
            value=f"[Full breakdown →]({jump_url})",
            inline=False,
        )

    await job.channel.send(embed=embed, reference=job.message)

    # Clean up reactions
    for emoji in PROCESSING_EMOJI:
        await safe_remove_react(job.message, emoji)
    await safe_react(job.message, "\u2705")  # ✅

    log.info("[%s] Done — posted 3 embeds", job.video_id)


async def _llm_call(prompt: str, max_tokens: int) -> str:
    """One LLM chat-completion request.

    Raises PermanentError on 4xx (won't recover on retry); RuntimeError on 5xx
    or transport errors (worker will retry).
    """
    assert http
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    async with http.post(f"{LLM_API}/chat/completions", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            # 4xx OR known-permanent error signature → no retry. Some
            # OpenAI-compatible servers return 500 with `exceed_context_size_error`
            # in the body — match the body too.
            if (400 <= resp.status < 500) or _is_permanent_remote_error(body):
                raise PermanentError(f"LLM rejected request ({resp.status}): {body[:300]}")
            raise RuntimeError(f"LLM failed ({resp.status}): {body[:200]}")
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


def _chunk_transcript(transcript: str, max_chars: int) -> list[str]:
    """Split transcript on line boundaries, each chunk ≤ max_chars.

    The transcript is line-oriented (`[MM:SS] text` per line); breaking on
    `\\n` preserves segment integrity so the model never sees half a sentence.
    """
    if len(transcript) <= max_chars:
        return [transcript]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in transcript.split("\n"):
        # +1 accounts for the "\n" that "\n".join will add back
        ln = len(line) + 1
        if cur_len + ln > max_chars and cur:
            chunks.append("\n".join(cur))
            cur = [line]
            cur_len = ln
        else:
            cur.append(line)
            cur_len += ln
    if cur:
        chunks.append("\n".join(cur))
    return chunks


# Minimum chunk size when adaptive halving — below this, give up rather
# than recurse forever. 4000 chars is ~2000 tokens which is small enough
# that any sane model can handle it; if THAT fails, the model is broken.
_MIN_CHUNK_CHARS = 4000


def _is_context_overflow(exc: Exception) -> bool:
    """True if the exception looks like an LLM context-size overflow."""
    s = str(exc).lower()
    return (
        "exceed_context_size" in s
        or "context_length_exceeded" in s
        or "exceeds the available context" in s
        or ("token" in s and ("ctx" in s or "context" in s) and "exceed" in s)
    )


async def summarize(
    transcript: str,
    prompt_template: str,
    max_tokens: int,
    *,
    reduce_template: str | None = None,
    _budget: int | None = None,
    _preamble: str = "",
    **kwargs,
) -> str:
    """Summarize a transcript, splitting + recombining if it exceeds budget.

    - Single-call path when transcript fits in `_budget` (defaults to
      `LLM_INPUT_CHAR_BUDGET`).
    - Otherwise: map per-chunk with `prompt_template`, then either:
        * reduce_template=None  → concatenate chunk summaries raw (use for
          chronological output like chapters where order is meaningful).
        * reduce_template=<str> → run a final pass to merge/dedupe partials.

    Self-correcting on context overflow: if a single map call hits a
    context-size error despite the calculated budget, the budget is halved
    and the same call is re-attempted as a multi-chunk map-reduce. The
    reduce step (when supplied) is preserved through the recursion so brief
    and key_points still get a final coherent pass — only chapters skip it
    by design.

    Duration / content density don't matter: the LLM tells us when to split
    smaller and we obey.
    """
    budget = _budget if _budget is not None else LLM_INPUT_CHAR_BUDGET
    chunks = _chunk_transcript(transcript, budget)

    if len(chunks) == 1:
        # Single-call path. If the LLM rejects with context overflow, fall
        # through to a smaller budget — this re-enters summarize() with a
        # halved budget, which forces the multi-chunk path and preserves
        # reduce_template handling.
        prompt = _preamble + prompt_template.format(transcript=chunks[0], **kwargs)
        try:
            return await _llm_call(prompt, max_tokens)
        except PermanentError as e:
            if not _is_context_overflow(e):
                raise
            if budget <= _MIN_CHUNK_CHARS:
                log.error(
                    "Chunk at minimum size (%d chars) still overflows context "
                    "— giving up",
                    budget,
                )
                raise
            new_budget = budget // 2
            log.warning(
                "Single-chunk call overflowed context; halving budget %d → %d "
                "and re-entering map-reduce path",
                budget, new_budget,
            )
            return await summarize(
                transcript, prompt_template, max_tokens,
                reduce_template=reduce_template,
                _budget=new_budget, _preamble=_preamble, **kwargs,
            )

    log.info(
        "LLM input %d chars > budget %d → splitting into %d chunks for map-reduce",
        len(transcript), budget, len(chunks),
    )

    # Map: per-chunk summarization, sequential to avoid GPU thrash on a
    # single-instance llm-compose backend. Each call recurses through
    # summarize() with no reduce_template so an overflow on one chunk
    # halves only that chunk, not the whole pipeline.
    partials: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        chunk_preamble = _preamble + CHUNK_PREAMBLE.format(n=i, total=len(chunks))
        partial = await summarize(
            chunk, prompt_template, max_tokens,
            reduce_template=None,  # map step never reduces
            _budget=budget, _preamble=chunk_preamble, **kwargs,
        )
        partials.append(partial)

    combined = "\n\n---\n\n".join(partials)

    if reduce_template is None:
        # Caller wants raw concatenation (e.g. chronological chapters).
        return combined

    # Run the reduce step. If combined partials still exceed budget, recurse
    # — same machinery handles it. Critical: pass reduce_template through so
    # the deeper recursion still produces a coherent reduce, not concat.
    return await summarize(
        combined, reduce_template, max_tokens,
        reduce_template=reduce_template if len(combined) > budget else None,
        _budget=budget, **kwargs,
    )


# ─── Transcript cache ─────────────────────────────────────────────────────────

# Cache file format (line-prefixed metadata + body):
#   # title: <title>
#   # status: <whisper status>
#   # duration: <seconds>
#   <blank line>
#   <transcript body>


def _cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.txt"


def write_cache(video_id: str, title: str, status: str, transcript: str, duration: int) -> None:
    """Persist transcript to disk for reuse across retries / future runs."""
    try:
        _cache_path(video_id).write_text(
            f"# title: {title}\n"
            f"# status: {status}\n"
            f"# duration: {duration}\n"
            f"\n"
            f"{transcript}"
        )
    except OSError as e:
        log.warning("[%s] Cache write failed: %s", video_id, e)


def _derive_duration_from_transcript(transcript: str) -> int:
    """Estimate duration in seconds from the last [H:MM:SS] / [MM:SS] timestamp
    in the transcript. Used when the cache file pre-dates duration storage.
    Returns 0 if no timestamps found.
    """
    # Match all [HH:MM:SS] or [MM:SS] anchored at line starts. The last such
    # match is the start time of the last segment — close enough to total
    # duration for display purposes (within ~1 segment of the true end).
    matches = re.findall(r"^\[(\d{1,3}):(\d{2})(?::(\d{2}))?\]", transcript, re.MULTILINE)
    if not matches:
        return 0
    h_or_m, m_or_s, maybe_s = matches[-1]
    if maybe_s:
        # [H:MM:SS]
        return int(h_or_m) * 3600 + int(m_or_s) * 60 + int(maybe_s)
    # [MM:SS]
    return int(h_or_m) * 60 + int(m_or_s)


def read_cache(video_id: str) -> tuple[str, str, str, int] | None:
    """Read cached transcript if present and not expired.

    Returns (title, status, transcript, duration) or None.
    """
    path = _cache_path(video_id)
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > CACHE_TTL:
            return None
        text = path.read_text()
    except OSError:
        return None

    title = ""
    status = ""
    duration = 0
    body_start = 0
    for i, line in enumerate(text.splitlines(keepends=True)):
        if line.startswith("# title: "):
            title = line[len("# title: "):].rstrip("\n")
        elif line.startswith("# status: "):
            status = line[len("# status: "):].rstrip("\n")
        elif line.startswith("# duration: "):
            try:
                duration = int(line[len("# duration: "):].strip())
            except ValueError:
                duration = 0
        elif line.strip() == "":
            body_start = sum(len(l) for l in text.splitlines(keepends=True)[:i + 1])
            break
        else:
            # Header parsing done at first non-comment line
            break

    transcript = text[body_start:] if body_start else text
    if not transcript.strip():
        return None
    # Backward compat: older cache files used "# {title}\n# {status}\n\n..."
    # without explicit keys. Fall back to those if the new keys weren't found.
    if not title and text.startswith("# "):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("# ") and lines[1].startswith("# "):
            title = lines[0][2:]
            status = lines[1][2:]
            transcript = "\n".join(lines[3:]) if len(lines) > 3 else ""

    # Duration fallback: if the cache file didn't record one (legacy format
    # or just-missing key), derive it from the transcript's last timestamp.
    # This keeps the embed footer accurate even for cache files written
    # before the duration field existed.
    if duration <= 0:
        duration = _derive_duration_from_transcript(transcript)
    return title, status, transcript, duration


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
        # First sentence of description for topic specificity. Require whitespace
        # after the period so version numbers like "v1.5 release" don't truncate.
        m = re.search(r"[.!?](\s|$)", description)
        first_sentence = description[:m.start()] if m else description[:100]
        query = f"{title} — {first_sentence}"

    payload = {
        "query": query,
        "type": "auto",
        "numResults": EXA_NUM_RESULTS,
        "contents": {
            "highlights": True,
            "text": {"maxCharacters": EXA_MAX_CHARACTERS},
        },
        "excludeDomains": EXA_EXCLUDE_DOMAINS,
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





# Whisper's prompt context window is 448 tokens total. initial_prompt and
# hotwords share that budget. Stay well under by capping the prompt at
# ~600 chars (≈ 150 tokens) and skipping hotwords entirely.
INITIAL_PROMPT_CHAR_CAP = 600


def build_initial_prompt(title: str, web_context: str) -> str:
    """Build a compact natural-language sentence with key proper nouns for
    Whisper's initial_prompt. Hard-capped at INITIAL_PROMPT_CHAR_CAP chars to
    stay within Whisper's 448-token prompt context.

    Term selection is content-agnostic: extract capitalised tokens and quoted
    phrases from the reference text, then rank them by FREQUENCY in that
    reference. The most-mentioned terms are the most likely to recur in the
    video's audio and most worth seeding into the decoder. No hardcoded
    stopword list — frequency naturally demotes generic words like "The",
    "Players", "System" because the same generic word also gets matched in
    every other paragraph and isn't actually domain-specific.

    A token gets priority if it appears in the title (it's clearly central to
    the video). Anything appearing only once in the reference is dropped as
    noise. Works for any language using Latin-script capitalisation
    conventions; non-capitalising scripts (CJK, Arabic, etc.) yield no
    capitalised terms and we cleanly fall through to title-only.
    """
    if not web_context:
        return ""

    # Extract candidate spans: capitalised words/phrases and quoted strings.
    # We keep multi-word capitalised phrases as units ("Path of Exile") so the
    # decoder learns the joint sequence, not the individual words.
    spans: list[str] = []
    spans.extend(re.findall(r"\b[A-Z][a-zA-Z'-]{2,}(?:\s+[A-Z][a-zA-Z'-]{2,})*\b", web_context))
    spans.extend(re.findall(r'"([^"]{3,40})"', web_context))

    if not spans:
        return title[:INITIAL_PROMPT_CHAR_CAP]

    # Frequency rank — case-insensitive, but emit canonical (first-seen) form.
    from collections import Counter
    canonical: dict[str, str] = {}
    for s in spans:
        canonical.setdefault(s.lower(), s)
    counts = Counter(s.lower() for s in spans)

    # Boost terms that appear in the title (they're central to the video).
    title_lower = title.lower()

    def _score(key: str) -> tuple[int, int]:
        c = counts[key]
        if key in title_lower:
            c *= 3
        # Primary: count, secondary: length (prefer longer phrases on tie).
        return (c, len(key))

    # Drop hapax legomena — single mentions in reference are noise, not
    # entities the speaker is going to repeat throughout the video. The
    # frequency-min-2 filter is the content-agnostic replacement for the
    # old hardcoded English stopword list.
    candidates = [k for k, n in counts.items() if n >= 2]
    candidates.sort(key=_score, reverse=True)

    # Use a language-agnostic format: title followed by a comma-separated
    # term list, no English framing sentence. Whisper's `initial_prompt` is
    # used to bias the decoder; a Spanish/Japanese/etc. video shouldn't be
    # primed with English filler like "This video is about ...".
    base = f"{title}. "
    suffix = "."
    budget = INITIAL_PROMPT_CHAR_CAP - len(base) - len(suffix)
    if budget <= 0 or not candidates:
        return title[:INITIAL_PROMPT_CHAR_CAP]

    picked: list[str] = []
    used = 0
    for key in candidates:
        term = canonical[key]
        addition = (", " if picked else "") + term
        if used + len(addition) > budget:
            break
        picked.append(term)
        used += len(addition)

    if not picked:
        return ""
    return base + ", ".join(picked) + suffix


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


# ─── VLM frame description helpers ────────────────────────────────────────────


def _build_vlm_prompt(user_text: str) -> str:
    """Wrap a user's steering text into a per-frame VLM prompt.

    The VLM receives one image per call and our prompt tells it what to
    describe. Default (no user steering) is the generic VLM_FRAME_PROMPT
    on the server. With user steering, we ask the VLM to focus on what
    the user asked about, while still keeping the output concise enough
    to fit alongside many frames in the eventual summary prompt.
    """
    return (
        "Describe what is happening in this video frame, focusing on this "
        f"user request: {user_text!r}. Keep your answer to 1-2 sentences. "
        "Be factual; do not speculate about content not visible in the "
        "frame. If the frame is unrelated to the user's request, say so "
        "briefly."
    )


async def _fetch_descriptions(file_path: str, cleanup: bool = True,
                              prompt: str | None = None) -> dict:
    """Call whisper service /api/describe and return parsed result.

    `prompt` overrides the per-frame VLM prompt (default is the server's
    VLM_FRAME_PROMPT). Used by the user-prompt path to steer the VLM
    toward what the Discord user asked about.

    Raises PermanentError on 4xx, RuntimeError on 5xx/timeout/network.
    """
    assert http
    payload = {
        "file_path": file_path,
        "cleanup": cleanup,
        "fps_interval": VLM_FPS_INTERVAL,
        "max_frames": VLM_MAX_FRAMES,
    }
    if prompt:
        payload["prompt"] = prompt
    async with http.post(
        f"{WHISPER_API}/api/describe",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=VLM_TIMEOUT),
    ) as resp:
        if resp.status != 200:
            try:
                body = await resp.json()
            except Exception:
                body = {"error": await resp.text()}
            err = str(body.get("error", resp.status))
            if (400 <= resp.status < 500) or _is_permanent_remote_error(err):
                raise PermanentError(f"Describe rejected ({resp.status}): {err}")
            raise RuntimeError(f"Describe failed: {err}")
        return await resp.json()


def _format_descriptions(descriptions: list[dict]) -> str:
    """Render VLM frame descriptions as `[H:MM:SS] text` lines, matching the
    whisper transcript format exactly so existing summary prompts work
    unmodified."""
    lines = []
    for d in descriptions:
        ts = format_duration(int(d.get("timestamp", 0)))
        text = (d.get("text") or "").strip()
        if not text or text == "[frame description unavailable]":
            continue
        lines.append(f"[{ts}] {text}")
    return "\n".join(lines)


# Match the leading [H:MM:SS] / [MM:SS] timestamp on a transcript line.
_TS_LINE_RE = re.compile(r"^\[(\d{1,3}):(\d{2})(?::(\d{2}))?\]\s*(.*)$")


def _parse_ts(line: str) -> tuple[int, str] | None:
    """Return (seconds, rest_of_line) if line starts with a [H:MM:SS] / [MM:SS]
    marker, else None.
    """
    m = _TS_LINE_RE.match(line)
    if not m:
        return None
    a, b, c, _rest = m.groups()
    if c is not None:
        secs = int(a) * 3600 + int(b) * 60 + int(c)
    else:
        secs = int(a) * 60 + int(b)
    return secs, line


def _interleave_by_timestamp(speech_text: str, visual_text: str) -> str:
    """Merge speech + visual transcripts by timestamp.

    Both inputs are line-oriented `[H:MM:SS] text`. The merged output is
    chronologically ordered. Speech wins on tie (same second) — visual
    descriptions land between speech segments. Lines without parseable
    timestamps are appended in order at the end (rare; safety net).
    """
    speech_lines = [_parse_ts(l) for l in speech_text.splitlines() if l.strip()]
    visual_lines = [_parse_ts(l) for l in visual_text.splitlines() if l.strip()]
    speech_pairs = [(s, l) for x in speech_lines for (s, l) in [x] if x is not None]
    visual_pairs = [(s, l) for x in visual_lines for (s, l) in [x] if x is not None]
    # speech tagged 0, visual tagged 1 → stable sort puts speech before visual on tie
    merged = sorted(
        [(s, 0, l) for s, l in speech_pairs] +
        [(s, 1, l) for s, l in visual_pairs]
    )
    return "\n".join(line for (_s, _kind, line) in merged)


async def _cleanup_remote_file(file_path: str) -> None:
    """Best-effort cleanup of a file on the whisper container.

    Used when we asked /api/transcribe with cleanup=False (so the VLM had a
    chance to use the file) and now no longer need it. Idempotent on the
    server side; swallows errors here since failure just leaves a temp file
    that gets reaped at container restart.
    """
    if not file_path:
        return
    assert http
    try:
        async with http.post(
            f"{WHISPER_API}/api/cleanup",
            json={"file_path": file_path},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning("Remote cleanup non-200 for %s: %d %s",
                            file_path, resp.status, body[:200])
    except Exception as e:
        log.warning("Remote cleanup failed for %s: %s", file_path, e)


# ─── Output sanitization ──────────────────────────────────────────────────────

# Domains we trust to link out to. Only these can appear in posted output.
# Everything else gets stripped to prevent prompt-injection-driven phishing.
# Configurable via ALLOWED_LINK_HOSTS env (CSV).
_ALLOWED_LINK_HOSTS = _csv_env(
    "ALLOWED_LINK_HOSTS",
    "youtube.com,www.youtube.com,m.youtube.com,music.youtube.com,youtu.be,"
    "twitch.tv,clips.twitch.tv,vimeo.com",
)

# Markdown link [text](url) and bare URL patterns
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_BARE_URL_RE = re.compile(r"(?<![(\[])\bhttps?://[^\s)\]]+", re.IGNORECASE)


def _is_allowed_link(url: str) -> bool:
    """True if `url`'s host is in the allowlist."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return host.lower() in _ALLOWED_LINK_HOSTS
    except Exception:
        return False


def sanitize_llm_output(text: str) -> str:
    """Strip untrusted links from LLM output.

    Defense-in-depth against prompt-injection-driven phishing: an attacker who
    convinces the LLM to inject `[click here](https://evil.com)` into a
    summary would otherwise see Discord render it as a clickable link. We
    keep markdown links only when their target is on `_ALLOWED_LINK_HOSTS`,
    and demote bare URLs from non-allowed hosts to bracketed plain text.
    """
    def _replace_md(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if _is_allowed_link(url):
            return m.group(0)
        # Drop the link target; keep the visible text. If text is empty, fall
        # back to a domain marker so it's clear something was elided.
        return label or "[link removed]"

    text = _MD_LINK_RE.sub(_replace_md, text)

    def _replace_bare(m: re.Match) -> str:
        url = m.group(0)
        if _is_allowed_link(url):
            return url
        # Strip protocol so Discord doesn't auto-linkify; mark visibly.
        return "[link removed]"

    text = _BARE_URL_RE.sub(_replace_bare, text)
    return text


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
