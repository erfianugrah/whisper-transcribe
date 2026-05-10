"""
Discord TL;DW Bot — watches for YouTube links, transcribes via local
WhisperX service, summarizes via local LLM, posts result as embed.

Posts three embeds per video:
  - Brief: one-paragraph TL;DW
  - Key Points: bullet-point breakdown
  - Chapters: chronological section-by-section summary
"""

import asyncio
import threading
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
from discord import app_commands
from discord.ext import commands

class _JsonFormatter(logging.Formatter):
    """Single-line JSON per log record. Easy to grep, ship, or feed to a
    real log aggregator later. Adds any `extra=` fields verbatim."""

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        import json as _json
        from datetime import datetime, timezone
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                payload[k] = v
        return _json.dumps(payload, default=str, ensure_ascii=False)


_LOG_JSON = os.environ.get("LOG_JSON", "0").strip().lower() in {"1", "true", "yes", "on"}
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_root = logging.getLogger()
_root.setLevel(_LOG_LEVEL)
_handler = logging.StreamHandler()
if _LOG_JSON:
    _handler.setFormatter(_JsonFormatter())
else:
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    ))
_root.handlers = [_handler]
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
    channel: discord.TextChannel       # always present (where embeds post)
    submitter_id: int                  # Discord user.id of the requester
    submitter_name: str = ""           # for log lines / mentions
    user_prompt: str = ""              # steering text (see process())
    diarize: bool = False              # opt-in via /transcribe; default off
    model_override: str | None = None  # per-channel config can override LLM_MODEL
    # Source of the request. Exactly one of these is set; helpers in the
    # `_ack_*` family branch on which is non-None.
    message: discord.Message | None = None     # set when on_message-triggered
    interaction: discord.Interaction | None = None  # set when slash-triggered


# Maximum length of user-prompt text we'll honour (truncated above this).
# Keeps Discord-side lyrical messages from blowing the prompt budget on
# both VLM and summary calls. Picked to fit comfortably alongside the
# transcript content within LLM_INPUT_CHAR_BUDGET.
USER_PROMPT_MAX_CHARS = int(os.environ.get("USER_PROMPT_MAX_CHARS", "1500"))


# ─── Rate limiting ────────────────────────────────────────────────────────────


# Per-user sliding-window rate limit. Stored in-memory only — bot restart
# resets counters. Acceptable because (a) the queue capacity already bounds
# concurrent abuse, (b) restart-flooding is a different attack model that
# would need persistent storage to defend against.
MAX_JOBS_PER_USER_PER_HOUR = int(os.environ.get("MAX_JOBS_PER_USER_PER_HOUR", "5"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "20"))
# Discord user IDs (CSV) that bypass the per-user rate limit. Empty by default.
RATE_LIMIT_BYPASS_USERS = {
    int(x.strip()) for x in os.environ.get("RATE_LIMIT_BYPASS_USERS", "").split(",")
    if x.strip().isdigit()
}

from collections import deque, defaultdict

# Sliding-window store: deque per user. Single-threaded asyncio worker
# accesses these; no lock needed (cooperative scheduling, no preemption).
_user_jobs: dict[int, deque[float]] = defaultdict(deque)


def _rate_limit_check(user_id: int) -> tuple[bool, str]:
    """Returns (allowed, reason). `allowed=False` rejects with `reason`.

    Two checks:
      1. Total queue cap (independent of user) — protects against
         collective overload.
      2. Per-user sliding window (60 min) — protects against single-user
         spam.
    Bypass list (RATE_LIMIT_BYPASS_USERS) skips the per-user check but
    still enforces the queue cap (so admins can't crash the bot either).
    """
    if queue.qsize() >= MAX_QUEUE_SIZE:
        return False, (
            f"Queue is full ({queue.qsize()}/{MAX_QUEUE_SIZE} jobs pending). "
            f"Try again once existing jobs complete."
        )
    if user_id in RATE_LIMIT_BYPASS_USERS:
        return True, ""
    now = time.time()
    cutoff = now - 3600
    dq = _user_jobs[user_id]
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= MAX_JOBS_PER_USER_PER_HOUR:
        oldest_in_window = dq[0]
        retry_in = int((oldest_in_window + 3600) - now)
        mins = retry_in // 60
        return False, (
            f"Rate limit: {len(dq)}/{MAX_JOBS_PER_USER_PER_HOUR} jobs in the "
            f"last hour. Try again in ~{mins} min."
        )
    return True, ""


def _rate_limit_record(user_id: int) -> None:
    """Record that a job was just queued for this user."""
    _user_jobs[user_id].append(time.time())


# ─── Per-channel config ───────────────────────────────────────────────────────


# JSON config file kept in the bot-cache volume so it survives restarts.
# Schema: {channel_id_str: {"model": str, "vlm_enabled": bool, "diarize": bool}}
# Missing keys fall back to env defaults at job time.
CHANNELS_CONFIG_PATH = CACHE_DIR / "channels.json"
_channels_lock = threading.Lock()


def _load_channels_config() -> dict:
    """Read channels.json. Returns {} on missing/malformed file."""
    if not CHANNELS_CONFIG_PATH.exists():
        return {}
    try:
        return json_mod.loads(CHANNELS_CONFIG_PATH.read_text())
    except (OSError, json_mod.JSONDecodeError) as e:
        log.warning("channels.json read failed (%s) — treating as empty", e)
        return {}


def _save_channels_config(cfg: dict) -> None:
    with _channels_lock:
        try:
            CHANNELS_CONFIG_PATH.write_text(json_mod.dumps(cfg, indent=2, sort_keys=True))
        except OSError as e:
            log.error("channels.json write failed: %s", e)


def get_channel_config(channel_id: int) -> dict:
    """Look up a channel's overrides. Returns {} if no entry."""
    return _load_channels_config().get(str(channel_id), {})


def set_channel_config(channel_id: int, **fields) -> dict:
    """Update a channel's config; returns the merged result. Pass
    `field=None` to remove a key, or omit to leave unchanged.
    """
    cfg = _load_channels_config()
    entry = dict(cfg.get(str(channel_id), {}))
    for k, v in fields.items():
        if v is None:
            entry.pop(k, None)
        else:
            entry[k] = v
    if entry:
        cfg[str(channel_id)] = entry
    else:
        cfg.pop(str(channel_id), None)
    _save_channels_config(cfg)
    return entry


# ─── Job-source helpers (message vs. interaction) ────────────────────────────
# Jobs come from on_message reactions OR slash-command interactions. The
# helpers below abstract over both so the worker doesn't need to care.


async def _ack_queued(job: Job, position: int) -> None:
    """Acknowledge that a job has been queued."""
    if job.message is not None:
        await safe_react(job.message, "\u23f3")  # ⏳
    elif job.interaction is not None:
        try:
            await job.interaction.followup.send(
                f"Queued `{job.video_id}` (position {position} in queue).",
                ephemeral=False,
            )
        except discord.HTTPException as e:
            log.warning("Failed to send queue ack: %s", e)


async def _job_react(job: Job, emoji: str) -> None:
    if job.message is not None:
        await safe_react(job.message, emoji)


async def _job_remove_react(job: Job, emoji: str) -> None:
    if job.message is not None:
        await safe_remove_react(job.message, emoji)


async def _job_reply(job: Job, text: str) -> None:
    """Send a textual reply for failure / status messages."""
    if job.message is not None:
        try:
            await job.channel.send(text, reference=job.message)
        except discord.HTTPException as e:
            log.warning("reply failed: %s", e)
    elif job.interaction is not None:
        try:
            await job.interaction.followup.send(text)
        except discord.HTTPException as e:
            log.warning("interaction reply failed: %s", e)


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
    try:
        await _sync_slash_commands()
    except Exception as e:
        # Slash sync failure shouldn't prevent the legacy on_message
        # path from working — log and continue.
        log.error("Slash command sync failed: %s", e)
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

    # Per-channel config: model + diarize + vlm overrides
    chan_cfg = get_channel_config(message.channel.id)

    def _new_job(url, video_id):
        return Job(
            url=url, video_id=video_id,
            channel=message.channel,
            submitter_id=message.author.id,
            submitter_name=str(message.author),
            diarize=chan_cfg.get("diarize", False),
            model_override=chan_cfg.get("model"),
            message=message,
        )

    # YouTube URLs → extract video ID for timestamp linking.
    for m in YT_PATTERN.finditer(message.content):
        all_urls.append(m.group(0))
        video_id = m.group(1)
        if video_id in seen:
            continue
        seen.add(video_id)
        url = f"https://www.youtube.com/watch?v={video_id}"
        jobs_to_queue.append(_new_job(url, video_id))

    # Other video platform URLs
    for url_match in VIDEO_URL_PATTERN.finditer(message.content):
        url = url_match.group(1)
        all_urls.append(url)
        if any(d in url for d in ("youtube.com", "youtu.be")):
            continue
        path_parts = [p for p in url.rstrip("/").split("/")
                      if p and "." not in p and "//" not in p]
        vid = re.sub(r"[^\w-]", "", path_parts[-1])[:20] if path_parts else "unknown"
        if vid in seen:
            continue
        seen.add(vid)
        jobs_to_queue.append(_new_job(url, vid))

    if not jobs_to_queue:
        await bot.process_commands(message)
        return

    user_prompt = _extract_user_prompt(message.content, all_urls)
    if user_prompt:
        for job in jobs_to_queue:
            job.user_prompt = user_prompt
        log.info("User prompt detected (%d chars): %s",
                 len(user_prompt), user_prompt[:80])

    # Rate-limit check happens per-message — refuse the whole batch if
    # the user has hit the cap, even if they posted multiple URLs at once.
    ok, reason = _rate_limit_check(message.author.id)
    if not ok:
        await safe_react(message, "\U0001f6ab")  # 🚫
        try:
            await message.channel.send(
                f"❌ {message.author.mention} {reason}",
                reference=message, mention_author=False,
            )
        except discord.HTTPException:
            pass
        return

    for job in jobs_to_queue:
        _rate_limit_record(job.submitter_id)
        await queue.put(job)
        await _ack_queued(job, queue.qsize())
        log.info("Queued %s from %s (channel=%s)",
                 job.video_id, job.submitter_name, message.channel.id)


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
            for emoji in PROCESSING_EMOJI:
                await _job_remove_react(job, emoji)
            await _job_react(job, "\u274c")  # ❌
            await _job_reply(
                job,
                f"Failed to process `{job.video_id}` ({attempts}): "
                f"{type(last_error).__name__}: {last_error}",
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
        await _job_react(job, "\U0001f3a7")  # 🎧

        transcribe_payload = {
            "file_path": file_path,
            "model": WHISPER_MODEL,
            # Don't cleanup yet — VLM fallback (below) may need the file.
            "cleanup": False,
            "return_file": False,  # bot uses transcript text directly
            "diarize": job.diarize,
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
    await _job_react(job, "\U0001f9e0")  # 🧠

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

    # Apply per-channel model override for the duration of this job's
    # summarize calls. ContextVar isolates per-task so concurrent jobs
    # don't trample each other.
    _token = _model_override.set(job.model_override) if job.model_override else None
    try:
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
    finally:
        if _token is not None:
            _model_override.reset(_token)
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
        # Build the "Requested by …" line; works for both message + slash sources
        if job.message is not None:
            requester = job.message.author.mention
        elif job.interaction is not None:
            requester = job.interaction.user.mention
        else:
            requester = f"<@{job.submitter_id}>"
        header = discord.Embed(
            title=f"{truncate(title, 240)}",
            url=job.url,
            description=f"Requested by {requester} in <#{job.channel.id}>",
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

    if job.user_prompt:
        embed.add_field(
            name="User request",
            value=truncate(job.user_prompt, 1000),
            inline=False,
        )

    if use_split and detail_msg:
        embed.add_field(
            name="",
            value=f"[Full breakdown →]({detail_msg.jump_url})",
            inline=False,
        )

    # Attach a "Rename speakers" button when diarization labels are present.
    view = None
    if job.diarize and _has_speaker_labels(transcript):
        view = SpeakerRenameView(job_video_id=job.video_id, channel_id=job.channel.id)

    if job.message is not None:
        await job.channel.send(embed=embed, reference=job.message, view=view)
    else:
        await job.channel.send(embed=embed, view=view)

    for emoji in PROCESSING_EMOJI:
        await _job_remove_react(job, emoji)
    await _job_react(job, "\u2705")  # ✅

    log.info("[%s] Done — posted 3 embeds", job.video_id)


# Per-task model override (set by process() per Job). Reads at _llm_call
# time via .get(). ContextVar lets us override without threading a `model`
# kwarg through every recursive summarize() call.
import contextvars
_model_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_model_override", default=None,
)


async def _llm_call(prompt: str, max_tokens: int) -> str:
    """One LLM chat-completion request.

    Raises PermanentError on 4xx (won't recover on retry); RuntimeError on 5xx
    or transport errors (worker will retry).
    """
    assert http
    payload = {
        "model": _model_override.get() or LLM_MODEL,
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


# ─── Speaker rename UI (Button + Modal) ──────────────────────────────────────


# Match `[SPEAKER_xx]` / `[F-SPEAKER_xx]` etc. labels in transcripts.
_SPEAKER_TAG_RE = re.compile(r"\[([A-Z]?-?SPEAKER_\d{1,3})\]")


def _has_speaker_labels(transcript: str) -> bool:
    return bool(_SPEAKER_TAG_RE.search(transcript))


def _extract_speaker_labels(transcript: str) -> list[str]:
    """Return unique speaker tags in document order (max 5 — Modal cap)."""
    seen: list[str] = []
    for tag in _SPEAKER_TAG_RE.findall(transcript):
        if tag not in seen:
            seen.append(tag)
        if len(seen) >= 5:  # Modal supports max 5 TextInputs
            break
    return seen


class SpeakerRenameModal(discord.ui.Modal, title="Rename speakers"):
    """Modal with one TextInput per speaker. Submitting kicks off a rerun."""

    def __init__(self, video_id: str, channel_id: int, speakers: list[str]):
        super().__init__(timeout=600)
        self._video_id = video_id
        self._channel_id = channel_id
        self._speakers = speakers
        self._inputs: list[discord.ui.TextInput] = []
        for sp in speakers:
            field = discord.ui.TextInput(
                label=sp[:45],
                placeholder=f"Real name for {sp}",
                required=False,
                max_length=64,
            )
            self.add_item(field)
            self._inputs.append(field)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        renames = {
            sp: inp.value.strip()
            for sp, inp in zip(self._speakers, self._inputs)
            if inp.value and inp.value.strip()
        }
        if not renames:
            await interaction.response.send_message(
                "No names entered — skipping rename.", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        try:
            await _apply_rename(interaction, self._video_id, self._channel_id, renames)
        except Exception as e:
            log.error("Rename apply failed: %s", e)
            await interaction.followup.send(
                f"Rename failed: {type(e).__name__}: {e}", ephemeral=True
            )


class SpeakerRenameView(discord.ui.View):
    """Persistent View attached to a brief embed when diarize=True."""

    def __init__(self, job_video_id: str, channel_id: int):
        super().__init__(timeout=None)  # persistent until restart
        self._video_id = job_video_id
        self._channel_id = channel_id

    @discord.ui.button(label="Rename speakers", style=discord.ButtonStyle.secondary, emoji="🏷️")
    async def rename_button(
        self, interaction: discord.Interaction, button: discord.ui.Button,
    ) -> None:
        cached = read_cache(self._video_id)
        if cached is None:
            await interaction.response.send_message(
                "Transcript no longer in cache — can't rename.", ephemeral=True
            )
            return
        _, _, transcript, _ = cached
        speakers = _extract_speaker_labels(transcript)
        if not speakers:
            await interaction.response.send_message(
                "No speaker labels found in this transcript.", ephemeral=True
            )
            return
        await interaction.response.send_modal(
            SpeakerRenameModal(self._video_id, self._channel_id, speakers)
        )


async def _apply_rename(
    interaction: discord.Interaction,
    video_id: str,
    channel_id: int,
    renames: dict,
) -> None:
    """Apply rename map to cached transcript + post a fresh brief embed.

    Strategy: text-replace `[SPEAKER_xx]` → `[NewName]` in the cached
    transcript, write back to cache, then re-summarize the brief only
    (key_points/chapters keep their original labels — they'd require
    re-summarisation on the LLM, costly).
    """
    cached = read_cache(video_id)
    if cached is None:
        await interaction.followup.send("Transcript expired.", ephemeral=True)
        return
    title, status, transcript, duration = cached

    # Apply renames atomically (longest first to avoid prefix collisions).
    for old, new in sorted(renames.items(), key=lambda kv: -len(kv[0])):
        transcript = transcript.replace(f"[{old}]", f"[{new}]")
    write_cache(video_id, title, status, transcript, duration)

    # Re-summarize brief only (cheapest; gives users immediate feedback).
    duration_str = format_duration(duration)
    brief = await summarize(
        transcript, PROMPT_BRIEF, LLM_MAX_TOKENS_BRIEF,
        reduce_template=REDUCE_BRIEF,
        title=title, duration=duration_str, reference_block="",
    )
    brief = sanitize_llm_output(brief)

    embed = discord.Embed(
        title=f"TL;DW (renamed): {truncate(title, 240)}",
        description=truncate(brief, 4000),
        color=0xFF0000,
    )
    embed.set_footer(text=f"{duration_str} | renamed: {', '.join(renames.keys())}")
    channel = bot.get_channel(channel_id)
    if channel is not None:
        await channel.send(embed=embed)
    await interaction.followup.send(
        f"Renamed: {', '.join(f'{o}→{n}' for o, n in renames.items())}",
        ephemeral=True,
    )


# ─── Slash commands ──────────────────────────────────────────────────────────


# Optional guild-scoped sync (instant). Set DISCORD_GUILD_ID for fast
# iteration during development; leave unset for global commands (1h
# propagation, but works in DMs and across all servers).
DISCORD_GUILD_ID = int(os.environ.get("DISCORD_GUILD_ID", "0"))


def _job_from_interaction(
    interaction: discord.Interaction,
    url: str,
    *,
    user_prompt: str = "",
    diarize: bool = False,
    model_override: str | None = None,
) -> Job | None:
    """Build a Job from an interaction. Returns None on URL parse failure."""
    # Try YouTube first for canonical video_id
    m = YT_PATTERN.search(url)
    if m:
        video_id = m.group(1)
        canonical = f"https://www.youtube.com/watch?v={video_id}"
        return Job(
            url=canonical, video_id=video_id,
            channel=interaction.channel, submitter_id=interaction.user.id,
            submitter_name=str(interaction.user),
            user_prompt=user_prompt, diarize=diarize,
            model_override=model_override, interaction=interaction,
        )
    # Fallback for other platforms
    m = VIDEO_URL_PATTERN.search(url)
    if not m:
        return None
    full_url = m.group(1)
    path_parts = [p for p in full_url.rstrip("/").split("/")
                  if p and "." not in p and "//" not in p]
    vid = re.sub(r"[^\w-]", "", path_parts[-1])[:20] if path_parts else "unknown"
    return Job(
        url=full_url, video_id=vid,
        channel=interaction.channel, submitter_id=interaction.user.id,
        submitter_name=str(interaction.user),
        user_prompt=user_prompt, diarize=diarize,
        model_override=model_override, interaction=interaction,
    )


@bot.tree.command(name="summarize", description="Transcribe + summarise a video")
@app_commands.describe(
    url="Video URL (YouTube, Twitch, Vimeo, etc.)",
    prompt="Optional steering: tell the bot what to focus on (forces VLM)",
    model="Override LLM_MODEL for this run (advanced)",
)
async def cmd_summarize(
    interaction: discord.Interaction,
    url: str,
    prompt: str | None = None,
    model: str | None = None,
) -> None:
    await interaction.response.defer()
    ok, reason = _rate_limit_check(interaction.user.id)
    if not ok:
        await interaction.followup.send(f"❌ {reason}", ephemeral=True)
        return

    chan_cfg = get_channel_config(interaction.channel.id)
    effective_model = model or chan_cfg.get("model")
    job = _job_from_interaction(
        interaction, url,
        user_prompt=(prompt or "").strip()[:USER_PROMPT_MAX_CHARS],
        model_override=effective_model,
    )
    if job is None:
        await interaction.followup.send(
            f"❌ Couldn't parse a supported video URL from: {url}", ephemeral=True
        )
        return
    _rate_limit_record(interaction.user.id)
    await queue.put(job)
    await _ack_queued(job, queue.qsize())
    log.info("Slash /summarize queued %s from %s (model=%s, prompt=%s)",
             job.video_id, job.submitter_name,
             effective_model or "default", bool(job.user_prompt))


@bot.tree.command(name="transcribe", description="Transcribe a video (no summary), with optional speaker diarization")
@app_commands.describe(
    url="Video URL",
    diarize="Identify and label different speakers (slower; adds rename button)",
)
async def cmd_transcribe(
    interaction: discord.Interaction,
    url: str,
    diarize: bool = False,
) -> None:
    # /transcribe is just /summarize with diarize on. We still produce
    # the brief/key_points/chapters embeds — the diarize flag flows
    # through to whisper for labelling.
    await interaction.response.defer()
    ok, reason = _rate_limit_check(interaction.user.id)
    if not ok:
        await interaction.followup.send(f"❌ {reason}", ephemeral=True)
        return

    chan_cfg = get_channel_config(interaction.channel.id)
    job = _job_from_interaction(
        interaction, url,
        diarize=diarize or chan_cfg.get("diarize", False),
        model_override=chan_cfg.get("model"),
    )
    if job is None:
        await interaction.followup.send(
            f"❌ Couldn't parse a supported video URL from: {url}", ephemeral=True
        )
        return
    _rate_limit_record(interaction.user.id)
    await queue.put(job)
    await _ack_queued(job, queue.qsize())
    log.info("Slash /transcribe queued %s from %s (diarize=%s)",
             job.video_id, job.submitter_name, job.diarize)


@bot.tree.command(name="status", description="Show queue + service health")
async def cmd_status(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    assert http
    try:
        async with http.get(f"{WHISPER_API}/api/status",
                            timeout=aiohttp.ClientTimeout(total=5)) as resp:
            wstatus = await resp.json() if resp.status == 200 else {"status": "down"}
    except Exception as e:
        wstatus = {"status": f"unreachable: {e}"}

    user_dq = _user_jobs.get(interaction.user.id, deque())
    msg = (
        f"**Queue**: {queue.qsize()}/{MAX_QUEUE_SIZE} jobs\n"
        f"**Your usage**: {len(user_dq)}/{MAX_JOBS_PER_USER_PER_HOUR} in the last hour\n"
        f"**Whisper**: {wstatus.get('status', '?')} on {wstatus.get('device', '?')}\n"
        f"**Vision**: {wstatus.get('vision', {}).get('model', 'unconfigured')}"
    )
    await interaction.followup.send(msg, ephemeral=True)


@bot.tree.command(name="find", description="Search past summaries by keyword")
@app_commands.describe(query="Keywords to search for (case-insensitive substring)")
async def cmd_find(interaction: discord.Interaction, query: str) -> None:
    await interaction.response.defer(ephemeral=True)
    q = query.lower().strip()
    if len(q) < 3:
        await interaction.followup.send("Query must be ≥3 characters.", ephemeral=True)
        return

    matches = []
    for f in CACHE_DIR.glob("*.txt"):
        try:
            text = f.read_text()
        except OSError:
            continue
        if q in text.lower():
            # Pull the title from header line: "# title: ..."
            title = ""
            for line in text.splitlines()[:5]:
                if line.startswith("# title: "):
                    title = line[len("# title: "):]
                    break
            video_id = f.stem
            url = f"https://www.youtube.com/watch?v={video_id}"  # YT-style; non-YT works too
            matches.append((video_id, title or video_id, url))
            if len(matches) >= 10:
                break

    if not matches:
        await interaction.followup.send(f"No matches for `{query}`.", ephemeral=True)
        return

    lines = [f"Found {len(matches)} match{'es' if len(matches) != 1 else ''} for `{query}`:"]
    for vid, title, url in matches:
        lines.append(f"- [{truncate(title, 80)}]({url}) (`{vid}`)")
    await interaction.followup.send("\n".join(lines), ephemeral=True)


@bot.tree.command(name="config", description="Configure this channel's bot defaults (admin)")
@app_commands.describe(
    model="Default LLM model id for this channel (empty = clear override)",
    vlm="Force VLM enrichment for every video in this channel",
    diarize="Enable speaker diarization by default in this channel",
    show="Just print the current config without changing it",
)
async def cmd_config(
    interaction: discord.Interaction,
    model: str | None = None,
    vlm: bool | None = None,
    diarize: bool | None = None,
    show: bool = False,
) -> None:
    # Discord-side permission check: requires Manage Channel.
    perms = interaction.channel.permissions_for(interaction.user)
    if not perms.manage_channels:
        await interaction.response.send_message(
            "❌ This command requires the **Manage Channel** permission.",
            ephemeral=True,
        )
        return
    if show or (model is None and vlm is None and diarize is None):
        cfg = get_channel_config(interaction.channel.id)
        if not cfg:
            txt = "No overrides for this channel — using global defaults."
        else:
            txt = "Current config:\n" + "\n".join(f"  **{k}**: `{v}`" for k, v in cfg.items())
        await interaction.response.send_message(txt, ephemeral=True)
        return

    fields: dict = {}
    if model is not None:
        fields["model"] = model.strip() or None  # empty string clears
    if vlm is not None:
        fields["vlm_enabled"] = vlm
    if diarize is not None:
        fields["diarize"] = diarize
    new_cfg = set_channel_config(interaction.channel.id, **fields)
    if new_cfg:
        txt = "Updated config:\n" + "\n".join(f"  **{k}**: `{v}`" for k, v in new_cfg.items())
    else:
        txt = "Config cleared — using global defaults."
    await interaction.response.send_message(txt, ephemeral=True)
    log.info("Channel %s config updated by %s: %s",
             interaction.channel.id, interaction.user, fields)


# Sync commands on startup. Called once after the worker is up.
async def _sync_slash_commands():
    if DISCORD_GUILD_ID:
        guild = discord.Object(id=DISCORD_GUILD_ID)
        bot.tree.copy_global_to(guild=guild)
        synced = await bot.tree.sync(guild=guild)
        log.info("Slash commands synced to guild %d (%d commands)",
                 DISCORD_GUILD_ID, len(synced))
    else:
        synced = await bot.tree.sync()
        log.info("Slash commands synced globally (%d commands; ~1 hour propagation)",
                 len(synced))


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)
