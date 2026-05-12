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

    Production note: when the bot runs under `compose env_file: ./bot/.env`,
    Compose has already loaded those values into the container's process
    env before this code executes — so this loader becomes a no-op (every
    setdefault hits an existing key). It exists for non-Docker dev runs
    (`python bot/main.py` from the repo root with a sibling .env file).
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
SCRAPER_API = os.environ.get("SCRAPER_API_URL", "http://localhost:11235")
FLARESOLVERR_API = os.environ.get("FLARESOLVERR_API_URL",
                                  "http://localhost:8191/v1")
SCRAPER_TIMEOUT = int(os.environ.get("SCRAPER_TIMEOUT", "120"))
FLARESOLVERR_TIMEOUT = int(os.environ.get("FLARESOLVERR_TIMEOUT", "90"))
# Scraped article body cap. Articles longer than this hit map-reduce in
# summarize() — the budget calc applies as it does for transcripts.
SCRAPED_BODY_CHAR_CAP = int(os.environ.get("SCRAPED_BODY_CHAR_CAP", "200000"))
LLM_API = os.environ.get("LLM_API_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-26B-A4B-it-Q4_K_M")
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

# ─── Server-side job queue ────────────────────────────────────────────────────
# Whisper now exposes a Valkey-backed FIFO queue at /api/jobs*. The bot
# submits via POST /api/jobs (returns 202 + job_id) and polls
# GET /api/jobs/{id} until the job is terminal. The queue serialises
# across all consumers (us, MCP, Gradio UI, ad-hoc curl) so there's no
# busy-wait dance against 409s.
#
# JOB_POLL_INTERVAL — seconds between status polls while a job is queued
# or running. 3s is a good tradeoff: fast enough that ⏳ → 🎧 transitions
# feel snappy, slow enough that we don't hammer /api/jobs/{id} for a
# multi-hour transcription.
JOB_POLL_INTERVAL = int(os.environ.get("JOB_POLL_INTERVAL", "3"))

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
LLM_MAX_TOKENS_CHAPTERS = int(os.environ.get("LLM_MAX_TOKENS_CHAPTERS", "5000"))

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

# Smart VLM gates — density alone misfires on long-form content (livestreams,
# podcasts, lectures) where natural quiet stretches drag the average down
# even when the absolute speech content is plenty. Any of these guards
# triggering will skip VLM (unless the user explicitly forces it via prompt).
#
# Livestreams: yt-dlp tags VOD'd streams with `was_live=true`. Streamer
# voice carries the meaning even when density looks "sparse"; gameplay/music
# gaps aren't "silent video" in the VLM sense.
VLM_SKIP_LIVESTREAMS = os.environ.get("VLM_SKIP_LIVESTREAMS", "1").strip().lower() in {"1", "true", "yes", "on"}
# Absolute transcript size — if there's already this much text, VLM frame
# descriptions add diminishing returns. 25k chars ≈ 5k tokens, plenty for
# a meaningful summary. Default sized to match a 35-min talk at normal pace.
#
# CRITICAL: this is intentionally the *only* speech-sufficiency gate.
# A naive duration cap ("skip VLM if video > 90min") would blackhole
# genuinely silent long content (8h ASMR, hour-long music videos, etc).
# Long + low-text means VLM is *more* needed, not less.
VLM_MIN_TEXT_CHARS = int(os.environ.get("VLM_MIN_TEXT_CHARS", "25000"))

# ─── YouTube comments ────────────────────────────────────────────────────────
# When enabled, the bot asks the whisper service to fetch top YT comments
# alongside the video download (yt-dlp --get-comments). Comments are filtered
# (substantive + creator-hearted prioritised) and summarised into a
# "Community reaction" embed posted alongside the existing brief / key_points
# / chapters embeds. On by default; channels can opt out via /config.
YT_COMMENTS_ENABLED = os.environ.get("YT_COMMENTS_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
YT_COMMENTS_MAX = int(os.environ.get("YT_COMMENTS_MAX", "100"))
YT_COMMENTS_SORT = os.environ.get("YT_COMMENTS_SORT", "top")
# Minimum comment text length (chars) to count as substantive. Below this,
# a comment is almost certainly noise: "first", "lol", emoji-only.
YT_COMMENT_MIN_CHARS = int(os.environ.get("YT_COMMENT_MIN_CHARS", "40"))
# Top-N substantive comments fed to the LLM after filtering+sorting.
YT_COMMENT_SUMMARY_TOP_N = int(os.environ.get("YT_COMMENT_SUMMARY_TOP_N", "30"))

# Upper bound on submit-and-poll-against-/api/jobs. Long videos on slow
# models can exceed the global session timeout (900s); this caps the total
# time the bot waits for whisper before giving up the poll loop and letting
# the worker either retry or fail the job. The server keeps running it
# regardless — we just stop listening.
TRANSCRIBE_TIMEOUT = int(os.environ.get("TRANSCRIBE_TIMEOUT", "1800"))

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
    PROMPT_BRIEF_WEB, PROMPT_KEY_POINTS_WEB, PROMPT_SECTIONS,
    REDUCE_BRIEF_WEB, REDUCE_KEY_POINTS_WEB, REDUCE_SECTIONS,
    PROMPT_BRIEF_REDDIT, PROMPT_KEY_POINTS_REDDIT, PROMPT_SECTIONS_REDDIT,
    REDUCE_BRIEF_REDDIT, REDUCE_KEY_POINTS_REDDIT, REDUCE_SECTIONS_REDDIT,
    PROMPT_YT_COMMENTS, REDUCE_YT_COMMENTS,
    PROMPT_BRIEF_SILENT, PROMPT_KEY_POINTS_SILENT, PROMPT_CHAPTERS_SILENT,
    REDUCE_BRIEF_SILENT, REDUCE_KEY_POINTS_SILENT,
    PROMPT_LITMUS,
    CHUNK_PREAMBLE,
)


# ─── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    url: str
    video_id: str  # also used as cache key for web jobs (URL-hash for kind="web")
    # `Messageable` covers TextChannel, DMChannel, Thread — anything we can
    # `.send()` to. Slash commands in DMs land here with a DMChannel; on_message
    # in DMs would too if we ever drop the guild gate. resolve_summary_channel
    # already handles the no-guild case.
    channel: "discord.abc.Messageable"
    submitter_id: int                  # Discord user.id of the requester
    submitter_name: str = ""           # for log lines / mentions
    user_prompt: str = ""              # steering text (see process())
    diarize: bool = False              # opt-in via /transcribe; default off
    vlm_enabled: bool = True           # per-channel override; falls back to global VLM_ENABLED
    yt_comments_enabled: bool = True   # per-channel override; falls back to global YT_COMMENTS_ENABLED
    model_override: str | None = None  # per-channel config can override LLM_MODEL
    # Job kind: "video" (default — yt-dlp + whisper + summarize) or "web"
    # (crawl4ai + summarize). Worker dispatches on this discriminant.
    kind: str = "video"
    # True when the user explicitly asked for a summary (reply-trigger,
    # slash command). False when the URL was just posted in a watched
    # channel and the bot picked it up automatically. Controls failure
    # behaviour: explicit jobs that turn out not to be video URLs fall
    # through to the web pipeline; auto-paste jobs fail silently to avoid
    # spamming the channel with "this isn't a video" messages every time
    # someone shares a Reddit text post.
    explicit_request: bool = False
    # Source of the request. Exactly one of these is set; helpers in the
    # `_ack_*` family branch on which is non-None. Validated in __post_init__.
    message: discord.Message | None = None     # set when on_message-triggered
    interaction: discord.Interaction | None = None  # set when slash-triggered

    def __post_init__(self) -> None:
        # Discriminated-union invariant: exactly one source.
        if (self.message is None) == (self.interaction is None):
            raise ValueError(
                "Job must have exactly one of message/interaction set "
                "(got message=%r, interaction=%r)" % (
                    self.message is not None, self.interaction is not None,
                )
            )
        if self.kind not in ("video", "web", "litmus"):
            raise ValueError(
                f"Job.kind must be 'video' | 'web' | 'litmus', got {self.kind!r}"
            )


# Maximum length of user-prompt text we'll honour (truncated above this).
# Keeps Discord-side lyrical messages from blowing the prompt budget on
# both VLM and summary calls. Picked to fit comfortably alongside the
# transcript content within LLM_INPUT_CHAR_BUDGET.
USER_PROMPT_MAX_CHARS = int(os.environ.get("USER_PROMPT_MAX_CHARS", "1500"))

# Below this many non-whitespace chars, the message-text-minus-URLs is
# considered too short to be meaningful steering (e.g. a bare "lol" or just
# punctuation). Caller falls through to default automatic processing.
USER_PROMPT_MIN_CHARS = 3


# ─── Rate limiting ────────────────────────────────────────────────────────────


# Per-user sliding-window rate limit. Stored in-memory only — bot restart
# resets counters. Acceptable because (a) the queue capacity already bounds
# concurrent abuse, (b) restart-flooding is a different attack model that
# would need persistent storage to defend against.
# Per-user sliding-window cap. Default 20 because the chained-reply flow
# (`tldr litmus` queues 2 jobs at once) burns through smaller caps quickly,
# especially when users iterate on a few videos in a session. Raise via
# env or use RATE_LIMIT_BYPASS_USERS for admins.
MAX_JOBS_PER_USER_PER_HOUR = int(os.environ.get("MAX_JOBS_PER_USER_PER_HOUR", "20"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "40"))
# Discord user IDs (CSV) that bypass the per-user rate limit. Empty by default.
RATE_LIMIT_BYPASS_USERS = {
    int(x.strip()) for x in os.environ.get("RATE_LIMIT_BYPASS_USERS", "").split(",")
    if x.strip().isdigit()
}

from collections import deque, defaultdict

# Sliding-window store: deque per user. Single-threaded asyncio worker
# accesses these; no lock needed (cooperative scheduling, no preemption).
_user_jobs: dict[int, deque[float]] = defaultdict(deque)


def _rate_limit_check(user_id: int, count: int = 1) -> tuple[bool, str]:
    """Returns (allowed, reason). `allowed=False` rejects with `reason`.

    Two checks:
      1. Total queue cap (independent of user) — protects against
         collective overload.
      2. Per-user sliding window (60 min) — protects against single-user
         spam.
    Bypass list (RATE_LIMIT_BYPASS_USERS) skips the per-user check but
    still enforces the queue cap (so admins can't crash the bot either).

    `count` is the number of jobs the caller wants to enqueue atomically
    (e.g. chained `tldr litmus` reply requests two jobs in one go). The
    cap check uses `count` so we reject all-or-nothing rather than
    partial-fail mid-batch.
    """
    if queue.qsize() + count > MAX_QUEUE_SIZE:
        return False, (
            f"Queue is full — would push {queue.qsize()}+{count} past "
            f"{MAX_QUEUE_SIZE}. Try again once existing jobs complete."
        )
    if user_id in RATE_LIMIT_BYPASS_USERS:
        return True, ""
    now = time.time()
    cutoff = now - 3600
    dq = _user_jobs[user_id]
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) + count > MAX_JOBS_PER_USER_PER_HOUR:
        oldest_in_window = dq[0] if dq else now
        retry_in = int((oldest_in_window + 3600) - now)
        mins = max(0, retry_in // 60)
        return False, (
            f"Rate limit: {len(dq)}+{count} jobs would exceed "
            f"{MAX_JOBS_PER_USER_PER_HOUR}/hour. Try again in ~{mins} min."
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


# ─── Per-guild config (server-wide overrides) ────────────────────────────────


# Same shape as channels.json but keyed by guild_id. Used for server-wide
# settings that don't make sense at the channel level (e.g. a single
# "summaries archive" channel for the whole server).
GUILDS_CONFIG_PATH = CACHE_DIR / "guilds.json"
_guilds_lock = threading.Lock()


def _load_guilds_config() -> dict:
    if not GUILDS_CONFIG_PATH.exists():
        return {}
    try:
        return json_mod.loads(GUILDS_CONFIG_PATH.read_text())
    except (OSError, json_mod.JSONDecodeError) as e:
        log.warning("guilds.json read failed (%s) — treating as empty", e)
        return {}


def _save_guilds_config(cfg: dict) -> None:
    with _guilds_lock:
        try:
            GUILDS_CONFIG_PATH.write_text(json_mod.dumps(cfg, indent=2, sort_keys=True))
        except OSError as e:
            log.error("guilds.json write failed: %s", e)


def get_guild_config(guild_id: int) -> dict:
    """Look up server-wide overrides. Returns {} if no entry."""
    return _load_guilds_config().get(str(guild_id), {})


def set_guild_config(guild_id: int, **fields) -> dict:
    """Update a guild's config; returns the merged result. Pass field=None
    to remove a key.
    """
    cfg = _load_guilds_config()
    entry = dict(cfg.get(str(guild_id), {}))
    for k, v in fields.items():
        if v is None:
            entry.pop(k, None)
        else:
            entry[k] = v
    if entry:
        cfg[str(guild_id)] = entry
    else:
        cfg.pop(str(guild_id), None)
    _save_guilds_config(cfg)
    return entry


def resolve_summary_channel(job_channel: "discord.TextChannel") -> "discord.TextChannel":
    """Pick the right channel for detail embeds (Key Points + Chapters).

    Precedence (most specific → least):
      1. Per-guild override (`guilds.json[<gid>].summary_channel`).
      2. Global env (`SUMMARY_CHANNEL`) — ONLY if same guild as the job.
      3. Original channel where the request came in.

    Both overrides are guarded by a guild-affinity check: if the configured
    summary channel lives in a different server than the job, fall through
    rather than leak detail embeds across server boundaries. This makes
    `SUMMARY_CHANNEL` safe to set even on a multi-server bot — it'll only
    route inside the one server that contains it.
    """
    guild = getattr(job_channel, "guild", None)
    if guild is None:
        # DM / no-guild context — nowhere meaningful to redirect to.
        return job_channel

    def _same_guild(channel) -> bool:
        cg = getattr(channel, "guild", None)
        return cg is not None and cg.id == guild.id

    # Per-guild override
    sc_id = get_guild_config(guild.id).get("summary_channel")
    if sc_id:
        ch = bot.get_channel(int(sc_id))
        if ch is not None and _same_guild(ch):
            return ch
        if ch is not None:
            log.warning(
                "Per-guild summary_channel %s is in a different guild than "
                "the job (%s vs %s) — ignoring; this should never happen "
                "since /serverconfig validates guild affinity.",
                sc_id, getattr(getattr(ch, "guild", None), "id", "?"), guild.id,
            )

    # Global env fallback — must also match guild
    if SUMMARY_CHANNEL:
        ch = bot.get_channel(SUMMARY_CHANNEL)
        if ch is not None and _same_guild(ch):
            return ch
        # Different guild → silently fall through (operator may have set
        # SUMMARY_CHANNEL for one specific server's archive; jobs from
        # other servers stay in-channel).

    return job_channel


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
    # Strip any http(s):// URL whole — up to next whitespace. The `urls`
    # list contains URLs as YT_PATTERN / VIDEO_URL_PATTERN matched them,
    # which can stop short of the full URL (e.g. YT_PATTERN ends at the
    # 11-char video ID, leaving query strings like `&pp=ygUJQXNt...` in
    # the message). Without this whole-URL strip, those tails get treated
    # as user prompts and pollute the VLM / summary prompts.
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    # Drop Discord mentions, channel refs, custom emoji shorthand
    text = re.sub(r"<[@#:][^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < USER_PROMPT_MIN_CHARS:
        return ""
    return text[:USER_PROMPT_MAX_CHARS]


# ─── Bot ──────────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
queue: asyncio.Queue[Job] = asyncio.Queue()
# Shared session; populated in on_ready(). Each network helper guards with an
# explicit `if http is None: raise` so the check survives `python -O`.
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


# ─── Reply-trigger ("tldr"/"summarize") for URL summaries ────────────────────
# Users post a URL → reply "tldr" → bot scrapes + summarises that URL.
# Strict matching: message body must be ONLY the keyword (with optional
# trailing punctuation), to avoid accidental triggers in a normal sentence
# like "give me a tldr of …".
REPLY_TRIGGER_RE = re.compile(r"^\s*(tldr|summarize|summarise)[!.\s]*$",
                              re.IGNORECASE)

# Reply-trigger for the AI litmus test: surfaces stylistic + metadata
# signals that an article may be LLM-generated or LLM-heavy-edited. Same
# strict match — only fires when the reply body is JUST the keyword.
LITMUS_TRIGGER_RE = re.compile(r"^\s*litmus[!.?\s]*$", re.IGNORECASE)

# Canonical mapping for chained replies. Keys are exact lowercase keywords
# the user might type; values are the kind_hint passed to the dispatch.
# Same kind_hint may have multiple keyword aliases (tldr/summarize/summarise).
_TRIGGER_KEYWORD_MAP = {
    "tldr": "summary",
    "summarize": "summary",
    "summarise": "summary",
    "litmus": "litmus",
}


def _parse_trigger_keywords(content: str) -> list[str]:
    """Return the list of distinct kind_hints when a reply body is composed
    ENTIRELY of recognised trigger keywords (any order, any number,
    optional punctuation between them). Else empty list.

    Examples:
      'tldr'             → ['summary']
      'TLDR.'            → ['summary']
      'tldr litmus'      → ['summary', 'litmus']
      'litmus tldr'      → ['litmus', 'summary']  (preserves order)
      'tldr tldr'        → ['summary']             (deduplicated)
      'summarize litmus' → ['summary', 'litmus']
      'give me a tldr'   → []  (extra non-keyword words → no fire)
      ''                 → []

    Order is preserved so chained dispatch matches the user's typing intent
    (e.g. `tldr litmus` queues summary first, then litmus). Dedup prevents
    double-charging the rate limit when a user accidentally types
    `tldr tldr`.
    """
    if not content or not content.strip():
        return []
    # Tokenise on whitespace + punctuation. \w includes digits + underscores
    # so "tldr 123" produces ['tldr', '123'] and `123` (not a keyword) falls
    # into the reject branch — keeps the trigger strict.
    tokens = re.findall(r"\w+", content)
    if not tokens:
        return []
    seen: set[str] = set()
    hints: list[str] = []
    for tok in tokens:
        kw = _TRIGGER_KEYWORD_MAP.get(tok.lower())
        if kw is None:
            # Any non-keyword token (word, number, mention, etc.) →
            # reject the whole reply (sentence-style mention shouldn't fire).
            return []
        if kw not in seen:
            seen.add(kw)
            hints.append(kw)
    return hints

# Bare URL extractor — used when on_message-triggered URL discovery doesn't
# match a video domain. Permissive on host, restrictive on scheme. Excludes
# Discord-internal URLs (channels.com etc.) by host filter at the call site.
_ANY_URL_RE = re.compile(r"https?://[^\s<>()\[\]]+", re.IGNORECASE)


def _hash_url(url: str) -> str:
    """Stable 11-char id from a URL, used as cache key for web jobs."""
    import hashlib
    return "w" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def _is_video_url(url: str) -> bool:
    """True iff the URL host matches a domain in VIDEO_DOMAINS.

    Coarse domain-membership check. Use _is_clearly_video_url() instead
    when deciding routing — many of the listed domains (reddit.com,
    twitter.com, instagram.com) host both videos AND text posts, and the
    bare host check sends text posts down the video pipeline where
    yt-dlp fails noisily.
    """
    return bool(VIDEO_URL_PATTERN.search(url))


# URL-shape patterns that strongly indicate the URL points at a single
# video. Used for routing — anything matching here goes to the video
# pipeline; everything else (including text posts on video-hosting
# platforms) goes to the web pipeline.
#
# Curated per platform from observation of real URLs. Ordering doesn't
# matter; first match wins.
_CLEAR_VIDEO_URL_PATTERNS = (
    # YouTube + shorts/live (also covered by YT_PATTERN; duplicated for clarity)
    re.compile(r"https?://(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?|shorts/|live/|embed/)", re.IGNORECASE),
    re.compile(r"https?://youtu\.be/[\w-]{11}", re.IGNORECASE),
    # Twitch — VODs, clips, live channels (live channels are clearly video too).
    # Bare twitch.tv/<channel> matches a live stream; twitch.tv/<channel>/about
    # / /schedule etc are profile pages — exclude those by requiring the path
    # to be just a single segment.
    re.compile(r"https?://clips\.twitch\.tv/", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?twitch\.tv/videos/\d+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?twitch\.tv/[\w-]+/?(?:\?|$)", re.IGNORECASE),
    # Vimeo numeric video IDs. Matches:
    #   vimeo.com/123, www.vimeo.com/123, player.vimeo.com/video/123,
    #   vimeo.com/channels/staffpicks/123
    re.compile(r"https?://(?:www\.|player\.|m\.)?vimeo\.com/(?:video/|channels/[\w-]+/)?\d+", re.IGNORECASE),
    # TikTok video URLs (vm.tiktok.com short links + /@user/video/<id> long form)
    re.compile(r"https?://vm\.tiktok\.com/", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+", re.IGNORECASE),
    # Reddit's video host (NOT reddit.com text posts!)
    re.compile(r"https?://v\.redd\.it/", re.IGNORECASE),
    # Dailymotion (`/video/<id>` is the canonical video path)
    re.compile(r"https?://(?:www\.)?dailymotion\.com/video/", re.IGNORECASE),
    re.compile(r"https?://dai\.ly/", re.IGNORECASE),
    # Rumble + Odysee + Kick — `/v/...` or `/<channel>/<slug>` paths
    re.compile(r"https?://(?:www\.)?rumble\.com/v[\w.-]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?odysee\.com/@[\w-]+(?:[:.][\w-]+)?/[\w-]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?kick\.com/[\w-]+/(?:videos|clips)/", re.IGNORECASE),
    # Bilibili
    re.compile(r"https?://(?:www\.)?bilibili\.com/video/", re.IGNORECASE),
    re.compile(r"https?://b23\.tv/", re.IGNORECASE),
    # SoundCloud (audio counts as video for our purposes — yt-dlp handles it)
    re.compile(r"https?://(?:www\.|m\.)?soundcloud\.com/[\w-]+/[\w-]+", re.IGNORECASE),
)


def _is_clearly_video_url(url: str) -> bool:
    """True iff the URL shape unambiguously points at a video / audio file.

    Stricter than _is_video_url. Use this for ROUTING decisions — it
    prevents reddit.com text posts from being treated as videos just
    because reddit.com is in VIDEO_DOMAINS.

    False negatives are acceptable here: an unrecognised video URL just
    means the bot routes it to the web pipeline first; the
    NotAVideoError fallback in process() can still upgrade it to video
    if Crawl4AI extracts video metadata. False positives (text URLs
    classified as video) are NOT acceptable — that's the bug we're
    fixing.
    """
    if not url:
        return False
    return any(p.search(url) for p in _CLEAR_VIDEO_URL_PATTERNS)


def _extract_first_url(text: str) -> str | None:
    """Return the first non-Discord URL found in `text`, or None."""
    if not text:
        return None
    for m in _ANY_URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?")  # trim trailing sentence punctuation
        # Skip Discord-internal links — those aren't articles
        if "discord.com" in url or "discordapp.com" in url:
            continue
        return url
    return None


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if ALLOWED_CHANNELS and message.channel.id not in ALLOWED_CHANNELS:
        return

    # Reply-trigger paths. Recognised keywords:
    #   tldr / summarize / summarise → summary flow (kind="web" or "video")
    #   litmus                       → AI litmus flow (kind="litmus")
    # Chaining: a reply body composed of multiple keywords (e.g. `tldr litmus`)
    # fires BOTH flows in order. _parse_trigger_keywords returns [] when the
    # body contains any non-keyword word, preventing accidental triggers in
    # sentences like "give me a tldr".
    if message.reference is not None:
        hints = _parse_trigger_keywords(message.content or "")
        if hints:
            await _handle_reply_trigger(message, kind_hints=hints)
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
            vlm_enabled=chan_cfg.get("vlm_enabled", VLM_ENABLED),
            yt_comments_enabled=chan_cfg.get(
                "yt_comments_enabled", YT_COMMENTS_ENABLED
            ),
            model_override=chan_cfg.get("model"),
            # Auto-paste — not an explicit user request. NotAVideoError
            # will silently drop instead of falling through to web,
            # because users posting reddit/twitter text-post URLs
            # generally don't expect a summary unless they ask.
            explicit_request=False,
            message=message,
        )

    # YouTube URLs → extract video ID for timestamp linking. YT_PATTERN is
    # already strict (matches /watch, /shorts, /live, youtu.be paths) so
    # all matches are real videos.
    for m in YT_PATTERN.finditer(message.content):
        all_urls.append(m.group(0))
        video_id = m.group(1)
        if video_id in seen:
            continue
        seen.add(video_id)
        url = f"https://www.youtube.com/watch?v={video_id}"
        jobs_to_queue.append(_new_job(url, video_id))

    # Other video platform URLs — only auto-trigger when the URL SHAPE
    # clearly points at a video. Reddit/Twitter/Instagram domains are in
    # VIDEO_DOMAINS but most URLs on those hosts are text posts that
    # would just produce a noisy yt-dlp failure.
    for url_match in VIDEO_URL_PATTERN.finditer(message.content):
        url = url_match.group(1)
        all_urls.append(url)
        if any(d in url for d in ("youtube.com", "youtu.be")):
            continue
        if not _is_clearly_video_url(url):
            # Text post / profile page on a video-hosting domain. Don't
            # auto-summarise; user can still ask via `tldr` reply.
            continue
        vid = _derive_video_id(url)
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


# ─── Reply-trigger handler ───────────────────────────────────────────────────


async def _resolve_referenced_message(message: discord.Message) -> discord.Message | None:
    """Return the message that `message` is replying to.

    discord.py auto-resolves it for cached references; for older replies the
    server fetch is required. Either way, returns None if unreachable
    (deleted, no permission, etc.).
    """
    ref = message.reference
    if ref is None:
        return None
    # Already-resolved reference (the common case)
    resolved = getattr(ref, "resolved", None)
    if isinstance(resolved, discord.Message):
        return resolved
    # Fall back to channel.fetch_message for stale references
    if ref.message_id is None:
        return None
    try:
        return await message.channel.fetch_message(ref.message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
        log.info("Reply-trigger: couldn't fetch referenced message %s: %s",
                 ref.message_id, e)
        return None


async def _handle_reply_trigger(
    message: discord.Message,
    kind_hints: list[str] | str = "summary",
) -> None:
    """User replied to a message with one or more trigger keywords. Find
    the URL in the replied-to message ONCE; build one Job per kind_hint
    and enqueue them all.

    `kind_hints`:
      str          → single hint (legacy single-keyword behaviour)
      list[str]    → chained reply (e.g. ["summary", "litmus"]). Each hint
                     produces its own Job; rate limit charged per Job;
                     queueing is atomic (rejected as a batch if any of
                     the cap checks would be violated).

    Hint values:
      "summary" → tldr / summarize / summarise. Routes to video or web
                  depending on URL shape.
      "litmus"  → AI-litmus-test reply. Always routes to kind="litmus"
                  regardless of URL — even YouTube URLs get litmus'd
                  (the bot pulls the page, not the video file).
    """
    if isinstance(kind_hints, str):
        kind_hints = [kind_hints]
    if not kind_hints:
        return  # caller bug; nothing to do
    referenced = await _resolve_referenced_message(message)
    if referenced is None:
        try:
            await message.channel.send(
                "❌ Couldn't read the message you replied to. "
                "Try replying directly to the message containing the URL.",
                reference=message, mention_author=False,
            )
        except discord.HTTPException:
            pass
        return

    url = _extract_first_url(referenced.content)
    if url is None:
        try:
            await message.channel.send(
                "❌ No URL found in the message you replied to.",
                reference=message, mention_author=False,
            )
        except discord.HTTPException:
            pass
        return

    chan_cfg = get_channel_config(message.channel.id)

    # Rate-limit + queue-cap apply equally to web/video/litmus jobs.
    # Atomic batch: reject the whole reply if it would push past either cap.
    ok, reason = _rate_limit_check(message.author.id, count=len(kind_hints))
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

    # Build one Job per hint. URL parsing happens once above; per-hint we
    # only flip the kind discriminator and pick the right ID scheme.
    jobs: list[Job] = []
    for hint in kind_hints:
        if hint == "litmus":
            # Litmus is always a web-style fetch — even when the URL points
            # at a video, we want to inspect the page (text, byline,
            # AdSense, domain age), not transcribe the audio.
            job = Job(
                url=url, video_id=_hash_url(url),
                channel=message.channel,
                submitter_id=message.author.id,
                submitter_name=str(message.author),
                model_override=chan_cfg.get("model"),
                kind="litmus",
                explicit_request=True,
                message=message,
            )
        elif _is_clearly_video_url(url):
            # Routing: only send to video pipeline when the URL shape
            # clearly points at a video. Reddit/Twitter/Instagram URLs go
            # to web by default — most of them are text posts. The
            # NotAVideoError fallback in the worker upgrades to/from web
            # if the routing turns out wrong (e.g. a tweet that does have
            # an embedded video and the user replied tldr to it).
            m_yt = YT_PATTERN.search(url)
            if m_yt:
                video_id = m_yt.group(1)
                canonical = f"https://www.youtube.com/watch?v={video_id}"
            else:
                canonical = url
                video_id = _derive_video_id(url)
            job = Job(
                url=canonical, video_id=video_id,
                channel=message.channel,
                submitter_id=message.author.id,
                submitter_name=str(message.author),
                diarize=chan_cfg.get("diarize", False),
                vlm_enabled=chan_cfg.get("vlm_enabled", VLM_ENABLED),
                yt_comments_enabled=chan_cfg.get(
                    "yt_comments_enabled", YT_COMMENTS_ENABLED
                ),
                model_override=chan_cfg.get("model"),
                kind="video",
                explicit_request=True,  # user typed tldr → wants a summary
                message=message,
            )
        else:
            job = Job(
                url=url, video_id=_hash_url(url),
                channel=message.channel,
                submitter_id=message.author.id,
                submitter_name=str(message.author),
                model_override=chan_cfg.get("model"),
                kind="web",
                explicit_request=True,
                message=message,
            )
        jobs.append(job)

    for job in jobs:
        _rate_limit_record(job.submitter_id)
        await queue.put(job)
        await _ack_queued(job, queue.qsize())
        log.info("Reply-trigger queued %s (%s) from %s (channel=%s)",
                 job.video_id, job.kind, job.submitter_name, message.channel.id)
    if len(jobs) > 1:
        log.info("Chained reply: %d jobs queued from %s",
                 len(jobs), message.author)


# ─── Web scraper client ──────────────────────────────────────────────────────


# Patterns in scraped Markdown that indicate a Cloudflare interstitial /
# challenge page. When seen in the Crawl4AI output, fall back to FlareSolverr.
_CF_CHALLENGE_MARKERS = (
    "Just a moment",
    "Checking your browser",
    "challenges.cloudflare.com",
    "cf-challenge",
    "ddos protection by cloudflare",
    "Enable JavaScript and cookies to continue",
    "cf_chl_opt",
)


def _looks_like_cf_challenge(text: str) -> bool:
    """Heuristic: extracted Markdown that's just a CF interstitial.

    A real article body is usually >500 chars and contains paragraph text;
    a CF challenge page is short and dominated by the marker phrases above.
    Both checks together avoid false positives on long articles that happen
    to mention Cloudflare in passing.
    """
    if not text or len(text) > 2000:
        return False
    return any(m.lower() in text.lower() for m in _CF_CHALLENGE_MARKERS)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_WS_RE = re.compile(r"\s+")
_HTML_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL,
)


def _html_to_text(html: str) -> str:
    """Crude HTML → plain text. Used only for the FlareSolverr fallback path
    (rare; primary scrape returns Markdown directly).

    Intentional non-goals: handle every weird tag, preserve list/heading
    structure perfectly. Goal is "give the LLM something readable when CF
    has us cornered". For better quality on the hot path we use Crawl4AI's
    readability extractor.
    """
    text = _HTML_SCRIPT_STYLE_RE.sub(" ", html)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html_mod.unescape(text)
    return _HTML_WS_RE.sub(" ", text).strip()


def _derive_title_from_markdown(md: str, url: str) -> str:
    """Pull the first H1 from the markdown, or fall back to the URL host."""
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()[:300]
    from urllib.parse import urlparse
    return urlparse(url).hostname or url


async def _fetch_via_crawl4ai(url: str) -> str | None:
    """Hit Crawl4AI /md. Returns the Markdown body, or None on failure /
    obvious CF challenge. Permanent errors raise PermanentError.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    payload = {
        "url": url,
        "f": "fit",   # readability-based extraction (cleanest)
        "c": "0",     # cache mode: bypass server-side cache (we cache locally)
    }
    try:
        async with http.post(
            f"{SCRAPER_API}/md",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=SCRAPER_TIMEOUT),
        ) as resp:
            if resp.status == 400:
                body = await resp.text()
                raise PermanentError(f"Scraper rejected URL: {body[:200]}")
            if resp.status != 200:
                body = await resp.text()
                log.warning("Crawl4AI %d for %s: %s", resp.status, url, body[:200])
                return None
            data = await resp.json()
    except asyncio.TimeoutError:
        log.warning("Crawl4AI timeout for %s", url)
        return None
    except aiohttp.ClientError as e:
        log.warning("Crawl4AI transport error for %s: %s", url, e)
        return None

    md = (data.get("markdown") or "").strip()
    if not md or _looks_like_cf_challenge(md):
        return None
    return md


async def _fetch_via_flaresolverr(url: str) -> str | None:
    """Fallback path. Asks FlareSolverr to solve any CF challenge and return
    the resolved HTML; we then strip to plain text. Returns None on failure.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    payload = {
        "cmd": "request.get",
        "url": url,
        "maxTimeout": FLARESOLVERR_TIMEOUT * 1000,  # ms
    }
    try:
        async with http.post(
            FLARESOLVERR_API,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=FLARESOLVERR_TIMEOUT + 30),
        ) as resp:
            if resp.status != 200:
                log.warning("FlareSolverr %d for %s", resp.status, url)
                return None
            data = await resp.json()
    except asyncio.TimeoutError:
        log.warning("FlareSolverr timeout for %s", url)
        return None
    except aiohttp.ClientError as e:
        log.warning("FlareSolverr transport error for %s: %s", url, e)
        return None

    if data.get("status") != "ok":
        log.warning("FlareSolverr non-ok for %s: %s", url, data.get("message"))
        return None
    solution = data.get("solution") or {}
    html = solution.get("response") or ""
    if not html:
        return None
    text = _html_to_text(html)
    if not text or len(text) < 200:
        # Probably still a challenge or login wall — give up rather than feed
        # the LLM a few hundred bytes of nav-bar text.
        return None
    return text


# ─── Reddit-specific scraper ──────────────────────────────────────────────────
# Reddit URLs benefit from a structured fetch (OP post + comments) instead of
# generic HTML scraping. The anonymous JSON API exposes everything we need
# without auth. For link posts, we additionally fetch the target article via
# the generic scraper and compose both into a single Markdown blob.

_REDDIT_POST_RE = re.compile(
    r"https?://(?:(?:www|old|new|sh|np)\.)?reddit\.com/r/[\w-]+/comments/(\w+)/",
    re.IGNORECASE,
)

# Reddit's anonymous endpoint rate-limits per User-Agent; the default aiohttp
# UA gets 429-d. Override to identify ourselves cleanly.
REDDIT_UA = os.environ.get(
    "REDDIT_USER_AGENT",
    "whisper-transcribe-bot/1.0 (TL;DW summary bot)"
)
REDDIT_TOP_COMMENTS = int(os.environ.get("REDDIT_TOP_COMMENTS", "10"))
REDDIT_REPLY_DEPTH = int(os.environ.get("REDDIT_REPLY_DEPTH", "1"))
REDDIT_TIMEOUT = int(os.environ.get("REDDIT_TIMEOUT", "30"))


def _is_reddit_post_url(url: str) -> bool:
    """True iff url is a /r/<sub>/comments/<id>/... post URL."""
    return bool(_REDDIT_POST_RE.search(url or ""))


async def _fetch_reddit_json(url: str) -> list | None:
    """Hit Reddit's anonymous JSON endpoint for a post URL. Returns the parsed
    array `[post_listing, comments_listing]` or None on failure.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    base = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if not base.endswith(".json"):
        base = base + ".json"
    # `raw_json=1` disables HTML-entity escaping in selftext / body fields.
    # `limit=50` + `depth=2` cap payload size.
    json_url = base + "?raw_json=1&limit=50&depth=2"
    headers = {"User-Agent": REDDIT_UA}
    try:
        async with http.get(
            json_url, headers=headers,
            timeout=aiohttp.ClientTimeout(total=REDDIT_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                log.warning("Reddit JSON %d for %s", resp.status, url)
                return None
            data = await resp.json(content_type=None)  # reddit serves text/json
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        log.warning("Reddit JSON error for %s: %s", url, e)
        return None
    if not isinstance(data, list) or len(data) < 2:
        log.warning("Reddit JSON: unexpected shape for %s", url)
        return None
    return data


def _format_reddit_comment(node: dict, depth: int, max_depth: int) -> str:
    """Render a comment node + replies up to max_depth as Markdown. Skips
    deleted/removed/AutoModerator. Returns "" when nothing useful renders.
    """
    if node.get("kind") != "t1":
        return ""
    data = node.get("data") or {}
    body = (data.get("body") or "").strip()
    if body in ("", "[deleted]", "[removed]"):
        return ""
    author = data.get("author") or "[deleted]"
    score = data.get("score", 0)
    indent = "  " * depth
    # Collapse very long bodies — comments aren't articles
    if len(body) > 2000:
        body = body[:2000] + "…"
    lines = [f"{indent}- **u/{author}** ({score} pts): {body}"]
    if depth < max_depth:
        replies = data.get("replies")
        if isinstance(replies, dict):
            children = (replies.get("data") or {}).get("children") or []
            for child in children:
                rendered = _format_reddit_comment(child, depth + 1, max_depth)
                if rendered:
                    lines.append(rendered)
    return "\n".join(lines)


def _build_reddit_markdown(post_data: dict,
                           comment_nodes: list,
                           article_md: str | None,
                           article_url: str,
                           article_error: str | None) -> tuple[str, str]:
    """Compose the post + comments + (optional) linked article into one
    Markdown blob. Returns (title, body).
    """
    title = post_data.get("title") or "Reddit post"
    subreddit = post_data.get("subreddit") or ""
    author = post_data.get("author") or "[deleted]"
    selftext = (post_data.get("selftext") or "").strip()
    is_self = bool(post_data.get("is_self"))
    score = post_data.get("score", 0)
    num_comments = post_data.get("num_comments", 0)

    parts: list[str] = []

    # Article section (link posts that target a non-Reddit URL and scraped OK)
    if not is_self and article_url:
        if article_md:
            from urllib.parse import urlparse
            host = urlparse(article_url).hostname or article_url
            parts.append(f"# Linked article ({host})\n\n{article_md.strip()}")
        elif article_error:
            parts.append(
                f"# Linked article: {article_url}\n\n"
                f"*Article unreachable: {article_error}*"
            )

    # Reddit post section
    post_lines = [f"# Reddit discussion — r/{subreddit}"]
    post_lines.append(
        f"**Posted by u/{author}** "
        f"({score} pts, {num_comments} comments): **{title}**"
    )
    if selftext:
        post_lines.append("")
        post_lines.append(selftext)
    parts.append("\n".join(post_lines))

    # Top comments by score
    scored: list[tuple[int, dict]] = []
    for node in comment_nodes:
        if node.get("kind") != "t1":
            continue
        data = node.get("data") or {}
        body = (data.get("body") or "").strip()
        if body in ("", "[deleted]", "[removed]"):
            continue
        scored.append((data.get("score", 0), node))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    top = scored[:REDDIT_TOP_COMMENTS]

    if top:
        comment_lines = [f"## Top {len(top)} comments"]
        for _s, node in top:
            rendered = _format_reddit_comment(node, 0, REDDIT_REPLY_DEPTH)
            if rendered:
                comment_lines.append(rendered)
                comment_lines.append("")
        parts.append("\n".join(comment_lines).rstrip())

    return title, "\n\n".join(parts)


async def _fetch_reddit(url: str) -> tuple[str, str]:
    """Reddit-specific path: OP + top comments + linked article (if any).

    Raises RuntimeError on JSON-API failure so the caller falls back to the
    generic Crawl4AI path.
    """
    data = await _fetch_reddit_json(url)
    if data is None:
        raise RuntimeError(f"Reddit JSON API returned no data for {url}")

    post_children = ((data[0] or {}).get("data") or {}).get("children") or []
    if not post_children:
        raise RuntimeError(f"Reddit JSON: empty post listing for {url}")
    post_data = (post_children[0] or {}).get("data") or {}

    comment_children = ((data[1] or {}).get("data") or {}).get("children") or []

    # If it's a link post, fetch the target article alongside.
    article_md: str | None = None
    article_error: str | None = None
    article_url = ""
    is_self = bool(post_data.get("is_self"))
    candidate = (post_data.get("url") or "").strip()
    if not is_self and candidate and not candidate.startswith((
        "https://www.reddit.com", "https://reddit.com",
        "https://i.redd.it", "https://v.redd.it",
        "https://i.imgur.com",  # raw images aren't articles
    )):
        article_url = candidate
        log.info("[reddit] link post → fetching target: %s", article_url)
        try:
            article_md = await _fetch_via_crawl4ai(article_url)
            if article_md is None:
                article_md = await _fetch_via_flaresolverr(article_url)
            if article_md is None:
                article_error = "scraper returned empty / CF challenge"
        except PermanentError as e:
            article_error = f"permanent: {e}"[:200]
        except Exception as e:
            article_error = f"transient: {type(e).__name__}: {e}"[:200]
        if article_md:
            log.info("[reddit] linked article scraped: %d chars", len(article_md))
        else:
            log.info("[reddit] linked article unreachable: %s", article_error)

    title, body = _build_reddit_markdown(
        post_data, comment_children, article_md, article_url, article_error,
    )
    return title, body


# ─── HackerNews-specific scraper ──────────────────────────────────────────────
# Same shape as Reddit: structured fetch (post + linked article + comments)
# via the public Firebase-backed API. Anonymous, no auth, no rate-limits worth
# worrying about. Uses the same Reddit-style prompts in process_url since the
# composed Markdown looks the same.

_HN_POST_RE = re.compile(
    r"https?://news\.ycombinator\.com/item\?id=(\d+)",
    re.IGNORECASE,
)
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
HN_TOP_COMMENTS = int(os.environ.get("HN_TOP_COMMENTS", "10"))
HN_REPLY_DEPTH = int(os.environ.get("HN_REPLY_DEPTH", "1"))
HN_TIMEOUT = int(os.environ.get("HN_TIMEOUT", "20"))


def _is_hn_post_url(url: str) -> bool:
    """True iff url is an HN /item?id=<n> URL."""
    return bool(_HN_POST_RE.search(url or ""))


async def _fetch_hn_item(item_id: int | str) -> dict | None:
    """Fetch one HN item via the Firebase API. None on failure."""
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    url = f"{HN_API_BASE}/item/{item_id}.json"
    try:
        async with http.get(
            url, timeout=aiohttp.ClientTimeout(total=HN_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
    except (asyncio.TimeoutError, aiohttp.ClientError):
        return None
    return data if isinstance(data, dict) else None


async def _fetch_hn_comments(kid_ids: list,
                             max_count: int,
                             max_depth: int,
                             current_depth: int = 0) -> list[dict]:
    """Recursively fetch HN comments up to max_depth + max_count.

    HN returns kids in posting order; we don't have per-comment scores
    (HN's API doesn't expose them) so we keep posting order for top-level
    threads and trust HN's ranking.
    """
    if not kid_ids or current_depth > max_depth:
        return []
    # Cap how many siblings we even fetch at this level
    ids = kid_ids[:max_count]
    items = await asyncio.gather(*(_fetch_hn_item(k) for k in ids),
                                 return_exceptions=False)
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Skip dead / deleted comments
        if item.get("dead") or item.get("deleted"):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # Depth-1 children
        children: list[dict] = []
        if current_depth < max_depth and item.get("kids"):
            # Fewer replies per comment than top-level cap
            children = await _fetch_hn_comments(
                item["kids"], max_count=5,
                max_depth=max_depth, current_depth=current_depth + 1,
            )
        out.append({
            "by": item.get("by") or "[deleted]",
            "text": text,
            "kids": children,
        })
    return out


def _format_hn_comment(comment: dict, depth: int = 0) -> str:
    """Render an HN comment + replies as Markdown bullets. HN comment
    `text` is HTML (<p>, <i>, etc.) — strip to plain text.
    """
    body = _html_to_text(comment.get("text") or "")
    if len(body) > 2000:
        body = body[:2000] + "…"
    indent = "  " * depth
    lines = [f"{indent}- **{comment.get('by', '[deleted]')}**: {body}"]
    for child in comment.get("kids", []):
        rendered = _format_hn_comment(child, depth + 1)
        if rendered:
            lines.append(rendered)
    return "\n".join(lines)


def _build_hn_markdown(post: dict,
                       comments: list[dict],
                       article_md: str | None,
                       article_url: str,
                       article_error: str | None) -> tuple[str, str]:
    """Compose HN post + linked article + comments into Markdown.

    Mirrors `_build_reddit_markdown`'s shape so process_url's Reddit-aware
    prompts work for HN content too (community discussion + linked article).
    """
    title = post.get("title") or "HackerNews post"
    by = post.get("by") or "[deleted]"
    score = post.get("score", 0)
    descendants = post.get("descendants", 0)  # total comment count
    selftext_html = post.get("text") or ""
    selftext = _html_to_text(selftext_html) if selftext_html else ""
    is_self = not article_url  # link posts have a non-self url

    parts: list[str] = []

    # Linked article section (Ask HN / Show HN with no URL skip this)
    if article_url:
        if article_md:
            from urllib.parse import urlparse
            host = urlparse(article_url).hostname or article_url
            parts.append(f"# Linked article ({host})\n\n{article_md.strip()}")
        elif article_error:
            parts.append(
                f"# Linked article: {article_url}\n\n"
                f"*Article unreachable: {article_error}*"
            )

    # HN post section — use same heading shape as Reddit so the prompt's
    # "Reddit discussion" instruction applies. The prompt is platform-
    # agnostic about the discussion content; only the heading wording
    # changes for clarity.
    post_lines = [f"# HackerNews discussion (news.ycombinator.com)"]
    post_lines.append(
        f"**Submitted by {by}** ({score} pts, {descendants} comments): "
        f"**{title}**"
    )
    if selftext:
        post_lines.append("")
        post_lines.append(selftext)
    parts.append("\n".join(post_lines))

    # Top comments
    if comments:
        comment_lines = [f"## Top {len(comments)} comments"]
        for c in comments:
            rendered = _format_hn_comment(c, depth=0)
            if rendered:
                comment_lines.append(rendered)
                comment_lines.append("")
        parts.append("\n".join(comment_lines).rstrip())

    return title, "\n\n".join(parts)


async def _fetch_hn(url: str) -> tuple[str, str]:
    """HackerNews-specific path: post + linked article + top comments.

    Raises RuntimeError on API failure so caller falls back to generic.
    """
    m = _HN_POST_RE.search(url)
    if not m:
        raise RuntimeError(f"Couldn't parse HN item id from {url}")
    item_id = m.group(1)

    post = await _fetch_hn_item(item_id)
    if post is None:
        raise RuntimeError(f"HN API returned nothing for item {item_id}")

    if post.get("type") not in ("story", "ask", "show", "job"):
        # Comment URL — fetch parent story instead
        parent_id = post.get("parent")
        if parent_id:
            log.info("[hn] %s is a comment; fetching parent story %s",
                     item_id, parent_id)
            parent = await _fetch_hn_item(parent_id)
            if parent:
                post = parent

    # Optional article fetch — `url` field present iff this is a link post
    article_md: str | None = None
    article_error: str | None = None
    article_url = (post.get("url") or "").strip()
    if article_url:
        log.info("[hn] link post → fetching target: %s", article_url)
        try:
            article_md = await _fetch_via_crawl4ai(article_url)
            if article_md is None:
                article_md = await _fetch_via_flaresolverr(article_url)
            if article_md is None:
                article_error = "scraper returned empty / CF challenge"
        except PermanentError as e:
            article_error = f"permanent: {e}"[:200]
        except Exception as e:
            article_error = f"transient: {type(e).__name__}: {e}"[:200]
        if article_md:
            log.info("[hn] linked article scraped: %d chars", len(article_md))
        else:
            log.info("[hn] linked article unreachable: %s", article_error)

    # Top-level comment ids (no per-comment score on HN; use HN's order)
    kid_ids = post.get("kids") or []
    comments = await _fetch_hn_comments(
        kid_ids, max_count=HN_TOP_COMMENTS, max_depth=HN_REPLY_DEPTH,
    )

    title, body = _build_hn_markdown(
        post, comments, article_md, article_url, article_error,
    )
    return title, body


async def fetch_article(url: str) -> tuple[str, str]:
    """Scrape `url` and return (title, body_text).

    Routing:
      1. Reddit post URLs → JSON API + linked-article fetch + top comments.
      2. HackerNews post URLs → Firebase API + linked-article fetch +
         top comments.
      3. Everything else → Crawl4AI; FlareSolverr on CF block / failure.

    Raises PermanentError on 4xx (bad URL, scheme reject) — caller should NOT
    retry. Raises RuntimeError on total failure across both backends.
    """
    if _is_reddit_post_url(url):
        try:
            title, body = await _fetch_reddit(url)
            log.info("[scrape] reddit ok: %s (%d chars)", url, len(body))
            return title, body[:SCRAPED_BODY_CHAR_CAP]
        except RuntimeError as e:
            # Reddit-specific path failed (JSON 5xx, malformed payload).
            # Fall through to generic Crawl4AI scrape so the user still gets
            # SOMETHING, even if it's just whatever readability extracts.
            log.warning("[scrape] reddit path failed (%s) — falling back to generic", e)

    if _is_hn_post_url(url):
        try:
            title, body = await _fetch_hn(url)
            log.info("[scrape] hn ok: %s (%d chars)", url, len(body))
            return title, body[:SCRAPED_BODY_CHAR_CAP]
        except RuntimeError as e:
            log.warning("[scrape] hn path failed (%s) — falling back to generic", e)

    md = await _fetch_via_crawl4ai(url)
    if md is not None:
        title = _derive_title_from_markdown(md, url)
        log.info("[scrape] crawl4ai ok: %s (%d chars)", url, len(md))
        return title, md[:SCRAPED_BODY_CHAR_CAP]

    log.info("[scrape] crawl4ai miss/CF — trying FlareSolverr for %s", url)
    text = await _fetch_via_flaresolverr(url)
    if text is not None:
        from urllib.parse import urlparse
        title = urlparse(url).hostname or url
        log.info("[scrape] flaresolverr ok: %s (%d chars)", url, len(text))
        return title, text[:SCRAPED_BODY_CHAR_CAP]

    raise RuntimeError(
        f"Both scrapers failed for {url} — site likely behind hard "
        f"Turnstile or down."
    )


# ─── Worker ───────────────────────────────────────────────────────────────────


class PermanentError(Exception):
    """Errors that will fail identically on retry (4xx, oversized inputs, etc.)."""


class NotAVideoError(PermanentError):
    """yt-dlp determined the URL doesn't point at a video.

    Subclass of PermanentError so it skips the transient-retry loop, but
    the worker handles it specially: explicit user requests fall through
    to the web pipeline (the user asked for a summary, give them one);
    auto-paste jobs silently drop (URL probably wasn't intended for the
    bot at all).
    """


# Subset of _PERMANENT_REMOTE_PATTERNS that specifically mean "this URL is not
# a video". Used to distinguish "yt-dlp can't handle this URL at all" from
# "yt-dlp could but the content is gated/unavailable" (private video, members
# only, geo-blocked, etc. — those should NOT fall through to web because the
# article-page version would be just as restricted).
_NOT_A_VIDEO_PATTERNS = (
    "Unsupported URL",
    "is not a valid URL",
    "No video could be found in this tweet",
    "No video could be found in this",
    "There's no video in this post",
    "No media found",
    "Post does not contain any media",
    "No video formats found",
    "no video formats found",
)


def _is_not_a_video_error(text: str) -> bool:
    """True iff the error message specifically means 'this URL is not a video'.

    Excludes gating/availability errors (private, members-only, geo-blocked)
    — those fail the same way on the article page so falling through to web
    wouldn't help.
    """
    return any(p in text for p in _NOT_A_VIDEO_PATTERNS)


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





# Reaction emoji used during processing — cleaned up on completion or failure.
# Both video and web flows reuse the queued (⏳) and summarising (🧠) emoji;
# the middle "fetching content" emoji differs by kind so users can tell at a
# glance what the bot is doing:
#   🎧  video / audio download (yt-dlp + whisper)
#   📰  web article scrape (crawl4ai / flaresolverr)
PROCESSING_EMOJI_VIDEO = "\U0001f3a7"  # 🎧
PROCESSING_EMOJI_WEB = "\U0001f4f0"    # 📰
# Cleanup list covers BOTH so a kind switch (e.g. NotAVideoError fall-through)
# leaves no stale reactions behind.
PROCESSING_EMOJI = (
    "\u23f3",                  # ⏳ queued
    PROCESSING_EMOJI_VIDEO,    # 🎧 video fetch
    PROCESSING_EMOJI_WEB,      # 📰 web fetch
    "\U0001f9e0",              # 🧠 summarising
)


async def _submit_and_poll_transcribe(payload: dict, job: "Job") -> dict:
    """Submit a transcription job to the server-side queue, poll until done,
    return the result dict.

    Maps server-side terminal status to bot exceptions:
      done       → returns the result dict
      failed (permanent=true)  → PermanentError (no retry)
      failed (permanent=false) → RuntimeError (worker retries)
      cancelled  → PermanentError (someone cancelled out-of-band)

    Reactions:
      queued → ⏳ already set by the caller before submit
      running → ⏳ swapped to 🎧 (PROCESSING_EMOJI_VIDEO) the first time
                we see status=running; caller already does this before
                calling, so this is just a safety net.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")

    # Submit
    async with http.post(
        f"{WHISPER_API}/api/jobs",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        if resp.status == 503:
            # Valkey is down on the whisper side. The legacy /api/transcribe
            # path still works but we don't fall through automatically —
            # the operator should be aware and fix the queue.
            raise RuntimeError("Whisper queue backend unavailable")
        if resp.status != 202:
            try:
                body = await resp.json()
            except Exception:
                body = {"error": await resp.text()}
            err = str(body.get("error", resp.status))
            if (400 <= resp.status < 500) or body.get("permanent") \
                    or _is_permanent_remote_error(err):
                raise PermanentError(f"Job submit rejected ({resp.status}): {err}")
            raise RuntimeError(f"Job submit failed ({resp.status}): {err}")
        sub = await resp.json()

    job_id = sub.get("job_id")
    if not job_id:
        raise RuntimeError(f"Job submit returned no job_id: {sub}")
    log.info("[%s] queued as %s (position=%s)",
             job.video_id, job_id, sub.get("position"))

    # Poll until terminal. /api/jobs/{id} is cheap (single HGETALL on
    # valkey) so 3s ticks are fine even at sustained load. TRANSCRIBE_TIMEOUT
    # is the upper bound on total poll duration — past that we give up on
    # the job (the server keeps running it; we just stop listening).
    deadline = time.monotonic() + TRANSCRIBE_TIMEOUT
    prev_status = None
    transient_errors = 0
    while time.monotonic() < deadline:
        # Transient network blips (whisper restarting, momentary DNS
        # flake) must NOT kill the poll — the job is still running on the
        # server. We catch generic Exception (aiohttp.ClientError,
        # asyncio.TimeoutError, OSError) and retry within the deadline.
        # If errors persist for too long we eventually escalate.
        try:
            async with http.get(
                f"{WHISPER_API}/api/jobs/{job_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 404:
                    raise RuntimeError(f"job {job_id} vanished from server")
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"poll {job_id} failed ({resp.status}): {body[:200]}"
                    )
                data = await resp.json()
        except RuntimeError:
            # 404 / non-200 is authoritative — re-raise.
            raise
        except Exception as e:
            transient_errors += 1
            # 20 consecutive transient errors at 3s poll interval ≈ 1 min
            # of total outage — escalate so we don't tie up the worker
            # for hours on a wedged whisper.
            if transient_errors >= 20:
                raise RuntimeError(
                    f"poll {job_id} keeps failing: {e}"
                ) from e
            log.debug("[%s] poll blip #%d (%s); retrying",
                      job.video_id, transient_errors, e)
            await asyncio.sleep(JOB_POLL_INTERVAL)
            continue
        transient_errors = 0  # reset on success

        status = data.get("status")
        if status != prev_status:
            log.info("[%s] job %s -> %s", job.video_id, job_id, status)
            prev_status = status

        if status == "done":
            result = data.get("result") or {}
            if data.get("cached") or result.get("cached"):
                log.info("[%s] (transcript came from server cache)", job.video_id)
            return result
        if status == "failed":
            err = data.get("error", "unknown")
            if data.get("permanent"):
                raise PermanentError(f"Transcription failed: {err}")
            raise RuntimeError(f"Transcription failed: {err}")
        if status == "cancelled":
            raise PermanentError(f"job {job_id} cancelled")
        # queued or running — keep polling.
        await asyncio.sleep(JOB_POLL_INTERVAL)

    # Hit TRANSCRIBE_TIMEOUT before the job reached a terminal state. The
    # server keeps running it; we just stop listening. RuntimeError (not
    # PermanentError) so the worker retries — the next attempt's poll might
    # catch the same job (now done) or the worker bumps it forward.
    raise RuntimeError(
        f"poll {job_id} exceeded TRANSCRIBE_TIMEOUT ({TRANSCRIBE_TIMEOUT}s); "
        f"server still has the job — check /api/jobs/{job_id}"
    )


async def worker():
    """Sequential worker — one job at a time. Video jobs are GPU-bound (whisper);
    web jobs share the same LLM proxy used by video summaries, so even though
    they don't compete for GPU they still serialise on the model and we keep
    a single worker for predictable throughput.

    Routing fallback: if a video job's download fails with NotAVideoError
    (yt-dlp says "Unsupported URL" / "No video found"), the URL probably
    points at an article instead. For explicit user requests we re-dispatch
    as a web job and continue. For auto-paste we silently drop — the user
    didn't ask for a summary, no need to spam the channel with errors.
    """
    while True:
        job = await queue.get()
        last_error = None
        silent_drop = False
        # GPU contention is now handled server-side by the queue at
        # /api/jobs — busy-wait branches are gone. This loop only handles
        # truly transient errors (network blips, LLM timeouts, etc.).
        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                    log.info("[%s] Retry %d/%d in %ds...", job.video_id, attempt, MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                # Recompute handler each attempt — NotAVideoError flips kind.
                if job.kind == "litmus":
                    handler = process_litmus
                elif job.kind == "web":
                    handler = process_url
                else:
                    handler = process
                await handler(job)
                last_error = None
                break
            except NotAVideoError as e:
                if job.explicit_request:
                    # User explicitly asked for a summary; URL isn't a
                    # video → try the web pipeline. Reset video_id to the
                    # URL hash so cache key matches the web pipeline.
                    log.info(
                        "[%s] not a video — falling through to web pipeline: %s",
                        job.video_id, e,
                    )
                    job.kind = "web"
                    job.video_id = _hash_url(job.url)
                    # Don't count this as a retry — it's a routing change.
                    # `continue` re-enters the loop with the new handler.
                    continue
                else:
                    # Auto-paste of a non-video URL. User didn't ask for a
                    # summary; don't spam the channel with an error embed.
                    log.info("[%s] not a video (auto-paste) — silent drop: %s",
                             job.video_id, e)
                    silent_drop = True
                    break
            except PermanentError as e:
                last_error = e
                log.error("[%s] Permanent failure (no retry): %s", job.video_id, e)
                break
            except Exception as e:
                last_error = e
                log.warning("[%s] Attempt %d failed: %s", job.video_id, attempt + 1, e)

        if silent_drop:
            # Remove the ⏳ reaction so the message looks clean
            for emoji in PROCESSING_EMOJI:
                await _job_remove_react(job, emoji)
        elif last_error:
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
    if http is None: raise RuntimeError("HTTP session not initialised")

    # 1. Check whisper service status
    async with http.get(f"{WHISPER_API}/api/status") as resp:
        if resp.status != 200:
            raise RuntimeError("Whisper service unavailable")

    # 1a. Cache lookup — skip download+transcribe if we have a fresh transcript
    cached = read_cache(job.video_id)
    file_path = None
    # Comments aren't cached today; cache hits skip the Community Reaction
    # embed. First-time runs (cache miss) populate this from the yt-dlp
    # response and use it after the main 3-style summary gather.
    raw_comments: list[dict] = []
    if cached is not None:
        title, status, transcript, duration = cached
        log.info("[%s] Cache hit (%d chars, '%s')", job.video_id, len(transcript), title)
    else:
        # 2. Download. Keep the video stream alongside audio when VLM is
        # enabled — /api/describe needs a video file to extract frames
        # from. When VLM is off, audio-only WAV (smaller, current default).
        # Optionally also fetch top YT comments for the "Community reaction"
        # embed (default on; per-channel opt-out via /config).
        log.info("[%s] Downloading%s%s...", job.video_id,
                 " (audio+video)" if job.vlm_enabled else "",
                 f" + comments(top {YT_COMMENTS_MAX})" if job.yt_comments_enabled else "")
        download_payload = {
            "url": job.url,
            "keep_video": job.vlm_enabled,
            "include_comments": job.yt_comments_enabled,
            "comments_max": YT_COMMENTS_MAX,
            "comments_sort": YT_COMMENTS_SORT,
        }
        async with http.post(
            f"{WHISPER_API}/api/yt-download",
            json=download_payload,
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
                    # Distinguish "URL isn't a video" from other permanent
                    # errors (private/members-only/geo-blocked). The worker
                    # falls through to web pipeline only on the former.
                    if _is_not_a_video_error(err):
                        raise NotAVideoError(
                            f"URL is not a video ({resp.status}): {err}"
                        )
                    raise PermanentError(f"Download failed ({resp.status}): {err}")
                raise RuntimeError(f"Download failed: {err}")
            dl = await resp.json()

        title = dl.get("title", job.video_id)
        duration = dl.get("duration", 0)
        file_path = dl["filename"]
        # Livestream signal from yt-dlp metadata. Plumbed through the
        # /api/yt-download response. `was_live=True` means VOD'd livestream
        # — natural quiet stretches (gameplay, music, audience interaction)
        # are normal; speech-density-based VLM trigger would misfire here.
        was_live = bool(dl.get("was_live", False))
        live_status = str(dl.get("live_status", ""))
        # Comments are present only when include_comments=True was sent AND
        # yt-dlp succeeded in extracting them. Empty list / missing key both
        # mean "no comments to summarise" (e.g. comments disabled on the
        # video, age-gated, or the comment-fetch path failed silently).
        raw_comments = dl.get("comments") or []

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
        await _job_react(job, PROCESSING_EMOJI_VIDEO)  # 🎧

        transcribe_payload = {
            "file_path": file_path,
            "model": WHISPER_MODEL,
            # Don't cleanup yet — VLM fallback (below) may need the file.
            "cleanup": False,
            "return_file": False,  # bot uses transcript text directly
            "diarize": job.diarize,
            "consumer": "discord-bot",
        }
        if initial_prompt:
            transcribe_payload["initial_prompt"] = initial_prompt

        # Submit to the server-side queue + poll until terminal. The queue
        # serialises across all consumers (us, MCP, Gradio, ad-hoc curl) so
        # there's no busy-wait dance against 409s any more.
        try:
            result = await _submit_and_poll_transcribe(transcribe_payload, job)
        except PermanentError:
            await _cleanup_remote_file(file_path)
            raise
        except Exception:
            await _cleanup_remote_file(file_path)
            raise

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

        # Smart VLM gates — see the VLM_* constants for rationale. Each
        # guard short-circuits the density check. Logged at INFO so it's
        # obvious when VLM was skipped and why (otherwise users see a
        # "speech_density: 5.3 → sparse" log and wonder why VLM didn't fire).
        vlm_skip_reason: str | None = None
        if VLM_SKIP_LIVESTREAMS and was_live:
            vlm_skip_reason = f"livestream (live_status={live_status or 'was_live'})"
        elif len(transcript.strip()) >= VLM_MIN_TEXT_CHARS:
            vlm_skip_reason = (f"transcript {len(transcript.strip())} chars "
                               f">= VLM_MIN_TEXT_CHARS {VLM_MIN_TEXT_CHARS}")

        density_triggers_vlm = density < SPEECH_DENSITY_SPARSE
        # User force overrides all gates — if they asked about visuals, give
        # them visuals even on a 5h livestream.
        run_vlm = job.vlm_enabled and (
            user_forced_vlm
            or (density_triggers_vlm and vlm_skip_reason is None)
        )
        if density_triggers_vlm and vlm_skip_reason and not user_forced_vlm:
            log.info("[%s] VLM skipped: %s (density %.1f would have triggered)",
                     job.video_id, vlm_skip_reason, density)

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
                visual_text = _format_vlm_output(desc_result)
                # Prefer scene-count for status when the new pipeline ran;
                # fall back to frame_count for legacy responses.
                scenes = desc_result.get("scenes") or []
                unit = (
                    f"{len(scenes)} scenes ({desc_result.get('frame_count', 0)} frames)"
                    if scenes
                    else f"{desc_result.get('frame_count', 0)} frames"
                )
                if visual_only:
                    transcript = visual_text
                    status = (status or "") + f" | visual-only ({unit})"
                else:
                    # Hybrid: interleave speech and visual lines by timestamp.
                    transcript = _interleave_by_timestamp(transcript, visual_text)
                    tag = "user-enriched" if user_forced_vlm else "hybrid"
                    status = (status or "") + f" | {tag} (+{unit})"
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

    # Pick prompt family based on transcript shape:
    #   - Speech-heavy → standard PROMPT_BRIEF / KEY_POINTS / CHAPTERS
    #   - Visual-heavy (VLM scene markers present) → SILENT variants that
    #     LEAD with content identity (title + channel + OCR) and treat
    #     visual descriptions as supporting context rather than the spine.
    is_visual_heavy = _is_visual_heavy_transcript(transcript)
    channel_name = await _fetch_channel_name(job.video_id) if is_visual_heavy else ""
    if is_visual_heavy:
        log.info("[%s] Visual-heavy transcript detected — using silent-video prompts",
                 job.video_id)
        prompt_brief = PROMPT_BRIEF_SILENT
        prompt_key_points = PROMPT_KEY_POINTS_SILENT
        prompt_chapters = PROMPT_CHAPTERS_SILENT
        reduce_brief = REDUCE_BRIEF_SILENT
        reduce_key_points = REDUCE_KEY_POINTS_SILENT
    else:
        prompt_brief = PROMPT_BRIEF
        prompt_key_points = PROMPT_KEY_POINTS
        prompt_chapters = PROMPT_CHAPTERS
        reduce_brief = REDUCE_BRIEF
        reduce_key_points = REDUCE_KEY_POINTS

    # Apply per-channel model override for the duration of this job's
    # summarize calls. ContextVar isolates per-task so concurrent jobs
    # don't trample each other.
    _token = _model_override.set(job.model_override) if job.model_override else None
    try:
        brief, key_points, chapters_raw = await asyncio.gather(
            summarize(
                transcript, prompt_brief, LLM_MAX_TOKENS_BRIEF,
                reduce_template=reduce_brief,
                title=title, duration=duration_str, reference_block=ref_block,
                channel=channel_name,
            ),
            summarize(
                transcript, prompt_key_points, LLM_MAX_TOKENS_KEY_POINTS,
                reduce_template=reduce_key_points,
                title=title, duration=duration_str, reference_block=ref_block,
                char_cap=SUMMARY_CHAR_CAP,
                channel=channel_name,
            ),
            summarize(
                transcript, prompt_chapters, LLM_MAX_TOKENS_CHAPTERS,
                reduce_template=None,  # chapters are time-ordered; concat preserves chronology
                title=title, duration=duration_str, reference_block=ref_block,
                tail_start=tail_start, char_cap=SUMMARY_CHAR_CAP,
                channel=channel_name,
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
    detail_channel = resolve_summary_channel(job.channel)
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

    # 4th embed: Community Reaction (top YT comments). Only when:
    #   - the job opted in (job.yt_comments_enabled, default true)
    #   - the server returned a non-empty comment list
    #   - filtering left enough substance to summarise
    # Cache hits skip this — comments aren't persisted in the cache today.
    yt_comments_summary: str | None = None
    if job.yt_comments_enabled and raw_comments:
        filtered = filter_yt_comments(raw_comments)
        log.info("[%s] YT comments: %d raw, %d after filter",
                 job.video_id, len(raw_comments), len(filtered))
        if filtered:
            comment_md = format_yt_comments(filtered)
            try:
                yt_comments_summary = await summarize(
                    comment_md, PROMPT_YT_COMMENTS, LLM_MAX_TOKENS_KEY_POINTS,
                    reduce_template=REDUCE_YT_COMMENTS,
                    title=title, duration=duration_str,
                    char_cap=SUMMARY_CHAR_CAP,
                )
                yt_comments_summary = sanitize_llm_output(yt_comments_summary)
            except Exception as e:
                # Non-fatal — log and skip the embed. Main video summary
                # still posts. (Comment summary is supplementary.)
                log.warning("[%s] Comment summary failed (non-fatal): %s",
                            job.video_id, e)
                yt_comments_summary = None
    if yt_comments_summary:
        await send_long_embed(
            detail_channel, "Community Reaction",
            yt_comments_summary, 0xFF3366,  # YT-red-pink, distinct from chapters
        )

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

    # Count what we actually posted. Base: TL;DW + Key Points + Chapters = 3.
    # +1 for the split-summary header card. +1 when Community Reaction fired.
    posted_count = 3 + (1 if use_split else 0) + (1 if yt_comments_summary else 0)
    log.info("[%s] Done — posted %d embeds%s%s",
             job.video_id, posted_count,
             " (split summary)" if use_split else "",
             " + Community Reaction" if yt_comments_summary else "")


async def process_url(job: Job):
    """Web-article handler. Mirrors process() but skips download/transcribe/VLM.

    Prompt selection:
    - Reddit URLs (or scraped bodies that contain Reddit-discussion markers)
      → Reddit-flavoured prompts that surface BOTH the linked article AND
      community comment reactions.
    - Everything else → generic article prompts (PROMPT_*_WEB +
      PROMPT_SECTIONS).

    Cache strategy: same {hash}.txt format as transcripts. The hash here is
    the URL hash, so two identical URL summary requests share a cache hit;
    different URLs to the same article (canonicalisation differences) miss
    each other — acceptable, we don't try to canonicalise URLs.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")

    # 1. Cache lookup
    cached = read_cache(job.video_id)
    if cached is not None:
        title, status, body, _duration = cached
        log.info("[%s] Cache hit (%d chars, '%s')", job.video_id, len(body), title)
    else:
        # 2. Scrape via Crawl4AI → FlareSolverr fallback. Errors here are
        # transient by default (network / scraper container down). Hard
        # failures (4xx scheme reject) raise PermanentError from
        # _fetch_via_crawl4ai.
        log.info("[%s] Scraping %s ...", job.video_id, job.url)
        await _job_react(job, PROCESSING_EMOJI_WEB)  # 📰 — distinguishes web from audio fetch
        title, body = await fetch_article(job.url)
        status = f"scraped {len(body)} chars"
        log.info("[%s] Scraped: %s (%d chars)", job.video_id, title, len(body))
        # Persist before LLM step so a transient LLM failure doesn't force a
        # re-scrape on retry. duration=0 for web jobs (no runtime concept).
        write_cache(job.video_id, title, status, body, 0)

    # 3. Optional: Exa context for terminology spelling. Same rationale as
    # video pipeline — gives the LLM a glossary without polluting the body.
    web_context = await search_topic_context(title)
    if web_context:
        log.info("[%s] Got web context (%d chars)", job.video_id, len(web_context))

    # 4. Reference + user-prompt blocks (same convention as process()).
    ref_block = ""
    if web_context:
        ref_block = (
            "Reference material — USE FOR SPELLING/TERMINOLOGY ONLY. "
            "Do NOT copy facts, dates, numbers, or claims from this into the summary. "
            "Summary content must come exclusively from the article below.\n"
            "<reference>\n"
            f"{web_context[:REFERENCE_CHAR_CAP]}\n"
            "</reference>\n\n"
        )

    if job.user_prompt:
        ref_block = (
            "The Discord user who requested this summary specifically asked: "
            "<user_request>\n"
            f"{job.user_prompt}\n"
            "</user_request>\n"
            "Honour that request when shaping your output — emphasise the "
            "aspects they're interested in, while still covering the rest of "
            "the article. The user_request is steering, not data to "
            "summarise.\n\n"
        ) + ref_block

    from urllib.parse import urlparse
    source = urlparse(job.url).hostname or job.url

    # Reddit content has a multi-source structure (linked article + OP + top
    # comments) that the generic web prompts ignore — they treat the whole
    # blob as one article and drop the comment discussion. Detect "discussion
    # thread" content (Reddit OR HackerNews — both produce the same
    # structural shape: linked article + post + top comments) and switch to
    # discussion-aware prompts.
    is_discussion_content = (
        _is_reddit_post_url(job.url)
        or _is_hn_post_url(job.url)
        or "# Reddit discussion" in body
        or "# HackerNews discussion" in body
        or ("## Top " in body and " comments" in body)
    )
    if is_discussion_content:
        log.info("[%s] Discussion-thread content detected — using "
                 "discussion-aware prompts", job.video_id)
        prompt_brief = PROMPT_BRIEF_REDDIT
        prompt_key_points = PROMPT_KEY_POINTS_REDDIT
        prompt_sections = PROMPT_SECTIONS_REDDIT
        reduce_brief = REDUCE_BRIEF_REDDIT
        reduce_key_points = REDUCE_KEY_POINTS_REDDIT
        reduce_sections = REDUCE_SECTIONS_REDDIT
    else:
        prompt_brief = PROMPT_BRIEF_WEB
        prompt_key_points = PROMPT_KEY_POINTS_WEB
        prompt_sections = PROMPT_SECTIONS
        reduce_brief = REDUCE_BRIEF_WEB
        reduce_key_points = REDUCE_KEY_POINTS_WEB
        reduce_sections = REDUCE_SECTIONS

    log.info("[%s] Summarizing article (%d chars)...", job.video_id, len(body))
    await _job_react(job, "\U0001f9e0")  # 🧠

    _token = _model_override.set(job.model_override) if job.model_override else None
    try:
        brief, key_points, sections = await asyncio.gather(
            summarize(
                body, prompt_brief, LLM_MAX_TOKENS_BRIEF,
                reduce_template=reduce_brief,
                title=title, source=source, reference_block=ref_block,
            ),
            summarize(
                body, prompt_key_points, LLM_MAX_TOKENS_KEY_POINTS,
                reduce_template=reduce_key_points,
                title=title, source=source, reference_block=ref_block,
                char_cap=SUMMARY_CHAR_CAP,
            ),
            summarize(
                body, prompt_sections, LLM_MAX_TOKENS_CHAPTERS,
                reduce_template=reduce_sections,
                title=title, source=source, reference_block=ref_block,
                char_cap=SUMMARY_CHAR_CAP,
            ),
        )
    finally:
        if _token is not None:
            _model_override.reset(_token)

    brief = sanitize_llm_output(brief)
    key_points = sanitize_llm_output(key_points)
    sections = sanitize_llm_output(sections)

    # 5. Post embeds. Same routing rule as video: detail goes to summary
    # channel (when configured for this server), brief stays in-channel.
    detail_channel = resolve_summary_channel(job.channel)
    use_split = detail_channel.id != job.channel.id

    detail_msg = None
    if use_split:
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
            color=0x4A90E2,  # blue — visual distinguisher from video (red)
        )
        header.set_footer(text=source)
        detail_msg = await detail_channel.send(embed=header)

    await send_long_embed(detail_channel, "Key Points", key_points, 0x357ABD)
    await send_long_embed(detail_channel, "Sections", sections, 0x5DADE2)

    # Brief embed in the original channel
    embed = discord.Embed(
        title=f"TL;DR: {truncate(title, 240)}",
        url=job.url,
        description=truncate(brief, 4000),
        color=0x4A90E2,
    )
    embed.set_footer(text=f"{source} | {status}")

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

    if job.message is not None:
        await job.channel.send(embed=embed, reference=job.message)
    else:
        await job.channel.send(embed=embed)

    for emoji in PROCESSING_EMOJI:
        await _job_remove_react(job, emoji)
    await _job_react(job, "\u2705")  # ✅

    log.info("[%s] Done — posted web summary", job.video_id)


# ─── AI litmus handler ───────────────────────────────────────────────────────


_SEVERITY_DOT = {"low": "🟢", "med": "🟡", "high": "🔴"}


def _format_litmus_signals(signals: dict,
                           adsense_detected: bool,
                           author: str | None,
                           domain_severity: str,
                           domain_text: str) -> str:
    """Render the regex + metadata signals as a readable bullet list for
    the embed. Each line: `<dot> <signal name>: <details>`.
    """
    lines: list[str] = []

    def line(name: str, sev: str, detail: str) -> None:
        dot = _SEVERITY_DOT.get(sev, "⚪")
        lines.append(f"{dot} **{name}**: {detail}")

    if "too_short" in signals:
        line("Article too short", "low",
             f"only {signals['too_short']['len']} chars; stylistic detection "
             f"needs ~200+ chars of prose.")
        # No further regex signals are meaningful on a stub
        line("Domain age", domain_severity, domain_text)
        if adsense_detected:
            line("AdSense", "med", "ad-network markers detected in HTML")
        if author:
            line("Author byline", "low", f"present ({author})")
        else:
            line("Author byline", "med", "no `<meta name=author>` or rel=author")
        return "\n".join(lines)

    # Stylistic
    if "llm_tic_phrases" in signals:
        s = signals["llm_tic_phrases"]
        examples = ", ".join(f"`{p}` ×{c}" for p, c in s["examples"][:4])
        line("LLM-tic phrases", s["severity"],
             f"{s['count']} hits ({s['density_per_1k']}/1k words). {examples}")
    else:
        line("LLM-tic phrases", "low", "0 hits")

    if "buzzwords" in signals:
        s = signals["buzzwords"]
        examples = ", ".join(f"`{p}` ×{c}" for p, c in s["examples"][:4])
        line("Generic buzzwords", s["severity"],
             f"{s['count']} hits ({s['density_per_1k']}/1k words). {examples}")
    else:
        line("Generic buzzwords", "low", "0 hits")

    if "hedges" in signals:
        s = signals["hedges"]
        line("Hedge phrases", s["severity"],
             f"{s['count']} hits (e.g. \"it's worth noting\")")
    else:
        line("Hedge phrases", "low", "0 hits")

    if "em_dash_density" in signals:
        s = signals["em_dash_density"]
        line("Em-dash density", s["severity"],
             f"{s['count']} dashes ({s['density_per_1k']}/1k words; "
             f"typical human ≈ 2-3)")
    else:
        line("Em-dash density", "low", "within typical-human range")

    if "listicle_structure" in signals:
        s = signals["listicle_structure"]
        line("Listicle structure", s["severity"],
             f"{s['headings']} headings, {s['bullets']} bullets "
             f"({s['density_per_1k']}/1k words)")
    else:
        line("Listicle structure", "low", "prose-heavy, not heading-heavy")

    if "low_substance" in signals:
        s = signals["low_substance"]
        line("Substance", s["severity"],
             f"few specific markers — quotes:{s['quotes']}, "
             f"named titles:{s['named_titles']}, year refs:{s['year_refs']}, "
             f"$ refs:{s['monetary_refs']}, %:{s['percentages']} "
             f"({s['density_per_1k']}/1k words)")
    else:
        line("Substance", "low", "specific quotes / dates / numbers present")

    # Metadata
    line("Domain age", domain_severity, domain_text)

    if author:
        line("Author byline", "low", f"present ({author})")
    else:
        line("Author byline", "med", "no `<meta name=author>` or rel=author")

    if adsense_detected:
        line("AdSense", "med", "ad-network markers detected in HTML")
    else:
        line("AdSense", "low", "not detected")

    return "\n".join(lines)


def _signals_summary_for_prompt(signals: dict,
                                adsense_detected: bool,
                                author: str | None,
                                domain_severity: str,
                                domain_text: str) -> str:
    """Compact signals summary fed into PROMPT_LITMUS as `{signals_summary}`.
    Same content as the embed format but more compact and without emoji.
    """
    parts = []
    for name, info in signals.items():
        sev = info.get("severity", "low")
        detail_bits = []
        for k, v in info.items():
            if k in ("severity", "examples"):
                continue
            detail_bits.append(f"{k}={v}")
        parts.append(f"- {name} ({sev}): {', '.join(detail_bits)}")
    parts.append(f"- domain_age ({domain_severity}): {domain_text}")
    parts.append(f"- author_byline: {'present (' + author + ')' if author else 'missing'}")
    parts.append(f"- adsense: {'detected' if adsense_detected else 'not detected'}")
    return "\n".join(parts)


# Hard-cap article excerpt sent to the LLM for the qualitative read.
# We don't need the full body — a representative excerpt is enough to gauge
# voice / structure / substance, and full bodies blow context for marginal
# qualitative gain.
LITMUS_EXCERPT_CHARS = int(os.environ.get("LITMUS_EXCERPT_CHARS", "8000"))


async def process_litmus(job: Job):
    """Surface stylistic + metadata signals that a web article may be
    LLM-generated or LLM-heavy-edited. Forensic output (signals + qualitative
    read), no verdict.

    Pipeline:
      1. Scrape via fetch_article (reuses Crawl4AI / FlareSolverr / Reddit
         path).
      2. Regex pre-pass over the scraped Markdown — LLM-tic phrases,
         em-dash density, hedge phrases, buzzwords, listicle structure,
         substance markers (positive signal).
      3. Parallel metadata fetch — Crawl4AI /html for AdSense + author,
         Wayback /available for domain age.
      4. Aggregate severity. If the score lands in the ambiguous middle
         range, call the LLM for a qualitative description.
      5. Compose a forensic embed: signals list + qualitative read +
         caveat about detection unreliability.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")

    # Scraping reaction — same 📰 emoji as the web flow
    await _job_react(job, PROCESSING_EMOJI_WEB)

    log.info("[%s] Litmus: scraping %s", job.video_id, job.url)
    title, body = await fetch_article(job.url)
    log.info("[%s] Litmus: scraped %d chars", job.video_id, len(body))

    # Regex pre-pass + metadata fetches concurrently
    signals = _regex_signals(body)
    raw_html, wayback_ts = await asyncio.gather(
        _fetch_raw_html(job.url),
        _wayback_first_seen(job.url),
        return_exceptions=False,
    )
    adsense_detected = _detect_adsense(raw_html or "")
    author = _extract_author_from_html(raw_html or "")
    domain_severity, domain_text = _domain_age_severity(wayback_ts)

    score = _aggregate_severity(
        signals,
        adsense_detected=adsense_detected,
        author_present=bool(author),
        domain_severity=domain_severity,
    )
    log.info("[%s] Litmus: severity score %d (skip-below=%d, skip-above=%d)",
             job.video_id, score, LITMUS_SKIP_LLM_BELOW, LITMUS_SKIP_LLM_ABOVE)

    # Qualitative LLM read only for the ambiguous middle range
    qualitative: str | None = None
    if LITMUS_SKIP_LLM_BELOW < score < LITMUS_SKIP_LLM_ABOVE:
        await _job_react(job, "\U0001f9e0")  # 🧠
        log.info("[%s] Litmus: ambiguous → calling LLM for qualitative read",
                 job.video_id)
        from urllib.parse import urlparse
        excerpt = body[:LITMUS_EXCERPT_CHARS]
        signals_summary = _signals_summary_for_prompt(
            signals, adsense_detected, author, domain_severity, domain_text,
        )
        _token = _model_override.set(job.model_override) if job.model_override else None
        try:
            qualitative = await summarize(
                excerpt, PROMPT_LITMUS, LLM_MAX_TOKENS_BRIEF,
                reduce_template=None,  # single-pass; excerpt is bounded
                title=title,
                source=urlparse(job.url).hostname or job.url,
                signals_summary=signals_summary,
                reference_block="",
            )
            qualitative = sanitize_llm_output(qualitative)
        except Exception as e:
            log.warning("[%s] Litmus LLM call failed (non-fatal): %s",
                        job.video_id, e)
            qualitative = None
        finally:
            if _token is not None:
                _model_override.reset(_token)
    elif score <= LITMUS_SKIP_LLM_BELOW:
        log.info("[%s] Litmus: low score → skipping LLM (clear-clean)", job.video_id)
    else:
        log.info("[%s] Litmus: high score → skipping LLM (clear-LLM-style)", job.video_id)

    # Compose embed
    from urllib.parse import urlparse
    source = urlparse(job.url).hostname or job.url
    body_lines = [
        "**Detected signals:**",
        _format_litmus_signals(
            signals, adsense_detected, author, domain_severity, domain_text,
        ),
    ]
    if qualitative:
        body_lines.append(f"\n**Qualitative read** (`{job.model_override or LLM_MODEL}`):")
        body_lines.append(qualitative.strip())
    elif score <= LITMUS_SKIP_LLM_BELOW:
        body_lines.append(
            "\n*Few stylistic markers detected — typical-human-range. "
            "Skipped LLM qualitative read (signals were clear).*"
        )
    else:
        body_lines.append(
            "\n*Multiple strong LLM-style markers detected. "
            "Skipped LLM qualitative read (signals were clear).*"
        )

    body_lines.append(
        "\n⚠️ *AI detection is fundamentally unreliable. False positives are "
        "common on careful technical writing; false negatives common on "
        "lightly-edited LLM output. Treat this as forensic signals only, "
        "not a verdict.*"
    )

    description = "\n".join(body_lines)
    embed = discord.Embed(
        title=f"🔍 Litmus: {truncate(title, 200)}",
        url=job.url,
        description=truncate(description, 4000),
        color=0xA855F7,  # purple — distinct from video (red), web (blue), reddit
    )
    embed.set_footer(text=f"{source} | severity score {score}")

    # Litmus output is always a single embed in the original channel — no
    # split to summary_channel (it's not a "summary"; it's a forensic note).
    if job.message is not None:
        await job.channel.send(embed=embed, reference=job.message)
    else:
        await job.channel.send(embed=embed)

    for emoji in PROCESSING_EMOJI:
        await _job_remove_react(job, emoji)
    await _job_react(job, "\u2705")  # ✅
    log.info("[%s] Litmus done — score %d, llm=%s",
             job.video_id, score, "yes" if qualitative else "no")


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
    if http is None: raise RuntimeError("HTTP session not initialised")
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


# Common English stopwords + domain-frequent generics. Removed before
# Jaccard similarity comparison so paraphrased near-duplicates dedup
# against each other instead of being kept as "distinct" content.
_DEDUP_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "are", "from", "have",
    "has", "was", "were", "been", "being", "into", "their", "they",
    "which", "what", "when", "where", "while", "would", "could", "should",
    "but", "not", "any", "all", "one", "two", "out", "use", "used",
    # Domain-frequent — every summary mentions these so they don't
    # discriminate between distinct bullets.
    "video", "trailer", "story", "scene", "narrative",
})


def _dedup_compare_set(line: str) -> set[str]:
    """Word set used for Jaccard similarity. Strips bullet markers,
    lowercases, keeps alpha tokens ≥3 chars, removes stopwords."""
    s = re.sub(r"^[\s\-\*\d.•]+", "", line.strip()).lower()
    words = re.findall(r"[a-z']{3,}", s)
    return {w for w in words if w not in _DEDUP_STOPWORDS}


def _dedup_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedup_lines(text: str, similarity_threshold: float = 0.5) -> str:
    """Drop duplicate and near-duplicate lines from LLM output.

    Local LLMs occasionally enter degenerate loops where they emit
    paraphrased variations of the same sentence many times (especially
    on repetitive input like VLM-only transcripts of silent videos).
    Two-stage dedup catches both:

    1. Exact-match (key = lowercase, bullet-markers stripped). Cheap,
       O(1) lookup. Catches verbatim repeats.
    2. Jaccard word-set similarity (stopwords + domain-generic words
       removed). Catches paraphrase loops where the LLM rewords the
       same point — "X is a creative performance" vs "X is an
       innovative showcase". Default threshold 0.5 ≈ "half the
       distinctive words overlap".

    Short lines (<20 chars after normalisation) and lines with fewer
    than 4 distinctive words pass through unchanged — typically section
    headers, timestamp markers, or single-word emphasis. Logs the count
    of dropped lines when it fires.
    """
    if not text:
        return text
    seen_exact: set[str] = set()
    seen_sets: list[set[str]] = []
    out: list[str] = []
    dropped = 0
    for line in text.splitlines(keepends=False):
        key = re.sub(r"^[\s\-\*\d.•]+", "", line.strip()).lower()
        if not key or len(key) < 20:
            out.append(line)
            continue
        if key in seen_exact:
            dropped += 1
            continue
        word_set = _dedup_compare_set(line)
        if len(word_set) >= 4:
            is_dup = False
            for prev in seen_sets:
                if _dedup_jaccard(word_set, prev) >= similarity_threshold:
                    is_dup = True
                    break
            if is_dup:
                dropped += 1
                continue
            seen_sets.append(word_set)
        seen_exact.add(key)
        out.append(line)
    if dropped > 0:
        log.info("Dedup: dropped %d duplicate/near-duplicate lines", dropped)
    return _strip_incomplete_trailing_sentence("\n".join(out))


def _strip_incomplete_trailing_sentence(text: str) -> str:
    """Drop the last line if it ends mid-sentence (no terminal
    punctuation). LLMs occasionally truncate when they hit max_tokens
    while in the middle of a paraphrase loop — that final fragment
    ("The video is a creative and innovative performance that") is
    ugly and adds nothing. Safer to drop it than to leave it dangling.
    """
    if not text or not text.strip():
        return text
    lines = text.rstrip().splitlines()
    if len(lines) < 2:
        return text
    last = lines[-1].rstrip()
    if not last:
        return text
    # Sentence enders include English + CJK punctuation + colon/semi-colon
    # (some markdown bullets legitimately end with `:` to introduce a list).
    if last[-1] in ".!?。！？:;":
        return text
    # If the last line is very short (a label or fragment), it might be
    # legitimate; only drop if it looks like a sentence fragment (>3 words).
    if len(last.split()) <= 3:
        return text
    # Drop the trailing fragment
    log.info("Dedup: dropped trailing mid-sentence fragment: %r", last[:60])
    return "\n".join(lines[:-1])


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
            return _dedup_lines(await _llm_call(prompt, max_tokens))
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

    # Map: per-chunk summarization, sequential WITHIN this style. Note that
    # process() runs three styles (brief / key_points / chapters) concurrently
    # via asyncio.gather, so the LLM still sees up to 3 in-flight requests at
    # any time — sequencing here only bounds chunk fan-out within a single
    # style. Each call recurses through summarize() with no reduce_template
    # so an overflow on one chunk halves only that chunk, not the whole
    # pipeline.
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
        # Dedup applies here too — chapters from neighbouring chunks
        # sometimes generate the same heading at the boundary, and that
        # dedups cleanly.
        return _dedup_lines(combined)

    # Run the reduce step. If combined partials still exceed budget, recurse
    # — same machinery handles it. Critical: pass reduce_template through so
    # the deeper recursion still produces a coherent reduce, not concat.
    # The recursive call's own return path is already deduped.
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


# Pattern that uniquely identifies VLM-scene-rendered transcript content.
# _format_scenes emits `[t1-t2] (N frames) <desc>` for multi-frame scenes
# and `[t] <desc> — text on screen: "..."` when OCR is present. Detecting
# either marker tells us this transcript came from the visual pipeline,
# not from whisper speech.
_VISUAL_HEAVY_MARKER_RE = re.compile(
    r"\[\d+:\d{2}-\d+:\d{2}\]"        # time-range marker (multi-frame scene)
    r"|\(\d+ frames\)"                 # frame-count annotation
    r"|text on screen:",               # OCR annotation
)


def _is_visual_heavy_transcript(text: str) -> bool:
    """True iff transcript was generated primarily from VLM scene
    descriptions (vs whisper speech). Used to switch to silent-video
    prompts that lead with content identity instead of "main thesis".
    Works on both fresh transcripts and cache hits (markers persist).
    """
    if not text:
        return False
    return bool(_VISUAL_HEAVY_MARKER_RE.search(text))


_CHANNEL_NAME_RE = re.compile(
    r'"ownerChannelName"\s*:\s*"([^"]+)"'
    r'|<meta\s+itemprop="author"[^>]+content="([^"]+)"'
    r'|<link\s+itemprop="name"\s+content="([^"]+)"',
    re.IGNORECASE,
)


async def _fetch_channel_name(video_id: str) -> str:
    """Pull the YouTube channel name for a video.

    For silent-video summaries the channel name is high-signal — it
    often tells you immediately what KIND of content the video is
    (e.g. "Shittyflute" = comedy parody, "VEVO" = music video,
    "ASMR Glow" = ASMR). The brief / key-points prompts use it as
    grounding context the VLM can't provide on its own.

    Returns empty string on any failure — silent-video prompts handle
    missing channel gracefully.
    """
    if http is None or not video_id:
        return ""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en"}
        async with http.get(
            url, headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return ""
            page = await resp.text()
    except (asyncio.TimeoutError, aiohttp.ClientError):
        return ""
    m = _CHANNEL_NAME_RE.search(page)
    if not m:
        return ""
    # Three alternation groups; take the first non-None
    return next((g for g in m.groups() if g), "").strip()[:120]


async def fetch_video_description(video_id: str) -> str:
    """Fetch full YouTube video description from page JSON data."""
    if http is None: raise RuntimeError("HTTP session not initialised")
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
    if http is None: raise RuntimeError("HTTP session not initialised")
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


# Discord embed limits per message: title 256, description 4096, fields 25,
# field name 256, field value 1024, footer 2048, AND total payload ≤ 6000
# chars summed across title + description + fields + footer + author. If we
# fill description to 4000 + a 256-char title + a 1000-char field + 200-char
# footer we're at 5456 — fine. The guard below trims description to keep us
# safely under the total limit even if title/footer grow.
EMBED_TOTAL_LIMIT = 6000


def _safe_description_len(title: str, footer: str = "", extra: int = 0) -> int:
    """How many description chars we can spend without breaching the
    6000-char total cap. `extra` is reserved for fields the caller will add.
    """
    overhead = len(title or "") + len(footer or "") + extra
    return max(256, EMBED_TOTAL_LIMIT - overhead - 64)  # 64 char safety margin


async def send_long_embed(channel, title: str, content: str, color: int):
    """Send embed, splitting into continuation embeds if >4000 chars.

    Each chunk respects the 4096 description cap AND the 6000 total payload
    cap. Title is the only other contributor here (no fields, no footer),
    so we cap at min(4000, 6000 - len(title)).
    """
    cap = min(4000, _safe_description_len(title))
    chunks = split_content(content, cap)
    for i, chunk in enumerate(chunks):
        t = title if i == 0 else f"{title} (cont.)"
        embed = discord.Embed(title=t, description=chunk, color=color)
        await channel.send(embed=embed)


# Matches timestamps the LLM might output. Order matters: bracketed forms
# first (so we don't double-replace), then bare forms anchored at line
# start or after a markdown bold marker.
TIMESTAMP_RE = re.compile(
    r"\[(\d{1,3}):(\d{2}):(\d{2})\]"         # [H:MM:SS]
    r"|\[(\d{1,3}):(\d{2})\]"                # [MM:SS]
    r"|(?:^|\*\*)(\d{1,3}):(\d{2}):(\d{2})"  # bare H:MM:SS
    r"|(?:^|\*\*)(\d{1,3}):(\d{2})(?=\s)",   # bare MM:SS followed by space
    re.MULTILINE
)

# Normalisation pre-pass: bracketed expressions that are NOT a clean
# `[H:MM:SS]` or `[MM:SS]` but DO contain a timestamp inside (e.g.
# `[0 and 0:05:46]` from an LLM that conflated two moments). Strip the
# noise and keep only the first valid timestamp so linkify_timestamps
# below renders it as a clickable link.
_MALFORMED_TS_BRACKET_RE = re.compile(
    r"\[(?P<inner>[^\[\]]*?)\]"
)
_INNER_TS_RE = re.compile(r"(\d{1,3}):(\d{2})(?::(\d{2}))?")


def _normalize_chapter_timestamps(text: str) -> str:
    """Find `[...]` brackets containing a timestamp and replace with a
    clean `[H:MM:SS]` / `[MM:SS]`. Brackets without a timestamp are left
    alone (they're presumably real markdown link text or other content).
    """
    def fix(m: "re.Match") -> str:
        inner = m.group("inner")
        # If the inner content already matches a clean timestamp pattern,
        # leave it alone — saves work and avoids accidental rewrites.
        if re.fullmatch(r"\d{1,3}:\d{2}(?::\d{2})?", inner.strip()):
            return m.group(0)
        ts = _INNER_TS_RE.search(inner)
        if not ts:
            return m.group(0)
        a, b, c = ts.groups()
        if c is not None:
            return f"[{int(a)}:{int(b):02d}:{int(c):02d}]"
        return f"[{int(a)}:{int(b):02d}]"
    return _MALFORMED_TS_BRACKET_RE.sub(fix, text)


def linkify_timestamps(text: str, video_id: str) -> str:
    """Replace timestamps with clickable YouTube timestamp links.

    Runs a normalisation pre-pass first to clean up malformed brackets
    like `[0 and 0:05:46]` (LLM artefact) → `[0:05:46]`.
    """
    text = _normalize_chapter_timestamps(text)
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


# ─── AI litmus test ──────────────────────────────────────────────────────────
# Surfaces stylistic + metadata signals that an article may be LLM-generated
# or LLM-heavy-edited. Forensic output (signals + qualitative read) — not a
# verdict; AI detection is fundamentally unreliable. The honest framing is
# "here are signals; you decide", not "this is AI".

# Phrases that LLMs (especially RLHF-trained chat models) reach for far more
# than human writers. Many have legitimate uses; what matters is DENSITY in
# a single article. Pulled from observed LLM output patterns + published
# style-detection research.
LLM_TIC_PHRASES = (
    # Vocabulary tics
    "delve into", "delving into", "delve deeper",
    "tapestry of", "rich tapestry",
    "navigate the landscape", "navigating the landscape",
    "underscore the importance", "underscores the importance",
    "shed light on", "shed some light",
    "in the realm of", "in the world of",
    "in today's fast-paced world", "in the digital age",
    "stand the test of time",
    "embark on", "embark upon",
    "myriad of", "a plethora of",
    "showcase", "showcasing",
    "whether it be", "be it",
    # Conclusion / transition tics
    "in conclusion", "to conclude", "in essence",
    "moreover", "furthermore",
    "ultimately",
    # Hedging tics
    "it's worth noting", "it is worth noting",
    "it's important to note", "it is important to note",
    "it should be noted", "it bears mentioning",
    # Buzzword adjectives — separate scoring; see _BUZZWORDS below
)

# Buzzwords scored separately because their density curve differs (a single
# "robust" is fine; five in one article is a tic). Stacked buzzwords are a
# strong stylistic marker.
LLM_BUZZWORDS = (
    "robust", "seamless", "comprehensive", "cutting-edge", "cutting edge",
    "game-changer", "game changer", "revolutionary", "transformative",
    "leverage", "leveraging", "elevate", "elevating",
    "commendable", "noteworthy", "paramount", "pivotal",
    "unparalleled", "unprecedented",
)

# Hedge phrases scored separately; LLMs over-hedge to seem cautious.
LLM_HEDGE_PHRASES = (
    "it's important to note",
    "it is important to note",
    "it's worth noting",
    "it is worth noting",
    "it should be noted",
    "it is worth mentioning",
    "it bears mentioning",
    "it's crucial to remember",
    "it's essential to consider",
    "it goes without saying",
)

# Severity → numeric score for aggregation (higher = stronger LLM signal)
_SEVERITY_SCORE = {"low": 0, "med": 1, "high": 2}
# Aggregate-score thresholds for the "ambiguous → call LLM" decision.
# Sum of all signals' severity scores. Two signals = clearly clean = skip;
# eight = clearly LLM-style = skip; in between = call the LLM for nuance.
LITMUS_SKIP_LLM_BELOW = int(os.environ.get("LITMUS_SKIP_LLM_BELOW", "2"))
LITMUS_SKIP_LLM_ABOVE = int(os.environ.get("LITMUS_SKIP_LLM_ABOVE", "8"))

# Wayback API timeout — short, since Wayback is occasionally slow and we
# don't want to hold up the whole litmus job.
WAYBACK_TIMEOUT = int(os.environ.get("WAYBACK_TIMEOUT", "8"))


def _per_1000(count: int, total_words: int) -> float:
    if total_words <= 0:
        return 0.0
    return (count * 1000.0) / total_words


def _count_phrases(text: str, phrases) -> tuple[int, list[tuple[str, int]]]:
    """Return (total_count, [(phrase, count), ...]) for case-insensitive
    word-boundaried matches. Phrases with hyphens / apostrophes are
    matched literally within word boundaries.
    """
    hits: list[tuple[str, int]] = []
    total = 0
    for phrase in phrases:
        # Word boundary on each end; phrase may contain spaces/hyphens.
        pat = r"(?:^|[^\w])" + re.escape(phrase) + r"(?:[^\w]|$)"
        c = len(re.findall(pat, text, re.IGNORECASE))
        if c > 0:
            hits.append((phrase, c))
            total += c
    return total, hits


def _regex_signals(text: str) -> dict:
    """Run cheap regex / structural detectors on scraped article text.

    Returns dict of {signal_name: {count|density|examples|severity}}.
    Only includes signals that fired (i.e. count > 0) — keeps the embed
    short. Severities are calibrated against typical-human baseline:
      low  = within human range
      med  = elevated; one of several signals would be unremarkable
      high = beyond typical-human density
    """
    signals: dict = {}
    if not text or len(text) < 200:
        # Too short for stylistic analysis — return marker
        signals["too_short"] = {"len": len(text), "severity": "low"}
        return signals

    word_count = len(re.findall(r"\b\w+\b", text))

    # 1. LLM tic phrases
    tic_count, tic_hits = _count_phrases(text, LLM_TIC_PHRASES)
    if tic_count > 0:
        density = _per_1000(tic_count, word_count)
        sev = "high" if density >= 3 else ("med" if density >= 1 else "low")
        signals["llm_tic_phrases"] = {
            "count": tic_count,
            "density_per_1k": round(density, 2),
            "examples": sorted(tic_hits, key=lambda kv: -kv[1])[:6],
            "severity": sev,
        }

    # 2. Buzzwords
    buzz_count, buzz_hits = _count_phrases(text, LLM_BUZZWORDS)
    if buzz_count > 0:
        density = _per_1000(buzz_count, word_count)
        sev = "high" if density >= 3 else ("med" if density >= 1 else "low")
        signals["buzzwords"] = {
            "count": buzz_count,
            "density_per_1k": round(density, 2),
            "examples": sorted(buzz_hits, key=lambda kv: -kv[1])[:6],
            "severity": sev,
        }

    # 3. Hedge phrases (slightly different threshold — denser human writing
    # uses these legitimately, but more than 4 in one article is unusual).
    hedge_count, hedge_hits = _count_phrases(text, LLM_HEDGE_PHRASES)
    if hedge_count > 0:
        sev = "high" if hedge_count >= 5 else ("med" if hedge_count >= 2 else "low")
        signals["hedges"] = {
            "count": hedge_count,
            "examples": hedge_hits[:4],
            "severity": sev,
        }

    # 4. Em-dash density. — character + literal `--`. LLMs (notably GPT
    # family) over-use em-dashes; humans average ~2-3 per 1000 words.
    em_count = text.count("—") + text.count(" -- ")
    em_density = _per_1000(em_count, word_count)
    if em_density >= 2:
        sev = "high" if em_density >= 8 else ("med" if em_density >= 5 else "low")
        signals["em_dash_density"] = {
            "count": em_count,
            "density_per_1k": round(em_density, 2),
            "severity": sev,
        }

    # 5. Listicle structure — heading + bullet ratio per 1000 words. High
    # density of structural markers in short prose is a "content farm" tell.
    headings = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
    bullets = len(re.findall(r"^\s*[-*+]\s", text, re.MULTILINE))
    structure_density = _per_1000(headings + bullets, word_count)
    if structure_density >= 5:
        sev = "high" if structure_density >= 15 else ("med" if structure_density >= 10 else "low")
        signals["listicle_structure"] = {
            "headings": headings,
            "bullets": bullets,
            "density_per_1k": round(structure_density, 2),
            "severity": sev,
        }

    # 6. Substance markers (POSITIVE — presence reduces overall LLM signal).
    # Specific quotes, named individuals, dated events, numeric specifics.
    # Counted as a single "substance" signal with severity inverted: more
    # substance = lower severity (= less LLM-like).
    quoted = len(re.findall(r'"[^"\n]{20,}"', text))   # >=20 char quotes
    named = len(re.findall(r"\b(?:Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|Sen\.|Rep\.)\s+[A-Z][a-zA-Z'-]+", text))
    cited_speakers = len(re.findall(r"according to\s+[A-Z]", text))
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    money = len(re.findall(r"\$\s?\d", text))
    percentages = len(re.findall(r"\b\d+(?:\.\d+)?\s?%", text))
    substance_total = quoted + named + cited_speakers + years + money + percentages
    substance_density = _per_1000(substance_total, word_count)
    if substance_density < 2:
        # Severely lacking substance — strong LLM signal
        signals["low_substance"] = {
            "quotes": quoted, "named_titles": named,
            "year_refs": years, "monetary_refs": money,
            "percentages": percentages,
            "density_per_1k": round(substance_density, 2),
            "severity": "high" if substance_density < 0.5 else "med",
        }

    return signals


# ─── Litmus metadata signals (Wayback, AdSense, author meta) ─────────────────


_AUTHOR_META_RE = re.compile(
    r"<meta\s+(?:[^>]+\s+)?(?:name|property)=['\"](author|article:author)['\"]"
    r"\s+content=['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_REL_AUTHOR_RE = re.compile(
    r"<a[^>]+rel=['\"]author['\"][^>]*>\s*([^<]+?)\s*</a>",
    re.IGNORECASE,
)
_ADSENSE_PATTERNS = (
    "adsbygoogle",
    "googleads.g.doubleclick",
    "pagead2.googlesyndication",
    "data-ad-client",
    "data-ad-slot",
    'class="adsbygoogle"',
)


def _extract_author_from_html(html: str) -> str | None:
    """Pull a likely author byline from raw HTML. Returns the author string
    or None when none of the standard markers fire.
    """
    if not html:
        return None
    m = _AUTHOR_META_RE.search(html)
    if m:
        return m.group(2).strip()
    m = _REL_AUTHOR_RE.search(html)
    if m:
        return m.group(1).strip()
    return None


def _detect_adsense(html: str) -> bool:
    """True iff the raw HTML contains AdSense / DoubleClick markers."""
    if not html:
        return False
    low = html.lower()
    return any(p in low for p in _ADSENSE_PATTERNS)


async def _fetch_raw_html(url: str) -> str | None:
    """Pull the rendered HTML via Crawl4AI's /html endpoint. Used for
    AdSense / author-meta detection that the markdown-extraction path
    discards. Returns None on failure (litmus continues without these
    signals).
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    try:
        async with http.post(
            f"{SCRAPER_API}/html",
            json={"url": url},
            timeout=aiohttp.ClientTimeout(total=SCRAPER_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                log.warning("Crawl4AI /html %d for %s", resp.status, url)
                return None
            data = await resp.json()
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        log.warning("Crawl4AI /html error for %s: %s", url, e)
        return None
    return data.get("html") or None


async def _wayback_first_seen(url: str) -> str | None:
    """Earliest archive.org snapshot timestamp for this URL.

    Returns YYYYMMDDhhmmss string or None. Uses the `available` endpoint
    asking for a 2000-01-01 starting timestamp — Wayback returns the
    closest snapshot AFTER that, which (for established domains) is the
    earliest archive.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    try:
        async with http.get(
            "http://archive.org/wayback/available",
            params={"url": url, "timestamp": "20000101"},
            timeout=aiohttp.ClientTimeout(total=WAYBACK_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
    except (asyncio.TimeoutError, aiohttp.ClientError, ValueError):
        return None
    snap = (data.get("archived_snapshots") or {}).get("closest") or {}
    return snap.get("timestamp")


def _domain_age_severity(timestamp: str | None) -> tuple[str, str]:
    """Map a Wayback timestamp to (severity, human-readable description).

    Brand-new domains pumping out content are a strong AI-content tell
    (cheap RSS-mill style). Domains older than ~3 years are well-aged.
    """
    if not timestamp or len(timestamp) < 6:
        return ("med", "no archive found")
    try:
        from datetime import datetime, timezone
        first = datetime.strptime(timestamp[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_days = (now - first).days
    except ValueError:
        return ("low", f"first archived {timestamp[:8]}")
    if age_days < 180:
        return ("high", f"first archived {timestamp[:6]} (<6 months ago)")
    if age_days < 730:
        return ("med", f"first archived {timestamp[:6]} (<2 years ago)")
    return ("low", f"first archived {timestamp[:6]} ({age_days // 365}+ years ago)")


def _aggregate_severity(signals: dict, *,
                        adsense_detected: bool,
                        author_present: bool,
                        domain_severity: str) -> int:
    """Sum severity scores across all signals. Used to decide whether to
    call the LLM for a qualitative read (called when 2 < score < 8).
    """
    total = 0
    for s in signals.values():
        total += _SEVERITY_SCORE.get(s.get("severity", "low"), 0)
    if adsense_detected:
        total += 1
    if not author_present:
        total += 1  # missing byline is a soft tell
    total += _SEVERITY_SCORE.get(domain_severity, 0)
    return total


# ─── YT comment filtering + rendering ────────────────────────────────────────


def _is_emoji_only(text: str) -> bool:
    """Heuristic: comment is just emoji + whitespace + minor punctuation.

    Real comments contain at least some Latin / CJK / Cyrillic letters or
    digits. Pure emoji + ❤️ lol-style reactions don't add summarisation
    signal beyond "people reacted".
    """
    stripped = re.sub(r"[\s\W\d_]+", "", text, flags=re.UNICODE)
    # If after stripping symbols/digits/whitespace nothing's left, it's noise
    return len(stripped) < 5


def filter_yt_comments(comments: list[dict],
                       min_chars: int = YT_COMMENT_MIN_CHARS,
                       top_n: int = YT_COMMENT_SUMMARY_TOP_N) -> list[dict]:
    """Filter + rank YT comments for summarisation.

    Filter (drop):
      - Empty / deleted / shorter than min_chars
      - Pure emoji / punctuation noise
      - Bot-style spam ("First!", single-word reactions)

    Rank (highest first):
      1. Pinned (creator-pinned, signals importance)
      2. Hearted by uploader (creator-engaged)
      3. Top-level high-likes
      4. Replies to high-engagement threads

    Returns the top-N after filter+rank.
    """
    if not comments:
        return []

    keep: list[dict] = []
    for c in comments:
        text = (c.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        if _is_emoji_only(text):
            continue
        # Spam patterns
        low = text.lower()
        if low in {"first", "first!", "early gang", "who's here in 2026"}:
            continue
        keep.append(c)

    # Sort: pinned > hearted > likes desc, with ties broken by likes
    def _rank(c: dict) -> tuple:
        pinned = 1 if c.get("is_pinned") else 0
        hearted = 1 if c.get("is_favorited") else 0
        likes = int(c.get("like_count") or 0)
        # Negative for descending sort under tuple comparison
        return (-pinned, -hearted, -likes)

    keep.sort(key=_rank)
    return keep[:top_n]


def format_yt_comments(comments: list[dict]) -> str:
    """Render filtered comments as Markdown for the LLM. Pinned and
    creator-hearted comments get tagged so the model can weigh them.
    """
    lines = []
    for c in comments:
        author = c.get("author") or "[unknown]"
        likes = c.get("like_count") or 0
        text = (c.get("text") or "").strip()
        if len(text) > 1500:
            text = text[:1500] + "…"
        tags = []
        if c.get("is_pinned"):
            tags.append("📌pinned")
        if c.get("is_favorited"):
            tags.append("❤️creator-hearted")
        if c.get("author_is_uploader"):
            tags.append("creator-replied")
        is_reply = c.get("parent") and c.get("parent") != "root"
        prefix = "  - " if is_reply else "- "
        tag_str = (" [" + ", ".join(tags) + "]") if tags else ""
        lines.append(f"{prefix}**{author}** ({likes} likes){tag_str}: {text}")
    return "\n".join(lines)


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
    if http is None: raise RuntimeError("HTTP session not initialised")
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
    """Render per-frame VLM descriptions as `[H:MM:SS] text` lines.

    Legacy / fallback format for when the server didn't return scenes[].
    Matches the whisper transcript format so existing summary prompts
    work unmodified.
    """
    lines = []
    for d in descriptions:
        ts = format_duration(int(d.get("timestamp", 0)))
        text = (d.get("text") or "").strip()
        if not text or text == "[frame description unavailable]":
            continue
        lines.append(f"[{ts}] {text}")
    return "\n".join(lines)


def _format_scenes(scenes: list[dict]) -> str:
    """Render scene-clustered VLM output as `[H:MM:SS-H:MM:SS] text` lines.

    Each scene is one line — the server already pre-clustered consecutive
    frames into semantic scenes and synthesized a single description per
    cluster, so input to the summarizer LLM is compact (no redundant
    per-frame descriptions) and time-anchored.

    Single-frame scenes use `[H:MM:SS]` (point timestamp); multi-frame
    scenes use `[H:MM:SS-H:MM:SS]` (range). The downstream linkify
    timestamp regex picks up the starting timestamp from either form.

    When the scene has OCR text (titles, credits, captions), it's
    appended as `text on screen: "..."` so the summary LLM can ground
    specific names / titles in the actual on-screen text rather than
    the VLM's vague paraphrase.
    """
    lines = []
    for s in scenes:
        start = float(s.get("start", 0))
        end = float(s.get("end", start))
        text = (s.get("description") or "").strip()
        if not text or text == "[frame description unavailable]":
            continue
        if end - start >= 1.0:
            ts = f"[{format_duration(int(start))}-{format_duration(int(end))}]"
        else:
            ts = f"[{format_duration(int(start))}]"
        n = int(s.get("frame_count", 1))
        suffix = f" ({n} frames)" if n > 1 else ""
        ocr = (s.get("ocr") or "").strip()
        # Truncate noisy OCR (e.g. wall of fast-changing captions)
        if len(ocr) > 400:
            ocr = ocr[:400] + "…"
        ocr_suffix = f" — text on screen: \"{ocr}\"" if ocr else ""
        lines.append(f"{ts}{suffix} {text}{ocr_suffix}")
    return "\n".join(lines)


def _format_vlm_output(desc_result: dict) -> str:
    """Pick the right renderer based on server response shape.

    Server ≥ scene-cluster pipeline returns `scenes[]`; older versions
    only return `descriptions[]`. This helper keeps the bot working
    against either.
    """
    scenes = desc_result.get("scenes")
    if scenes:
        return _format_scenes(scenes)
    return _format_descriptions(desc_result.get("descriptions") or [])


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

    Used when we asked /api/jobs with cleanup=False (so the VLM had a
    chance to use the file) and now no longer need it. Idempotent on the
    server side; swallows errors here since failure just leaves a temp file
    that gets reaped at container restart.
    """
    if not file_path:
        return
    if http is None: raise RuntimeError("HTTP session not initialised")
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
# Bare URL match: lookbehind excludes `(`, `[`, `<` so we don't double-match
# URLs already inside markdown links or Discord no-preview wrappers `<...>`.
# Trailing exclude class adds `>` so `<https://x.com>` doesn't consume the
# closing bracket and leave a dangling `<`.
_BARE_URL_RE = re.compile(r"(?<![(\[<])\bhttps?://[^\s)\]>]+", re.IGNORECASE)
# Discord no-preview form: `<https://...>`. Strip the wrapper if the inner
# URL is disallowed; preserve as a normal link if allowed.
_NOPREVIEW_URL_RE = re.compile(r"<(https?://[^\s>]+)>", re.IGNORECASE)


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

    Order matters: handle Discord's `<https://...>` no-preview form FIRST so
    the wrapper is consumed atomically (preventing dangling `<` from a
    bare-URL match that would otherwise eat through the `>`).
    """
    def _replace_md(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if _is_allowed_link(url):
            return m.group(0)
        # Drop the link target; keep the visible text. If text is empty, fall
        # back to a domain marker so it's clear something was elided.
        return label or "[link removed]"

    text = _MD_LINK_RE.sub(_replace_md, text)

    def _replace_nopreview(m: re.Match) -> str:
        url = m.group(1)
        if _is_allowed_link(url):
            return m.group(0)  # keep entire `<url>` form
        return "[link removed]"

    text = _NOPREVIEW_URL_RE.sub(_replace_nopreview, text)

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


def _derive_video_id(full_url: str) -> str:
    """Derive a stable opaque ID for a non-YouTube URL.

    Tries the last path segment first (handles `/status/12345`, `/v/abc`).
    Falls back to a short hash of the full URL when that segment looks
    like a profile/handle (no digits, short) — prevents collisions when
    multiple distinct videos from the same author are posted.
    """
    import hashlib
    path_parts = [p for p in full_url.rstrip("/").split("/")
                  if p and "." not in p and "//" not in p]
    last = re.sub(r"[^\w-]", "", path_parts[-1])[:20] if path_parts else ""
    # Heuristic: real video IDs are mostly digits or alphanumeric tokens of
    # decent length (>=5 chars and contain a digit). A plain author handle
    # would be e.g. "elonmusk" → no digits → collision risk.
    if last and len(last) >= 5 and any(c.isdigit() for c in last):
        return last
    # Hash fallback — short but unambiguous.
    return "u" + hashlib.sha1(full_url.encode("utf-8")).hexdigest()[:11]


def _job_from_interaction(
    interaction: discord.Interaction,
    url: str,
    *,
    user_prompt: str = "",
    diarize: bool = False,
    vlm_enabled: bool | None = None,
    yt_comments_enabled: bool | None = None,
    model_override: str | None = None,
) -> Job | None:
    """Build a Job from an interaction. Returns None on URL parse failure.

    Slash commands are always explicit_request=True — the user typed the
    command and the URL.
    """
    eff_vlm = VLM_ENABLED if vlm_enabled is None else vlm_enabled
    eff_comments = YT_COMMENTS_ENABLED if yt_comments_enabled is None else yt_comments_enabled
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
            vlm_enabled=eff_vlm,
            yt_comments_enabled=eff_comments,
            model_override=model_override,
            explicit_request=True,
            interaction=interaction,
        )
    # Fallback for other platforms
    m = VIDEO_URL_PATTERN.search(url)
    if not m:
        return None
    full_url = m.group(1)
    return Job(
        url=full_url, video_id=_derive_video_id(full_url),
        channel=interaction.channel, submitter_id=interaction.user.id,
        submitter_name=str(interaction.user),
        user_prompt=user_prompt, diarize=diarize,
        vlm_enabled=eff_vlm,
        yt_comments_enabled=eff_comments,
        model_override=model_override,
        explicit_request=True,
        interaction=interaction,
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
        vlm_enabled=chan_cfg.get("vlm_enabled", VLM_ENABLED),
        yt_comments_enabled=chan_cfg.get("yt_comments_enabled", YT_COMMENTS_ENABLED),
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
        vlm_enabled=chan_cfg.get("vlm_enabled", VLM_ENABLED),
        yt_comments_enabled=chan_cfg.get("yt_comments_enabled", YT_COMMENTS_ENABLED),
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
    if http is None: raise RuntimeError("HTTP session not initialised")
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
    yt_comments="Fetch + summarise YouTube comments (extra Community Reaction embed)",
    show="Just print the current config without changing it",
)
async def cmd_config(
    interaction: discord.Interaction,
    model: str | None = None,
    vlm: bool | None = None,
    diarize: bool | None = None,
    yt_comments: bool | None = None,
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
    if show or (model is None and vlm is None and diarize is None and yt_comments is None):
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
    if yt_comments is not None:
        fields["yt_comments_enabled"] = yt_comments
    new_cfg = set_channel_config(interaction.channel.id, **fields)
    if new_cfg:
        txt = "Updated config:\n" + "\n".join(f"  **{k}**: `{v}`" for k, v in new_cfg.items())
    else:
        txt = "Config cleared — using global defaults."
    await interaction.response.send_message(txt, ephemeral=True)
    log.info("Channel %s config updated by %s: %s",
             interaction.channel.id, interaction.user, fields)


@bot.tree.command(name="serverconfig", description="Configure server-wide bot defaults (admin)")
@app_commands.describe(
    summary_channel="Channel to receive Key Points + Chapters embeds for THIS server",
    clear="Clear all server-wide overrides (revert to global defaults)",
    show="Print current server config without changing anything",
)
async def cmd_serverconfig(
    interaction: discord.Interaction,
    summary_channel: discord.TextChannel | None = None,
    clear: bool = False,
    show: bool = False,
) -> None:
    if interaction.guild_id is None:
        await interaction.response.send_message(
            "❌ This command must be used in a server, not a DM.",
            ephemeral=True,
        )
        return
    # Server-level permission gate. interaction.user inside a guild is a
    # discord.Member; .guild_permissions reflects the user's effective
    # role-based permissions.
    perms = getattr(interaction.user, "guild_permissions", None)
    if perms is None or not perms.manage_guild:
        await interaction.response.send_message(
            "❌ This command requires the **Manage Server** permission.",
            ephemeral=True,
        )
        return

    if show or (summary_channel is None and not clear):
        cfg = get_guild_config(interaction.guild_id)
        if not cfg:
            txt = "No server-wide overrides — using global defaults."
        else:
            lines = ["Current server config:"]
            for k, v in cfg.items():
                if k.endswith("_channel"):
                    lines.append(f"  **{k}**: <#{v}>")
                else:
                    lines.append(f"  **{k}**: `{v}`")
            txt = "\n".join(lines)
        await interaction.response.send_message(txt, ephemeral=True)
        return

    if clear:
        # Wipe every key by passing None for known fields.
        set_guild_config(interaction.guild_id, summary_channel=None)
        await interaction.response.send_message(
            "Server config cleared — using global defaults.", ephemeral=True
        )
        log.info("Guild %s config cleared by %s",
                 interaction.guild_id, interaction.user)
        return

    # summary_channel is set — validate and save
    if summary_channel.guild.id != interaction.guild_id:
        await interaction.response.send_message(
            "❌ Summary channel must be in this server.", ephemeral=True
        )
        return
    me = summary_channel.guild.me
    chan_perms = summary_channel.permissions_for(me)
    if not (chan_perms.send_messages and chan_perms.embed_links):
        await interaction.response.send_message(
            f"❌ I don't have **Send Messages** + **Embed Links** in "
            f"<#{summary_channel.id}>. Grant me access there first, then "
            f"re-run this command.",
            ephemeral=True,
        )
        return
    set_guild_config(interaction.guild_id, summary_channel=summary_channel.id)
    await interaction.response.send_message(
        f"Updated: detail embeds will now post to <#{summary_channel.id}>.",
        ephemeral=True,
    )
    log.info("Guild %s config updated by %s: summary_channel=%s",
             interaction.guild_id, interaction.user, summary_channel.id)


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
