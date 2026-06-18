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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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

import voice  # voice-call live transcription (import-safe even if ext absent)

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
WHISPER_LIVE_URL = os.environ.get("WHISPER_LIVE_URL", "http://localhost:7861")
# WebSocket form derived from the HTTP URL (http→ws, https→wss).
WHISPER_LIVE_WS = WHISPER_LIVE_URL.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
SCRAPER_API = os.environ.get("SCRAPER_API_URL", "http://localhost:11235")
FLARESOLVERR_API = os.environ.get("FLARESOLVERR_API_URL",
                                  "http://localhost:8191/v1")
SCRAPER_TIMEOUT = int(os.environ.get("SCRAPER_TIMEOUT", "120"))
FLARESOLVERR_TIMEOUT = int(os.environ.get("FLARESOLVERR_TIMEOUT", "90"))
# Tier 3 + 4 archive fallbacks. Triggered only after Crawl4AI AND
# FlareSolverr both fail. Both are anti-bot-resilient (DataDome, Akamai
# Bot Manager, hardened Turnstile) because they fetch a pre-rendered
# snapshot rather than hitting the live site. Set ENABLE_ARCHIVE_FALLBACKS=0
# to disable the whole tier (e.g. if archive.org is suffering a long
# outage and degrading the user-visible failure path).
ENABLE_ARCHIVE_FALLBACKS = os.environ.get("ENABLE_ARCHIVE_FALLBACKS", "1") != "0"
WAYBACK_API = os.environ.get(
    "WAYBACK_API_URL", "https://archive.org/wayback/available",
)
WAYBACK_TIMEOUT = int(os.environ.get("WAYBACK_TIMEOUT", "30"))
ARCHIVE_PH_BASE = os.environ.get("ARCHIVE_PH_BASE_URL", "https://archive.ph")
ARCHIVE_PH_TIMEOUT = int(os.environ.get("ARCHIVE_PH_TIMEOUT", "60"))
# Stable UA for archive endpoints. archive.org tightens anonymous quotas
# and the API docs ask integrators to identify themselves; archive.ph
# rejects empty / "python-urllib" UAs entirely.
ARCHIVE_USER_AGENT = os.environ.get(
    "ARCHIVE_USER_AGENT",
    "Mozilla/5.0 (compatible; whisper-transcribe-bot/1.0; "
    "+https://github.com/erfianugrah/whisper-transcribe)",
)
# Scraped article body cap. Articles longer than this hit map-reduce in
# summarize() — the budget calc applies as it does for transcripts.
SCRAPED_BODY_CHAR_CAP = int(os.environ.get("SCRAPED_BODY_CHAR_CAP", "200000"))
# Minimum scraped body length before we treat a scrape as successful.
# Anti-bot products (DataDome, Akamai Bot Manager, PerimeterX, hardened CF
# Turnstile) often serve a tiny challenge stub that survives readability
# extraction as a few dozen chars of nothing — e.g. Reuters via Crawl4AI
# returns just `reuters.com\n` (11 chars). Without a floor, the bot would
# feed that stub to the LLM and produce a useless "no content to summarise"
# embed. A real article body is virtually always >500 chars after
# readability; 200 is a conservative floor that still passes short blog
# posts. Set to 0 to disable.
MIN_SCRAPED_BODY_CHARS = int(os.environ.get("MIN_SCRAPED_BODY_CHARS", "200"))
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
    PROMPT_BRIEF_IMAGE, PROMPT_KEY_POINTS_IMAGE,
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
    # Translation policy:
    #   "auto" (default): server runs a 30s LID pre-pass; if source is
    #                     non-English, translates to English. CS-FLEURS
    #                     (arXiv:2509.14161) shows whisper handles
    #                     code-switched audio well as translation but
    #                     badly as ASR — auto-translate gets us coherent
    #                     transcripts for the bot's summarisation use case.
    #   True            : force task=translate regardless of source.
    #   False           : force task=transcribe (preserve source language).
    # Exposed on /transcribe + /summarize slash commands.
    translate: object = "auto"         # "auto" | True | False
    # User-requested cache bypass. When True:
    #   - bot's per-video file cache is skipped on read
    #   - /api/jobs payload sets refresh=true so the server skips its
    #     Valkey transcript cache
    #   - successful results still OVERWRITE the cache, so subsequent
    #     non-refresh runs see the new transcript
    # Exposed as `refresh: bool` on /summarize and /transcribe.
    refresh: bool = False
    vlm_enabled: bool = True           # per-channel override; falls back to global VLM_ENABLED
    yt_comments_enabled: bool = True   # per-channel override; falls back to global YT_COMMENTS_ENABLED
    model_override: str | None = None  # per-channel config can override LLM_MODEL
    # Job kind: "video" (default — yt-dlp + whisper + summarize) or "web"
    # (crawl4ai + summarize). Worker dispatches on this discriminant.
    kind: str = "video"
    # Image-attachment jobs (kind="image") carry a list of CDN URLs the bot
    # downloads from. Empty for non-image kinds. The first attachment's
    # filename is used as the title; multi-image jobs render bullets per
    # image in the key-points embed.
    image_urls: list[str] = field(default_factory=list)
    image_filenames: list[str] = field(default_factory=list)
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
        if self.kind not in ("video", "web", "litmus", "image", "live"):
            raise ValueError(
                f"Job.kind must be 'video' | 'web' | 'litmus' | 'image' | 'live', "
                f"got {self.kind!r}"
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


# ─── LLM health gate ──────────────────────────────────────────────────────────
# The bot needs the LLM endpoint (LLM_API_URL) to do any summarisation work.
# When the endpoint is unreachable, we want fast, clear failure rather than
# the 130s retry-ladder behaviour the old worker had (10 + 30 + 90 s of pure
# sleep per request, fanned out across multiple summary styles).
#
# Two-pronged design:
#   1. _llm_call wraps aiohttp transport errors → LLMOfflineError, a
#      PermanentError subclass that the worker fails fast on (no retry).
#      Marking the LLM unhealthy at the same time.
#   2. A background probe (llm_health_loop) polls /v1/models periodically.
#      Healthy: 60s interval. Unhealthy: 15s interval (fast recovery).
#   3. The queue gate (_rate_limit_check) rejects new submissions while
#      unhealthy with a clear user-facing message.
#
# Net effect on the user: a failed run posts a "❌ LLM offline" message in
# ~1s instead of ~140s; subsequent requests reject in <1ms with the same
# message until the probe flips back to healthy.

LLM_PROBE_HEALTHY_INTERVAL = int(os.environ.get("LLM_PROBE_HEALTHY_INTERVAL", "60"))
LLM_PROBE_UNHEALTHY_INTERVAL = int(os.environ.get("LLM_PROBE_UNHEALTHY_INTERVAL", "15"))
LLM_PROBE_TIMEOUT = float(os.environ.get("LLM_PROBE_TIMEOUT", "5"))


@dataclass
class _LLMHealthState:
    """Bot's view of whether the LLM endpoint is reachable. Single-threaded
    asyncio access — no lock needed."""
    healthy: bool = True                # start optimistic; probe corrects on boot
    last_check_at: float = 0.0          # last probe (success OR failure)
    last_recovery_at: float = 0.0       # last healthy → unhealthy → healthy edge
    last_failure_reason: str = ""       # for the user-facing message
    failure_count: int = 0              # consecutive probe failures


_llm_health = _LLMHealthState()


def _mark_llm_unhealthy(reason: str) -> None:
    """Flag LLM as unreachable. Idempotent. On the healthy→unhealthy edge
    we log a warning so operators see the transition in container logs."""
    _llm_health.failure_count += 1
    _llm_health.last_failure_reason = reason
    _llm_health.last_check_at = time.time()
    if _llm_health.healthy:
        log.warning("LLM endpoint marked unhealthy: %s", reason)
        _llm_health.healthy = False


def _mark_llm_healthy() -> None:
    """Flag LLM as reachable. Logs recovery on the unhealthy→healthy edge."""
    if not _llm_health.healthy:
        log.info("LLM endpoint recovered: %s", LLM_API)
        _llm_health.last_recovery_at = time.time()
    _llm_health.healthy = True
    _llm_health.failure_count = 0
    _llm_health.last_check_at = time.time()


async def _probe_llm() -> bool:
    """One-shot transport probe of LLM_API_URL/models. Returns True on 200.

    Uses a short timeout so an unhealthy endpoint doesn't slow the probe
    loop. Catches every aiohttp/asyncio error class — at probe time we
    just want a boolean reachable/not-reachable answer."""
    if http is None:
        return False  # session not initialised yet; main loop will retry
    try:
        async with http.get(
            f"{LLM_API}/models",
            timeout=aiohttp.ClientTimeout(total=LLM_PROBE_TIMEOUT),
        ) as resp:
            return resp.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return False


async def llm_health_loop() -> None:
    """Background task: probe LLM_API_URL periodically + maintain
    _llm_health state. Started from on_ready."""
    # Boot-time probe so the "Bot ready" log line is followed by an
    # explicit reachability status. Discord stays connected either way.
    ok = await _probe_llm()
    if ok:
        _mark_llm_healthy()
        log.info("LLM endpoint reachable at %s (boot probe OK)", LLM_API)
    else:
        _mark_llm_unhealthy(f"boot probe failed against {LLM_API}")
        log.warning(
            "LLM endpoint NOT reachable at %s — bot will reject new "
            "summary work with a degraded-mode message and re-probe "
            "every %ds until recovered.",
            LLM_API, LLM_PROBE_UNHEALTHY_INTERVAL,
        )

    while True:
        interval = (
            LLM_PROBE_HEALTHY_INTERVAL if _llm_health.healthy
            else LLM_PROBE_UNHEALTHY_INTERVAL
        )
        await asyncio.sleep(interval)
        if await _probe_llm():
            _mark_llm_healthy()
        else:
            _mark_llm_unhealthy(f"probe failed against {LLM_API}")


def _llm_offline_user_reason() -> str:
    """User-facing reason string when LLM is unreachable. Used by
    _rate_limit_check and by the worker's failure handler."""
    if _llm_health.last_recovery_at:
        ago = int(time.time() - _llm_health.last_recovery_at)
        when = f" (last reachable ~{ago // 60} min ago)" if ago > 60 else ""
    else:
        when = ""
    return (
        f"🚧 LLM backend offline{when}. Bot is auto-probing every "
        f"{LLM_PROBE_UNHEALTHY_INTERVAL}s; resubmit when it's back."
    )


def _rate_limit_check(user_id: int, count: int = 1) -> tuple[bool, str]:
    """Returns (allowed, reason). `allowed=False` rejects with `reason`.

    Three checks, in order of severity:
      1. LLM endpoint reachable — short-circuits everything else when
         the LLM is down (no point queueing work we can't fulfil).
      2. Total queue cap (independent of user) — protects against
         collective overload.
      3. Per-user sliding window (60 min) — protects against single-user
         spam.
    Bypass list (RATE_LIMIT_BYPASS_USERS) skips the per-user check but
    still enforces the queue cap and the LLM gate.

    `count` is the number of jobs the caller wants to enqueue atomically
    (e.g. chained `tldr litmus` reply requests two jobs in one go). The
    cap check uses `count` so we reject all-or-nothing rather than
    partial-fail mid-batch.
    """
    if not _llm_health.healthy:
        return False, _llm_offline_user_reason()
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


async def _is_live_stream(url: str) -> bool:
    """Ask whisper-live whether `url` is a currently-airing live stream.

    Fails safe (returns False) on any error — if whisper-live is down or
    the probe errors, the job falls through to the normal video pipeline
    rather than getting stuck. Only currently-live streams return True;
    VOD'd streams ('was_live') route to the normal video pipeline."""
    if http is None:
        return False
    try:
        async with http.get(
            f"{WHISPER_LIVE_URL}/probe",
            params={"url": url},
            timeout=aiohttp.ClientTimeout(total=35),
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            return bool(data.get("is_live"))
    except Exception:
        return False


# ─── Retry view (LLM-offline rejections) ──────────────────────────────────────
# When the queue gate rejects a job because the LLM endpoint is unreachable,
# we attach a single "Retry when ready" button to the rejection message so
# the user can resubmit with one click instead of re-pasting the URL.
#
# Rate-limit / queue-full rejections don't get the retry button — there's
# nothing the user can do via a click that retyping wouldn't also fix, and
# a button that just hits the rate limit again would be misleading.


@dataclass
class _RetrySpec:
    """Lightweight snapshot of a Job's submission inputs. Stored in the
    RetryJobsView so a button click can rebuild the Job without holding
    a stale Job reference (Job contains Discord message/interaction refs
    that may be invalid by the time the user clicks retry)."""
    url: str
    kind: str                      # "video" | "web" | "litmus" | "image" | "live"
    video_id: str                  # YT id or URL hash, same convention as Job
    diarize: bool = False
    vlm_enabled: bool = True
    yt_comments_enabled: bool = True
    user_prompt: str = ""
    model_override: str | None = None
    translate: object = "auto"
    refresh: bool = False
    explicit_request: bool = True  # retries are always explicit (user clicked)
    # Image-attachment retries carry the same CDN URL list as the source Job.
    # Discord attachment URLs are signed but valid for ~24h, long enough for
    # the 30-min RetryJobsView TIMEOUT to keep working in practice.
    image_urls: tuple[str, ...] = ()
    image_filenames: tuple[str, ...] = ()


def _job_to_retry_spec(job: "Job") -> _RetrySpec:
    """Extract the parts of a Job that survive being held in a View."""
    return _RetrySpec(
        url=job.url,
        kind=job.kind,
        video_id=job.video_id,
        diarize=job.diarize,
        vlm_enabled=job.vlm_enabled,
        yt_comments_enabled=job.yt_comments_enabled,
        user_prompt=job.user_prompt,
        model_override=job.model_override,
        translate=job.translate,
        refresh=job.refresh,
        explicit_request=job.explicit_request,
        image_urls=tuple(job.image_urls),
        image_filenames=tuple(job.image_filenames),
    )


# Sentinel prefix the RetryJobsView appends to the original rejection
# message when a click fails (LLM still offline / queue still full).
# Found-and-replaced on subsequent failed clicks so we refresh the
# timestamp rather than stacking lines.
_RETRY_FOOTER_MARKER = "\n· last retry "


class RetryJobsView(discord.ui.View):
    """Single-click resubmit for jobs rejected by the LLM-offline gate.

    Only the original submitter can click. Once a click succeeds in
    queueing the jobs, the button disables itself so a user can't queue
    the same job twice (the queued copies still respect the rate limit).
    """

    # 30 min — long enough that a typical LLM outage (model swap, OOM
    # recovery, container restart) doesn't strand a queued retry, short
    # enough that the View doesn't pile up indefinitely on the bot's
    # in-memory state if the LLM stays down for hours.
    TIMEOUT = float(os.environ.get("RETRY_VIEW_TIMEOUT", "1800"))

    def __init__(self, specs: list[_RetrySpec], target_user_id: int):
        super().__init__(timeout=self.TIMEOUT)
        self.specs = specs
        self.target_user_id = target_user_id

    @discord.ui.button(label="Retry when ready", style=discord.ButtonStyle.primary, emoji="🔁")
    async def retry_button(
        self, interaction: discord.Interaction, button: discord.ui.Button,
    ) -> None:
        # Only the original submitter can retry — otherwise a stranger
        # could burn the submitter's rate-limit budget by clicking.
        if interaction.user.id != self.target_user_id:
            await interaction.response.send_message(
                f"Only <@{self.target_user_id}> can retry this submission.",
                ephemeral=True,
            )
            return

        # Re-check the gate (LLM health + rate limit + queue cap). The
        # whole point of the button is that state may have changed since
        # the rejection was posted.
        ok, reason = _rate_limit_check(self.target_user_id, count=len(self.specs))
        if not ok:
            # Still failing. Edit the original rejection message in place
            # with a refreshed "last tried" footer rather than sending a
            # new ephemeral on every click — repeated clicks would otherwise
            # stack up multiple identical ephemerals in the channel.
            # Discord's "(edited)" badge gives the user enough feedback
            # that the click registered. Button stays enabled.
            now_hms = time.strftime("%H:%M:%S")
            base = interaction.message.content if interaction.message else ""
            # Strip any prior footer so we refresh rather than concat.
            if _RETRY_FOOTER_MARKER in base:
                base = base.split(_RETRY_FOOTER_MARKER, 1)[0].rstrip()
            new_content = (
                f"{base}{_RETRY_FOOTER_MARKER}{now_hms} — still offline"
            )
            try:
                await interaction.response.edit_message(
                    content=new_content, view=self,
                )
            except (discord.HTTPException, discord.NotFound):
                # Fallback: ephemeral if the original message was deleted
                # or we lost edit permission. Rare path.
                try:
                    await interaction.followup.send(
                        f"⏳ {reason}", ephemeral=True,
                    )
                except discord.HTTPException:
                    pass
            return

        # Gate passes — defer the response (we'll send a followup with
        # the queued ack) and disable the button to prevent duplicates.
        await interaction.response.defer(thinking=False)
        for c in self.children:
            c.disabled = True
        try:
            # Replace the view on the original rejection message so the
            # disabled button reflects the consumed retry.
            await interaction.edit_original_response(view=self)
        except (discord.HTTPException, discord.NotFound):
            pass

        # Rebuild Jobs. Each one points at the button-click interaction
        # for ack + worker output. interaction.followup is valid for ~15
        # min from the click, which comfortably covers transcribe +
        # summarise on any sane GPU.
        for spec in self.specs:
            job = Job(
                url=spec.url, video_id=spec.video_id,
                channel=interaction.channel,
                submitter_id=interaction.user.id,
                submitter_name=str(interaction.user),
                user_prompt=spec.user_prompt,
                diarize=spec.diarize,
                vlm_enabled=spec.vlm_enabled,
                yt_comments_enabled=spec.yt_comments_enabled,
                model_override=spec.model_override,
                translate=spec.translate,
                refresh=spec.refresh,
                kind=spec.kind,
                explicit_request=spec.explicit_request,
                image_urls=list(spec.image_urls),
                image_filenames=list(spec.image_filenames),
                interaction=interaction,
            )
            _rate_limit_record(job.submitter_id)
            await queue.put(job)
            await _ack_queued(job, queue.qsize())
            log.info(
                "Retry-button queued %s (%s) from %s (channel=%s)",
                job.video_id, job.kind, job.submitter_name,
                interaction.channel.id,
            )


def _maybe_retry_view(
    specs: list[_RetrySpec],
    target_user_id: int,
) -> "RetryJobsView | None":
    """Return a RetryJobsView when the reason for rejection is recoverable
    by clicking (i.e. LLM-offline). Returns None for non-recoverable rejects
    (rate limit, queue full) where a click would just hit the same wall."""
    if _llm_health.healthy:
        # LLM is reachable → the gate failed for a different reason
        # (rate limit, queue cap). A button can't bypass either.
        return None
    return RetryJobsView(specs=specs, target_user_id=target_user_id)


# ─── Inflight job tracking ────────────────────────────────────────────────────
# Lightweight in-memory map of jobs that are queued or running on the bot side.
# Single asyncio loop owns this — no lock needed (cooperative scheduling).
# Backs the `/progress`, `/cancel`, and `/queue` slash commands and gives the
# bot enough context to render ETA + phase to the submitter.
#
# Lifecycle (one entry per Job):
#   _inflight_register(job)  ← from _ack_queued (after queue.put)
#   _inflight_phase(...)     ← at each react site in worker / process_* paths
#   _inflight_remove(...)    ← at terminal react site (✅ / ❌ / silent drop)
#
# `cancel_requested` is a soft flag — the worker checks it at phase transitions
# and aborts cleanly before hitting expensive steps. The whisper service has
# its own DELETE /api/jobs/{id} which we forward when the server job_id is
# known; once server-side transcription has started, we can no longer cancel
# (whisperX has no safe interruption point), and the user is told so.

PHASE_QUEUED = "queued"
PHASE_DOWNLOADING = "downloading"
PHASE_TRANSCRIBING = "transcribing"
PHASE_SCRAPING = "scraping"
PHASE_SUMMARIZING = "summarizing"


@dataclass
class InflightEntry:
    """Live state for one in-flight job. Stored in _inflight by bot_video_id."""
    bot_video_id: str
    url: str
    kind: str                    # video / web / litmus
    submitter_id: int
    submitter_name: str
    channel_id: int
    queued_at: float
    started_at: float | None = None      # set when worker picks up
    phase: str = PHASE_QUEUED
    server_job_id: str | None = None     # whisper-side job id (after submit)
    title: str | None = None             # post-download (video) / post-scrape
    duration: int | None = None          # seconds; post-download
    cancel_requested: bool = False       # /cancel sets this


_inflight: dict[str, InflightEntry] = {}

# Rolling per-kind average of completed-job durations, used to estimate ETA
# while a job is queued. Capped to recent N. Keyed by kind so video jobs (a
# few minutes typically) don't get conflated with web jobs (a few seconds).
_completed_durations: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))


def _inflight_register(job: "Job") -> None:
    """Register a freshly-queued job. No-op on duplicates (replaces)."""
    _inflight[job.video_id] = InflightEntry(
        bot_video_id=job.video_id,
        url=job.url,
        kind=job.kind,
        submitter_id=job.submitter_id,
        submitter_name=job.submitter_name,
        channel_id=getattr(job.channel, "id", 0),
        queued_at=time.time(),
        phase=PHASE_QUEUED,
    )


def _inflight_phase(job: "Job", phase: str, **fields) -> None:
    """Update phase + arbitrary fields on the inflight entry. No-op if missing
    (defensive — race-free in single-threaded asyncio but guards manual calls
    against a removed entry).
    """
    entry = _inflight.get(job.video_id)
    if entry is None:
        return
    entry.phase = phase
    for k, v in fields.items():
        if hasattr(entry, k):
            setattr(entry, k, v)
    # First time we transition out of "queued" is the actual start.
    if phase != PHASE_QUEUED and entry.started_at is None:
        entry.started_at = time.time()


def _inflight_remove(video_id: str, *, kind: str | None = None) -> None:
    """Drop a finished job. Records the runtime for ETA estimation when the
    job has a started_at + kind (so cached/instant returns don't skew the avg).
    """
    entry = _inflight.pop(video_id, None)
    if entry is None:
        return
    if entry.started_at is not None:
        runtime = time.time() - entry.started_at
        # Sub-second runtimes are cache-hit replies; skip — they'd skew ETA low.
        if runtime >= 1.0:
            _completed_durations[kind or entry.kind].append(runtime)


def _inflight_user(user_id: int) -> list[InflightEntry]:
    """All entries owned by a given Discord user, queued-first then started."""
    entries = [e for e in _inflight.values() if e.submitter_id == user_id]
    # Stable order: running first (started_at not None), then queued, oldest first.
    entries.sort(key=lambda e: (e.started_at is None, e.queued_at))
    return entries


def _avg_runtime(kind: str) -> float | None:
    """Rolling-average runtime for a kind. None when we have <2 samples."""
    dq = _completed_durations.get(kind)
    if not dq or len(dq) < 2:
        return None
    return sum(dq) / len(dq)


def _format_relative(seconds: float) -> str:
    """Human-readable elapsed/ETA. Always positive (callers pass abs)."""
    s = int(round(max(0.0, seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    h, rem = divmod(s, 3600)
    return f"{h}h{rem // 60:02d}m"


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


# ─── Per-user config (own defaults) ──────────────────────────────────────────


# Per-user overrides — a third layer of config under channel + guild. Useful
# for users who consistently want a particular translate mode or model
# regardless of which channel they post in. Precedence at job-build time:
#   explicit slash arg > channel config > user config > env default.
# Storage shape mirrors channels.json: {user_id_str: {field: value, ...}}.
USERS_CONFIG_PATH = CACHE_DIR / "users.json"
_users_lock = threading.Lock()


def _load_users_config() -> dict:
    if not USERS_CONFIG_PATH.exists():
        return {}
    try:
        return json_mod.loads(USERS_CONFIG_PATH.read_text())
    except (OSError, json_mod.JSONDecodeError) as e:
        log.warning("users.json read failed (%s) — treating as empty", e)
        return {}


def _save_users_config(cfg: dict) -> None:
    with _users_lock:
        try:
            USERS_CONFIG_PATH.write_text(json_mod.dumps(cfg, indent=2, sort_keys=True))
        except OSError as e:
            log.error("users.json write failed: %s", e)


def get_user_config(user_id: int) -> dict:
    """Look up a user's overrides. Returns {} if no entry."""
    return _load_users_config().get(str(user_id), {})


def set_user_config(user_id: int, **fields) -> dict:
    """Update a user's config; returns the merged result. field=None removes."""
    cfg = _load_users_config()
    entry = dict(cfg.get(str(user_id), {}))
    for k, v in fields.items():
        if v is None:
            entry.pop(k, None)
        else:
            entry[k] = v
    if entry:
        cfg[str(user_id)] = entry
    else:
        cfg.pop(str(user_id), None)
    _save_users_config(cfg)
    return entry


def _effective_config(user_id: int, channel_id: int) -> dict:
    """Merge user + channel configs. Channel wins on conflict (channel is
    'more specific' than user — a channel admin's policy outranks a user's
    personal preference). Explicit slash args still take precedence over
    both at the call site (callers do `arg or cfg.get(...)`).

    Returns a flat dict with keys: model, diarize, vlm_enabled, yt_comments_enabled.
    Missing keys mean 'use env default'.
    """
    user = get_user_config(user_id) if user_id else {}
    chan = get_channel_config(channel_id) if channel_id else {}
    # Per-channel config wins; per-user fills the gaps.
    return {**user, **chan}


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
    # Register the job in the inflight map before reacting so /progress and
    # /cancel see it immediately even if the user fires those commands
    # between the queue.put and the worker pick-up.
    _inflight_register(job)
    if job.message is not None:
        await safe_react(job.message, "\u23f3")  # ⏳
    elif job.interaction is not None:
        try:
            await job.interaction.followup.send(
                f"Queued `{job.video_id}` (position {position} in queue). "
                f"Track with `/progress`, cancel with `/cancel`.",
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


async def _job_reply(
    job: Job, text: str, view: discord.ui.View | None = None,
) -> None:
    """Send a textual reply for failure / status messages.

    `view` (optional) attaches Discord components (buttons, selects) to the
    reply — used by the worker's LLM-offline failure handler to surface a
    one-click resubmit button after a mid-pipeline LLM crash.
    """
    # discord.py's send / followup.send accept `view=MISSING` as the "no
    # view" sentinel; passing `view=None` works on modern versions but
    # branching keeps us safe across pin upgrades.
    kwargs: dict[str, object] = {}
    if view is not None:
        kwargs["view"] = view
    if job.message is not None:
        try:
            await job.channel.send(text, reference=job.message, **kwargs)
        except discord.HTTPException as e:
            log.warning("reply failed: %s", e)
    elif job.interaction is not None:
        try:
            await job.interaction.followup.send(text, **kwargs)
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
_voice_enabled = False  # set True once voice slash commands are registered
# Shared session; populated in on_ready(). Each network helper guards with an
# explicit `if http is None: raise` so the check survives `python -O`.
http: aiohttp.ClientSession | None = None


async def _voice_post_summary(thread, transcript_text: str) -> None:
    """Phase 3: summarise a finished voice call and post it into the thread.

    Injected into voice.register_voice_commands so the LLM + embed machinery
    stays in main (where summarize()/prompts/send_long_embed live). Best-effort:
    a summary failure must never break the /transcribe-leave teardown.
    """
    text = (transcript_text or "").strip()
    if len(text) < 40:  # too little was said to be worth a summary
        return
    try:
        brief = await summarize(
            text, PROMPT_BRIEF, LLM_MAX_TOKENS_BRIEF,
            reduce_template=REDUCE_BRIEF,
            title="Voice call", duration="", reference_block="",
        )
        brief = sanitize_llm_output(brief)
    except Exception as e:
        log.error("voice: summary generation failed: %s", e)
        return
    if brief.strip():
        await send_long_embed(thread, "\U0001f399\ufe0f Voice call summary", brief, 0x5865F2)


@bot.event
async def on_ready():
    global http
    http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900))
    bot.loop.create_task(worker())
    bot.loop.create_task(cache_cleanup_loop())
    # LLM health probe — boots optimistic (healthy), corrects on first probe.
    # Without this, the first failed user request would be the only signal
    # to operators that the LLM endpoint is unreachable (and the worker
    # would burn 130s of retry backoff before posting "❌ Failed").
    bot.loop.create_task(llm_health_loop())
    # Register voice-transcription slash commands BEFORE the sync so they ship
    # in the same tree push. Guarded against on_ready firing twice on reconnect
    # (tree.command raises on duplicate registration). No-op unless
    # VOICE_TRANSCRIBE_ENABLED + the extension + libopus are all present.
    global _voice_enabled
    if not _voice_enabled:
        try:
            _voice_enabled = voice.register_voice_commands(
                bot, summarize_cb=_voice_post_summary)
        except Exception as e:
            log.error("voice: registration failed: %s", e)
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

    # Per-channel + per-user config: model / diarize / vlm overrides.
    # Channel config takes precedence over user config (channel admin's
    # policy outranks a personal preference).
    cfg = _effective_config(message.author.id, message.channel.id)

    def _new_job(url, video_id):
        return Job(
            url=url, video_id=video_id,
            channel=message.channel,
            submitter_id=message.author.id,
            submitter_name=str(message.author),
            diarize=cfg.get("diarize", False),
            vlm_enabled=cfg.get("vlm_enabled", VLM_ENABLED),
            yt_comments_enabled=cfg.get(
                "yt_comments_enabled", YT_COMMENTS_ENABLED
            ),
            model_override=cfg.get("model"),
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
        # LLM-offline rejections get a Retry button so the user doesn't
        # have to re-paste the URL when the endpoint comes back.
        retry_view = _maybe_retry_view(
            [_job_to_retry_spec(j) for j in jobs_to_queue],
            target_user_id=message.author.id,
        )
        try:
            await message.channel.send(
                f"❌ {message.author.mention} {reason}",
                reference=message, mention_author=False,
                view=retry_view,
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

    # Image-attachment fallback: when there's no URL but the referenced
    # message has image attachments, route to the image OCR+VLM flow.
    # Litmus on an image doesn't make sense (no domain age, no AdSense,
    # no byline) so we silently drop the `litmus` hint when only images
    # are present — a chained `tldr litmus` on an image still produces
    # the image summary.
    image_urls: list[str] = []
    image_filenames: list[str] = []
    if url is None:
        image_urls, image_filenames = _extract_image_attachments(referenced)

    if url is None and not image_urls:
        try:
            await message.channel.send(
                "❌ No URL or image attachment found in the message you "
                "replied to.",
                reference=message, mention_author=False,
            )
        except discord.HTTPException:
            pass
        return

    # Image-only path: drop hints that don't apply, keep only "summary".
    if url is None and image_urls:
        image_hints = [h for h in kind_hints if h == "summary"]
        if not image_hints:
            # User typed `litmus` on an image-only message — explain and stop.
            try:
                await message.channel.send(
                    "❌ The `litmus` test is for URLs, not images. "
                    "Try `tldr` for an image summary.",
                    reference=message, mention_author=False,
                )
            except discord.HTTPException:
                pass
            return
        if len(image_hints) != len(kind_hints):
            # Chained `tldr litmus` on an image — keep the summary, log that
            # litmus was dropped.
            log.info(
                "Reply-trigger: dropping non-image-compatible hints (%s) "
                "on image-only message",
                [h for h in kind_hints if h not in image_hints],
            )
        kind_hints = image_hints

    cfg = _effective_config(message.author.id, message.channel.id)

    # Build one Job per hint. URL parsing happens once above; per-hint we
    # only flip the kind discriminator and pick the right ID scheme.
    # Built BEFORE the rate-limit check so the retry view (attached to
    # an LLM-offline rejection) has the specs available.
    jobs: list[Job] = []
    for hint in kind_hints:
        if url is None and image_urls:
            # Image-attachment summary. video_id derived from the first
            # attachment URL so retries dedupe sensibly on the inflight map.
            job = Job(
                url=image_urls[0],  # used as cache key + log identifier
                video_id=_hash_url(image_urls[0]),
                channel=message.channel,
                submitter_id=message.author.id,
                submitter_name=str(message.author),
                model_override=cfg.get("model"),
                kind="image",
                explicit_request=True,
                image_urls=list(image_urls),
                image_filenames=list(image_filenames),
                message=message,
            )
        elif hint == "litmus":
            # Litmus is always a web-style fetch — even when the URL points
            # at a video, we want to inspect the page (text, byline,
            # AdSense, domain age), not transcribe the audio.
            job = Job(
                url=url, video_id=_hash_url(url),
                channel=message.channel,
                submitter_id=message.author.id,
                submitter_name=str(message.author),
                model_override=cfg.get("model"),
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
                diarize=cfg.get("diarize", False),
                vlm_enabled=cfg.get("vlm_enabled", VLM_ENABLED),
                yt_comments_enabled=cfg.get(
                    "yt_comments_enabled", YT_COMMENTS_ENABLED
                ),
                model_override=cfg.get("model"),
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
                model_override=cfg.get("model"),
                kind="web",
                explicit_request=True,
                message=message,
            )
        jobs.append(job)

    # Rate-limit + queue-cap apply equally to web/video/litmus jobs.
    # Atomic batch: reject the whole reply if it would push past either cap.
    ok, reason = _rate_limit_check(message.author.id, count=len(kind_hints))
    if not ok:
        await safe_react(message, "\U0001f6ab")  # 🚫
        retry_view = _maybe_retry_view(
            [_job_to_retry_spec(j) for j in jobs],
            target_user_id=message.author.id,
        )
        try:
            await message.channel.send(
                f"❌ {message.author.mention} {reason}",
                reference=message, mention_author=False,
                view=retry_view,
            )
        except discord.HTTPException:
            pass
        return

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


# Patterns in scraped Markdown / stripped HTML that indicate a bot-protection
# interstitial (Cloudflare, DataDome, Akamai Bot Manager, PerimeterX). When
# seen in the Crawl4AI output we fall back to FlareSolverr; when seen in
# FlareSolverr's resolved HTML we reject too (FlareSolverr only handles
# Cloudflare, so a DataDome/Akamai stub passing through it is still useless).
_BOT_CHALLENGE_MARKERS = (
    # Cloudflare — "Just a moment" path is the classic JS challenge.
    "Just a moment",
    "Checking your browser",
    "challenges.cloudflare.com",
    "cf-challenge",
    "ddos protection by cloudflare",
    "Enable JavaScript and cookies to continue",
    "cf_chl_opt",
    # Cloudflare CAPTCHA / Turnstile path — wording differs from the JS
    # challenge above. archive.ph in particular shells out to this when
    # rate-limiting our IP, with the title "One more step" + body
    # "Please complete the security check to access <site>".
    "One more step",
    "Please complete the security check",
    "Why do I have to complete a CAPTCHA?",
    "Completing the CAPTCHA proves you are a human",
    # DataDome (e.g. Reuters, Allociné, many French/EU news sites).
    # `var dd={'rt':` is their script init; `captcha-delivery.com` is their
    # challenge endpoint.
    "captcha-delivery.com",
    "var dd={'rt':",
    # Akamai Bot Manager. `_abck=~-1~` is the canonical "unsolved
    # challenge" cookie sentinel — won't appear in legitimate article
    # text the way bare `_abck` would (security blogs about Akamai
    # bypass routinely mention the cookie name itself).
    "ak-challenge",
    "_abck=~-1~",
    # PerimeterX / HUMAN
    "_pxhd",
    "px-captcha",
)


def _looks_like_bot_challenge(text: str) -> bool:
    """Heuristic: extracted body that's just a bot-protection interstitial.

    A real article body is usually >500 chars of paragraph text; a challenge
    stub is short and dominated by the marker phrases above. Both checks
    together avoid false positives on long articles that happen to mention
    one of these products in passing. The upper-length cap is generous
    (4000 chars) because some Akamai stubs embed a chunk of bm.js text.
    """
    if not text or len(text) > 4000:
        return False
    return any(m.lower() in text.lower() for m in _BOT_CHALLENGE_MARKERS)


# Backwards-compat alias — `_looks_like_cf_challenge` was the original name
# back when only Cloudflare was on the radar. Kept so any external scripts
# or in-flight branches don't break; safe to remove on the next major.
_CF_CHALLENGE_MARKERS = _BOT_CHALLENGE_MARKERS
_looks_like_cf_challenge = _looks_like_bot_challenge


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
    if not md:
        return None
    if _looks_like_bot_challenge(md):
        log.warning("Crawl4AI returned bot-challenge stub for %s — "
                    "falling back to FlareSolverr", url)
        return None
    if MIN_SCRAPED_BODY_CHARS and len(md) < MIN_SCRAPED_BODY_CHARS:
        # Hardened anti-bot products (DataDome, Akamai) often pass the
        # marker check because their challenge stub is just `<title>` text
        # after readability extraction (e.g. Reuters → "reuters.com").
        # An article that genuinely renders to <200 chars is rare enough
        # that it's worth retrying via FlareSolverr.
        log.warning("Crawl4AI returned too-short body (%d chars, floor %d) "
                    "for %s — treating as failed extraction",
                    len(md), MIN_SCRAPED_BODY_CHARS, url)
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
    if not text:
        return None
    if _looks_like_bot_challenge(text):
        # FlareSolverr only handles Cloudflare — DataDome/Akamai stubs sail
        # through it unchanged. Reject so the caller surfaces an honest
        # "scrapers failed" error rather than summarising boilerplate.
        log.warning("FlareSolverr returned bot-challenge stub for %s", url)
        return None
    if MIN_SCRAPED_BODY_CHARS and len(text) < MIN_SCRAPED_BODY_CHARS:
        # Probably still a challenge or login wall — give up rather than feed
        # the LLM a few hundred bytes of nav-bar text.
        log.warning("FlareSolverr returned too-short body (%d chars, "
                    "floor %d) for %s", len(text), MIN_SCRAPED_BODY_CHARS, url)
        return None
    return text


# ─── Tier 3/4: Archive fallbacks ──────────────────────────────────────────────
# When Crawl4AI + FlareSolverr both fail (anti-bot product won and stayed
# won), try fetching a pre-rendered snapshot from archive.org's Wayback
# Machine, then archive.ph. Both have several upsides:
#
#   • Snapshot HTML has already cleared DataDome/Akamai/CF at archive time
#     — we never hit the live origin, never see the challenge.
#   • Wayback exposes an "id_" raw-bytes view that strips its UI chrome,
#     so Crawl4AI's readability extractor sees the article DOM cleanly.
#   • archive.ph re-renders each snapshot as a static HTML mirror; no JS
#     execution required to view.
#
# Trade-offs:
#   • Wayback snapshots can lag the live URL by hours-to-days. For news
#     this is usually fine (articles rarely change after publication);
#     for live blogs / fast-updating pages it's a known limitation.
#   • Wayback rate-limits anonymous traffic aggressively (~15 req/min/IP
#     in our testing). On 429 we silently fall through to archive.ph
#     rather than retrying — the live channel might be a single high-
#     traffic Discord server hitting the same IP.
#   • archive.ph has no official API and occasionally serves Cloudflare
#     challenges of its own. Crawl4AI's headful Playwright clears those.
#
# Both tiers are gated on ENABLE_ARCHIVE_FALLBACKS so they can be turned
# off without a redeploy.


# Inserts the Wayback `id_` modifier after the timestamp, which switches
# the response from "snapshot wrapped in Wayback chrome" to "raw archived
# bytes". Important for clean readability extraction.
_WAYBACK_TS_RE = re.compile(r"(/web/\d{14})/")


def _wayback_raw_url(snapshot_url: str) -> str:
    """Convert a normal Wayback URL to its `id_` raw form.

    Input:  http://web.archive.org/web/20231213155408/https://www.reuters.com/
    Output: https://web.archive.org/web/20231213155408id_/https://www.reuters.com/

    Also coerces the scheme to HTTPS because the API still returns http:// URLs
    from time to time.
    """
    url = snapshot_url.replace("http://web.archive.org", "https://web.archive.org", 1)
    return _WAYBACK_TS_RE.sub(r"\1id_/", url, count=1)


async def _fetch_via_wayback(url: str) -> str | None:
    """Tier 3 fallback: Wayback Machine.

    1. Ask the availability API for the closest snapshot of `url`.
    2. Rewrite to the `id_` raw form (no Wayback nav chrome).
    3. Hand that URL to Crawl4AI so its readability extractor pulls the
       article body. We re-use the existing Crawl4AI path so MIN_SCRAPED_
       BODY_CHARS + bot-challenge detection apply automatically.

    Returns None on:
      • Rate limit (429) — falls through to archive.ph silently.
      • No snapshot available — common for very recent / niche URLs.
      • Crawl4AI fails on the snapshot URL.
      • Network / timeout errors.

    Never raises — this is a best-effort tier. PermanentError from the
    downstream Crawl4AI call is suppressed too (we don't want a malformed
    snapshot URL to short-circuit the entire fallback chain).
    """
    # Kill-switch is evaluated FIRST so flipping ENABLE_ARCHIVE_FALLBACKS
    # off cleanly disables the tier without depending on http session
    # initialisation order (matters for tests + standalone import).
    if not ENABLE_ARCHIVE_FALLBACKS:
        return None
    if http is None:
        raise RuntimeError("HTTP session not initialised")

    try:
        async with http.get(
            WAYBACK_API,
            params={"url": url},
            timeout=aiohttp.ClientTimeout(total=WAYBACK_TIMEOUT),
            headers={"User-Agent": ARCHIVE_USER_AGENT},
        ) as resp:
            if resp.status == 429:
                log.warning("Wayback rate-limited (429) for %s — skipping tier", url)
                return None
            if resp.status != 200:
                log.warning("Wayback availability %d for %s", resp.status, url)
                return None
            data = await resp.json(content_type=None)
    except asyncio.TimeoutError:
        log.warning("Wayback availability timeout for %s", url)
        return None
    except aiohttp.ClientError as e:
        log.warning("Wayback transport error for %s: %s", url, e)
        return None
    except (ValueError, json_mod.JSONDecodeError) as e:
        log.warning("Wayback non-JSON response for %s: %s", url, e)
        return None

    snap = (data.get("archived_snapshots") or {}).get("closest") or {}
    snap_url = snap.get("url")
    if not snap.get("available") or not snap_url:
        log.info("Wayback has no snapshot for %s", url)
        return None

    raw_url = _wayback_raw_url(snap_url)
    log.info("Wayback snapshot %s for %s — re-extracting", snap.get("timestamp"), url)
    try:
        return await _fetch_via_crawl4ai(raw_url)
    except PermanentError as e:
        # A 4xx from Crawl4AI on the snapshot URL shouldn't kill the
        # whole chain — archive.ph is still worth trying.
        log.warning("Wayback snapshot rejected by Crawl4AI for %s: %s", url, e)
        return None


async def _fetch_via_archive_ph(url: str) -> str | None:
    """Tier 4 fallback: archive.ph (archive.today).

    No official API — we hit `/newest/<url>`, which 302s to the latest
    snapshot. archive.ph snapshots are static HTML with the article body
    intact, so Crawl4AI's readability extractor handles them well.

    Returns None on any failure. Never raises (best-effort tier — when
    this fails fetch_article raises a final RuntimeError).
    """
    if not ENABLE_ARCHIVE_FALLBACKS:
        return None
    if http is None:
        raise RuntimeError("HTTP session not initialised")

    # archive.ph stores by URL exact-match. Percent-encode the embedded URL
    # so reserved chars (`?`, `&`, `=`, `#`) don't get reinterpreted as
    # archive.ph's own query-string by yarl/aiohttp's URL parser:
    #   raw : archive.ph/newest/https://r.com/x?utm=1
    #     -> path=/newest/https://r.com/x  query=utm=1   (lookup misses
    #        the tracker-tagged URL because archive.ph keys by exact URL).
    #   safe: archive.ph/newest/https://r.com/x%3Futm%3D1
    #     -> path is preserved end-to-end, archive.ph decodes it back to
    #        the original URL for snapshot lookup.
    # `safe=':/'` keeps the scheme separator and path slashes readable in
    # logs; only the actually-reserved chars get escaped.
    from urllib.parse import quote
    archive_url = (
        f"{ARCHIVE_PH_BASE.rstrip('/')}/newest/{quote(url, safe=':/')}"
    )
    log.info("Trying archive.ph for %s → %s", url, archive_url)
    try:
        return await _fetch_via_crawl4ai(archive_url)
    except PermanentError as e:
        log.warning("archive.ph URL rejected by Crawl4AI for %s: %s", url, e)
        return None


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

    Routing (live origins first, archive snapshots last):
      1. Reddit post URLs → JSON API + linked-article fetch + top comments.
      2. HackerNews post URLs → Firebase API + linked-article fetch +
         top comments.
      3. Generic Crawl4AI on the live URL.
      4. FlareSolverr on the live URL (CF challenge bypass).
      5. Wayback Machine snapshot (DataDome/Akamai-resilient — never touches
         the live origin). Gated on ENABLE_ARCHIVE_FALLBACKS.
      6. archive.ph snapshot (no official API but tends to mirror sites
         Wayback misses). Gated on ENABLE_ARCHIVE_FALLBACKS.

    Raises PermanentError on 4xx (bad URL, scheme reject) — caller should NOT
    retry. Raises RuntimeError on total failure across every backend.
    """
    from urllib.parse import urlparse

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
        title = urlparse(url).hostname or url
        log.info("[scrape] flaresolverr ok: %s (%d chars)", url, len(text))
        return title, text[:SCRAPED_BODY_CHAR_CAP]

    # Tiers 5 + 6: archive snapshots. Anti-bot-resilient (DataDome, Akamai,
    # hardened Turnstile) because the snapshot was rendered before the
    # protection challenge — we re-fetch the cached HTML from the archive
    # rather than hitting the origin. Both return None on any failure
    # rather than raising; we only escalate to RuntimeError after both miss.
    if ENABLE_ARCHIVE_FALLBACKS:
        log.info("[scrape] FlareSolverr miss — trying Wayback for %s", url)
        md = await _fetch_via_wayback(url)
        if md is not None:
            title = _derive_title_from_markdown(md, url)
            log.info("[scrape] wayback ok: %s (%d chars)", url, len(md))
            return title, md[:SCRAPED_BODY_CHAR_CAP]

        log.info("[scrape] Wayback miss — trying archive.ph for %s", url)
        md = await _fetch_via_archive_ph(url)
        if md is not None:
            title = _derive_title_from_markdown(md, url)
            log.info("[scrape] archive.ph ok: %s (%d chars)", url, len(md))
            return title, md[:SCRAPED_BODY_CHAR_CAP]

    raise RuntimeError(
        f"All scrapers failed for {url} — site likely behind hardened "
        f"anti-bot protection (DataDome/Akamai/Turnstile) with no "
        f"archive snapshot available."
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


class IsLiveError(Exception):
    """process() determined the URL is a currently-airing live stream.

    Caught by worker() (mirroring NotAVideoError) to re-route the job to
    kind='live' / process_live, which streams via whisper-live instead of
    downloading a never-terminating file. Deliberately a plain Exception
    (not PermanentError) so its except clause — placed before the
    NotAVideoError/PermanentError handlers — catches it as a routing signal.
    """


# Pattern matching upstream 5xx errors propagated through whisper's
# /api/image and /api/describe endpoints. The LLM proxy returns 502
# during model swaps (~30-45s window) and the bot's default retry
# backoff (10/30/90) frequently lands the retry mid-swap. When we
# see one of these in the error message, the worker bumps the next
# retry delay to at least 90s so the retry has a chance to land on
# a stable proxy. Match both the proxy's own 502 and llama-server's
# 503-during-load shape.
_UPSTREAM_5XX_PATTERN = re.compile(
    r"VLM HTTP 50[2345]|upstream error|Remote end closed connection|"
    r"llama_server.*loading|model is loading",
    re.IGNORECASE,
)


def _is_upstream_5xx(err: BaseException) -> bool:
    """True if the error is an LLM proxy / llama-server transient 5xx
    that benefits from a longer-than-default retry wait."""
    return bool(_UPSTREAM_5XX_PATTERN.search(str(err)))


class LLMOfflineError(PermanentError):
    """LLM endpoint is unreachable at the transport level (connection
    refused, DNS failure, no route to host, transport timeout).

    Subclass of PermanentError so the worker skips the retry-backoff
    ladder. Backoff (10 + 30 + 90 s) is the right policy for a 503 from
    a loaded model — counterproductive against a port nobody's listening
    on. The periodic LLM health probe (llm_health_loop) re-flips the
    bot to healthy when the endpoint comes back, and queued/new jobs
    are rejected up-front via _rate_limit_check until then.
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
PROCESSING_EMOJI_IMAGE = "\U0001f5bc\ufe0f"  # 🖼️ image OCR + VLM
PROCESSING_EMOJI_LIVE = "\U0001f399\ufe0f"   # 🎙️ live transcription
# Cleanup list covers ALL kinds so a kind switch (e.g. NotAVideoError
# fall-through) leaves no stale reactions behind.
PROCESSING_EMOJI = (
    "\u23f3",                  # ⏳ queued
    PROCESSING_EMOJI_VIDEO,    # 🎧 video fetch
    PROCESSING_EMOJI_WEB,      # 📰 web fetch
    PROCESSING_EMOJI_IMAGE,    # 🖼️ image processing
    PROCESSING_EMOJI_LIVE,     # 🎙️ live transcription
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
    # Surface the server-side job id so /cancel can forward DELETE later.
    _inflight_phase(job, PHASE_TRANSCRIBING, server_job_id=job_id)

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
        cancelled_pre_run = False
        # Pre-run cancel check — user may have hit /cancel while the job was
        # waiting. Drop without invoking the handler.
        _entry_pre = _inflight.get(job.video_id)
        if _entry_pre is not None and _entry_pre.cancel_requested:
            log.info("[%s] Cancelled before run (user-requested)", job.video_id)
            cancelled_pre_run = True
        # When not cancelled, started_at gets set by the handler's first
        # phase transition (download/scrape react). Brief gap between
        # queue.get() and that react is at most a few hundred ms.
        # GPU contention is now handled server-side by the queue at
        # /api/jobs — busy-wait branches are gone. This loop only handles
        # truly transient errors (network blips, LLM timeouts, etc.).
        for attempt in range(MAX_RETRIES + 1):
            if cancelled_pre_run:
                break
            try:
                if attempt > 0:
                    delay = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                    # LLM proxy model swaps take ~30-45s and return 502
                    # during the window. The default 10s/30s/90s backoff
                    # consistently lands the retry mid-swap. Bump the
                    # first two delays to >swap-time when the last error
                    # was an upstream 5xx, so the retry has a chance to
                    # land on a stable proxy. Caps at the configured
                    # delay if the user explicitly raised RETRY_BACKOFF.
                    if last_error is not None and _is_upstream_5xx(last_error):
                        delay = max(delay, 90)
                        log.info(
                            "[%s] Retry %d/%d in %ds (upstream 5xx — "
                            "waiting past model swap window)...",
                            job.video_id, attempt, MAX_RETRIES, delay,
                        )
                    else:
                        log.info("[%s] Retry %d/%d in %ds...",
                                 job.video_id, attempt, MAX_RETRIES, delay)
                    await asyncio.sleep(delay)
                # Recompute handler each attempt — NotAVideoError flips kind.
                if job.kind == "litmus":
                    handler = process_litmus
                elif job.kind == "web":
                    handler = process_url
                elif job.kind == "image":
                    handler = process_image
                elif job.kind == "live":
                    handler = process_live
                else:
                    handler = process
                await handler(job)
                last_error = None
                break
            except IsLiveError as e:
                log.info("[%s] live stream — re-routing to live pipeline: %s",
                         job.video_id, e)
                job.kind = "live"
                # Routing change, not a retry — re-enter loop with new handler.
                continue
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

        if cancelled_pre_run:
            # User cancelled while queued. Clear reactions and ack the user.
            for emoji in PROCESSING_EMOJI:
                await _job_remove_react(job, emoji)
            try:
                await _job_reply(job, f"⏹️ `{job.video_id}` cancelled before run.")
            except Exception:
                pass
        elif silent_drop:
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
            # LLM-offline failures get a clearer user-facing message
            # than the generic "LLMOfflineError: LLM unreachable at X"
            # since the user's next move (resubmit later) differs from
            # other permanent failures (which are typically content-side
            # and won't be fixed by resubmitting).
            if isinstance(last_error, LLMOfflineError):
                reply = _llm_offline_user_reason()
                # Attach a Retry button so the user gets a one-click
                # resubmit when the LLM is back, mirroring the queue-gate
                # rejection UX. The retry path re-enters worker() and will
                # cache-hit the transcribe step (same video_id + translate),
                # so the user only pays the summarisation cost on retry.
                retry_view = RetryJobsView(
                    specs=[_job_to_retry_spec(job)],
                    target_user_id=job.submitter_id,
                )
            else:
                reply = (
                    f"Failed to process `{job.video_id}` ({attempts}): "
                    f"{type(last_error).__name__}: {last_error}"
                )
                retry_view = None
            await _job_reply(job, reply, view=retry_view)
        # Terminal: drop from inflight whether success, error, silent drop,
        # or cancellation. Records runtime for the rolling ETA average when
        # the job actually ran.
        _inflight_remove(job.video_id, kind=job.kind)
        queue.task_done()


async def process(job: Job):
    if http is None: raise RuntimeError("HTTP session not initialised")

    # 1. Check whisper service status
    async with http.get(f"{WHISPER_API}/api/status") as resp:
        if resp.status != 200:
            raise RuntimeError("Whisper service unavailable")

    # 1a. Cache lookup — skip download+transcribe if we have a fresh transcript.
    # `refresh=true` on the Job bypasses this (user explicitly asked for a
    # fresh run via /summarize refresh:true). The result still overwrites
    # the cache below so subsequent runs benefit.
    cached = None if job.refresh else read_cache(job.video_id, job.translate)
    if job.refresh:
        log.info("[%s] Cache bypass (refresh=true)", job.video_id)
    file_path = None
    # Comments aren't cached today; cache hits skip the Community Reaction
    # embed. First-time runs (cache miss) populate this from the yt-dlp
    # response and use it after the main 3-style summary gather.
    raw_comments: list[dict] = []
    if cached is not None:
        title, status, transcript, duration = cached
        log.info("[%s] Cache hit (%d chars, '%s')", job.video_id, len(transcript), title)
    else:
        # 1b. Live-stream gate. A live stream download never terminates, so
        # detect-and-reroute BEFORE the download. Cache hits skip this (a
        # cached transcript means the stream already finished as a VOD).
        # _is_live_stream fails safe to False, so a down whisper-live just
        # means we attempt the normal video path.
        if await _is_live_stream(job.url):
            raise IsLiveError(job.url)
        # 2. Download. Keep the video stream alongside audio when VLM is
        # enabled — /api/describe needs a video file to extract frames
        # from. When VLM is off, audio-only WAV (smaller, current default).
        # Optionally also fetch top YT comments for the "Community reaction"
        # embed (default on; per-channel opt-out via /config).
        log.info("[%s] Downloading%s%s...", job.video_id,
                 " (audio+video)" if job.vlm_enabled else "",
                 f" + comments(top {YT_COMMENTS_MAX})" if job.yt_comments_enabled else "")
        _inflight_phase(job, PHASE_DOWNLOADING)
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
        # Now that we know what the video actually is, surface it in /progress.
        _inflight_phase(job, PHASE_DOWNLOADING, title=title, duration=duration)
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
        _inflight_phase(job, PHASE_TRANSCRIBING)

        transcribe_payload = {
            "file_path": file_path,
            "model": WHISPER_MODEL,
            # Don't cleanup yet — VLM fallback (below) may need the file.
            "cleanup": False,
            "return_file": False,  # bot uses transcript text directly
            "diarize": job.diarize,
            # Server resolves "auto" via a 30s LID pre-pass and translates
            # to English for non-English sources. See docs/design/multilingual.md.
            "translate": job.translate,
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

        # Persist to cache (whatever combination of speech / visual we ended up with).
        # Keyed by translate mode so the three variants (auto/translate/native)
        # don't collide — see _cache_path docstring for the bug this prevents.
        write_cache(job.video_id, title, status, transcript, duration, job.translate)

    # 5. Summarize in multiple styles (concurrent — model handles full context)
    log.info("[%s] Summarizing (%d chars)...", job.video_id, len(transcript))
    await _job_react(job, "\U0001f9e0")  # 🧠
    _inflight_phase(job, PHASE_SUMMARIZING, title=title, duration=duration)

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
    # The view carries the translate variant so rename hits the correct
    # cache file (translate=auto/translate/native each have their own file).
    view = None
    if job.diarize and _has_speaker_labels(transcript):
        view = SpeakerRenameView(
            job_video_id=job.video_id,
            channel_id=job.channel.id,
            translate=job.translate,
        )

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

    # 1. Cache lookup (skipped on refresh=true). Web jobs don't use the
    # translate dimension (article text stays in source language) so they
    # always cache under the "auto" key.
    cached = None if job.refresh else read_cache(job.video_id)
    if job.refresh:
        log.info("[%s] Cache bypass (refresh=true)", job.video_id)
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
        _inflight_phase(job, PHASE_SCRAPING)
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
    _inflight_phase(job, PHASE_SUMMARIZING, title=title)

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


# ─── Image attachment handler (OCR + VLM + LLM summary) ──────────────────
# Triggered by replying `tldr` / `summarize` to a Discord message that
# carries image attachments (and no URL). For each attachment we:
#   1. Download the bytes via the bot's aiohttp session.
#   2. POST multipart to whisper's /api/image — the service runs EasyOCR
#      (faithful text extraction) and the configured VLM (scene description)
#      in parallel and returns both.
#   3. Format into a compact <images> block (one section per attachment).
#   4. Run the standard summarize() over that block with the IMAGE prompts.
#   5. Post a single TL;DR embed; if any OCR text was extracted, attach
#      a verbatim "Text in image" embed below it.

IMAGE_MAX_ATTACHMENTS = int(os.environ.get("IMAGE_MAX_ATTACHMENTS", "4"))
# Per-attachment byte cap on what the bot will forward to whisper. Discord
# free-tier attachments cap at 25 MB; Nitro at 50 MB. 32 MB matches the
# server-side IMAGE_MAX_BYTES default.
IMAGE_MAX_BYTES_PER_ATTACHMENT = int(os.environ.get(
    "IMAGE_MAX_BYTES_PER_ATTACHMENT", str(32 * 1024 * 1024)))
# Per-attachment timeout for the /api/image call. OCR is ~1-2s; VLM is the
# long pole (5-30s on a 7B model, longer on larger). 180s leaves headroom
# for queue waits on the LLM proxy.
IMAGE_API_TIMEOUT = int(os.environ.get("IMAGE_API_TIMEOUT", "180"))

# File extensions accepted as images. Mirrors _ALLOWED_IMAGE_CONTENT_TYPES
# on the server side but checked against attachment filenames (Discord's
# attachment.content_type can be missing for older attachments).
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif",
                     ".bmp", ".tif", ".tiff")


def _attachment_is_image(att: "discord.Attachment") -> bool:
    """True if a Discord attachment looks like a still image.

    Prefers att.content_type when present (Discord populates it from MIME
    sniffing); falls back to extension. Animated GIFs are accepted — OCR
    grabs the first frame, VLM describes the still.
    """
    ct = (att.content_type or "").lower()
    if ct.startswith("image/"):
        return True
    fn = (att.filename or "").lower()
    return fn.endswith(_IMAGE_EXTENSIONS)


def _extract_image_attachments(
    message: "discord.Message",
) -> tuple[list[str], list[str]]:
    """Return (urls, filenames) for image attachments on the message.

    Capped at IMAGE_MAX_ATTACHMENTS to bound LLM cost and Discord embed
    real-estate. Order preserved (matches the order Discord renders them).
    """
    urls: list[str] = []
    filenames: list[str] = []
    for att in message.attachments:
        if not _attachment_is_image(att):
            continue
        # Filter out oversized attachments at the source so the user sees
        # the rejection in logs and we don't waste a download round-trip.
        if att.size and att.size > IMAGE_MAX_BYTES_PER_ATTACHMENT:
            log.info(
                "Image attachment %s skipped: %d bytes > limit %d",
                att.filename, att.size, IMAGE_MAX_BYTES_PER_ATTACHMENT,
            )
            continue
        urls.append(att.url)
        filenames.append(att.filename or "image")
        if len(urls) >= IMAGE_MAX_ATTACHMENTS:
            break
    return urls, filenames


# Image-format magic-byte signatures. We check these on every downloaded
# attachment to catch Discord CDN edge cases where a signed URL returns
# 200 with a tiny HTML error stub (`{"message":"This content is no longer
# available."}` shape) instead of the real bytes — the bot used to forward
# those stubs to /api/image, which then handed them to the VLM, which 400'd
# with "Failed to load image". With a sniff here, the failure is surfaced
# as a clean PermanentError after the first attempt instead of burning
# four retries on the same stub.
_IMAGE_MAGIC_BYTES = (
    b"\x89PNG\r\n\x1a\n",          # PNG
    b"\xff\xd8\xff",                # JPEG (any flavour)
    b"GIF87a", b"GIF89a",            # GIF
    b"RIFF",                         # WebP (full check below)
    b"BM",                           # BMP
    b"II*\x00", b"MM\x00*",          # TIFF (little/big endian)
    b"\x00\x00\x00\x0cftypheic",     # HEIC (Discord auto-converts iOS uploads)
    b"\x00\x00\x00\x0cftypheix",
    b"\x00\x00\x00\x0cftypavif",     # AVIF
)


def _looks_like_image_bytes(data: bytes) -> bool:
    """Cheap magic-byte sniff to confirm `data` is a real image.

    Falsey for HTML error stubs, JSON error bodies, redirected login pages,
    and the empty/truncated payloads Discord's CDN occasionally returns
    for expired signed URLs.
    """
    if not data or len(data) < 16:
        return False
    for sig in _IMAGE_MAGIC_BYTES:
        if data.startswith(sig):
            # RIFF prefix is shared by WAV / AVI / WebP — require the
            # WEBP fourcc at byte offset 8 for true WebP.
            if sig == b"RIFF":
                return len(data) >= 12 and data[8:12] == b"WEBP"
            return True
    return False


async def _download_attachment_bytes(url: str) -> bytes:
    """Download a Discord CDN attachment. Returns raw bytes.

    Raises:
      PermanentError on 4xx, on non-image content-type, or on bytes that
        don't pass the magic-byte sniff (HTML error stub, JSON, empty).
        These never recover by retrying.
      RuntimeError on transient network failure (worker will retry).
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if 400 <= resp.status < 500:
                raise PermanentError(
                    f"attachment download failed {resp.status} — "
                    f"signed URL likely expired or attachment deleted"
                )
            if resp.status != 200:
                raise RuntimeError(
                    f"attachment download failed: HTTP {resp.status}"
                )
            # Cap upload size at the source via Content-Length BEFORE reading
            # — we don't want to buffer multi-GB into memory just to reject.
            # Discord populates Content-Length for all attachment responses.
            clen = int(resp.headers.get("content-length") or "0")
            if clen and clen > IMAGE_MAX_BYTES_PER_ATTACHMENT:
                raise PermanentError(
                    f"attachment exceeds size cap "
                    f"({clen} > {IMAGE_MAX_BYTES_PER_ATTACHMENT} bytes)"
                )
            # `resp.read()` reads the full body, length-prefixed. Earlier
            # versions used `resp.content.read(N)` which streams and can
            # return short on chunked-transfer or connection-edge cases —
            # observed delivering truncated PNGs to /api/image and tripping
            # PIL's "image file is truncated" path. `read()` always returns
            # the complete body or raises ClientPayloadError.
            data = await resp.read()
            if len(data) > IMAGE_MAX_BYTES_PER_ATTACHMENT:
                raise PermanentError(
                    f"attachment exceeds size cap "
                    f"({len(data)} > {IMAGE_MAX_BYTES_PER_ATTACHMENT} bytes)"
                )
            # Cross-check against Content-Length — if the body is shorter
            # than the server claimed, the transfer was interrupted. Raise
            # transient so a retry can grab the full thing.
            if clen and len(data) < clen:
                raise RuntimeError(
                    f"attachment download short: got {len(data)} bytes, "
                    f"server claimed {clen}"
                )
            ctype = (resp.headers.get("content-type") or "").lower()
            # Discord CDN returns the actual content-type for valid
            # attachments (image/png, image/webp, image/gif, ...). If it
            # comes back as text/html or application/json the body is an
            # error stub, not an image — surface a permanent failure.
            if ctype and not ctype.startswith(
                ("image/", "application/octet-stream", "binary/octet-stream"),
            ):
                preview = data[:200].decode("utf-8", errors="replace")
                raise PermanentError(
                    f"attachment download returned non-image "
                    f"content-type '{ctype}' — likely a CDN error page. "
                    f"Body preview: {preview!r}"
                )
            # Magic-byte sniff catches the case where content-type is
            # missing or generic but the bytes still aren't an image.
            if not _looks_like_image_bytes(data):
                preview = data[:64].hex()
                raise PermanentError(
                    f"attachment download returned {len(data)} bytes that "
                    f"are not a recognised image format. "
                    f"First bytes (hex): {preview}"
                )
            return data
    except asyncio.TimeoutError:
        raise RuntimeError("attachment download timed out")
    except aiohttp.ClientError as e:
        raise RuntimeError(f"attachment download transport error: {e}")


# Inline retry budget for transient upstream 5xx inside _call_image_api.
# Separate from the worker-level RETRY_BACKOFF: this catches the proxy
# mid-swap window without burning a full worker retry slot. Total worst
# case from these in-call retries: 2 * 90s = 3 min before the call
# returns to the worker loop (which may then do its own retries).
_IMAGE_API_INLINE_502_RETRIES = int(os.environ.get(
    "IMAGE_API_INLINE_502_RETRIES", "2"))
_IMAGE_API_INLINE_502_WAIT = int(os.environ.get(
    "IMAGE_API_INLINE_502_WAIT", "90"))


async def _call_image_api(
    data: bytes,
    filename: str,
    *,
    prompt: str | None = None,
) -> dict:
    """POST one image to whisper's /api/image. Returns the JSON body.

    Raises PermanentError on 422 permanent (VLM not configured) or 4xx
    rejections; RuntimeError on transient 5xx / transport / timeout.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    form = aiohttp.FormData()
    # Guess a content_type from filename; falls back to image/jpeg which
    # the server-side filter accepts for any image/*-shaped filename.
    import mimetypes
    ctype, _ = mimetypes.guess_type(filename)
    form.add_field(
        "file", data,
        filename=filename,
        content_type=ctype or "image/jpeg",
    )
    if prompt:
        form.add_field("prompt", prompt)
    # Inline retry loop for upstream 5xx — specifically the model-proxy
    # 502 that lands during a vision-model swap window. Without this,
    # the worker-level retry (10s/30s/90s) almost always lands the
    # next attempt mid-swap too.
    attempts = _IMAGE_API_INLINE_502_RETRIES + 1
    for inline_attempt in range(attempts):
        try:
            async with http.post(
                f"{WHISPER_API}/api/image",
                data=form,
                timeout=aiohttp.ClientTimeout(total=IMAGE_API_TIMEOUT),
            ) as resp:
                body = await resp.json(content_type=None)
                if resp.status == 200:
                    return body
                err = (body or {}).get("error", f"HTTP {resp.status}")
                if body and body.get("permanent") is True:
                    kind = body.get("kind") or ""
                    # VLM-not-configured: operator needs to fix the model;
                    # retrying never helps.
                    if kind == "vlm_not_configured":
                        raise PermanentError(
                            f"vision model can't accept images: {err}. "
                            f"Set LLM_VISION_MODEL to a multimodal model "
                            f"(or load with mmproj on llama-server)."
                        )
                    raise PermanentError(f"/api/image rejected: {err}")
                if 400 <= resp.status < 500:
                    raise PermanentError(f"/api/image {resp.status}: {err}")
                # 5xx — inline-retry if it looks like a proxy swap (502
                # with "upstream error" / "Remote end closed" in the body)
                # and we still have inline-retry budget.
                err_str = f"/api/image {resp.status}: {err}"
                is_swap_502 = (
                    resp.status == 502
                    and (_UPSTREAM_5XX_PATTERN.search(err_str) is not None)
                )
                if is_swap_502 and inline_attempt < attempts - 1:
                    log.info(
                        "/api/image got swap-502 (inline attempt %d/%d); "
                        "sleeping %ds to outlast model swap...",
                        inline_attempt + 1, attempts,
                        _IMAGE_API_INLINE_502_WAIT,
                    )
                    await asyncio.sleep(_IMAGE_API_INLINE_502_WAIT)
                    # IMPORTANT: aiohttp FormData is single-use; rebuild
                    # for the next attempt.
                    form = aiohttp.FormData()
                    form.add_field(
                        "file", data,
                        filename=filename,
                        content_type=ctype or "image/jpeg",
                    )
                    if prompt:
                        form.add_field("prompt", prompt)
                    continue
                raise RuntimeError(err_str)
        except asyncio.TimeoutError:
            raise RuntimeError(f"/api/image timed out after {IMAGE_API_TIMEOUT}s")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"/api/image transport error: {e}")


def _format_image_block(
    *,
    index: int,
    filename: str,
    width: int,
    height: int,
    description: str,
    ocr: str,
) -> str:
    """Render one image's extracted content into the <images> block format
    described in PROMPT_BRIEF_IMAGE. Both description and OCR are included
    when present; absent fields render as `(none)` to keep the LLM honest
    about what was actually extracted (vs hallucinating absent content).
    """
    dim = f"{width}×{height}" if width and height else "unknown size"
    description = (description or "").strip() or "(no description available)"
    ocr_block = (ocr or "").strip()
    # EasyOCR joins snippets with " | " — split back to one-per-line so the
    # LLM sees discrete tokens rather than one wall of text.
    if ocr_block:
        ocr_lines = [s.strip() for s in ocr_block.split(" | ") if s.strip()]
        ocr_rendered = "\n".join(ocr_lines)
    else:
        ocr_rendered = "(no text detected)"
    return (
        f"## Image {index} ({filename}, {dim})\n"
        f"[Description] {description}\n"
        f"[Text on screen]\n{ocr_rendered}"
    )


# Min total OCR chars across all images before we also render the
# "Text in image" verbatim embed and run the key-points LLM pass.
# Below this, the brief alone covers the content (a meme / photo with a
# 2-word caption doesn't need a separate verbatim block).
IMAGE_OCR_VERBATIM_MIN_CHARS = int(os.environ.get(
    "IMAGE_OCR_VERBATIM_MIN_CHARS", "80"))


# Filenames Discord (and the OSes that paste into it) assign to clipboard
# screenshots / unsaved uploads. None of these carry information about the
# image content — "TL;DR: Image: image.png" reads as nothing. When we see
# one we drop the filename from the embed title and try to derive something
# meaningful from the OCR text instead.
#
# Matches: image.png, image.jpg, Untitled.png, clipboard.jpg, screenshot*,
# screen-shot*, Screen Shot YYYY-MM-DD*, paste-image.png, pasted_image.png,
# IMG_1234.JPG, DSC01234.JPG, PIC00001.JPG, 1234.png, 2024-05-26-*.png.
_GENERIC_IMAGE_NAME_PATTERN = re.compile(
    r"^(?:"
    r"image|untitled|clipboard|paste[d]?[_\-\s]*image|"
    r"screen[_\-\s]?shot.*|"                  # screenshot / Screen Shot YYYY-MM-DD at HH:MM:SS
    r"(?:img|dsc|pic|cam|p|photo)[_\-]?\d+|"  # IMG_1234, DSC01234, etc.
    r"\d{4}[_\-]\d{2}[_\-]\d{2}.*|"           # date-only stems
    r"\d+"                                    # pure number stems (1234.png)
    r")$",
    re.IGNORECASE,
)


def _is_generic_image_filename(filename: str) -> bool:
    """True if filename is a placeholder like 'image.png' / 'Untitled.jpg' /
    'IMG_1234.JPG' — carries no signal about the image content. Bot then
    falls back to OCR-derived title."""
    if not filename:
        return True
    stem = os.path.splitext(filename)[0].strip()
    if not stem:
        return True
    return bool(_GENERIC_IMAGE_NAME_PATTERN.match(stem))


def _ocr_title_snippet(ocr_text: str, max_chars: int = 80) -> str:
    """Extract a short, title-worthy phrase from EasyOCR output.

    EasyOCR joins detected text snippets with ' | '. The first snippet is
    usually the highest-up / left-most text in the image — typically a
    headline / username / window title / meme caption, which makes a
    much better Discord embed title than 'image.png'.

    Returns '' when no snippet meets the minimum-length bar (avoids
    1-2 character snippets like a stray '@' or 'X').
    """
    if not ocr_text:
        return ""
    first = ocr_text.split(" | ", 1)[0].strip()
    # Collapse internal whitespace runs so multi-line OCR doesn't render
    # as a multi-line embed title (Discord renders \n inside titles).
    first = re.sub(r"\s+", " ", first)
    # Strip trailing isolated punctuation / symbols when they sit AFTER
    # a space — these are typically OCR-chunk artifacts where the next
    # neighbouring word (e.g. "@username") got split off by the " | "
    # join. "Kate Tungusova @" → "Kate Tungusova". Doesn't touch valid
    # trailing punctuation glued to a word like "Hello world!" because
    # we require a leading space.
    first = re.sub(r"\s+[^\w\s]{1,3}$", "", first).rstrip()
    if len(first) < 10:
        return ""
    if len(first) <= max_chars:
        return first
    # Truncate on a word boundary if possible — nicer than mid-word.
    cut = first[: max_chars - 1].rsplit(" ", 1)[0]
    if len(cut) < max_chars // 2:
        cut = first[: max_chars - 1]
    return cut + "…"


def _derive_image_title(
    filenames: list[str],
    ocr_parts: list[tuple[str, str]],
) -> str:
    """Build a human-friendly title for the image-summary embed.

    Priority:
    1. First image's OCR snippet (when OCR found enough text).
    2. Meaningful filename (when not on the generic-placeholder list).
    3. Fallback to "Image" / "N images".

    Multi-image jobs always carry the count, optionally suffixed with
    an OCR snippet from the first image.
    """
    n = len(filenames)
    # First OCR snippet (any image, ordered as ocr_parts — which matches
    # attachment order).
    ocr_snippet = ""
    for _fn, ocr in ocr_parts:
        snippet = _ocr_title_snippet(ocr)
        if snippet:
            ocr_snippet = snippet
            break

    primary_name = filenames[0] if filenames else ""
    is_generic = _is_generic_image_filename(primary_name)
    name_stem = ""
    if not is_generic and primary_name:
        stem = os.path.splitext(primary_name)[0]
        # Make filename-style separators read as English: 'cool_pic-v2' →
        # 'cool pic v2'. Single-letter underscores like 'a_b' get joined.
        name_stem = re.sub(r"[_\-]+", " ", stem).strip()

    if n == 1:
        if ocr_snippet:
            return ocr_snippet
        if name_stem:
            return name_stem
        return "Image"
    # Multi-image
    base = f"{n} images"
    if ocr_snippet:
        return f"{base} — {ocr_snippet}"
    if name_stem:
        return f"{base} ({name_stem} + {n - 1} more)"
    return base


async def process_image(job: Job):
    """Image-attachment handler. OCR + VLM-describe each attachment via
    /api/image, then summarize with the standard LLM.

    Cache strategy: image jobs are NOT cached. Discord CDN URLs are signed
    and rotate; the same conceptual image posted twice gets a different
    URL and a different video_id. Caching would only ever hit on the
    Retry-button path, which already short-circuits via the in-memory view.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")
    if not job.image_urls:
        raise PermanentError("image job has no attachments")

    # 1. Whisper service health check — cheap and prevents wasting an LLM
    # call when /api/image will 503.
    async with http.get(f"{WHISPER_API}/api/status") as resp:
        if resp.status != 200:
            raise RuntimeError("Whisper service unavailable")

    log.info(
        "[%s] Image job: %d attachment(s) — %s",
        job.video_id, len(job.image_urls),
        ", ".join(job.image_filenames),
    )
    await _job_react(job, PROCESSING_EMOJI_IMAGE)
    _inflight_phase(job, PHASE_SCRAPING)  # closest existing phase — download+VLM

    # 2. For each attachment: download → /api/image. Sequential (not
    # parallel) so a single VLM that serialises requests doesn't get
    # hammered, and so OOM on one image doesn't kill the others' progress.
    blocks: list[str] = []
    ocr_parts: list[tuple[str, str]] = []  # (filename, ocr) for verbatim embed
    total_ocr_chars = 0
    for i, (url, filename) in enumerate(
        zip(job.image_urls, job.image_filenames), start=1,
    ):
        log.info("[%s] Downloading attachment %d/%d: %s",
                 job.video_id, i, len(job.image_urls), filename)
        try:
            data = await _download_attachment_bytes(url)
        except PermanentError:
            # One bad attachment shouldn't kill the whole job if others
            # succeed. Skip and continue.
            log.warning("[%s] Attachment %d unavailable, skipping: %s",
                        job.video_id, i, filename)
            blocks.append(
                f"## Image {i} ({filename})\n"
                f"[Description] (could not be downloaded — attachment expired "
                f"or deleted)\n[Text on screen]\n(none)"
            )
            continue

        log.info("[%s] Processing attachment %d via /api/image (%d bytes)",
                 job.video_id, i, len(data))
        # User prompt threads through to the VLM — "focus on the X"
        # style steering, same UX as the video VLM path.
        vlm_prompt = _build_vlm_prompt(job.user_prompt) if job.user_prompt else None
        result = await _call_image_api(data, filename, prompt=vlm_prompt)

        description = (result.get("description") or "").strip()
        ocr = (result.get("ocr") or "").strip()
        width = int(result.get("width") or 0)
        height = int(result.get("height") or 0)
        total_ocr_chars += len(ocr)
        if ocr:
            ocr_parts.append((filename, ocr))

        blocks.append(_format_image_block(
            index=i, filename=filename,
            width=width, height=height,
            description=description, ocr=ocr,
        ))

    if not blocks:
        # All attachments failed to download — nothing to summarise.
        raise PermanentError(
            "no image attachments could be downloaded — "
            "Discord CDN URLs may have expired"
        )

    body = "\n\n".join(blocks)
    log.info(
        "[%s] Image extraction done: %d image(s), %d chars description+OCR, "
        "OCR=%d chars",
        job.video_id, len(blocks), len(body), total_ocr_chars,
    )

    # 3. LLM summary.
    ref_block = ""
    if job.user_prompt:
        ref_block = (
            "The Discord user who requested this summary specifically asked: "
            "<user_request>\n"
            f"{job.user_prompt}\n"
            "</user_request>\n"
            "Honour that request when shaping your output — emphasise the "
            "aspects they're interested in.\n\n"
        )

    await _job_react(job, "\U0001f9e0")  # 🧠 summarising
    _inflight_phase(job, PHASE_SUMMARIZING,
                    title=job.image_filenames[0] if job.image_filenames else None)

    _token = _model_override.set(job.model_override) if job.model_override else None
    try:
        # Brief is always produced. Key-points only when there's a
        # meaningful amount of OCR text (otherwise it duplicates the brief).
        coros = [summarize(
            body, PROMPT_BRIEF_IMAGE, LLM_MAX_TOKENS_BRIEF,
            reference_block=ref_block,
        )]
        do_key_points = total_ocr_chars >= IMAGE_OCR_VERBATIM_MIN_CHARS
        if do_key_points:
            coros.append(summarize(
                body, PROMPT_KEY_POINTS_IMAGE, LLM_MAX_TOKENS_KEY_POINTS,
                reference_block=ref_block,
                char_cap=SUMMARY_CHAR_CAP,
            ))
        results = await asyncio.gather(*coros)
    finally:
        if _token is not None:
            _model_override.reset(_token)

    brief = sanitize_llm_output(results[0])
    key_points = sanitize_llm_output(results[1]) if do_key_points else ""

    # 4. Post embeds. Title prefers OCR snippet over filename because
    # Discord clipboard pastes all arrive as "image.png" and the embed
    # title "TL;DR: image.png" carries zero information about content.
    n = len(job.image_urls)
    title = _derive_image_title(job.image_filenames, ocr_parts)
    detail_channel = resolve_summary_channel(job.channel)
    use_split = detail_channel.id != job.channel.id

    detail_msg = None
    # Detail-channel header (only when split routing is configured).
    if use_split and (key_points or total_ocr_chars):
        if job.message is not None:
            requester = job.message.author.mention
        elif job.interaction is not None:
            requester = job.interaction.user.mention
        else:
            requester = f"<@{job.submitter_id}>"
        header = discord.Embed(
            title=truncate(title, 240),
            description=f"Requested by {requester} in <#{job.channel.id}>",
            color=0x9B59B6,  # purple — distinguishes image from video/web/litmus
        )
        # Set the first image as the embed's image so the detail channel
        # has a visual anchor.
        if job.image_urls:
            header.set_image(url=job.image_urls[0])
        detail_msg = await detail_channel.send(embed=header)

    # Key-points embed (only when we ran the second LLM pass).
    if key_points:
        await send_long_embed(detail_channel, "Key Points", key_points, 0x8E44AD)

    # Verbatim OCR embed — only when the OCR is substantial. Short OCR (a
    # meme caption) is already in the brief; rendering it twice is noise.
    if total_ocr_chars >= IMAGE_OCR_VERBATIM_MIN_CHARS:
        if len(ocr_parts) == 1:
            verbatim = ocr_parts[0][1].replace(" | ", "\n")
        else:
            verbatim = "\n\n".join(
                f"**{fn}**\n{ocr.replace(' | ', chr(10))}"
                for fn, ocr in ocr_parts
            )
        await send_long_embed(
            detail_channel, "Text in image", verbatim, 0x7D3C98,
        )

    # Brief embed in the original channel.
    embed = discord.Embed(
        title=truncate(f"TL;DR: {title}", 240),
        description=truncate(brief, 4000),
        color=0x9B59B6,
    )
    footer_bits = [f"{n} image{'s' if n != 1 else ''}"]
    if total_ocr_chars:
        footer_bits.append(f"{total_ocr_chars} chars OCR")
    embed.set_footer(text=" | ".join(footer_bits))
    # Show the first image inline on the brief embed so users see what
    # was summarised at a glance.
    if job.image_urls:
        embed.set_thumbnail(url=job.image_urls[0])

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

    log.info("[%s] Done — posted image summary (%d image(s), brief=%d chars)",
             job.video_id, n, len(brief))


async def process_live(job: Job):
    """Stream-transcribe a currently-airing live URL via whisper-live.

    The bot is a thin WebSocket client: it sends the URL to whisper-live's
    /ws-url, which runs yt-dlp|ffmpeg internally and streams back transcript
    segments. When the live stream ends (yt-dlp exits), whisper-live sends a
    'done' frame; we then summarise and post (same prompts as video jobs).
    Live streams have no speaker labels, so no SpeakerRenameView is attached.
    """
    if http is None:
        raise RuntimeError("HTTP session not initialised")

    await _job_react(job, PROCESSING_EMOJI_LIVE)  # 🎙️
    _inflight_phase(job, PHASE_TRANSCRIBING)

    # 1. Title via probe (best-effort; the WS path doesn't return it).
    title = "Live Stream"
    try:
        async with http.get(
            f"{WHISPER_LIVE_URL}/probe", params={"url": job.url},
            timeout=aiohttp.ClientTimeout(total=35),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                title = data.get("title") or title
    except Exception:
        pass

    # 2. Stream: open WS, send URL, collect segments until 'done'.
    transcript_parts: list[str] = []
    stream_start = time.monotonic()
    async with http.ws_connect(f"{WHISPER_LIVE_WS}/ws-url", timeout=60) as ws:
        await ws.send_json({"url": job.url})
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data["type"] == "segment":
                    transcript_parts.append(data["text"])
                elif data["type"] == "done":
                    break
                elif data["type"] == "error":
                    raise RuntimeError(f"whisper-live: {data['message']}")
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                break

    if not transcript_parts:
        raise PermanentError("No speech detected in live stream — nothing to summarise.")

    transcript = " ".join(transcript_parts)
    duration = int(time.monotonic() - stream_start)
    duration_str = format_duration(duration)

    # 3. Summarise (mirror the video handler's brief + key_points calls).
    _inflight_phase(job, PHASE_SUMMARIZING)
    await _job_react(job, "\U0001f9e0")  # 🧠
    brief, key_points = await asyncio.gather(
        summarize(
            transcript, PROMPT_BRIEF, LLM_MAX_TOKENS_BRIEF,
            reduce_template=REDUCE_BRIEF,
            title=title, duration=duration_str, reference_block="",
        ),
        summarize(
            transcript, PROMPT_KEY_POINTS, LLM_MAX_TOKENS_KEY_POINTS,
            reduce_template=REDUCE_KEY_POINTS,
            title=title, duration=duration_str, reference_block="",
            char_cap=SUMMARY_CHAR_CAP,
        ),
    )
    brief = sanitize_llm_output(brief)
    key_points = sanitize_llm_output(key_points)

    # 4. Post embeds (no SpeakerRenameView — streaming mode has no speakers).
    detail_channel = resolve_summary_channel(job.channel)
    await send_long_embed(detail_channel, "Key Points", key_points, 0x9B59B6)

    embed = discord.Embed(
        title=f"TL;DW: {truncate(title, 240)}",
        url=job.url,
        description=truncate(brief, 4000),
        color=0x9B59B6,  # purple — distinct from video (red) / web (blue)
    )
    embed.set_footer(text=f"Live · {duration_str}")
    if job.message is not None:
        await job.channel.send(embed=embed, reference=job.message)
    else:
        await job.channel.send(embed=embed)

    for emoji in PROCESSING_EMOJI:
        await _job_remove_react(job, emoji)
    await _job_react(job, "\u2705")  # ✅
    log.info("[%s] Done — live summary posted (%d chars, %s)",
             job.video_id, len(transcript), duration_str)


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
    _inflight_phase(job, PHASE_SCRAPING)

    log.info("[%s] Litmus: scraping %s", job.video_id, job.url)
    title, body = await fetch_article(job.url)
    log.info("[%s] Litmus: scraped %d chars", job.video_id, len(body))
    _inflight_phase(job, PHASE_SCRAPING, title=title)

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
        _inflight_phase(job, PHASE_SUMMARIZING)
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

    Raises:
      - LLMOfflineError on transport-level failure (connection refused,
        DNS, timeout). PermanentError subclass → worker skips retry-ladder
        AND we mark the LLM unhealthy so subsequent submissions short-circuit
        in _rate_limit_check.
      - PermanentError on 4xx or known-permanent error signatures (won't
        recover on retry).
      - RuntimeError on 5xx / other HTTP errors (worker retries).
    """
    if http is None: raise RuntimeError("HTTP session not initialised")
    payload = {
        "model": _model_override.get() or LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    try:
        async with http.post(f"{LLM_API}/chat/completions", json=payload) as resp:
            status = resp.status
            body_for_classify = None
            if status != 200:
                body_for_classify = await resp.text()
            else:
                data = await resp.json()
    except (aiohttp.ClientConnectorError,
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectionError,
            asyncio.TimeoutError) as e:
        # Transport-level: nobody listening at the URL or connection torn
        # down before we got bytes back. Flag the endpoint unhealthy so
        # the queue gate stops accepting new jobs until the probe loop
        # confirms recovery, and fail fast instead of going through the
        # 130 s retry-backoff ladder for an error that won't fix itself
        # on retry.
        _mark_llm_unhealthy(f"{type(e).__name__}: {e}")
        raise LLMOfflineError(f"LLM unreachable at {LLM_API}: {e}") from e
    if status != 200:
        body = body_for_classify or ""
        # 4xx OR known-permanent error signature → no retry. Some
        # OpenAI-compatible servers return 500 with `exceed_context_size_error`
        # in the body — match the body too.
        if (400 <= status < 500) or _is_permanent_remote_error(body):
            raise PermanentError(f"LLM rejected request ({status}): {body[:300]}")
        raise RuntimeError(f"LLM failed ({status}): {body[:200]}")
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


def _translate_to_key(translate: object) -> str:
    """Map a `translate` payload value to a stable cache-key string.

    Different translate settings produce different transcripts (English vs
    source-language vs auto-decided) for the same video. The cache must
    key on this axis or we get the bug where /summarize translate:translate
    returns the previously-cached /summarize translate:auto result.
    """
    if translate is True:
        return "translate"
    if translate is False:
        return "native"
    return "auto"


def _cache_path(video_id: str, translate: object = "auto") -> Path:
    """Cache file path, keyed by (video_id, translate). Backward-compat:
    when translate="auto" we ALSO recognise the legacy `{video_id}.txt`
    (no translate suffix) in read_cache, since that's what older runs wrote.
    """
    key = _translate_to_key(translate)
    return CACHE_DIR / f"{video_id}.{key}.txt"


def write_cache(video_id: str, title: str, status: str, transcript: str,
                duration: int, translate: object = "auto") -> None:
    """Persist transcript to disk for reuse across retries / future runs."""
    try:
        _cache_path(video_id, translate).write_text(
            f"# title: {title}\n"
            f"# status: {status}\n"
            f"# duration: {duration}\n"
            f"# translate: {_translate_to_key(translate)}\n"
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


def read_cache(video_id: str, translate: object = "auto") -> tuple[str, str, str, int] | None:
    """Read cached transcript if present and not expired.

    Returns (title, status, transcript, duration) or None.

    Falls back to the legacy `{video_id}.txt` filename when the new
    translate-keyed path is missing AND the request is for translate="auto"
    — covers pre-translate-aware cache files from before the multilingual
    Tier 1 change. Old files expire naturally via CACHE_TTL.
    """
    path = _cache_path(video_id, translate)
    if not path.exists():
        # Back-compat: old filename without translate suffix. Only relevant
        # for translate="auto" since that was the only behaviour pre-Tier-1.
        legacy = CACHE_DIR / f"{video_id}.txt"
        if translate == "auto" and legacy.exists():
            path = legacy
        else:
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
        elif line.startswith("# translate: "):
            # Recognise so the parser keeps consuming header lines. The
            # value itself is informational only (the filename carries
            # the authoritative key).
            pass
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

    # Defensive: invalidate cached web-scrape entries whose body never met
    # the meaningful-content floor. Before the MIN_SCRAPED_BODY_CHARS guard
    # was added to _fetch_via_crawl4ai, anti-bot stubs (Reuters → DataDome
    # → "reuters.com\n", 11 chars) were happily persisted to cache and
    # served on every retry — pinning the user to a useless "no content to
    # summarise" embed forever. The status field discriminates: web jobs
    # always write "scraped N chars", video jobs write durations / formats.
    # Limited strictly to that prefix so a legitimately short transcript
    # (e.g. a 3-second clip) is never invalidated.
    if (status.startswith("scraped ")
            and MIN_SCRAPED_BODY_CHARS
            and len(transcript.strip()) < MIN_SCRAPED_BODY_CHARS):
        log.warning("[%s] Cached scrape body below floor (%d chars, "
                    "floor %d) — invalidating so retry re-scrapes",
                    video_id, len(transcript.strip()), MIN_SCRAPED_BODY_CHARS)
        try:
            path.unlink()
        except OSError:
            pass
        return None

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
    timestamps are dropped (rare; the formatter upstream always emits
    timestamps, but Whisper occasionally produces an empty-text segment
    that comes out without one in translate mode).
    """
    # Pre-existing bug fix: the previous form `(s, l) for x in lines for
    # (s, l) in [x] if x is not None` tried to unpack None BEFORE the
    # filter ran. Filter the None-returning _parse_ts results FIRST.
    speech_pairs = [
        parsed for parsed in (_parse_ts(l) for l in speech_text.splitlines() if l.strip())
        if parsed is not None
    ]
    visual_pairs = [
        parsed for parsed in (_parse_ts(l) for l in visual_text.splitlines() if l.strip())
        if parsed is not None
    ]
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

    def __init__(self, video_id: str, channel_id: int, speakers: list[str],
                 translate: object = "auto"):
        super().__init__(timeout=600)
        self._video_id = video_id
        self._channel_id = channel_id
        self._translate = translate
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
            await _apply_rename(interaction, self._video_id, self._channel_id,
                                renames, translate=self._translate)
        except Exception as e:
            log.error("Rename apply failed: %s", e)
            await interaction.followup.send(
                f"Rename failed: {type(e).__name__}: {e}", ephemeral=True
            )


class SpeakerRenameView(discord.ui.View):
    """Persistent View attached to a brief embed when diarize=True.

    Carries the translate variant so we look up the correct cache file —
    each translate setting (auto/translate/native) has its own file, and
    the rename only makes sense on the variant the user originally saw.
    """

    def __init__(self, job_video_id: str, channel_id: int,
                 translate: object = "auto"):
        super().__init__(timeout=None)  # persistent until restart
        self._video_id = job_video_id
        self._channel_id = channel_id
        self._translate = translate

    @discord.ui.button(label="Rename speakers", style=discord.ButtonStyle.secondary, emoji="🏷️")
    async def rename_button(
        self, interaction: discord.Interaction, button: discord.ui.Button,
    ) -> None:
        cached = read_cache(self._video_id, self._translate)
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
            SpeakerRenameModal(self._video_id, self._channel_id, speakers,
                               translate=self._translate)
        )


async def _apply_rename(
    interaction: discord.Interaction,
    video_id: str,
    channel_id: int,
    renames: dict,
    *,
    translate: object = "auto",
) -> None:
    """Apply rename map to cached transcript + post a fresh brief embed.

    Strategy: text-replace `[SPEAKER_xx]` → `[NewName]` in the cached
    transcript, write back to cache, then re-summarize the brief only
    (key_points/chapters keep their original labels — they'd require
    re-summarisation on the LLM, costly).

    `translate` selects which cache variant to rename — each translate
    setting (auto/translate/native) has its own file. Defaults to auto
    for back-compat with older view instances that didn't pass it.
    """
    cached = read_cache(video_id, translate)
    if cached is None:
        await interaction.followup.send("Transcript expired.", ephemeral=True)
        return
    title, status, transcript, duration = cached

    # Apply renames atomically (longest first to avoid prefix collisions).
    for old, new in sorted(renames.items(), key=lambda kv: -len(kv[0])):
        transcript = transcript.replace(f"[{old}]", f"[{new}]")
    write_cache(video_id, title, status, transcript, duration, translate)

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
# iteration during development; leave unset for global commands (~1h
# propagation, but works in DMs and across all servers).
#
# Accepts a comma-separated list so operators running the bot in
# multiple servers (e.g. testing + community) can get instant sync in
# all of them without waiting on global propagation. Single-ID values
# from older configs keep working unchanged.
DISCORD_GUILD_IDS: tuple[int, ...] = tuple(
    int(s.strip())
    for s in (os.environ.get("DISCORD_GUILD_ID", "") or "").split(",")
    if s.strip()
)


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
    translate: object = "auto",
    refresh: bool = False,
    vlm_enabled: bool | None = None,
    yt_comments_enabled: bool | None = None,
    model_override: str | None = None,
    kind: str = "video",
) -> Job | None:
    """Build a Job from an interaction. Returns None on URL parse failure.

    Slash commands are always explicit_request=True — the user typed the
    command and the URL.

    `translate`: "auto" (default) lets the server LID-detect and translate
    non-English; True forces translate; False forces transcribe-as-source.
    `refresh`: when True, both bot-side and server-side caches are skipped
    for this run. Result still overwrites the cache on success.
    `kind`: "video" (default), "web", or "litmus" — picks the worker
    handler. For "web" / "litmus" the URL doesn't need to be a video and
    the video_id is derived from a URL hash instead of YT/platform IDs.
    """
    eff_vlm = VLM_ENABLED if vlm_enabled is None else vlm_enabled
    eff_comments = YT_COMMENTS_ENABLED if yt_comments_enabled is None else yt_comments_enabled

    # Web + litmus paths accept any http(s) URL; video_id is a URL hash so
    # cache keys don't collide with video-platform IDs.
    if kind in ("web", "litmus"):
        any_url = _ANY_URL_RE.search(url)
        if not any_url:
            return None
        canonical = any_url.group(0).rstrip("),.;]")
        return Job(
            url=canonical, video_id=_hash_url(canonical),
            channel=interaction.channel, submitter_id=interaction.user.id,
            submitter_name=str(interaction.user),
            user_prompt=user_prompt, diarize=diarize, translate=translate,
            refresh=refresh,
            vlm_enabled=eff_vlm,
            yt_comments_enabled=eff_comments,
            model_override=model_override,
            kind=kind,
            explicit_request=True,
            interaction=interaction,
        )

    # Try YouTube first for canonical video_id
    m = YT_PATTERN.search(url)
    if m:
        video_id = m.group(1)
        canonical = f"https://www.youtube.com/watch?v={video_id}"
        return Job(
            url=canonical, video_id=video_id,
            channel=interaction.channel, submitter_id=interaction.user.id,
            submitter_name=str(interaction.user),
            user_prompt=user_prompt, diarize=diarize, translate=translate,
            refresh=refresh,
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
        user_prompt=user_prompt, diarize=diarize, translate=translate,
        refresh=refresh,
        vlm_enabled=eff_vlm,
        yt_comments_enabled=eff_comments,
        model_override=model_override,
        explicit_request=True,
        interaction=interaction,
    )


def _resolve_translate_choice(choice: str) -> object:
    """Map the slash-command Literal value to a `translate` payload value.

    Slash UI shows three options; payload accepts "auto" | True | False.
    """
    if choice == "translate":
        return True
    if choice == "native":
        return False
    return "auto"


# ─── Model autocomplete ──────────────────────────────────────────────────────
# Cache /v1/models so autocomplete doesn't slam the LLM proxy. Refresh on
# expiry (rare — operators don't swap models frequently). Failing fetches
# don't block — autocomplete just returns no suggestions, model: stays
# free-text. The LLM proxy validates the model name at request time anyway.

_MODEL_CACHE_TTL = 300  # seconds
_model_cache: dict[str, object] = {"models": [], "ts": 0.0}


async def _fetch_model_list() -> list[str]:
    """Live model list from the LLM proxy. Memoised for _MODEL_CACHE_TTL."""
    now = time.time()
    cached_ts = _model_cache.get("ts", 0.0)
    if isinstance(cached_ts, float) and now - cached_ts < _MODEL_CACHE_TTL:
        models = _model_cache.get("models")
        if isinstance(models, list):
            return models
    if http is None:
        return []
    try:
        async with http.get(
            f"{LLM_API}/models",
            timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
            if resp.status != 200:
                return _model_cache.get("models", []) or []  # serve stale
            data = await resp.json()
        # OpenAI-compatible: {"data": [{"id": "..."}, ...]}
        models = [
            str(m.get("id"))
            for m in data.get("data", [])
            if isinstance(m, dict) and m.get("id")
        ]
        _model_cache["models"] = models
        _model_cache["ts"] = now
        return models
    except Exception as e:
        log.debug("model list fetch failed: %s", e)
        return _model_cache.get("models", []) or []


async def _model_autocomplete(
    interaction: discord.Interaction, current: str,
) -> list[app_commands.Choice[str]]:
    """Autocomplete model: from the LLM proxy's /v1/models endpoint."""
    models = await _fetch_model_list()
    current_lc = (current or "").lower()
    out: list[app_commands.Choice[str]] = []
    for m in models:
        if current_lc and current_lc not in m.lower():
            continue
        out.append(app_commands.Choice(name=m[:100], value=m[:100]))
        if len(out) >= 25:
            break
    return out


@bot.tree.command(name="summarize", description="Transcribe + summarise a video")
@app_commands.describe(
    url="Video URL (YouTube, Twitch, Vimeo, etc.)",
    prompt="Optional steering: tell the bot what to focus on (forces VLM)",
    model="Override LLM_MODEL for this run (advanced)",
    translate=("Translation policy. 'auto' (default) translates non-English "
               "sources to English for cleaner summaries. 'translate' forces "
               "English output. 'native' preserves the source language."),
    refresh=("Skip cache and re-transcribe from scratch. Use this when a "
             "previous run was wrong or you changed translate mode."),
)
@app_commands.autocomplete(model=_model_autocomplete)
async def cmd_summarize(
    interaction: discord.Interaction,
    url: str,
    prompt: str | None = None,
    model: str | None = None,
    translate: Literal["auto", "translate", "native"] = "auto",
    refresh: bool = False,
) -> None:
    await interaction.response.defer()

    # Parse + build the Job BEFORE the rate-limit check so that an
    # LLM-offline rejection can attach a Retry button populated from
    # the parsed-and-validated job spec.
    cfg = _effective_config(interaction.user.id, interaction.channel.id)
    effective_model = model or cfg.get("model")
    job = _job_from_interaction(
        interaction, url,
        user_prompt=(prompt or "").strip()[:USER_PROMPT_MAX_CHARS],
        translate=_resolve_translate_choice(translate),
        refresh=refresh,
        vlm_enabled=cfg.get("vlm_enabled", VLM_ENABLED),
        yt_comments_enabled=cfg.get("yt_comments_enabled", YT_COMMENTS_ENABLED),
        model_override=effective_model,
    )
    if job is None:
        await interaction.followup.send(
            f"❌ Couldn't parse a supported video URL from: {url}", ephemeral=True
        )
        return

    ok, reason = _rate_limit_check(interaction.user.id)
    if not ok:
        retry_view = _maybe_retry_view([_job_to_retry_spec(job)],
                                       target_user_id=interaction.user.id)
        await interaction.followup.send(
            f"❌ {reason}", ephemeral=True, view=retry_view,
        )
        return

    _rate_limit_record(interaction.user.id)
    await queue.put(job)
    await _ack_queued(job, queue.qsize())
    log.info("Slash /summarize queued %s from %s (model=%s, prompt=%s, translate=%s, refresh=%s)",
             job.video_id, job.submitter_name,
             effective_model or "default", bool(job.user_prompt), translate, refresh)


@bot.tree.command(name="transcribe", description="Transcribe a video (no summary), with optional speaker diarization")
@app_commands.describe(
    url="Video URL",
    diarize="Identify and label different speakers (slower; adds rename button)",
    prompt="Optional steering: tell the bot what to focus on (forces VLM)",
    translate=("Translation policy. 'auto' (default) translates non-English "
               "sources to English. 'native' preserves the source language."),
    refresh="Skip cache and re-transcribe from scratch.",
)
async def cmd_transcribe(
    interaction: discord.Interaction,
    url: str,
    diarize: bool = False,
    prompt: str | None = None,
    translate: Literal["auto", "translate", "native"] = "auto",
    refresh: bool = False,
) -> None:
    # /transcribe is just /summarize with diarize on. We still produce
    # the brief/key_points/chapters embeds — the diarize flag flows
    # through to whisper for labelling. `prompt` matches /summarize
    # symmetry: forces VLM enrichment and steers the summary LLM.
    await interaction.response.defer()

    cfg = _effective_config(interaction.user.id, interaction.channel.id)
    job = _job_from_interaction(
        interaction, url,
        diarize=diarize or cfg.get("diarize", False),
        user_prompt=(prompt or "").strip()[:USER_PROMPT_MAX_CHARS],
        translate=_resolve_translate_choice(translate),
        refresh=refresh,
        vlm_enabled=cfg.get("vlm_enabled", VLM_ENABLED),
        yt_comments_enabled=cfg.get("yt_comments_enabled", YT_COMMENTS_ENABLED),
        model_override=cfg.get("model"),
    )
    if job is None:
        await interaction.followup.send(
            f"❌ Couldn't parse a supported video URL from: {url}", ephemeral=True
        )
        return

    ok, reason = _rate_limit_check(interaction.user.id)
    if not ok:
        retry_view = _maybe_retry_view([_job_to_retry_spec(job)],
                                       target_user_id=interaction.user.id)
        await interaction.followup.send(
            f"❌ {reason}", ephemeral=True, view=retry_view,
        )
        return

    _rate_limit_record(interaction.user.id)
    await queue.put(job)
    await _ack_queued(job, queue.qsize())
    log.info("Slash /transcribe queued %s from %s (diarize=%s, prompt=%s, translate=%s, refresh=%s)",
             job.video_id, job.submitter_name, job.diarize,
             bool(job.user_prompt), translate, refresh)


@bot.tree.command(name="web", description="Summarise a web article (non-video URL)")
@app_commands.describe(
    url="Article URL",
    prompt="Optional steering: focus the summary on a specific angle",
    model="Override LLM_MODEL for this run (advanced)",
    refresh="Skip cache and re-scrape from scratch.",
)
@app_commands.autocomplete(model=_model_autocomplete)
async def cmd_web(
    interaction: discord.Interaction,
    url: str,
    prompt: str | None = None,
    model: str | None = None,
    refresh: bool = False,
) -> None:
    """Slash equivalent of the `tldr` reply trigger — scrapes any web URL
    via Crawl4AI/FlareSolverr/Reddit-fetch and summarises. If the URL
    happens to be a clear video URL we still route through the web pipeline
    (use /summarize for video). Cache hits return instantly."""
    await interaction.response.defer()

    cfg = _effective_config(interaction.user.id, interaction.channel.id)
    effective_model = model or cfg.get("model")
    job = _job_from_interaction(
        interaction, url,
        user_prompt=(prompt or "").strip()[:USER_PROMPT_MAX_CHARS],
        model_override=effective_model,
        kind="web",
    )
    if job is None:
        await interaction.followup.send(
            f"❌ Couldn't parse a URL from: {url}", ephemeral=True
        )
        return

    ok, reason = _rate_limit_check(interaction.user.id)
    if not ok:
        retry_view = _maybe_retry_view([_job_to_retry_spec(job)],
                                       target_user_id=interaction.user.id)
        await interaction.followup.send(
            f"❌ {reason}", ephemeral=True, view=retry_view,
        )
        return

    _rate_limit_record(interaction.user.id)
    await queue.put(job)
    await _ack_queued(job, queue.qsize())
    log.info("Slash /web queued %s from %s (model=%s, prompt=%s, refresh=%s)",
             job.video_id, job.submitter_name,
             effective_model or "default", bool(job.user_prompt), refresh)


@bot.tree.command(name="litmus", description="AI litmus test — forensic signals on a web article")
@app_commands.describe(
    url="Article URL",
    model="Override LLM_MODEL for the qualitative read (advanced)",
)
@app_commands.autocomplete(model=_model_autocomplete)
async def cmd_litmus(
    interaction: discord.Interaction,
    url: str,
    model: str | None = None,
) -> None:
    """Slash equivalent of the `litmus` reply trigger. Surfaces stylistic +
    metadata signals that an article may be LLM-generated. Forensic output,
    no verdict. Always runs the web-fetch path even on video URLs (the
    bot inspects the page, not the audio)."""
    await interaction.response.defer()

    cfg = _effective_config(interaction.user.id, interaction.channel.id)
    job = _job_from_interaction(
        interaction, url,
        model_override=model or cfg.get("model"),
        kind="litmus",
    )
    if job is None:
        await interaction.followup.send(
            f"❌ Couldn't parse a URL from: {url}", ephemeral=True
        )
        return

    ok, reason = _rate_limit_check(interaction.user.id)
    if not ok:
        retry_view = _maybe_retry_view([_job_to_retry_spec(job)],
                                       target_user_id=interaction.user.id)
        await interaction.followup.send(
            f"❌ {reason}", ephemeral=True, view=retry_view,
        )
        return

    _rate_limit_record(interaction.user.id)
    await queue.put(job)
    await _ack_queued(job, queue.qsize())
    log.info("Slash /litmus queued %s from %s (model=%s)",
             job.video_id, job.submitter_name, model or "default")


@bot.tree.command(name="status", description="Show queue + service health")
@app_commands.describe(verbose="Also list the bot's queued + running jobs (titles, phases)")
async def cmd_status(interaction: discord.Interaction, verbose: bool = False) -> None:
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

    if verbose and _inflight:
        # Sorted: running first, then queued in submission order.
        entries = sorted(
            _inflight.values(),
            key=lambda e: (e.started_at is None, e.queued_at),
        )
        lines = ["", f"**Active + queued ({len(entries)}):**"]
        for i, e in enumerate(entries[:15], 1):
            label = e.title or e.bot_video_id
            elapsed = _format_relative(time.time() - (e.started_at or e.queued_at))
            marker = "▶" if e.started_at else "⏳"
            lines.append(
                f"`{i}.` {marker} **{truncate(label, 60)}** "
                f"({e.kind}, {e.phase}, {elapsed})"
            )
        if len(entries) > 15:
            lines.append(f"... and {len(entries) - 15} more")
        msg += "\n" + "\n".join(lines)

    await interaction.followup.send(msg, ephemeral=True)


@bot.tree.command(name="progress", description="Show your in-flight jobs with phase + ETA")
async def cmd_progress(interaction: discord.Interaction) -> None:
    """User-scoped view of every job they have queued or running. Shows
    title once available, current phase, elapsed time, and an ETA when
    one is computable (post-download for video; rolling average for
    queued jobs)."""
    await interaction.response.defer(ephemeral=True)
    entries = _inflight_user(interaction.user.id)
    if not entries:
        await interaction.followup.send(
            "You have no jobs queued or running. Use `/summarize`, "
            "`/transcribe`, `/web`, or `/litmus` to start one.",
            ephemeral=True,
        )
        return

    avg_queued_eta = _avg_runtime("video") or _avg_runtime("web") or 0.0
    now = time.time()
    lines = [f"**Your jobs ({len(entries)}):**"]
    for i, e in enumerate(entries, 1):
        label = e.title or e.bot_video_id
        if e.started_at is not None:
            elapsed = _format_relative(now - e.started_at)
            # ETA: phase-dependent
            eta_str = ""
            if e.phase == PHASE_TRANSCRIBING and e.duration:
                # WhisperX ~10x realtime on consumer GPU; quote conservatively
                remaining = max(0.0, e.duration / 8.0 - (now - e.started_at))
                if remaining > 1.0:
                    eta_str = f", ETA ~{_format_relative(remaining)}"
            elif e.phase in (PHASE_SUMMARIZING,):
                # Summarisation is bounded by LLM_TIMEOUT; rolling avg from
                # observed runtimes gives a better estimate than guessing.
                avg = _avg_runtime(e.kind) or 0.0
                if avg > 0:
                    remaining = max(0.0, avg - (now - e.started_at))
                    if remaining > 1.0:
                        eta_str = f", ETA ~{_format_relative(remaining)}"
            phase_emoji = {
                PHASE_DOWNLOADING: "⬇️",
                PHASE_TRANSCRIBING: "🎧",
                PHASE_SCRAPING: "📰",
                PHASE_SUMMARIZING: "🧠",
            }.get(e.phase, "▶")
            lines.append(
                f"`{i}.` {phase_emoji} **{truncate(label, 70)}**\n"
                f"     {e.phase} • elapsed {elapsed}{eta_str} • `{e.bot_video_id}`"
            )
        else:
            # Queued. Position is its index among queued entries in _inflight.
            queued_ahead = sum(
                1 for x in _inflight.values()
                if x.started_at is None and x.queued_at < e.queued_at
            )
            position = queued_ahead + 1
            wait = _format_relative(now - e.queued_at)
            eta_str = ""
            if avg_queued_eta > 0:
                eta_str = f", ETA ~{_format_relative(position * avg_queued_eta)}"
            lines.append(
                f"`{i}.` ⏳ **{truncate(label, 70)}**\n"
                f"     queued (position {position}) • waiting {wait}{eta_str} • "
                f"`{e.bot_video_id}`"
            )
    await interaction.followup.send("\n".join(lines), ephemeral=True)


async def _cancel_autocomplete(
    interaction: discord.Interaction, current: str,
) -> list[app_commands.Choice[str]]:
    """Autocomplete /cancel job: from the user's own inflight entries."""
    entries = _inflight_user(interaction.user.id)
    current_lc = (current or "").lower()
    choices: list[app_commands.Choice[str]] = []
    for e in entries[:25]:
        label_src = e.title or e.bot_video_id
        if current_lc and current_lc not in label_src.lower() and current_lc not in e.bot_video_id.lower():
            continue
        marker = "running" if e.started_at else "queued"
        label = f"[{marker}] {truncate(label_src, 80)}"
        choices.append(app_commands.Choice(name=label[:100], value=e.bot_video_id))
    return choices


@bot.tree.command(name="cancel", description="Cancel one of your queued or running jobs")
@app_commands.describe(job="Pick from your active jobs (autocomplete)")
@app_commands.autocomplete(job=_cancel_autocomplete)
async def cmd_cancel(interaction: discord.Interaction, job: str) -> None:
    """Soft-cancel a bot-side queued job (worker checks the flag before
    invoking the handler). For jobs that have already entered server-side
    transcription, forwards DELETE /api/jobs/{server_job_id} — the whisper
    service refuses to cancel in-flight transcription (no safe interrupt
    point in whisperX), so we surface the right error message in that case.
    """
    await interaction.response.defer(ephemeral=True)
    entry = _inflight.get(job)
    if entry is None:
        await interaction.followup.send(
            f"❌ Job `{job}` not found in the queue (might have just finished).",
            ephemeral=True,
        )
        return
    if entry.submitter_id != interaction.user.id:
        await interaction.followup.send(
            "❌ You can only cancel your own jobs.", ephemeral=True
        )
        return

    # Case 1: still queued on the bot side — set the soft flag.
    if entry.phase == PHASE_QUEUED:
        entry.cancel_requested = True
        await interaction.followup.send(
            f"⏹️ Will cancel `{job}` before it runs.", ephemeral=True
        )
        log.info("[%s] /cancel requested by %s (still queued bot-side)",
                 job, interaction.user)
        return

    # Case 2: server-side transcription has started; forward to whisper.
    if entry.server_job_id and entry.phase == PHASE_TRANSCRIBING:
        if http is None:
            await interaction.followup.send(
                "❌ HTTP session not ready; try again in a moment.",
                ephemeral=True,
            )
            return
        try:
            async with http.delete(
                f"{WHISPER_API}/api/jobs/{entry.server_job_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.json()
                if resp.status == 200:
                    entry.cancel_requested = True
                    await interaction.followup.send(
                        f"⏹️ Cancelled `{job}` (server-side queue).",
                        ephemeral=True,
                    )
                    log.info("[%s] /cancel forwarded DELETE for server job %s",
                             job, entry.server_job_id)
                    return
                msg = body.get("error", f"http {resp.status}")
                if "in-flight" in msg or resp.status == 409:
                    await interaction.followup.send(
                        f"❌ Too late — `{job}` is already transcribing and "
                        "whisperX has no safe interruption point. Wait for "
                        "it to finish or fail.",
                        ephemeral=True,
                    )
                    return
                await interaction.followup.send(
                    f"❌ Couldn't cancel: {msg}", ephemeral=True,
                )
        except Exception as e:
            await interaction.followup.send(
                f"❌ Cancel request failed: {e}", ephemeral=True,
            )
        return

    # Case 3: phase is downloading / scraping / summarising on the bot side.
    # No safe interruption point — the underlying HTTP request is in-flight.
    await interaction.followup.send(
        f"❌ Too late — `{job}` is in the `{entry.phase}` phase. "
        "Wait for it to finish or fail.",
        ephemeral=True,
    )


@bot.tree.command(name="queue", description="List everything in the bot queue (server-wide)")
async def cmd_queue(interaction: discord.Interaction) -> None:
    """Server-wide queue listing. Same fields as `/status verbose:true` but
    formatted as a standalone command for users who think 'queue', not
    'status verbose'."""
    await interaction.response.defer(ephemeral=True)
    if not _inflight:
        await interaction.followup.send(
            f"Queue empty. (`{queue.qsize()}/{MAX_QUEUE_SIZE}`)",
            ephemeral=True,
        )
        return
    entries = sorted(
        _inflight.values(),
        key=lambda e: (e.started_at is None, e.queued_at),
    )
    now = time.time()
    lines = [f"**Bot queue ({len(entries)} job{'s' if len(entries) != 1 else ''}):**"]
    for i, e in enumerate(entries[:20], 1):
        label = e.title or e.bot_video_id
        elapsed = _format_relative(now - (e.started_at or e.queued_at))
        marker = "▶" if e.started_at else "⏳"
        owner = e.submitter_name.split("#")[0] if e.submitter_name else "?"
        lines.append(
            f"`{i}.` {marker} **{truncate(label, 60)}** "
            f"(`{e.kind}`, {e.phase}, {elapsed}, @{owner})"
        )
    if len(entries) > 20:
        lines.append(f"... and {len(entries) - 20} more")
    await interaction.followup.send("\n".join(lines), ephemeral=True)


_HELP_TOPICS: dict[str, str] = {
    "overview": (
        "**TL;DW Bot — quick overview**\n\n"
        "Paste a video URL → bot transcribes + summarises automatically.\n"
        "Reply `tldr` to a message with a URL → summarises any article.\n"
        "Reply `tldr` to a message with image attachments → OCR + visual "
        "description + summary.\n"
        "Reply `litmus` → forensic AI-writing test on an article.\n\n"
        "**Slash commands:**\n"
        "• `/summarize url:` — video → 3 embeds (brief / key points / chapters)\n"
        "• `/transcribe url: diarize:true` — adds speaker labels + Rename button\n"
        "• `/web url:` — article summary (slash equivalent of `tldr` reply)\n"
        "• `/litmus url:` — AI litmus test (slash equivalent of `litmus` reply)\n"
        "• `/progress` — your active jobs with phase + ETA\n"
        "• `/cancel job:` — cancel a queued/transcribing job of yours\n"
        "• `/queue` — full bot queue (server-wide)\n"
        "• `/status` — queue depth + whisper/vision health\n"
        "• `/find query:` — search past summaries\n"
        "• `/help topic:` — more help on triggers, admin, limits, errors\n"
    ),
    "triggers": (
        "**Reply triggers (no URL needed in your reply)**\n\n"
        "Reply with one of these words to a message containing a URL "
        "or image attachments:\n"
        "• `tldr` / `summarize` / `summarise` — summary flow (video, web, "
        "or image OCR auto-detect)\n"
        "• `litmus` — AI litmus test on URLs only (forensic signals, no verdict)\n\n"
        "**Image summaries:** Reply `tldr` to a message with one or more "
        "image attachments → the bot extracts on-screen text via OCR, "
        "describes the scene with the vision model, and posts a single "
        "TL;DR embed plus a verbatim \"Text in image\" embed when there's "
        "enough OCR content to be worth showing separately.\n\n"
        "**Chained replies:** `tldr litmus` or `litmus tldr` (order preserved) fires both. "
        "Reply body must be only keywords + punctuation — sentences like "
        "\"give me a tldr\" intentionally don't trigger.\n\n"
        "**Auto-trigger on paste:** any URL whose shape clearly points at a "
        "video (YouTube watch/shorts, Twitch VODs, Vimeo, TikTok, etc.) "
        "starts a job. Text-post URLs on video domains don't auto-trigger; "
        "use `/web` or reply `tldr` for those."
    ),
    "admin": (
        "**Admin commands** (require permission)\n\n"
        "**Per-channel** (`Manage Channel`):\n"
        "• `/config model:` — override LLM for /summarize in this channel\n"
        "• `/config diarize:true` — enable speaker diarization by default\n"
        "• `/config vlm:false` — disable vision-language frame analysis\n"
        "• `/config yt_comments:false` — skip the Community Reaction embed\n"
        "• `/config show:true` (or no args) — print current config\n"
        "• `/config model:` (empty value) — clear that override\n\n"
        "**Per-server** (`Manage Server`):\n"
        "• `/serverconfig summary_channel:#name` — route Key Points + Chapters to a separate channel\n"
        "• `/serverconfig clear:true` — wipe all server overrides\n"
        "• `/serverconfig show:true` (or no args) — print current server config\n\n"
        "Configs persist across restarts (stored under bot's cache dir)."
    ),
    "limits": (
        "**Rate limits**\n\n"
        f"• `{MAX_JOBS_PER_USER_PER_HOUR}` jobs per user per hour (sliding window).\n"
        f"• `{MAX_QUEUE_SIZE}` total jobs in queue (server-wide cap).\n"
        "• Chained replies count per kind: `tldr litmus` = 2 slots, atomic.\n"
        "• Multiple URLs in one message = one slot each.\n\n"
        "Hitting the cap reacts 🚫 and replies with the retry timer. "
        "Trusted users can bypass via `RATE_LIMIT_BYPASS_USERS` env (queue cap "
        "still enforced — admins can't crash the bot either).\n\n"
        "Check your own usage with `/status`. Track live jobs with `/progress`."
    ),
    "errors": (
        "**Reactions during processing**\n\n"
        "• ⏳ queued (worker hasn't picked it up yet)\n"
        "• 🎧 downloading + transcribing video\n"
        "• 📰 scraping article / litmus fetch\n"
        "• 🧠 summarising (LLM call)\n"
        "• ✅ done — embed has been posted\n"
        "• 🚫 rate limit hit\n"
        "• ❌ permanent failure (private video, hard CAPTCHA, geo-blocked, etc.)\n\n"
        "For age-restricted videos, see `bot/.env.example` → `YT_DLP_COOKIES_FILE`. "
        "If you want richer signal than emojis, use `/progress`."
    ),
    "translate": (
        "**Translation policy** (`/summarize translate:`)\n\n"
        "• `auto` (default) — server runs a 30s LID pre-pass; non-English "
        "sources get translated to English. Best for downstream summarisation "
        "since the model is most fluent in English.\n"
        "• `translate` — force task=translate regardless of source.\n"
        "• `native` — preserve the source language (no translation).\n\n"
        "Each mode caches separately, so switching modes re-runs from scratch. "
        "Add `refresh:true` to bypass cache entirely."
    ),
}


@bot.tree.command(name="help", description="Explain what the bot does and how to use it")
@app_commands.describe(topic="Which area of help to show")
async def cmd_help(
    interaction: discord.Interaction,
    topic: Literal[
        "overview", "triggers", "admin", "limits", "errors", "translate",
    ] = "overview",
) -> None:
    """In-Discord help. Ephemeral so it doesn't clutter the channel."""
    body = _HELP_TOPICS.get(topic) or _HELP_TOPICS["overview"]
    await interaction.response.send_message(body, ephemeral=True)


def _parse_cache_header(text: str) -> dict[str, str]:
    """Extract the leading `# key: value` lines from a cache file. Stops at
    the first blank line. Used by /find, /recent, /redo to surface metadata
    without reading the full transcript."""
    hdr: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        if not line.startswith("# "):
            break
        kv = line[2:]
        if ":" not in kv:
            continue
        k, _, v = kv.partition(":")
        hdr[k.strip()] = v.strip()
    return hdr


# Heuristic: YT-style video_id is exactly 11 chars from the URL-safe base64
# alphabet (A-Za-z0-9_-). Anything else is treated as a hash (web/litmus
# don't cache litmus today, but the hashed video_id pattern matches /web).
_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _classify_cache_kind(stem: str) -> str:
    """Infer the kind for a cached file given the file stem (video_id).
    11-char YT-style ID → video; otherwise a URL hash → web. Litmus isn't
    cached today, so it never shows up here.
    """
    # Stem looks like `{video_id}.{translate_key}` — split off the translate
    # suffix to recover the video_id only.
    base = stem.rsplit(".", 1)[0]
    return "video" if _YT_ID_RE.match(base) else "web"


@bot.tree.command(name="find", description="Search past summaries by keyword")
@app_commands.describe(
    query="Keywords to search for (case-insensitive substring)",
    kind="Filter results by job kind",
    since_days="Only show results from the last N days",
    limit="Max results to return (1-25, default 10)",
)
async def cmd_find(
    interaction: discord.Interaction,
    query: str,
    kind: Literal["any", "video", "web"] = "any",
    since_days: int | None = None,
    limit: app_commands.Range[int, 1, 25] = 10,
) -> None:
    await interaction.response.defer(ephemeral=True)
    q = query.lower().strip()
    if len(q) < 3:
        await interaction.followup.send("Query must be ≥3 characters.", ephemeral=True)
        return
    cutoff = time.time() - (since_days * 86400) if since_days else 0.0

    matches: list[tuple[str, str, str, float, str]] = []
    for f in CACHE_DIR.glob("*.txt"):
        try:
            mtime = f.stat().st_mtime
            if cutoff and mtime < cutoff:
                continue
            file_kind = _classify_cache_kind(f.stem)
            if kind != "any" and kind != file_kind:
                continue
            text = f.read_text()
        except OSError:
            continue
        if q not in text.lower():
            continue
        hdr = _parse_cache_header(text)
        title = hdr.get("title", "") or f.stem
        # video_id is the part before the translate-key suffix.
        video_id = f.stem.rsplit(".", 1)[0]
        if file_kind == "video":
            url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            # We don't have the original URL stored for web hashes — point
            # at the cache file name as a placeholder. /redo can re-fetch.
            url = f"#{video_id}"
        matches.append((video_id, title, url, mtime, file_kind))

    # Sort newest first so /find behaves like a "what did we recently summarise about X" query.
    matches.sort(key=lambda m: m[3], reverse=True)
    matches = matches[:limit]

    if not matches:
        await interaction.followup.send(f"No matches for `{query}`.", ephemeral=True)
        return

    lines = [
        f"Found {len(matches)} match{'es' if len(matches) != 1 else ''} for "
        f"`{query}` (kind=`{kind}`{f', last {since_days}d' if since_days else ''}):"
    ]
    now = time.time()
    for vid, title, url, mtime, file_kind in matches:
        age = _format_relative(now - mtime)
        if url.startswith("#"):
            lines.append(f"- **{truncate(title, 80)}** (`{vid}` · {file_kind} · {age} ago)")
        else:
            lines.append(
                f"- [{truncate(title, 80)}]({url}) "
                f"(`{vid}` · {file_kind} · {age} ago)"
            )
    await interaction.followup.send("\n".join(lines), ephemeral=True)


@bot.tree.command(name="recent", description="Show the most recently cached summaries")
@app_commands.describe(
    kind="Filter by job kind",
    limit="How many recent jobs to list (1-25, default 10)",
)
async def cmd_recent(
    interaction: discord.Interaction,
    kind: Literal["any", "video", "web"] = "any",
    limit: app_commands.Range[int, 1, 25] = 10,
) -> None:
    """List cached transcripts/summaries by modification time. Server-wide
    view of recent activity — useful as a 'what did we just summarise?'
    glance. Use /find to keyword-search the same cache."""
    await interaction.response.defer(ephemeral=True)
    items: list[tuple[float, str, str, str]] = []
    for f in CACHE_DIR.glob("*.txt"):
        try:
            mtime = f.stat().st_mtime
            file_kind = _classify_cache_kind(f.stem)
            if kind != "any" and kind != file_kind:
                continue
            # Read only the header (cheap — small prefix). full read is fine
            # too; cache files are bounded and the bot doesn't ship a huge
            # archive. Header lines stop at the first blank line.
            with f.open("r") as fh:
                head = ""
                for _ in range(8):
                    line = fh.readline()
                    if not line.strip():
                        break
                    head += line
            hdr = _parse_cache_header(head)
            title = hdr.get("title", "") or f.stem
            items.append((mtime, f.stem.rsplit(".", 1)[0], title, file_kind))
        except OSError:
            continue
    items.sort(key=lambda x: x[0], reverse=True)
    items = items[:limit]
    if not items:
        await interaction.followup.send(
            "No cached summaries yet.", ephemeral=True,
        )
        return
    now = time.time()
    lines = [f"**Recent summaries ({len(items)}):**"]
    for mtime, vid, title, file_kind in items:
        age = _format_relative(now - mtime)
        if file_kind == "video":
            url = f"https://www.youtube.com/watch?v={vid}"
            lines.append(
                f"- [{truncate(title, 80)}]({url}) "
                f"(`{vid}` · {file_kind} · {age} ago)"
            )
        else:
            lines.append(
                f"- **{truncate(title, 80)}** "
                f"(`{vid}` · {file_kind} · {age} ago)"
            )
    await interaction.followup.send("\n".join(lines), ephemeral=True)


async def _recent_video_autocomplete(
    interaction: discord.Interaction, current: str,
) -> list[app_commands.Choice[str]]:
    """Autocomplete /redo video_id: from the cache. Newest-first, capped at 25.
    Free-text typing is still allowed — autocomplete is just a discovery aid.
    """
    current_lc = (current or "").lower()
    items: list[tuple[float, str, str]] = []
    for f in CACHE_DIR.glob("*.txt"):
        try:
            stem = f.stem.rsplit(".", 1)[0]
            if not _YT_ID_RE.match(stem):
                continue
            mtime = f.stat().st_mtime
            with f.open("r") as fh:
                head = "".join(fh.readline() for _ in range(6))
            title = _parse_cache_header(head).get("title", stem)
            if current_lc and current_lc not in title.lower() and current_lc not in stem.lower():
                continue
            items.append((mtime, stem, title))
        except OSError:
            continue
    items.sort(key=lambda x: x[0], reverse=True)
    return [
        app_commands.Choice(name=truncate(f"{title} ({vid})", 100), value=vid)
        for _, vid, title in items[:25]
    ]


@bot.tree.command(name="redo", description="Re-run a cached video summary with different options")
@app_commands.describe(
    video_id="Pick from your cached videos (autocomplete)",
    prompt="Optional steering text (forces VLM)",
    model="Override LLM_MODEL for this run",
    translate="Translation policy",
    refresh="Skip cache and re-transcribe (default true for /redo)",
)
@app_commands.autocomplete(
    video_id=_recent_video_autocomplete, model=_model_autocomplete,
)
async def cmd_redo(
    interaction: discord.Interaction,
    video_id: str,
    prompt: str | None = None,
    model: str | None = None,
    translate: Literal["auto", "translate", "native"] = "auto",
    refresh: bool = True,
) -> None:
    """Re-run a cached video job with different options without retyping
    the URL. Defaults `refresh=true` because the common case is "the model
    got something wrong, retry"; pass refresh:false to re-summarise from
    the existing transcript (saves the download+transcribe phases)."""
    await interaction.response.defer()
    if not _YT_ID_RE.match(video_id):
        await interaction.followup.send(
            f"❌ `{video_id}` isn't a YouTube-style id; use `/summarize url:` "
            "with the full URL for web jobs.",
            ephemeral=True,
        )
        return
    canonical = f"https://www.youtube.com/watch?v={video_id}"
    cfg = _effective_config(interaction.user.id, interaction.channel.id)
    effective_model = model or cfg.get("model")
    job = _job_from_interaction(
        interaction, canonical,
        user_prompt=(prompt or "").strip()[:USER_PROMPT_MAX_CHARS],
        translate=_resolve_translate_choice(translate),
        refresh=refresh,
        vlm_enabled=cfg.get("vlm_enabled", VLM_ENABLED),
        yt_comments_enabled=cfg.get("yt_comments_enabled", YT_COMMENTS_ENABLED),
        model_override=effective_model,
    )
    if job is None:
        await interaction.followup.send(
            f"❌ Couldn't build a job for `{video_id}`.", ephemeral=True,
        )
        return

    ok, reason = _rate_limit_check(interaction.user.id)
    if not ok:
        retry_view = _maybe_retry_view([_job_to_retry_spec(job)],
                                       target_user_id=interaction.user.id)
        await interaction.followup.send(
            f"❌ {reason}", ephemeral=True, view=retry_view,
        )
        return

    _rate_limit_record(interaction.user.id)
    await queue.put(job)
    await _ack_queued(job, queue.qsize())
    log.info("Slash /redo queued %s from %s (model=%s, prompt=%s, translate=%s, refresh=%s)",
             job.video_id, job.submitter_name,
             effective_model or "default", bool(job.user_prompt), translate, refresh)


@bot.tree.command(name="config", description="Configure this channel's bot defaults (admin)")
@app_commands.describe(
    model="Default LLM model id for this channel (empty = clear override)",
    vlm="Force VLM enrichment for every video in this channel",
    diarize="Enable speaker diarization by default in this channel",
    yt_comments="Fetch + summarise YouTube comments (extra Community Reaction embed)",
    show="Just print the current config without changing it",
)
@app_commands.autocomplete(model=_model_autocomplete)
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


@bot.tree.command(name="myconfig", description="Set your own defaults — applies in any channel where you use the bot")
@app_commands.describe(
    model="Your default LLM model (empty = clear override)",
    diarize="Default to speaker diarization on /transcribe",
    clear="Wipe all your overrides",
    show="Just print current overrides",
)
@app_commands.autocomplete(model=_model_autocomplete)
async def cmd_myconfig(
    interaction: discord.Interaction,
    model: str | None = None,
    diarize: bool | None = None,
    clear: bool = False,
    show: bool = False,
) -> None:
    """Per-user defaults. Apply in any channel where the bot picks up a
    URL or a slash command from this user. Channel config (`/config`) wins
    on conflict — that's intentional, a channel admin's policy outranks a
    personal preference.
    """
    if clear:
        # Wipe by setting all known fields to None.
        set_user_config(interaction.user.id, model=None, diarize=None)
        await interaction.response.send_message(
            "Your overrides cleared — using channel/global defaults.",
            ephemeral=True,
        )
        return

    if show or (model is None and diarize is None):
        cfg = get_user_config(interaction.user.id)
        if not cfg:
            txt = ("No personal overrides — using channel/global defaults.\n"
                   "Set with `/myconfig model:<name>` or `/myconfig diarize:true`.")
        else:
            txt = "Your overrides:\n" + "\n".join(
                f"  **{k}**: `{v}`" for k, v in cfg.items()
            )
        await interaction.response.send_message(txt, ephemeral=True)
        return

    fields: dict = {}
    if model is not None:
        fields["model"] = model.strip() or None
    if diarize is not None:
        fields["diarize"] = diarize
    new_cfg = set_user_config(interaction.user.id, **fields)
    if new_cfg:
        txt = "Your overrides updated:\n" + "\n".join(
            f"  **{k}**: `{v}`" for k, v in new_cfg.items()
        )
    else:
        txt = "Overrides cleared — using channel/global defaults."
    await interaction.response.send_message(txt, ephemeral=True)
    log.info("User %s myconfig: %s", interaction.user, fields)


# Sync commands on startup. Called once after the worker is up.
async def _sync_slash_commands():
    """Sync slash commands either globally (~1h propagation) or per-guild
    (instant). DISCORD_GUILD_ID env var accepts a comma-separated list so
    operators with multiple servers get instant sync in all of them.
    """
    if DISCORD_GUILD_IDS:
        # Per-guild try/except: a single guild the bot can't sync to (e.g.
        # invited without the applications.commands scope → 403 Missing Access)
        # must NOT abort syncing for the other guilds in the list.
        ok = 0
        for gid in DISCORD_GUILD_IDS:
            guild = discord.Object(id=gid)
            try:
                bot.tree.copy_global_to(guild=guild)
                synced = await bot.tree.sync(guild=guild)
            except discord.Forbidden as e:
                log.error(
                    "Slash sync skipped for guild %d: %s. Re-invite the bot to "
                    "that guild with the applications.commands scope, or drop it "
                    "from DISCORD_GUILD_ID.", gid, e,
                )
                continue
            except Exception as e:
                log.error("Slash sync failed for guild %d: %s", gid, e)
                continue
            ok += 1
            log.info("Slash commands synced to guild %d (%d commands)",
                     gid, len(synced))
        log.info("Slash sync complete: %d/%d guild(s) succeeded",
                 ok, len(DISCORD_GUILD_IDS))
    else:
        synced = await bot.tree.sync()
        log.info("Slash commands synced globally (%d commands; ~1 hour propagation)",
                 len(synced))


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN, log_handler=None)
