"""Media discovery + yt-dlp helpers (auth args, error classification)."""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("whisper-ui")


# ─── /media filesystem scan ───────────────────────────────────────────────────

MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "/media")
MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma",
    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".ts", ".flv",
}

_MEDIA_SCAN_TTL = int(os.environ.get("MEDIA_SCAN_TTL", "60"))
_media_scan_cache: dict = {"at": 0.0, "result": []}


def scan_media_files(force: bool = False) -> list[tuple[str, str]]:
    """Walk MEDIA_ROOT and return (display, full_path) tuples newest-first.

    Cached for MEDIA_SCAN_TTL seconds (default 60). Pass force=True to bypass.
    """
    now = time.time()
    if not force and (now - _media_scan_cache["at"]) < _MEDIA_SCAN_TTL:
        return _media_scan_cache["result"]
    if not os.path.isdir(MEDIA_ROOT):
        _media_scan_cache.update(at=now, result=[])
        return []
    found: list[str] = []
    for root, _dirs, files in os.walk(MEDIA_ROOT):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in MEDIA_EXTENSIONS:
                found.append(os.path.join(root, fname))
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    result = [(os.path.basename(p), p) for p in found]
    _media_scan_cache.update(at=now, result=result)
    return result


# ─── yt-dlp helpers ───────────────────────────────────────────────────────────

# Stderr patterns that will not recover on retry. The API layer maps these to
# HTTP 422 so clients (the bot) skip retries.
_PERMANENT_YT_DLP_PATTERNS = (
    "Sign in to confirm your age",            # age-gated, needs cookies
    "Private video",
    "Video unavailable",
    "This video is unavailable",
    "members-only content",
    "members only video",
    "This video has been removed",
    "blocked it on copyright grounds",
    "blocked it in your country",
    "country and is unavailable",
    "Premieres in",                            # not yet released
    "This live event will begin",              # future stream
    "Sign in to confirm you're not a bot",    # IP/account-flagged
    "Join this channel to get access",         # paid membership
    "Video is not available",
)


def is_permanent_yt_dlp_error(stderr: str) -> bool:
    return any(p in stderr for p in _PERMANENT_YT_DLP_PATTERNS)


def yt_dlp_auth_args() -> list[str]:
    """Build cookie/auth args for yt-dlp from environment.

    YT_DLP_COOKIES_FILE          — path to Netscape cookies.txt readable by
                                   the container (mount it in via compose).
    YT_DLP_COOKIES_FROM_BROWSER  — passed straight to --cookies-from-browser
                                   (only useful if the browser is installed
                                   in the container; default image does not
                                   include one).
    """
    args: list[str] = []
    cookies_file = os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
    if cookies_file:
        if os.path.isfile(cookies_file):
            args.extend(["--cookies", cookies_file])
        else:
            log.warning(
                f"YT_DLP_COOKIES_FILE set to {cookies_file!r} but file not found "
                f"— age-restricted videos will fail"
            )
    cookies_browser = os.environ.get("YT_DLP_COOKIES_FROM_BROWSER", "").strip()
    if cookies_browser:
        args.extend(["--cookies-from-browser", cookies_browser])
    return args
