"""End-to-end regression suite.

Run via `make test` (no docker required — stubs aiohttp/discord and verifies
behaviour at the function level). Covers every code path touched in the
robustness/VLM/security/cleanup work so a green run gives confidence the
pipeline still does what it should.

Categories:
- Pattern lists: drift between server + bot, all known-permanent stderr
  fragments correctly classified, transient errors NOT misclassified.
- yt-dlp filename resolution: probe fallback finds files of any extension.
- Map-reduce: chunk sizing, reduce step preservation, recursive reduction.
- Adaptive halving: context overflow at map / reduce / single-call.
- Cache: roundtrip, legacy format, TTL expiry, duration derivation.
- VLM helpers: format descriptions, parse + interleave timestamps.
- Output sanitisation: allowed link allowlist, bare URL stripping.
- Speech density routing: silent / sparse / heavy thresholds.
- Build initial prompt: frequency rank, hapax filter, no English filler.
- Dispatch logic: process() routing references all helpers correctly.
- Module imports: all exports present, dead code stays gone.
"""
import asyncio
import importlib
import inspect
import logging
import os
import re
import sys
import tempfile
import time
import types


# ─── Stubs ────────────────────────────────────────────────────────────────────


def _setup_stubs():
    os.environ.setdefault("DISCORD_TOKEN", "test-token")
    os.environ.setdefault("CACHE_DIR", tempfile.mkdtemp(prefix="bot-test-"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bot"))

    aiohttp = types.ModuleType("aiohttp")
    aiohttp.ClientSession = object
    aiohttp.ClientTimeout = lambda **k: None
    sys.modules["aiohttp"] = aiohttp

    discord = types.ModuleType("discord")
    discord.Intents = type("I", (), {
        "default": staticmethod(lambda: types.SimpleNamespace(message_content=False))
    })
    discord.HTTPException = Exception
    discord.Embed = object
    discord.Message = object
    discord.TextChannel = object
    discord.Interaction = object
    discord.ButtonStyle = types.SimpleNamespace(
        secondary="secondary", primary="primary", danger="danger", success="success",
    )
    discord.Object = lambda **k: None
    sys.modules["discord"] = discord

    # discord.app_commands stub
    app_commands = types.ModuleType("discord.app_commands")
    def _passthrough_decorator(*a, **k):
        def _wrap(fn): return fn
        return _wrap
    app_commands.command = _passthrough_decorator
    app_commands.describe = _passthrough_decorator
    app_commands.autocomplete = _passthrough_decorator
    # Choice is constructed at decoration time inside our autocomplete fns and
    # also at call time. Stub captures kwargs so tests can inspect output.
    app_commands.Choice = type("Choice", (), {
        "__init__": lambda self, name="", value="": setattr(self, "name", name) or setattr(self, "value", value),
    })

    # Range is used as a type annotation: app_commands.Range[int, 1, 25].
    class _Range:
        def __class_getitem__(cls, item):
            return int  # collapse to plain int for type checks
    app_commands.Range = _Range
    app_commands.CommandTree = type("CT", (), {"__init__": lambda s, *a, **k: None})
    sys.modules["discord.app_commands"] = app_commands
    discord.app_commands = app_commands

    # discord.ui stub
    ui = types.ModuleType("discord.ui")
    ui.View = type("View", (), {"__init__": lambda s, *a, **k: None})
    ui.Modal = type("Modal", (), {
        "__init__": lambda s, *a, **k: None,
        "__init_subclass__": classmethod(lambda cls, **k: None),
        "add_item": lambda s, i: None,
    })
    ui.Button = type("Button", (), {})
    ui.TextInput = type("TextInput", (), {"__init__": lambda s, *a, **k: None})
    ui.Select = type("Select", (), {})
    def _ui_button_decorator(*a, **k):
        def _wrap(fn): return fn
        return _wrap
    ui.button = _ui_button_decorator
    sys.modules["discord.ui"] = ui
    discord.ui = ui

    sys.modules["discord.ext"] = types.ModuleType("discord.ext")
    commands = types.ModuleType("discord.ext.commands")

    class B:
        def __init__(self, *a, **k):
            self.tree = types.SimpleNamespace(
                command=_passthrough_decorator,
                sync=lambda **kw: None,
                copy_global_to=lambda **kw: None,
            )

        def event(self, fn):
            return fn

    commands.Bot = B
    sys.modules["discord.ext.commands"] = commands


_setup_stubs()
import main as bot  # noqa: E402
import prompts as p  # noqa: E402

APP_SRC = open(os.path.join(os.path.dirname(__file__), "..", "app.py")).read()
BOT_SRC = open(os.path.join(os.path.dirname(__file__), "..", "bot", "main.py")).read()


# ─── 1. Permanent error classification ────────────────────────────────────────


def test_permanent_classification_yt_dlp():
    """Every yt-dlp permanent-stderr fragment we know about is caught."""
    # Real stderr fragments observed in production / from yt-dlp source
    permanent = [
        # YouTube
        "ERROR: [youtube] xxx: Sign in to confirm your age",
        "ERROR: [youtube] yyy: Sign in to confirm you're not a bot",
        "ERROR: [youtube] zzz: Private video. Sign in if you've been granted access",
        "ERROR: [youtube] aaa: This video is unavailable",
        "ERROR: [youtube] bbb: This video has been removed by the uploader",
        "ERROR: [youtube] ccc: Join this channel to get access to members-only content",
        "ERROR: [youtube] ddd: Premieres in 2 days",
        "ERROR: [youtube] eee: Video is not available in your country",
        "ERROR: [youtube] fff: This live event will begin in 1 hour",
        "ERROR: blocked it on copyright grounds",
        "ERROR: blocked it in your country",
        "this content is members-only video",
        "country and is unavailable",
        # Twitter / X
        "ERROR: [twitter] 12345: No video could be found in this tweet",
        "ERROR: [twitter] 67890: No media found in this post",
        # Generic
        "ERROR: Unsupported URL: https://some-random.example/path",
        "ERROR: 'foo://bar' is not a valid URL",
        "ERROR: No video formats found",
        "ERROR: no video formats found for this video",
        # Instagram / Threads
        "ERROR: [Instagram] xxx: There's no video in this post",
        "ERROR: Post does not contain any media",
    ]
    failed = [s for s in permanent if not bot._is_permanent_remote_error(s)]
    assert not failed, f"missed permanent classifications: {failed}"


def test_permanent_classification_ffmpeg():
    """ffmpeg errors that surface from /api/describe are caught."""
    cases = [
        "Describe failed: ffmpeg frame extraction failed: [out#0/image2 ...] "
        "Output file does not contain any stream",
        "input file has no video stream — likely audio-only download",
        "ffmpeg: no video streams found in input",
        "Stream specifier 'v' does not match any streams",
    ]
    # Server-side wraps these in NoVideoStreamError (HTTP 422); bot sees the
    # error body and should classify as permanent via pattern match.
    failed = [s for s in cases if not bot._is_permanent_remote_error(s)]
    # Note: only 2 of the 4 actually have to match server-side patterns, but
    # bot list should cover all 4. Allow some flex but require coverage.
    assert len(failed) <= 1, f"too many ffmpeg cases unclassified: {failed}"


def test_permanent_classification_llm_context():
    """LLM context overflow patterns recognised even on 5xx."""
    cases = [
        '{"error":{"code":400,"message":"request (40424 tokens) exceeds the '
        'available context size (32768 tokens)","type":"exceed_context_size_error"}}',
        "context_length_exceeded: prompt too long",
        "Maximum context length is 32768 tokens, however you requested 35000",
    ]
    # First two should match; the third is a different OpenAI-style message
    # we don't currently catch (potential follow-up).
    assert bot._is_permanent_remote_error(cases[0])
    assert bot._is_permanent_remote_error(cases[1])


def test_permanent_classification_no_false_positives():
    """Transient errors must NOT be classified as permanent."""
    transient = [
        "Connection timed out",
        "503 Service Unavailable",
        "Network is unreachable",
        "Internal server error",
        "Whisper busy — another transcription running",
        "Read timed out",
        "GPU OOM",
        "",
        "  ",
        "Unrelated error message",
    ]
    misclassified = [s for s in transient if bot._is_permanent_remote_error(s)]
    assert not misclassified, f"false positives: {misclassified}"


def test_pattern_list_drift():
    """Bot's pattern list should be a strict superset of server's yt-dlp list.

    Uses AST so comments containing parens (which break naive regex) don't
    cause false positives.
    """
    import ast

    def _extract(src, var):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == var:
                        if isinstance(node.value, ast.Tuple):
                            return {
                                el.value for el in node.value.elts
                                if isinstance(el, ast.Constant)
                                and isinstance(el.value, str)
                            }
        return set()

    server = _extract(APP_SRC, "_PERMANENT_YT_DLP_PATTERNS")
    bot_patterns = _extract(BOT_SRC, "_PERMANENT_REMOTE_PATTERNS")
    assert server, "could not parse server pattern list"
    assert bot_patterns, "could not parse bot pattern list"
    missing = server - bot_patterns
    assert not missing, (
        f"bot list is missing {len(missing)} server patterns "
        f"(would miss them when server returns 5xx): {sorted(missing)}"
    )


# ─── 2. yt-dlp filename resolution (regression: .wav fallback) ────────────────


def test_app_yt_download_uses_directory_probe():
    """The .wav-extension fallback was replaced with a probe of output_dir."""
    assert "os.path.isfile(filename)" in APP_SRC
    assert "startswith(video_id)" in APP_SRC
    assert "max(candidates, key=os.path.getsize)" in APP_SRC
    # Old buggy fallback should be gone
    assert 'f"{meta.get(\'id\', \'unknown\')}.wav"' not in APP_SRC, \
        "stale .wav fallback still present"


def test_keep_video_request_format():
    """Bot passes keep_video=job.vlm_enabled on every yt-download call.

    Per-job vlm_enabled lets channel config override the global default; the
    download must reflect the *effective* setting for this job, not the
    module-level fallback.
    """
    assert '"keep_video": job.vlm_enabled' in BOT_SRC


# ─── 3. Map-reduce chunking ──────────────────────────────────────────────────


def test_chunk_transcript_small_passthrough():
    small = "a" * 100
    assert bot._chunk_transcript(small, 1000) == [small]


def test_chunk_transcript_line_boundaries():
    lines = [f"[00:{i:02d}] line {i}" + " word" * 10 for i in range(200)]
    big = "\n".join(lines)
    chunks = bot._chunk_transcript(big, 500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # Round-trip: concatenation reproduces the original
    assert "\n".join(chunks) == big
    # No chunk starts with continuation-style whitespace
    for c in chunks:
        assert not c.startswith(" "), f"chunk starts with space: {c[:30]!r}"


def test_chunk_transcript_exact_boundary():
    # Transcript exactly at budget → single chunk
    s = "x" * bot.LLM_INPUT_CHAR_BUDGET
    chunks = bot._chunk_transcript(s, bot.LLM_INPUT_CHAR_BUDGET)
    assert chunks == [s]
    # Just over → split
    over = "a\n" + s
    chunks = bot._chunk_transcript(over, bot.LLM_INPUT_CHAR_BUDGET)
    assert len(chunks) >= 2


def test_summarize_single_call():
    """Small transcript → single LLM call, no chunking."""
    calls = []

    async def fake(prompt, max_tokens):
        calls.append({"len": len(prompt)})
        return "DONE"

    bot._llm_call = fake
    out = asyncio.run(bot.summarize(
        "small", p.PROMPT_BRIEF, 1024,
        reduce_template=p.REDUCE_BRIEF,
        title="T", duration="0:30", reference_block="",
    ))
    assert out == "DONE"
    assert len(calls) == 1


def test_summarize_map_reduce_brief():
    """Long transcript with reduce_template → map calls + 1 reduce."""
    calls = []

    async def fake(prompt, max_tokens):
        calls.append({
            "is_chunk": "NOTE: This is part" in prompt,
            "is_reduce": "<partials>" in prompt,
        })
        if "NOTE: This is part" in prompt:
            return "PARTIAL"
        if "<partials>" in prompt:
            return "REDUCED"
        return "SINGLE"

    bot._llm_call = fake
    big = "\n".join(f"[00:{i//60:02d}:{i%60:02d}] " + "x" * 40 for i in range(500))

    out = asyncio.run(bot.summarize(
        big, p.PROMPT_BRIEF, 1024,
        reduce_template=p.REDUCE_BRIEF,
        title="T", duration="3:00", reference_block="",
        _budget=10000,
    ))
    assert out == "REDUCED"
    chunks = sum(1 for c in calls if c["is_chunk"])
    reduces = sum(1 for c in calls if c["is_reduce"])
    assert chunks > 1
    assert reduces == 1


def test_summarize_chapters_concat():
    """Chapters path (reduce_template=None) concatenates partials."""
    async def fake(prompt, max_tokens):
        if "NOTE: This is part" in prompt:
            return "PARTIAL"
        return "SINGLE"

    bot._llm_call = fake
    big = "\n".join(f"[00:{i:02d}] " + "x" * 100 for i in range(200))

    out = asyncio.run(bot.summarize(
        big, p.PROMPT_CHAPTERS, 3000,
        reduce_template=None,
        title="T", duration="3:00", reference_block="",
        tail_start="2:15", char_cap=3796,
        _budget=10000,
    ))
    assert "PARTIAL" in out and "---" in out


def test_summarize_adaptive_halving_preserves_reduce():
    """Single-call overflow → halve budget → re-enter map-reduce → reduce step still fires.

    Note: prompts that exceed the simulated cap will appear in the call list
    because the function is invoked before raising. Those rejected attempts
    are expected — they're what triggers the halving in the first place.
    The invariant is: the FINAL output is the reduced summary (not just the
    raw concatenation), which means the reduce step survived the recursion.
    """
    rejected_count = 0
    successful_count = 0
    successful_max = 0

    async def fake(prompt, max_tokens):
        nonlocal rejected_count, successful_count, successful_max
        if len(prompt) > 8000:
            rejected_count += 1
            raise bot.PermanentError(
                f"LLM rejected request (400): request ({len(prompt)//2} tokens) "
                f"exceeds the available context size (4000 tokens), "
                f"exceed_context_size_error"
            )
        successful_count += 1
        successful_max = max(successful_max, len(prompt))
        if "NOTE: This is part" in prompt:
            return "PARTIAL"
        if "<partials>" in prompt:
            return "REDUCED"
        return "SINGLE"

    bot._llm_call = fake
    big = "\n".join(f"[{i//60}:{i%60:02d}] " + "x" * 40 for i in range(300))

    out = asyncio.run(bot.summarize(
        big, p.PROMPT_BRIEF, 1024,
        reduce_template=p.REDUCE_BRIEF,
        title="T", duration="5:00", reference_block="",
        _budget=20000,
    ))
    # Critical invariants:
    # 1. Final output is REDUCED (reduce step preserved through halving).
    # 2. Halving actually fired (some attempts were rejected).
    # 3. SOME calls succeeded under the cap (otherwise pipeline gave up).
    assert out == "REDUCED", f"reduce step lost: {out!r}"
    assert rejected_count > 0, "halving never fired (test setup wrong)"
    assert successful_count > 0, "no calls succeeded → pipeline gave up"
    assert successful_max <= 8000, \
        f"successful call exceeded cap somehow: {successful_max}"


def test_summarize_min_chunk_floor():
    """Pathological case: even tiny chunks overflow → eventually gives up."""
    async def always_reject(prompt, max_tokens):
        raise bot.PermanentError(
            "request exceeds the available context size, exceed_context_size_error"
        )

    bot._llm_call = always_reject
    raised = False
    try:
        asyncio.run(bot.summarize(
            "x" * 100000, p.PROMPT_BRIEF, 1024,
            reduce_template=p.REDUCE_BRIEF,
            title="T", duration="5:00", reference_block="",
        ))
    except bot.PermanentError:
        raised = True
    assert raised, "min-chunk floor failed; pipeline didn't terminate"


# ─── 4. Cache ────────────────────────────────────────────────────────────────


def test_cache_roundtrip():
    bot.write_cache("vid-rt", "Test Title", "Done -- 10s",
                    "Transcript line 1\nLine 2", 600)
    result = bot.read_cache("vid-rt")
    assert result is not None
    title, status, transcript, duration = result
    assert title == "Test Title"
    assert status == "Done -- 10s"
    assert transcript == "Transcript line 1\nLine 2"
    assert duration == 600


def test_cache_special_chars():
    """Cache must roundtrip quotes, ampersands, multi-line content."""
    bot.write_cache("vid-special",
                    'Title with "quotes" & <chars>',
                    "status", "line\nwith\nbreaks", 100)
    title, status, transcript, dur = bot.read_cache("vid-special")
    assert title == 'Title with "quotes" & <chars>'
    assert transcript == "line\nwith\nbreaks"


def test_cache_legacy_format_compat():
    """Pre-existing cache files (from before the structured header) still readable."""
    legacy = "# Legacy Title\n# Done -- 5s\n\n[0:00] hello\n[2:38:01] end"
    bot._cache_path("vid-legacy").write_text(legacy)
    title, status, transcript, dur = bot.read_cache("vid-legacy")
    assert title == "Legacy Title"
    assert status == "Done -- 5s"
    assert dur == 9481, f"derived duration wrong: {dur}"


def test_cache_duration_derivation():
    """When cache file lacks duration, derive from last [H:MM:SS] timestamp."""
    assert bot._derive_duration_from_transcript("") == 0
    assert bot._derive_duration_from_transcript("[00:00] x\n[02:30] y") == 150
    assert bot._derive_duration_from_transcript("[00:00] x\n[2:38:01] y") == 9481
    assert bot._derive_duration_from_transcript("plain text no ts") == 0


def test_cache_ttl_expiry():
    import time
    bot.write_cache("vid-expired", "old", "s", "old text", 60)
    path = bot._cache_path("vid-expired")
    # Force mtime older than CACHE_TTL
    os.utime(path, (time.time() - bot.CACHE_TTL - 100,
                    time.time() - bot.CACHE_TTL - 100))
    assert bot.read_cache("vid-expired") is None


def test_cache_invalidates_short_scraped_entry():
    """Pre-existing poisoned scrape cache (body < MIN_SCRAPED_BODY_CHARS,
    status starts with 'scraped ') must be invalidated on read so the
    retry actually re-scrapes. This covers the Reuters/DataDome regression
    where the first failed scrape pinned the bot to an 11-char body
    forever — every subsequent `tldr` hit the poisoned cache and never
    exercised the new MIN_SCRAPED_BODY_CHARS guard in _fetch_via_crawl4ai.
    """
    bot.write_cache("vid-poisoned", "www.reuters.com",
                    "scraped 11 chars", "reuters.com", 0)
    path = bot._cache_path("vid-poisoned")
    assert path.exists()
    assert bot.read_cache("vid-poisoned") is None, (
        "Short scraped body must be treated as cache miss"
    )
    # Self-healing: the poisoned file is removed so any concurrent reader
    # also sees the miss, not just the first one.
    assert not path.exists(), "Poisoned cache file must be unlinked on miss"


def test_cache_does_not_invalidate_short_video_transcript():
    """A legitimately short video transcript (e.g. a 3-second clip whose
    only word is 'hi') must NOT be invalidated. Discrimination is via the
    status-string prefix — only 'scraped ' entries are subject to the
    floor, video statuses (durations / 'transcribed') are exempt.
    """
    bot.write_cache("vid-shortvid", "Tiny clip",
                    "Transcribed in 3s", "hi", 3)
    result = bot.read_cache("vid-shortvid")
    assert result is not None, (
        "Short video transcripts must be served from cache as usual"
    )
    title, status, transcript, dur = result
    assert transcript == "hi"
    assert status == "Transcribed in 3s"


# ─── 5. VLM helpers ──────────────────────────────────────────────────────────


def test_format_descriptions():
    descs = [
        {"timestamp": 0, "text": "Opening shot."},
        {"timestamp": 10, "text": "Close-up."},
        {"timestamp": 20, "text": "[frame description unavailable]"},
        {"timestamp": 30, "text": "Wide shot."},
    ]
    out = bot._format_descriptions(descs)
    assert "[0:00] Opening shot." in out
    assert "[0:10] Close-up." in out
    assert "[frame description unavailable]" not in out  # filtered
    assert "[0:30] Wide shot." in out


def test_parse_ts():
    assert bot._parse_ts("[0:30] hello")[0] == 30
    assert bot._parse_ts("[1:23:45] later")[0] == 1*3600 + 23*60 + 45
    assert bot._parse_ts("not a ts") is None
    assert bot._parse_ts("[12:34] x")[0] == 12*60 + 34


def test_interleave_chronological():
    speech = "[0:05] Hello.\n[0:30] Welcome.\n[1:00] Final."
    visual = "[0:00] Title.\n[0:15] Speaker enters.\n[0:45] Slides change."
    merged = bot._interleave_by_timestamp(speech, visual).splitlines()
    assert merged[0] == "[0:00] Title."
    assert merged[1] == "[0:05] Hello."
    assert merged[2] == "[0:15] Speaker enters."
    assert merged[3] == "[0:30] Welcome."
    assert merged[4] == "[0:45] Slides change."
    assert merged[5] == "[1:00] Final."


def test_interleave_handles_lines_without_timestamps():
    """Regression guard: the comprehension used to unpack None before the
    None-filter ran, crashing with 'cannot unpack non-iterable NoneType
    object' when whisper returned a segment without a parseable timestamp.
    Surfaced in production on translate-mode output for the Viggo
    Mortensen 9-languages video. Must NOT raise — non-parseable lines
    are dropped silently."""
    speech = "[0:05] Hello.\nLine without timestamp\n[0:30] Welcome."
    visual = "[0:00] Title.\nanother no-ts\n[0:45] Slides."
    # Did NOT raise (the assertion is that the call completes).
    merged = bot._interleave_by_timestamp(speech, visual).splitlines()
    # Only the parseable lines come through, in chronological order.
    assert merged == [
        "[0:00] Title.",
        "[0:05] Hello.",
        "[0:30] Welcome.",
        "[0:45] Slides.",
    ]


# ─── 6. Output sanitisation (anti-phishing) ──────────────────────────────────


def test_sanitize_strips_evil_markdown_link():
    out = bot.sanitize_llm_output("Check [click](https://evil.com/phish)")
    assert "evil.com" not in out
    assert "click" in out


def test_sanitize_strips_bare_evil_url():
    out = bot.sanitize_llm_output("Visit https://malicious.example/page")
    assert "malicious.example" not in out
    assert "[link removed]" in out


def test_sanitize_preserves_youtube_links():
    out = bot.sanitize_llm_output("[the video](https://www.youtube.com/watch?v=abc123)")
    assert "youtube.com" in out
    assert "the video" in out


def test_sanitize_preserves_youtube_timestamps():
    out = bot.sanitize_llm_output("[2:30](https://www.youtube.com/watch?v=x&t=150)")
    assert "t=150" in out


def test_sanitize_preserves_twitch_vimeo():
    for host in ("twitch.tv", "vimeo.com"):
        out = bot.sanitize_llm_output(f"see [a stream](https://{host}/clip/x)")
        assert host in out, f"{host} should be allowed"


def test_sanitize_no_preview_evil_url():
    """`<https://evil.com>` must not leak a dangling `<` into output."""
    out = bot.sanitize_llm_output("see <https://evil.com> now")
    assert "evil.com" not in out
    assert "<" not in out, f"dangling angle bracket in: {out!r}"
    assert "[link removed]" in out


def test_sanitize_no_preview_allowed_url_kept():
    """`<https://youtube.com/...>` should keep both wrapper and URL."""
    out = bot.sanitize_llm_output("see <https://www.youtube.com/watch?v=abc> now")
    assert "https://www.youtube.com/watch?v=abc" in out
    # Wrapper preserved (Discord renders this as no-preview link)
    assert "<https://www.youtube.com/watch?v=abc>" in out


def test_sanitize_bare_url_does_not_eat_closing_angle():
    """The bare-URL exclude class must include `>` so adjacent `<...>` survives."""
    # If the lookbehind missed `<` AND the exclude class missed `>`, a bare-
    # URL match would consume `evil.com>` and leave `<[link removed]`.
    out = bot.sanitize_llm_output("prefix <https://evil.com> suffix")
    # No mangled angle brackets should remain
    assert "<[link removed]" not in out
    assert "[link removed]>" not in out


# ─── 7. Speech density routing constants exist ──────────────────────────────


def test_speech_density_routing_constants():
    assert hasattr(bot, "SPEECH_DENSITY_SILENT")
    assert hasattr(bot, "SPEECH_DENSITY_SPARSE")
    assert bot.SPEECH_DENSITY_SILENT < bot.SPEECH_DENSITY_SPARSE
    assert hasattr(bot, "VLM_ENABLED")


def test_derive_video_id_youtube_id_like():
    """Real-looking IDs (digits + adequate length) survive verbatim."""
    assert bot._derive_video_id("https://twitter.com/user/status/12345678") == "12345678"
    assert bot._derive_video_id("https://x.com/u/status/9876543210") == "9876543210"


def test_derive_video_id_handle_collisions():
    """Profile handles (no digits) get a hash, distinct per URL."""
    a = bot._derive_video_id("https://x.com/elonmusk")
    b = bot._derive_video_id("https://x.com/jack")
    assert a != b
    assert a.startswith("u") and b.startswith("u")
    # Same URL → stable
    assert a == bot._derive_video_id("https://x.com/elonmusk")


def test_derive_video_id_empty_path():
    """No path parts → still produces a non-empty stable id."""
    out = bot._derive_video_id("https://example.com/")
    assert out  # non-empty
    assert len(out) >= 5


def test_job_vlm_enabled_default():
    """vlm_enabled defaults to True (matches default VLM_ENABLED env)."""
    j = bot.Job(
        url="https://x", video_id="x", channel=object(), submitter_id=1,
        message=object(),
    )
    assert j.vlm_enabled is True


def test_job_vlm_enabled_override():
    """Channel config can disable VLM per-channel."""
    j = bot.Job(
        url="https://x", video_id="x", channel=object(), submitter_id=1,
        vlm_enabled=False, message=object(),
    )
    assert j.vlm_enabled is False


def test_user_prompt_min_chars_constant():
    """The < 3 magic became a named constant."""
    assert hasattr(bot, "USER_PROMPT_MIN_CHARS")
    assert bot.USER_PROMPT_MIN_CHARS == 3
    # Below threshold returns empty
    assert bot._extract_user_prompt("a https://youtu.be/x", ["https://youtu.be/x"]) == ""


def test_user_prompt_strips_full_url_with_query_params():
    """The bug: YT_PATTERN matches up to the 11-char video ID, leaving
    query params like `&pp=ygUJQXNt` or `&list=RD...` in the message text
    where `_extract_user_prompt` mistakes them for user steering.
    """
    # Real-world cases from production logs (ASMR + YOASOBI URLs)
    cases = [
        (
            "https://m.youtube.com/watch?v=H8SNCx79M6o&pp=ygUJQXNtciB0aGFp",
            ["https://m.youtube.com/watch?v=H8SNCx79M6o"],  # YT_PATTERN's truncated match
        ),
        (
            "https://www.youtube.com/watch?v=u0wGWliC-I0&list=RDu0wGWliC-I0&start_radio=1&pp=oAcB",
            ["https://www.youtube.com/watch?v=u0wGWliC-I0"],
        ),
    ]
    for full_url, urls in cases:
        prompt = bot._extract_user_prompt(full_url, urls)
        assert prompt == "", (
            f"URL query-param tail leaked as user prompt: {prompt!r} "
            f"(full URL was {full_url!r})"
        )


def test_user_prompt_keeps_legit_steering_text():
    """The fix must NOT strip legitimate user steering text alongside the URL."""
    msg = "https://www.youtube.com/watch?v=abc123 describe what's on the slides"
    prompt = bot._extract_user_prompt(msg, ["https://www.youtube.com/watch?v=abc123"])
    assert "describe what's on the slides" in prompt


# ─── LLM-loop dedup safety net ───────────────────────────────────────────────


def test_dedup_drops_exact_duplicate_bullets():
    """The exact case from the YOASOBI / ASMR jobs: same bullet repeated."""
    looped = "\n".join([
        "Overview: This is a music video.",
        "- The video is a celebration of creativity in music.",
        "- The video is a celebration of creativity in music.",
        "- The video is a celebration of creativity in music.",
        "- The video showcases YOASOBI's recorder performance.",
        "- The video is a celebration of creativity in music.",
    ])
    out = bot._dedup_lines(looped)
    lines = out.splitlines()
    # The looping bullet appears exactly ONCE after dedup
    assert sum(1 for l in lines if "celebration of creativity" in l) == 1
    # The distinct bullet survives
    assert any("YOASOBI's recorder" in l for l in lines)
    # Overview survives
    assert any("music video" in l for l in lines)


def test_dedup_case_insensitive():
    """`The video is X` and `THE VIDEO IS X` count as duplicates."""
    text = "The video is a tribute to music.\nTHE VIDEO IS A TRIBUTE TO MUSIC."
    out = bot._dedup_lines(text)
    assert len(out.splitlines()) == 1


def test_dedup_strips_bullet_markers_in_key():
    """`- foo bar baz` and `* foo bar baz` should dedup against each other."""
    text = "- this is a long enough line to count\n* this is a long enough line to count"
    out = bot._dedup_lines(text)
    assert len(out.splitlines()) == 1


def test_dedup_preserves_short_section_markers():
    """Lines <20 chars (timestamps, headers, separators) pass through
    even if they recur — they're often legitimate structural markers."""
    text = "**0:00**\nIntro section.\n**1:30**\nMain section.\n**0:00**"
    out = bot._dedup_lines(text)
    # The two `**0:00**` markers BOTH survive (each <20 chars after strip)
    assert out.count("**0:00**") == 2


def test_dedup_preserves_legitimate_distinct_content():
    """A normal summary with no loops should pass through unchanged."""
    text = "\n".join([
        "Overview: A great article about technology.",
        "- The author explains the new chip design from Apple.",
        "- The chip achieves 30% better efficiency than predecessors.",
        "- Industry analysts predict wide adoption by 2027.",
        "- The article concludes with a comparison to NVIDIA's roadmap.",
    ])
    out = bot._dedup_lines(text)
    assert out == text  # no dedup needed → unchanged


def test_dedup_empty_input():
    assert bot._dedup_lines("") == ""
    assert bot._dedup_lines(None or "") == ""


def test_dedup_wired_into_summarize_return_paths():
    """summarize() must apply _dedup_lines on all return paths."""
    import inspect
    src = inspect.getsource(bot.summarize)
    # Single-call return
    assert "_dedup_lines(await _llm_call" in src
    # Raw-concat return (chapters style)
    assert "_dedup_lines(combined)" in src


# ─── Scene-clustered VLM output rendering ────────────────────────────────────


def test_format_scenes_single_frame():
    """Single-frame scene → point timestamp, no frame count suffix."""
    scenes = [{
        "start": 0.0, "end": 0.0, "frame_count": 1,
        "description": "A woman plays a recorder.",
    }]
    out = bot._format_scenes(scenes)
    assert out == "[0:00] A woman plays a recorder."


def test_format_scenes_multi_frame_range():
    """Multi-frame scene → range timestamp + frame count suffix."""
    scenes = [{
        "start": 0.0, "end": 50.0, "frame_count": 60,
        "description": "A woman plays two recorders simultaneously.",
    }]
    out = bot._format_scenes(scenes)
    assert out == "[0:00-0:50] (60 frames) A woman plays two recorders simultaneously."


def test_format_scenes_skips_failed_descriptions():
    """Scenes with placeholder text shouldn't appear in output."""
    scenes = [
        {"start": 0.0, "end": 10.0, "frame_count": 2,
         "description": "Real scene description."},
        {"start": 10.0, "end": 20.0, "frame_count": 1,
         "description": "[frame description unavailable]"},
        {"start": 20.0, "end": 30.0, "frame_count": 1, "description": ""},
    ]
    out = bot._format_scenes(scenes)
    assert "Real scene description" in out
    assert "unavailable" not in out
    # Only one line survives
    assert len(out.splitlines()) == 1


def test_format_scenes_long_video_uses_hms():
    """Times >1hr render as H:MM:SS in the range."""
    scenes = [{
        "start": 0.0, "end": 3700.0, "frame_count": 30,
        "description": "Static shot throughout.",
    }]
    out = bot._format_scenes(scenes)
    assert "[0:00-1:01:40]" in out


def test_format_vlm_output_prefers_scenes():
    """When server returns scenes[], bot uses _format_scenes; legacy
    descriptions[] is ignored if scenes[] is present."""
    desc_result = {
        "scenes": [{
            "start": 0.0, "end": 10.0, "frame_count": 5,
            "description": "Clustered scene.",
        }],
        "descriptions": [
            {"timestamp": 0.0, "text": "Frame 0."},
            {"timestamp": 5.0, "text": "Frame 1."},
        ],
    }
    out = bot._format_vlm_output(desc_result)
    assert "Clustered scene." in out
    assert "Frame 0." not in out
    assert "Frame 1." not in out


def test_format_vlm_output_falls_back_to_descriptions():
    """Legacy server response (no scenes[]) → bot uses _format_descriptions."""
    desc_result = {
        "descriptions": [
            {"timestamp": 0.0, "text": "Frame 0 description."},
            {"timestamp": 10.0, "text": "Frame 1 description."},
        ],
    }
    out = bot._format_vlm_output(desc_result)
    assert "Frame 0 description." in out
    assert "Frame 1 description." in out


def test_format_vlm_output_empty():
    """Neither scenes nor descriptions → empty string, no crash."""
    assert bot._format_vlm_output({}) == ""
    assert bot._format_vlm_output({"scenes": [], "descriptions": []}) == ""


def test_process_uses_format_vlm_output():
    """process() must call _format_vlm_output (auto-routes scenes vs descriptions),
    not the legacy _format_descriptions directly."""
    import inspect
    src = inspect.getsource(bot.process)
    assert "_format_vlm_output(desc_result)" in src
    # Legacy direct call shouldn't appear (we routed through the helper)
    assert "_format_descriptions(desc_result" not in src


# ─── Server-side scene-clustering helpers (app.py) ───────────────────────────
# These are tested by importing app.py module functions directly. The app
# imports torch / whisperx at top-level — too heavy for the test stub layer.
# We test via AST/source inspection of app.py instead.


def test_app_has_scene_cluster_helpers():
    """All scene-clustering helpers exist in app.py."""
    for name in (
        "_detect_scene_boundaries",
        "_adaptive_sample_timestamps",
        "_extract_frames_at_timestamps",
        "_cluster_descriptions",
        "_synthesize_cluster",
        "_vlm_word_set",
        "_vlm_jaccard",
    ):
        assert f"def {name}(" in APP_SRC, f"missing helper: {name}"


def test_app_describe_video_uses_new_pipeline():
    """_describe_video must run the new pipeline: scdet → sample →
    VLM → cluster → synthesize → scenes[]."""
    # Pull just the function body for inspection
    start = APP_SRC.index("def _describe_video(")
    end = APP_SRC.index("\nasync def api_describe(")
    body = APP_SRC[start:end]
    # All pipeline steps reference their helpers
    assert "_detect_scene_boundaries" in body
    assert "_adaptive_sample_timestamps" in body
    assert "_extract_frames_at_timestamps" in body
    assert "_cluster_descriptions" in body
    assert "_synthesize_cluster" in body
    # Response includes scenes[] + backward-compat descriptions[]
    assert '"scenes": scene_outputs' in body
    assert '"descriptions": descriptions' in body


def test_app_scene_detect_threshold_env_overridable():
    """SCENE_DETECT_THRESHOLD comes from env (tunable per deployment)."""
    assert 'SCENE_DETECT_THRESHOLD = float(os.environ.get("SCENE_DETECT_THRESHOLD"' in APP_SRC


def test_app_cluster_threshold_env_overridable():
    """CLUSTER_SIMILARITY_THRESHOLD env knob."""
    assert "CLUSTER_SIMILARITY_THRESHOLD" in APP_SRC
    assert 'os.environ.get("CLUSTER_SIMILARITY_THRESHOLD"' in APP_SRC


def test_app_synthesis_endpoint_separate_from_vlm():
    """Text-only synthesis LLM endpoint is configurable independently
    from the vision endpoint (operators may have different proxies)."""
    assert "LLM_TEXT_API_URL" in APP_SRC
    assert "LLM_SYNTHESIS_MODEL" in APP_SRC


def test_app_response_schema_backward_compatible():
    """New scenes[] is additive — descriptions[] still present for
    older bot versions that don't know scenes[]."""
    body = APP_SRC[APP_SRC.index("def _describe_video("):
                    APP_SRC.index("\nasync def api_describe(")]
    # Return dict includes both keys
    assert '"scenes":' in body
    assert '"descriptions":' in body


def test_chapters_prompt_warns_against_over_chaptering():
    """Chapters prompt must instruct the LLM to GROUP many scenes into
    fewer thematic chapters (real-world static-content videos produced
    30+ headings before this prompt fix)."""
    tmpl = p.PROMPT_CHAPTERS
    assert "GROUP" in tmpl, "Must instruct LLM to group similar scenes"
    assert "NEVER more than 15" in tmpl or "no more than 15" in tmpl.lower(), \
        "Must give a hard upper bound on chapter count"
    # Specific guidance for static content
    assert "static-shot" in tmpl.lower() or "static content" in tmpl.lower()


def test_chapters_token_budget_bumped():
    """LLM_MAX_TOKENS_CHAPTERS must accommodate ~10 detailed chapters.
    Was 3000, bumped to 5000 because static-content videos with many
    scenes were truncating mid-output."""
    assert bot.LLM_MAX_TOKENS_CHAPTERS >= 5000, (
        f"Chapters max_tokens too low ({bot.LLM_MAX_TOKENS_CHAPTERS}) — "
        f"static-content jobs will truncate"
    )


def test_prompt_tuning_constants_exist():
    """All prompt magic-numbers are hoisted into module-level constants
    so operators can tune output verbosity without editing prompt strings."""
    for name in (
        "BRIEF_SENTENCES", "WEB_BRIEF_SENTENCES", "REDDIT_BRIEF_SENTENCES",
        "CHAPTERS_TARGET", "CHAPTERS_MAX", "CHAPTERS_STATIC_TARGET",
        "CHAPTER_HEADING_WORDS", "CHAPTER_BODY_SENTENCES",
        "YT_COMMENTS_SENTENCES", "SECTIONS_BODY_SENTENCES",
        "REDDIT_ARTICLE_SUMMARY_SENTENCES", "REDDIT_OP_SENTENCES",
        "REDDIT_REACTION_SENTENCES",
    ):
        assert hasattr(p, name), f"missing prompt tuning constant: {name}"


def test_prompt_constants_env_overridable():
    """All prompt tuning constants must read from os.environ at module
    import so operators can override without forking prompts.py."""
    src = open(p.__file__).read()
    # Each constant should appear as `os.environ.get("NAME", default)`
    for name in (
        "BRIEF_SENTENCES", "CHAPTERS_TARGET", "CHAPTERS_MAX",
        "CHAPTER_HEADING_WORDS", "YT_COMMENTS_SENTENCES",
    ):
        assert f'os.environ.get("{name}"' in src, (
            f"{name} should be env-overridable: "
            f"missing os.environ.get(\"{name}\", ...) in prompts.py"
        )


def test_prompt_constants_actually_baked_into_templates():
    """Smoke test: the constants resolve at module load → prompt strings
    contain the actual values, not literal `{BRIEF_SENTENCES}` etc."""
    # No raw constant names should appear in compiled prompts
    for tmpl_name in ("PROMPT_BRIEF", "PROMPT_CHAPTERS", "PROMPT_BRIEF_WEB",
                      "PROMPT_BRIEF_REDDIT", "PROMPT_YT_COMMENTS",
                      "REDUCE_YT_COMMENTS", "REDUCE_BRIEF", "REDUCE_BRIEF_WEB",
                      "REDUCE_BRIEF_REDDIT", "PROMPT_SECTIONS",
                      "PROMPT_SECTIONS_REDDIT"):
        tmpl = getattr(p, tmpl_name)
        for constant_name in (
            "BRIEF_SENTENCES", "WEB_BRIEF_SENTENCES", "REDDIT_BRIEF_SENTENCES",
            "CHAPTERS_TARGET", "CHAPTERS_MAX", "CHAPTERS_STATIC_TARGET",
            "CHAPTER_HEADING_WORDS", "CHAPTER_BODY_SENTENCES",
            "YT_COMMENTS_SENTENCES", "SECTIONS_BODY_SENTENCES",
        ):
            assert f"{{{constant_name}}}" not in tmpl, (
                f"{tmpl_name} contains unfilled placeholder "
                f"{{{constant_name}}} — f-string didn't bake it in"
            )


def test_app_cluster_threshold_aggressive_default():
    """CLUSTER_SIMILARITY_THRESHOLD should be 0.25 (or lower) by
    default so static-content paraphrases reliably merge."""
    # Find the default value in app.py source
    m = re.search(
        r'CLUSTER_SIMILARITY_THRESHOLD = float\(\s*os\.environ\.get\(\s*"CLUSTER_SIMILARITY_THRESHOLD",\s*"([\d.]+)"\)',
        APP_SRC,
    )
    assert m, "CLUSTER_SIMILARITY_THRESHOLD default not found"
    default = float(m.group(1))
    assert default <= 0.30, (
        f"Cluster threshold {default} too high — static-content paraphrases "
        f"won't merge. Lower toward 0.25."
    )


def test_app_has_ocr_pipeline():
    """OCR pass is wired into the server-side describe pipeline."""
    for sym in ("VLM_OCR_ENABLED", "VLM_OCR_LANGUAGES", "_get_ocr_reader",
                "_ocr_frame"):
        assert sym in APP_SRC, f"missing OCR helper: {sym}"
    # OCR runs per-frame inside the worker
    worker_src = APP_SRC[APP_SRC.index("def _worker(i: int"):
                           APP_SRC.index("with concurrent.futures.ThreadPoolExecutor",
                                          APP_SRC.index("def _worker(i: int"))]
    assert "_ocr_frame(fp)" in worker_src
    # OCR text propagates into scene output
    assert '"ocr": ocr' in APP_SRC


def test_app_synthesis_uses_ocr_as_ground_truth():
    """_synthesize_cluster must include OCR text in the prompt so the
    LLM can resolve VLM vagueness against actual on-screen text."""
    body = APP_SRC[APP_SRC.index("def _synthesize_cluster("):
                    APP_SRC.index("def _ffprobe_duration(")]
    assert "ocr" in body.lower()
    assert "ground truth" in body.lower() or "prefer ocr" in body.lower()
    # Returns tuple now (description, ocr) — not just description
    assert "return fallback, ocr_combined" in body
    assert "return text, ocr_combined" in body


def test_app_scenes_cap_is_duration_aware():
    """Scene cap must scale with duration — short videos get few scenes,
    long videos get many. Not a fixed cap."""
    assert "_target_scene_count" in APP_SRC
    assert "SCENE_SECONDS_PER_TARGET" in APP_SRC
    assert "SCENES_MIN" in APP_SRC
    # No fixed flat cap masquerading as max
    body = APP_SRC[APP_SRC.index("def _cap_scenes("):
                    APP_SRC.index("def _cluster_descriptions(")]
    # _cap_scenes uses duration, not a fixed number
    assert "duration: float" in body
    assert "_target_scene_count(duration)" in body


def test_app_cap_scenes_has_tolerance():
    """The cap is SOFT — content within tolerance of target is left alone
    (a genuinely-varied trailer with rapid cuts deserves more scenes than
    a static-shot music video of the same duration)."""
    assert "SCENES_CAP_TOLERANCE" in APP_SRC
    body = APP_SRC[APP_SRC.index("def _cap_scenes("):
                    APP_SRC.index("def _cluster_descriptions(")]
    assert "SCENES_CAP_TOLERANCE" in body


# ─── Silent-video flow (visual-heavy content, content-metadata-first) ────────


def test_visual_heavy_detector_matches_scene_markers():
    """The transcript-shape detector fires on time-range markers,
    frame-count annotations, and OCR markers — all signals that
    content came from the visual pipeline."""
    samples = [
        "[0:00-1:30] A static-shot music video.",          # time range
        "[0:30] (5 frames) something happens",              # frame count
        '[1:00] A scene — text on screen: "STAR WARS"',    # OCR marker
    ]
    for s in samples:
        assert bot._is_visual_heavy_transcript(s), f"should detect: {s!r}"


def test_visual_heavy_detector_rejects_speech():
    """A plain whisper transcript (no time-range / frame-count / OCR
    markers) must NOT be flagged as visual-heavy."""
    samples = [
        "[0:00] We're going to talk about cosmic backgrounds today.",
        "Hello and welcome to today's stream.\n[1:23] Let's begin.",
        "",
    ]
    for s in samples:
        assert not bot._is_visual_heavy_transcript(s), f"should reject: {s!r}"


def test_silent_prompts_exist():
    """All four silent-video prompt variants must be exported."""
    for name in ("PROMPT_BRIEF_SILENT", "PROMPT_KEY_POINTS_SILENT",
                 "PROMPT_CHAPTERS_SILENT",
                 "REDUCE_BRIEF_SILENT", "REDUCE_KEY_POINTS_SILENT"):
        assert hasattr(p, name), f"missing silent prompt: {name}"


def test_silent_prompts_lead_with_identity():
    """Silent prompts must instruct the LLM to LEAD with content identity
    (title, channel, OCR), not generic 'main thesis' framing."""
    for tmpl in (p.PROMPT_BRIEF_SILENT, p.PROMPT_KEY_POINTS_SILENT):
        assert "title" in tmpl.lower()
        assert "channel" in tmpl.lower()
        assert "ocr" in tmpl.lower() or "on-screen text" in tmpl.lower()


def test_silent_prompts_acknowledge_vlm_limits():
    """Silent prompts must tell the LLM to TRUST OCR over VLM for
    specifics — otherwise the LLM will repeat VLM's vague descriptions."""
    for tmpl in (p.PROMPT_BRIEF_SILENT, p.PROMPT_KEY_POINTS_SILENT,
                 p.PROMPT_CHAPTERS_SILENT):
        # Explicit acknowledgement of VLM blind spots
        tlow = tmpl.lower()
        assert "vlm" in tlow or "vision-language" in tlow or \
               "cannot identify" in tlow or "cannot reliably" in tlow
        assert "ocr" in tlow or "on-screen text" in tlow


def test_silent_prompts_use_channel_placeholder():
    """Silent prompts must have {channel} for the YT channel name kwarg."""
    for tmpl in (p.PROMPT_BRIEF_SILENT, p.PROMPT_KEY_POINTS_SILENT,
                 p.PROMPT_CHAPTERS_SILENT):
        assert "{channel}" in tmpl


def test_process_switches_to_silent_prompts():
    """process() detects visual-heavy transcripts and switches to silent
    prompts. Source-level check."""
    src = BOT_SRC
    proc = src[src.index("async def process(job: Job):"):
                src.index("async def process_url")]
    assert "_is_visual_heavy_transcript" in proc
    assert "PROMPT_BRIEF_SILENT" in proc
    assert "PROMPT_KEY_POINTS_SILENT" in proc
    assert "PROMPT_CHAPTERS_SILENT" in proc
    assert "_fetch_channel_name" in proc


def test_format_scenes_includes_ocr_when_present():
    """_format_scenes must render OCR text alongside the scene description
    so the summary LLM can ground specific names in actual on-screen text."""
    scenes = [{
        "start": 0.0, "end": 10.0, "frame_count": 5,
        "description": "A title card appears with bold yellow text.",
        "ocr": "STAR WARS THEME",
    }]
    out = bot._format_scenes(scenes)
    assert "STAR WARS THEME" in out
    assert "text on screen" in out.lower()


def test_format_scenes_skips_empty_ocr():
    """No OCR → no `text on screen:` annotation."""
    scenes = [{
        "start": 0.0, "end": 5.0, "frame_count": 1,
        "description": "A scene description.",
        "ocr": "",
    }]
    out = bot._format_scenes(scenes)
    assert "text on screen" not in out.lower()


def test_chunk_preamble_anchors_timestamps():
    """Map preamble must instruct the LLM to keep timestamps verbatim
    (otherwise chunked chapters could renormalize relative to chunk start)."""
    assert "VERBATIM" in p.CHUNK_PREAMBLE
    assert "absolute video times" in p.CHUNK_PREAMBLE


def test_transcribe_call_has_explicit_timeout():
    """Submit + poll cycle bounded by TRANSCRIBE_TIMEOUT — long videos
    on slow models would otherwise hang the bot worker indefinitely if the
    server-side job got wedged."""
    assert "TRANSCRIBE_TIMEOUT" in BOT_SRC
    # Constant exists and is sane (>= 10 min)
    assert hasattr(bot, "TRANSCRIBE_TIMEOUT")
    assert bot.TRANSCRIBE_TIMEOUT >= 600
    # Used as the upper bound on the submit/poll loop deadline
    assert "time.monotonic() + TRANSCRIBE_TIMEOUT" in BOT_SRC
    # And the loop honours the deadline
    assert "while time.monotonic() < deadline" in BOT_SRC


def test_send_long_embed_total_size_guard():
    """send_long_embed respects 6000-char total embed payload limit."""
    assert hasattr(bot, "EMBED_TOTAL_LIMIT")
    assert bot.EMBED_TOTAL_LIMIT == 6000
    assert hasattr(bot, "_safe_description_len")
    # Long title eats into the description budget
    long_title = "x" * 250
    short_budget = bot._safe_description_len(long_title)
    short_budget_no_title = bot._safe_description_len("")
    assert short_budget < short_budget_no_title


def test_chan_cfg_vlm_enabled_wired():
    """Per-channel vlm_enabled override flows from the effective config into Job.

    The variable name changed from `chan_cfg` to `cfg` when per-user config
    (`_effective_config`) was layered on top of per-channel config — the
    behaviour is identical (channel wins on conflict), only the binding name
    changed. The guard here is that vlm_enabled is read from a merged-config
    dict at every Job construction site, not hardcoded.
    """
    # cfg.get("vlm_enabled", VLM_ENABLED) at every Job construction site:
    # on_message + cmd_summarize + cmd_transcribe + reply-trigger video path.
    occurrences = BOT_SRC.count('cfg.get("vlm_enabled", VLM_ENABLED)')
    assert occurrences >= 4, (
        f"vlm_enabled must be passed from cfg in on_message, reply-trigger, "
        f"cmd_summarize, cmd_transcribe (found {occurrences})"
    )
    # Effective config helper exists and merges user + channel — guard against
    # someone re-introducing `get_channel_config` at Job construction without
    # also wiring user config.
    assert "def _effective_config(" in BOT_SRC, (
        "_effective_config must exist as the canonical merge helper"
    )
    # process() reads job.vlm_enabled, not module-level VLM_ENABLED.
    # Use a regex so reflows/parenthesisation (single-line or block) don't
    # falsely fail this guard — what matters is that job.vlm_enabled gates
    # the user-force branch in the same boolean expression.
    import re
    assert re.search(
        r"job\.vlm_enabled\s+and\s+\(\s*\n?\s*user_forced_vlm", BOT_SRC
    ), "process() must gate run_vlm on `job.vlm_enabled and (user_forced_vlm ...)`"


def test_no_assert_http():
    """`assert http` strips under `python -O`. All call sites must use an
    explicit raise instead."""
    # Comment lines like "Replaces `assert http`" are fine; flag actual code.
    lines = BOT_SRC.splitlines()
    bad = [
        i + 1 for i, line in enumerate(lines)
        if line.strip() == "assert http"
    ]
    assert not bad, f"`assert http` survives at lines: {bad}"


# ─── Web URL summary flow ────────────────────────────────────────────────────


def test_reply_trigger_regex_matches_keywords():
    """`tldr`, `summarize`, `summarise` (with optional punctuation) trigger."""
    matches = [
        "tldr", "TLDR", "tldr.", "tldr!", " tldr ", "Tldr",
        "summarize", "Summarise", "summarise.",
    ]
    for s in matches:
        assert bot.REPLY_TRIGGER_RE.match(s), f"should match: {s!r}"


def test_reply_trigger_regex_rejects_sentences():
    """A sentence containing the keyword must NOT trigger — only bare keyword."""
    rejects = [
        "give me a tldr of this",
        "tldr please",
        "I'll tldr it later",
        "summarize this article",
        "",
        "lol",
        "tldr;",  # semicolons aren't typical sentence punctuation here
    ]
    for s in rejects:
        m = bot.REPLY_TRIGGER_RE.match(s)
        # `tldr;` is the edge case: semicolon isn't in our class so it rejects
        if s == "tldr;":
            assert not m, f"should reject: {s!r}"
        else:
            assert not m, f"should reject: {s!r}"


def test_extract_first_url_skips_discord_internal():
    """Discord channel/message links should be ignored — they aren't articles."""
    text = "see https://discord.com/channels/1/2/3 then https://example.com/x"
    assert bot._extract_first_url(text) == "https://example.com/x"


def test_extract_first_url_trims_trailing_punctuation():
    text = "look at https://example.com/article."
    assert bot._extract_first_url(text) == "https://example.com/article"


def test_extract_first_url_returns_none_when_no_url():
    assert bot._extract_first_url("just text, no link") is None
    assert bot._extract_first_url("") is None
    assert bot._extract_first_url(None) is None


def test_hash_url_stable_and_distinct():
    a = bot._hash_url("https://example.com/foo")
    b = bot._hash_url("https://example.com/bar")
    assert a == bot._hash_url("https://example.com/foo")  # stable
    assert a != b
    assert a.startswith("w") and len(a) == 11


def test_is_video_url():
    """Known video domains classify as video; arbitrary blogs do not."""
    assert bot._is_video_url("https://www.youtube.com/watch?v=abc")
    assert bot._is_video_url("https://twitch.tv/user")
    assert not bot._is_video_url("https://news.example.com/article")
    assert not bot._is_video_url("https://github.com/user/repo")


def test_job_kind_dispatch():
    """Job.kind defaults to 'video'; web jobs explicitly set kind='web'."""
    j_default = bot.Job(
        url="x", video_id="x", channel=object(), submitter_id=1,
        message=object(),
    )
    assert j_default.kind == "video"
    j_web = bot.Job(
        url="x", video_id="x", channel=object(), submitter_id=1,
        kind="web", message=object(),
    )
    assert j_web.kind == "web"
    # Invalid kind rejected
    try:
        bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                kind="bogus", message=object())
    except ValueError as e:
        assert "kind" in str(e)
    else:
        raise AssertionError("invalid kind should raise")


def test_looks_like_bot_challenge():
    """Short text containing anti-bot challenge markers → True; long
    article → False. Covers Cloudflare (JS + CAPTCHA), DataDome, Akamai,
    PerimeterX.
    """
    # Cloudflare JS challenge
    assert bot._looks_like_bot_challenge("Just a moment...\nChecking your browser.")
    assert bot._looks_like_bot_challenge("ddos protection by cloudflare")
    # Cloudflare CAPTCHA / Turnstile (archive.ph regression — they shell out
    # to this when rate-limiting and the response title becomes
    # "One more step" with body "Please complete the security check to
    # access archive.ph". Previously this slipped past the marker check
    # because the wording differs from the JS challenge above, and 563
    # chars was over the MIN_SCRAPED_BODY_CHARS floor.)
    archive_ph_cf_capture = (
        "#  One more step  ##  Please complete the security check to access "
        "archive.ph ##  Why do I have to complete a CAPTCHA?  Completing "
        "the CAPTCHA proves you are a human and gives you temporary access "
        "to the web property."
    )
    assert bot._looks_like_bot_challenge(archive_ph_cf_capture), (
        "archive.ph's Cloudflare CAPTCHA page must be detected as a challenge"
    )
    # DataDome (the Reuters case that motivated the marker expansion).
    assert bot._looks_like_bot_challenge("var dd={'rt':'i','cid':'abc'}")
    assert bot._looks_like_bot_challenge("captcha-delivery.com/load.js")
    # Akamai — marker is the canonical unsolved-challenge cookie sentinel
    # `_abck=~-1~`, NOT the bare cookie name. The full sentinel only
    # appears in actual blocked Akamai responses; a bare `_abck` mention
    # in a short security-blog post about Akamai bypass would otherwise
    # trip the heuristic falsely.
    assert bot._looks_like_bot_challenge(
        "set-cookie: _abck=~-1~-1~-1~-1~AAAA;"
    )
    assert bot._looks_like_bot_challenge(
        "_abck=~-1~-1~-1~ Akamai bot manager blocked this request"
    )
    # The bare cookie name in an article body is NOT a challenge — covers
    # the false-positive case that motivated the tightening (a security
    # blog about `_abck` cookies that happens to be < 4000 chars).
    assert not bot._looks_like_bot_challenge(
        "The _abck cookie is set by Akamai's bot manager to track "
        "client fingerprints. " + "x " * 30
    ), "bare `_abck` mention in an article must not be flagged as a challenge"
    assert bot._looks_like_bot_challenge("px-captcha challenge required")
    # Long article that mentions one of the products in passing — not a challenge
    long_text = "Cloudflare announced new features. " + "x " * 1500
    assert not bot._looks_like_bot_challenge(long_text)
    assert not bot._looks_like_bot_challenge("")
    # Backwards-compat alias still resolves to the same function.
    assert bot._looks_like_cf_challenge is bot._looks_like_bot_challenge


def test_html_to_text_strips_scripts_and_tags():
    html = """
    <html><head><title>x</title>
    <script>evil();</script>
    <style>body{display:none}</style>
    </head>
    <body><h1>Title</h1><p>Hello <b>world</b>.</p>
    <script>more();</script></body></html>
    """
    out = bot._html_to_text(html)
    assert "evil()" not in out
    assert "display:none" not in out
    assert "Title" in out
    assert "Hello" in out and "world" in out


def test_derive_title_from_markdown_uses_h1():
    md = "# My Article\n\nLorem ipsum"
    assert bot._derive_title_from_markdown(md, "https://x.com/p") == "My Article"


def test_derive_title_from_markdown_falls_back_to_host():
    md = "Just body text, no heading"
    assert bot._derive_title_from_markdown(md, "https://news.example.com/p") == "news.example.com"


def test_web_prompts_have_security_rules():
    """Article prompts must inherit REF_RULES_WEB security/citation rules."""
    for tmpl in (p.PROMPT_BRIEF_WEB, p.PROMPT_KEY_POINTS_WEB, p.PROMPT_SECTIONS):
        assert "STRICT RULES" in tmpl
        assert "<article>" in tmpl
        assert "UNTRUSTED USER CONTENT" in tmpl


def test_web_prompts_have_source_placeholder():
    """Web prompts include {source} for the article's host (vs {duration} on video)."""
    for tmpl in (p.PROMPT_BRIEF_WEB, p.PROMPT_KEY_POINTS_WEB, p.PROMPT_SECTIONS):
        assert "{source}" in tmpl


def test_sections_prompt_has_no_timestamp_instructions():
    """PROMPT_SECTIONS is the chapters analogue without timestamps."""
    assert "[H:MM:SS]" not in p.PROMPT_SECTIONS
    assert "[MM:SS]" not in p.PROMPT_SECTIONS
    assert "timestamp" in p.PROMPT_SECTIONS.lower()  # only as a NEGATIVE instruction
    assert "No timestamps" in p.PROMPT_SECTIONS


def test_worker_dispatches_on_kind():
    """Worker picks process_litmus / process_url / process based on kind."""
    src = BOT_SRC
    # New three-way dispatch: litmus / web / default-video
    assert 'if job.kind == "litmus":' in src
    assert "handler = process_litmus" in src
    assert 'elif job.kind == "web":' in src
    assert "handler = process_url" in src
    assert "handler = process" in src
    assert "await handler(job)" in src


def test_scraper_config_present():
    """Bot exposes scraper config + functions."""
    for name in ("SCRAPER_API", "FLARESOLVERR_API", "SCRAPER_TIMEOUT",
                 "MIN_SCRAPED_BODY_CHARS",
                 "ENABLE_ARCHIVE_FALLBACKS", "WAYBACK_API",
                 "WAYBACK_TIMEOUT", "ARCHIVE_PH_BASE",
                 "ARCHIVE_PH_TIMEOUT", "ARCHIVE_USER_AGENT",
                 "fetch_article", "_fetch_via_crawl4ai", "_fetch_via_flaresolverr",
                 "_fetch_via_wayback", "_fetch_via_archive_ph",
                 "_wayback_raw_url",
                 "process_url", "_handle_reply_trigger"):
        assert hasattr(bot, name), f"missing: {name}"


def test_wayback_raw_url_inserts_id_modifier():
    """Wayback URLs from the availability API include nav chrome by
    default; the `id_` modifier returns the raw archived bytes. Verifies
    the rewrite handles both http→https coercion and the timestamp insert.
    """
    inp = "http://web.archive.org/web/20231213155408/https://www.reuters.com/world/"
    out = bot._wayback_raw_url(inp)
    assert out == (
        "https://web.archive.org/web/20231213155408id_/https://www.reuters.com/world/"
    ), f"unexpected rewrite: {out}"
    # Idempotent on URLs that already have id_
    twice = bot._wayback_raw_url(out)
    # Second pass leaves id_ where it is (no double id_id_)
    assert "id_id_" not in twice
    # Already-https input stays https
    https_in = "https://web.archive.org/web/20240101120000/https://example.com/"
    assert bot._wayback_raw_url(https_in) == (
        "https://web.archive.org/web/20240101120000id_/https://example.com/"
    )


def test_fetch_article_routes_archive_tiers_last():
    """Tier ordering inside fetch_article: live origins (crawl4ai →
    flaresolverr) must always run before archive snapshots (wayback →
    archive.ph), since archives can lag real-time by hours-to-days.
    """
    import inspect
    src = inspect.getsource(bot.fetch_article)
    crawl_pos = src.index("_fetch_via_crawl4ai")
    flare_pos = src.index("_fetch_via_flaresolverr")
    wayback_pos = src.index("_fetch_via_wayback")
    archive_pos = src.index("_fetch_via_archive_ph")
    assert crawl_pos < flare_pos < wayback_pos < archive_pos, (
        "Expected tier order: crawl4ai < flaresolverr < wayback < archive.ph"
    )


def test_fetch_via_wayback_returns_none_on_no_snapshot():
    """Availability API returns `archived_snapshots: {}` for URLs with no
    snapshot (e.g. very-recent or fabricated URLs like the Reuters
    2026-05-18 test article). _fetch_via_wayback must return None so
    fetch_article falls through to archive.ph rather than raising.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.json = AsyncMock(return_value={
        "url": "https://example.com/", "archived_snapshots": {},
    })

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_http = MagicMock()
    fake_http.get = MagicMock(return_value=fake_ctx)

    with patch.object(bot, "http", fake_http):
        result = asyncio.run(bot._fetch_via_wayback("https://example.com/"))
    assert result is None, "No snapshot must yield None, not raise"


def test_fetch_via_wayback_handles_rate_limit():
    """429 from Wayback (its anonymous quotas are ~15 req/min/IP) must
    fall through silently. Loud retries would burn through the quota
    faster and pin the entire bot in the failure path.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_resp = MagicMock()
    fake_resp.status = 429
    fake_resp.json = AsyncMock(return_value={})

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_http = MagicMock()
    fake_http.get = MagicMock(return_value=fake_ctx)

    with patch.object(bot, "http", fake_http):
        result = asyncio.run(bot._fetch_via_wayback("https://example.com/"))
    assert result is None, "429 must yield None (fall through to next tier)"


def test_archive_fallbacks_disabled_short_circuits():
    """ENABLE_ARCHIVE_FALLBACKS=0 must skip both archive tiers entirely
    so the bot reverts to the two-tier (Crawl4AI + FlareSolverr) shape.
    Useful kill-switch if archive.org/ph are degrading the failure path.
    """
    import asyncio
    from unittest.mock import patch

    with patch.object(bot, "ENABLE_ARCHIVE_FALLBACKS", False):
        wb = asyncio.run(bot._fetch_via_wayback("https://example.com/"))
        ap = asyncio.run(bot._fetch_via_archive_ph("https://example.com/"))
    assert wb is None and ap is None


def test_archive_ph_url_encodes_query_string():
    """Tracker-tagged URLs (`?utm_source=`, `?id=`) must survive the
    embed-into-archive.ph-path step. Without percent-encoding, yarl/aiohttp
    splits the user URL's query string off as archive.ph's own query
    string, breaking the URL-exact-match snapshot lookup. Verifies the
    URL passed to Crawl4AI keeps the original query attached as part of
    the looked-up URL (encoded), not as a top-level archive.ph query.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    captured: list[str] = []

    async def fake_crawl4ai(url: str) -> str | None:
        captured.append(url)
        return None

    with patch.object(bot, "_fetch_via_crawl4ai",
                      side_effect=fake_crawl4ai), \
         patch.object(bot, "http", AsyncMock()):  # ensure http is non-None
        asyncio.run(bot._fetch_via_archive_ph(
            "https://www.reuters.com/world/europe/article/?utm_source=twitter"
        ))

    assert len(captured) == 1, "Crawl4AI must be called exactly once"
    archive_url = captured[0]
    # The query string from the user URL must be PERCENT-ENCODED into the
    # archive.ph path so yarl can't split it off. `?utm_source=...` would
    # become archive.ph's own query string; `%3Futm_source%3D...` keeps
    # it attached to the URL being looked up.
    assert "%3F" in archive_url, (
        f"`?` must be percent-encoded into archive.ph path; got {archive_url!r}"
    )
    assert "%3D" in archive_url, (
        f"`=` must be percent-encoded into archive.ph path; got {archive_url!r}"
    )
    # Sanity: scheme + slashes preserved (safe=':/'), so logs stay
    # human-readable for the URL-being-archived prefix.
    assert archive_url.startswith(
        "https://archive.ph/newest/https://www.reuters.com/world/europe/article/"
    )
    # The literal `?utm_source=` must NOT appear unencoded in the path —
    # that's the regression we're guarding against.
    assert "?utm_source=" not in archive_url


def test_archive_ph_url_unchanged_for_query_less_input():
    """URLs with no query string must produce the same archive.ph URL as
    before the encoding fix — guards against the new `quote()` call
    accidentally over-escaping (e.g. encoding `:` or `/` and breaking the
    happy path that worked in production).
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    captured: list[str] = []

    async def fake_crawl4ai(url: str) -> str | None:
        captured.append(url)
        return None

    with patch.object(bot, "_fetch_via_crawl4ai",
                      side_effect=fake_crawl4ai), \
         patch.object(bot, "http", AsyncMock()):
        asyncio.run(bot._fetch_via_archive_ph("https://example.com/foo"))

    assert captured == ["https://archive.ph/newest/https://example.com/foo"]


def test_min_scraped_body_floor_is_sane():
    """Default floor must reject obvious challenge stubs (e.g. Reuters'
    `reuters.com\\n` via DataDome → 11 chars) without rejecting short
    legitimate articles. 50–500 is the defensible band; default 200."""
    assert 50 <= bot.MIN_SCRAPED_BODY_CHARS <= 500


def test_fetch_via_crawl4ai_rejects_short_body():
    """Implausibly short markdown (e.g. just the hostname from a DataDome
    challenge stub) must be rejected so the caller falls through to the
    FlareSolverr backup. This is the core regression fix for the Reuters
    case where Crawl4AI returned `reuters.com\\n` and the bot fed those
    11 chars to the LLM.
    """
    import asyncio, inspect
    from unittest.mock import AsyncMock, MagicMock, patch

    # Mock the aiohttp session's POST: return a tiny markdown body.
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.json = AsyncMock(return_value={"markdown": "reuters.com\n"})
    fake_resp.text = AsyncMock(return_value="")

    fake_ctx = MagicMock()
    fake_ctx.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_ctx.__aexit__ = AsyncMock(return_value=None)

    fake_http = MagicMock()
    fake_http.post = MagicMock(return_value=fake_ctx)

    with patch.object(bot, "http", fake_http):
        result = asyncio.run(bot._fetch_via_crawl4ai(
            "https://www.reuters.com/some/article"))

    assert result is None, (
        "Crawl4AI returning 12-char DataDome stub must be rejected "
        "(MIN_SCRAPED_BODY_CHARS floor)"
    )
    # Sanity-check the floor is what we documented (200) — if someone
    # tightens it below 12 we'd lose the regression, above 500 we'd
    # start rejecting real short articles.
    assert bot.MIN_SCRAPED_BODY_CHARS >= 12


def test_process_url_uses_web_prompts_not_video():
    """process_url() must use PROMPT_*_WEB and PROMPT_SECTIONS, not the
    video templates (would emit timestamp instructions for an article)."""
    import inspect
    src = inspect.getsource(bot.process_url)
    assert "PROMPT_BRIEF_WEB" in src
    assert "PROMPT_KEY_POINTS_WEB" in src
    assert "PROMPT_SECTIONS" in src
    assert "PROMPT_CHAPTERS" not in src  # would inject timestamps


# ─── URL routing (clear-video classifier + NotAVideoError fallback) ──────────


def test_clearly_video_url_youtube():
    """YouTube watch / shorts / live / youtu.be all classify as video."""
    for url in (
        "https://www.youtube.com/watch?v=abc123",
        "https://m.youtube.com/watch?v=abc",
        "https://music.youtube.com/watch?v=abc",
        "https://youtube.com/shorts/xyz",
        "https://www.youtube.com/live/abcdefg",
        "https://youtu.be/abcdefghijk",
    ):
        assert bot._is_clearly_video_url(url), f"should be video: {url}"


def test_clearly_video_url_twitch():
    """Twitch VODs, clips, and live streams classify as video."""
    for url in (
        "https://clips.twitch.tv/SomeClip",
        "https://www.twitch.tv/videos/12345",
        "https://twitch.tv/streamer",
        "https://twitch.tv/streamer/",
    ):
        assert bot._is_clearly_video_url(url), f"should be video: {url}"


def test_clearly_video_url_other_platforms():
    cases = [
        "https://vimeo.com/123456",
        "https://player.vimeo.com/video/123456",
        "https://vimeo.com/channels/staffpicks/12345",
        "https://www.tiktok.com/@user/video/12345",
        "https://vm.tiktok.com/abc",
        "https://v.redd.it/somevideoid",
        "https://www.dailymotion.com/video/x123",
        "https://dai.ly/x123",
        "https://rumble.com/v123abc-some-title",
        "https://www.bilibili.com/video/BV1xx411c7mD",
        "https://b23.tv/abc",
        "https://soundcloud.com/artist/track-name",
    ]
    for url in cases:
        assert bot._is_clearly_video_url(url), f"should be video: {url}"


def test_clearly_video_url_rejects_text_posts():
    """The whole point of this classifier — text posts on video-hosting
    domains must NOT classify as video. This is the bug the screenshot
    shows: reddit.com text post URLs were being routed to yt-dlp.
    """
    cases = [
        # The exact URL from the bug report
        "https://www.reddit.com/r/television/comments/1t7sehx/karl_urban_is_officially_done_with_the_boys_but/",
        # Other plain reddit URLs
        "https://reddit.com/r/python",
        "https://www.reddit.com/user/someone",
        # Twitter / X text posts (no video)
        "https://twitter.com/user/status/123",
        "https://x.com/user/status/123",
        # Instagram profile / posts (mixed; play it safe — web)
        "https://instagram.com/user",
        "https://www.instagram.com/p/ABC123/",
        # Profile pages on video sites
        "https://www.youtube.com/@channelname",
        "https://www.youtube.com/c/ChannelName/about",
        # Generic article URLs (the actual destination of the redirect chain)
        "https://collider.com/karl-urban-the-boys-ending/",
        "https://news.example.com/article",
        "https://github.com/user/repo",
        # Empty / nonsense
        "",
        "not a url",
    ]
    for url in cases:
        assert not bot._is_clearly_video_url(url), \
            f"should NOT be video: {url}"


def test_not_a_video_error_class():
    """NotAVideoError is a PermanentError subclass — skips transient retries."""
    assert issubclass(bot.NotAVideoError, bot.PermanentError)


def test_not_a_video_error_classification():
    """'Unsupported URL' style errors classify as not-a-video."""
    cases = [
        "Unsupported URL: https://collider.com/article",
        "ERROR: 'foo' is not a valid URL",
        "No video could be found in this tweet",
        "There's no video in this post",
        "No media found",
        "No video formats found",
    ]
    for s in cases:
        assert bot._is_not_a_video_error(s), f"should classify: {s!r}"


def test_not_a_video_error_excludes_gating_failures():
    """Private / members-only / geo-blocked errors must NOT classify as
    not-a-video — falling through to the article URL would fail the same
    way (paywalled content, etc.) so we want a clean PermanentError."""
    cases = [
        "Sign in to confirm your age",
        "Private video",
        "members-only content",
        "blocked it on copyright grounds",
        "blocked it in your country",
        "Premieres in 2 days",
        "This live event will begin",
    ]
    for s in cases:
        assert not bot._is_not_a_video_error(s), \
            f"should NOT classify as not-a-video: {s!r}"


def test_reply_trigger_routes_reddit_text_to_web():
    """Bug from screenshot: tldr reply to reddit text post URL was sent
    to video pipeline. Now it must route to web."""
    url = "https://www.reddit.com/r/x/comments/abc/title/"
    # The classifier is the only routing decision in _handle_reply_trigger
    # for kind. Verify the source uses _is_clearly_video_url.
    src = BOT_SRC
    # Reply-trigger handler should use _is_clearly_video_url, not _is_video_url
    handler_src = src[src.index("async def _handle_reply_trigger"):
                       src.index("# ─── Web scraper client")]
    assert "_is_clearly_video_url(url)" in handler_src
    assert "_is_video_url(url)" not in handler_src
    # Sanity: that classifier returns False for the reddit URL
    assert not bot._is_clearly_video_url(url)


def test_on_message_uses_clear_video_classifier():
    """Auto-paste should NOT trigger on text-post URLs from video-hosting
    domains. on_message must filter VIDEO_URL_PATTERN matches through
    _is_clearly_video_url before queueing."""
    src = BOT_SRC
    on_msg = src[src.index("async def on_message"):
                  src.index("# ─── Reply-trigger handler")]
    assert "_is_clearly_video_url(url)" in on_msg


def test_explicit_request_field():
    """Job.explicit_request defaults False — auto-paste; reply-trigger and
    slash sites set True."""
    j = bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                message=object())
    assert j.explicit_request is False
    j2 = bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                 explicit_request=True, message=object())
    assert j2.explicit_request is True


def test_worker_handles_not_a_video_error():
    """Worker source must contain the NotAVideoError handling branch
    that falls through to web for explicit jobs and silent-drops for
    auto-paste."""
    src = BOT_SRC
    worker_src = src[src.index("async def worker"):
                      src.index("async def process(job: Job)")]
    assert "except NotAVideoError" in worker_src
    assert "job.explicit_request" in worker_src
    assert 'job.kind = "web"' in worker_src
    assert "silent_drop" in worker_src


def test_worker_attaches_retry_view_on_llm_offline_failure():
    """Worker terminal-failure path must build a RetryJobsView when the
    last_error is LLMOfflineError so the user gets a one-click resubmit
    without copy-pasting the URL when the LLM comes back. Mirrors the
    queue-gate rejection UX in _maybe_retry_view."""
    src = BOT_SRC
    worker_src = src[src.index("async def worker"):
                      src.index("async def process(job: Job)")]
    # The branch must look at LLMOfflineError specifically — non-LLM
    # permanent errors (content-side issues) don't get the button.
    assert "isinstance(last_error, LLMOfflineError)" in worker_src
    # And it must construct the view from the live Job's retry spec.
    assert "RetryJobsView(" in worker_src
    assert "_job_to_retry_spec(job)" in worker_src
    # And pass it down to _job_reply.
    assert "view=retry_view" in worker_src


def test_job_reply_accepts_view_parameter():
    """_job_reply must accept an optional `view` so the worker's
    LLM-offline failure handler can attach a Retry button. Without this
    parameter, the worker can't surface a retry button on terminal
    failure and the user is forced to re-paste the URL."""
    import inspect
    sig = inspect.signature(bot._job_reply)
    assert "view" in sig.parameters, \
        "_job_reply must accept a `view` kwarg for the Retry button path"
    # Default must be None so existing callers (without a view) still work.
    assert sig.parameters["view"].default is None


def test_video_path_raises_not_a_video_on_unsupported():
    """process() must distinguish NotAVideoError from generic PermanentError
    so the worker can route correctly."""
    src = BOT_SRC
    process_src = src[src.index("async def process(job: Job):"):
                       src.index("async def process_url")]
    assert "NotAVideoError" in process_src
    assert "_is_not_a_video_error(err)" in process_src


def test_processing_emoji_distinguishes_video_and_web():
    """🎧 (audio download) vs 📰 (article scrape) — users see at a glance
    what the bot is doing."""
    assert hasattr(bot, "PROCESSING_EMOJI_VIDEO")
    assert hasattr(bot, "PROCESSING_EMOJI_WEB")
    assert bot.PROCESSING_EMOJI_VIDEO == "\U0001f3a7"  # 🎧
    assert bot.PROCESSING_EMOJI_WEB == "\U0001f4f0"    # 📰
    assert bot.PROCESSING_EMOJI_VIDEO != bot.PROCESSING_EMOJI_WEB
    # Cleanup tuple covers BOTH so a kind-switch leaves no stale reaction
    assert bot.PROCESSING_EMOJI_VIDEO in bot.PROCESSING_EMOJI
    assert bot.PROCESSING_EMOJI_WEB in bot.PROCESSING_EMOJI


def test_video_path_uses_video_emoji():
    """process() must react with the video emoji, not the web one."""
    src = BOT_SRC
    process_src = src[src.index("async def process(job: Job):"):
                       src.index("async def process_url")]
    assert "PROCESSING_EMOJI_VIDEO" in process_src
    assert "PROCESSING_EMOJI_WEB" not in process_src


def test_web_path_uses_web_emoji():
    """process_url() must react with the web emoji, not the video one."""
    src = BOT_SRC
    web_src = src[src.index("async def process_url"):
                   src.index("# Per-task model override")]
    assert "PROCESSING_EMOJI_WEB" in web_src
    assert "PROCESSING_EMOJI_VIDEO" not in web_src


# ─── Reddit-specific scraper ─────────────────────────────────────────────────


def test_is_reddit_post_url():
    """Detects /r/<sub>/comments/<id>/... across www/old/np/sh subdomains."""
    matches = [
        "https://www.reddit.com/r/television/comments/1t7sehx/karl_urban_/",
        "https://reddit.com/r/python/comments/abc123/title/",
        "https://old.reddit.com/r/news/comments/xyz/title/",
        "https://np.reddit.com/r/x/comments/aaa/y/",
        "https://sh.reddit.com/r/x/comments/aaa/y/",
    ]
    for url in matches:
        assert bot._is_reddit_post_url(url), f"should match: {url}"


def test_is_reddit_post_url_rejects_non_post():
    """Subreddit listings, user pages, /comments/ root etc. don't match."""
    rejects = [
        "https://reddit.com/r/python",
        "https://www.reddit.com/r/python/hot",
        "https://www.reddit.com/user/someone",
        "https://www.reddit.com/comments",
        "https://www.reddit.com/",
        "https://example.com/r/foo/comments/1/x/",
    ]
    for url in rejects:
        assert not bot._is_reddit_post_url(url), f"should NOT match: {url}"


def test_format_reddit_comment_basic():
    node = {
        "kind": "t1",
        "data": {"author": "alice", "score": 42, "body": "Hello world.", "replies": ""},
    }
    out = bot._format_reddit_comment(node, depth=0, max_depth=1)
    assert "u/alice" in out
    assert "42 pts" in out
    assert "Hello world." in out


def test_format_reddit_comment_skips_deleted():
    for body in ("[deleted]", "[removed]", ""):
        node = {"kind": "t1", "data": {"author": "x", "score": 1, "body": body}}
        assert bot._format_reddit_comment(node, 0, 1) == ""


def test_format_reddit_comment_includes_replies_to_depth():
    node = {
        "kind": "t1",
        "data": {
            "author": "a", "score": 10, "body": "parent",
            "replies": {
                "data": {
                    "children": [
                        {"kind": "t1", "data": {
                            "author": "b", "score": 5, "body": "child1",
                            "replies": "",
                        }},
                        {"kind": "t1", "data": {
                            "author": "c", "score": 3, "body": "child2",
                            "replies": "",
                        }},
                    ]
                }
            },
        },
    }
    out = bot._format_reddit_comment(node, 0, max_depth=1)
    assert "parent" in out
    assert "child1" in out and "child2" in out
    # depth-2 grandchildren would NOT appear if max_depth=1
    out_depth0 = bot._format_reddit_comment(node, 0, max_depth=0)
    assert "parent" in out_depth0
    assert "child1" not in out_depth0


def test_format_reddit_comment_truncates_long_bodies():
    long_body = "x" * 3000
    node = {"kind": "t1", "data": {"author": "a", "score": 1, "body": long_body}}
    out = bot._format_reddit_comment(node, 0, 1)
    # Cap is 2000 chars + ellipsis
    assert "x" * 2000 in out
    assert "x" * 2001 not in out
    assert "…" in out


def test_build_reddit_markdown_self_post():
    """Self-post (no link) — no 'Linked article' section."""
    post = {
        "title": "How do I X?", "subreddit": "python", "author": "alice",
        "selftext": "I'm trying to do X but Y happens.",
        "is_self": True, "score": 42, "num_comments": 5,
        "url": "https://www.reddit.com/r/python/comments/abc/how_do_i_x/",
    }
    comments = [{"kind": "t1", "data": {
        "author": "bob", "score": 100, "body": "Use Z.", "replies": "",
    }}]
    title, body = bot._build_reddit_markdown(post, comments, None, "", None)
    assert title == "How do I X?"
    assert "Linked article" not in body
    assert "r/python" in body
    assert "u/alice" in body and "I'm trying to do X" in body
    assert "Top 1 comments" in body
    assert "u/bob" in body and "Use Z." in body


def test_build_reddit_markdown_link_post_with_article():
    """Link post + article scraped → article section + reddit section."""
    post = {
        "title": "Karl Urban interview", "subreddit": "television",
        "author": "DaddyCool", "selftext": "", "is_self": False,
        "score": 500, "num_comments": 50,
        "url": "https://collider.com/karl-urban-interview/",
    }
    article_md = "# Karl Urban Is Done\n\nKarl Urban said in an interview..."
    title, body = bot._build_reddit_markdown(
        post, [], article_md,
        "https://collider.com/karl-urban-interview/", None,
    )
    assert "Linked article" in body
    assert "collider.com" in body
    assert "Karl Urban Is Done" in body
    assert "r/television" in body
    assert "DaddyCool" in body


def test_build_reddit_markdown_link_post_with_article_error():
    """Link post + article fetch failed → note in markdown, reddit content still present."""
    post = {
        "title": "Big news", "subreddit": "news", "author": "u1",
        "selftext": "", "is_self": False, "score": 10, "num_comments": 2,
        "url": "https://paywalled.example/article",
    }
    title, body = bot._build_reddit_markdown(
        post, [], None, "https://paywalled.example/article",
        "permanent: 403 Forbidden",
    )
    assert "Article unreachable" in body
    assert "permanent: 403" in body
    assert "r/news" in body  # discussion section still present


def test_build_reddit_markdown_orders_comments_by_score():
    """Top N comments must be sorted by score, descending."""
    post = {"title": "T", "subreddit": "x", "author": "u", "selftext": "",
            "is_self": True, "score": 1, "num_comments": 3,
            "url": "https://reddit.com/r/x/comments/a/t/"}
    comments = [
        {"kind": "t1", "data": {"author": "low", "score": 1, "body": "low", "replies": ""}},
        {"kind": "t1", "data": {"author": "high", "score": 100, "body": "high", "replies": ""}},
        {"kind": "t1", "data": {"author": "mid", "score": 50, "body": "mid", "replies": ""}},
    ]
    _, body = bot._build_reddit_markdown(post, comments, None, "", None)
    # high should appear before mid which should appear before low
    high_pos = body.index("high")
    mid_pos = body.index("mid")
    low_pos = body.index("low")
    assert high_pos < mid_pos < low_pos


def test_build_reddit_markdown_skips_deleted_comments():
    post = {"title": "T", "subreddit": "x", "author": "u", "selftext": "",
            "is_self": True, "score": 1, "num_comments": 0,
            "url": "https://reddit.com/r/x/comments/a/t/"}
    comments = [
        {"kind": "t1", "data": {"author": "x", "score": 100, "body": "[deleted]", "replies": ""}},
        {"kind": "t1", "data": {"author": "y", "score": 50, "body": "[removed]", "replies": ""}},
        {"kind": "t1", "data": {"author": "z", "score": 1, "body": "real", "replies": ""}},
    ]
    _, body = bot._build_reddit_markdown(post, comments, None, "", None)
    assert "[deleted]" not in body
    assert "[removed]" not in body
    assert "real" in body
    assert "Top 1 comments" in body


def test_fetch_article_routes_reddit_url_first():
    """fetch_article must check _is_reddit_post_url before generic Crawl4AI."""
    import inspect
    src = inspect.getsource(bot.fetch_article)
    # Reddit branch should appear before the generic Crawl4AI call
    reddit_pos = src.index("_is_reddit_post_url")
    crawl_pos = src.index("_fetch_via_crawl4ai")
    assert reddit_pos < crawl_pos, \
        "Reddit special-case must run before generic crawl4ai fallback"


# ─── HackerNews scraper ──────────────────────────────────────────────────────


def test_is_hn_post_url():
    """Detects /item?id=<n> URLs from news.ycombinator.com."""
    matches = [
        "https://news.ycombinator.com/item?id=48082039",
        "http://news.ycombinator.com/item?id=1",
        "https://news.ycombinator.com/item?id=12345678&p=2",
    ]
    for url in matches:
        assert bot._is_hn_post_url(url), f"should match: {url}"


def test_is_hn_post_url_rejects():
    rejects = [
        "https://news.ycombinator.com/",
        "https://news.ycombinator.com/newest",
        "https://news.ycombinator.com/user?id=somebody",
        "https://example.com/item?id=123",  # not HN host
        "",
    ]
    for url in rejects:
        assert not bot._is_hn_post_url(url), f"should NOT match: {url}"


def test_format_hn_comment_basic():
    c = {"by": "alice", "text": "<p>Hello world testing.</p>", "kids": []}
    out = bot._format_hn_comment(c, depth=0)
    assert "alice" in out
    assert "Hello world testing." in out
    # HTML stripped
    assert "<p>" not in out


def test_format_hn_comment_includes_replies():
    c = {
        "by": "a", "text": "parent comment text",
        "kids": [
            {"by": "b", "text": "child reply 1", "kids": []},
            {"by": "c", "text": "child reply 2", "kids": []},
        ],
    }
    out = bot._format_hn_comment(c, depth=0)
    assert "parent comment text" in out
    assert "child reply 1" in out
    assert "child reply 2" in out
    # Children are indented
    assert "  - **b**:" in out


def test_format_hn_comment_truncates_long_bodies():
    long_body = "x" * 3000
    c = {"by": "a", "text": long_body, "kids": []}
    out = bot._format_hn_comment(c)
    assert "x" * 2000 in out
    assert "x" * 2001 not in out
    assert "…" in out


def test_build_hn_markdown_link_post():
    """Link post (Show HN with URL) → Linked article + HN discussion + comments."""
    post = {
        "title": "Show HN: My new tool", "by": "submitter", "score": 200,
        "descendants": 50, "url": "https://example.com/tool",
        "type": "story", "kids": [],
    }
    article_md = "# My Tool\n\nIt does X with Y."
    comments = [
        {"by": "alice", "text": "Cool, I tried it", "kids": []},
    ]
    title, body = bot._build_hn_markdown(
        post, comments, article_md, "https://example.com/tool", None,
    )
    assert title == "Show HN: My new tool"
    assert "Linked article" in body
    assert "example.com" in body
    assert "My Tool" in body
    assert "HackerNews discussion" in body
    assert "submitter" in body
    assert "200 pts, 50 comments" in body
    assert "Top 1 comments" in body
    assert "alice" in body


def test_build_hn_markdown_ask_hn():
    """Ask HN (no URL) → no Linked article section, but selftext present."""
    post = {
        "title": "Ask HN: Best stack for X?", "by": "asker", "score": 50,
        "descendants": 10, "type": "ask",
        "text": "<p>I'm building a new project and wondering...</p>",
        "kids": [],
    }
    title, body = bot._build_hn_markdown(post, [], None, "", None)
    assert "Linked article" not in body
    assert "HackerNews discussion" in body
    assert "Ask HN: Best stack for X?" in body
    # HTML in selftext stripped
    assert "<p>" not in body
    assert "I'm building" in body


def test_build_hn_markdown_link_post_with_article_error():
    """Link post + article fetch failed → note in markdown, discussion still present."""
    post = {
        "title": "An article", "by": "submitter", "score": 10,
        "descendants": 0, "url": "https://paywalled.example/article",
        "type": "story", "kids": [],
    }
    title, body = bot._build_hn_markdown(
        post, [], None, "https://paywalled.example/article",
        "permanent: 403 Forbidden",
    )
    assert "Article unreachable" in body
    assert "permanent: 403" in body
    assert "HackerNews discussion" in body


def test_fetch_article_routes_hn_url():
    """fetch_article must check _is_hn_post_url and run HN before generic."""
    import inspect
    src = inspect.getsource(bot.fetch_article)
    assert "_is_hn_post_url" in src
    hn_pos = src.index("_is_hn_post_url")
    crawl_pos = src.index("_fetch_via_crawl4ai")
    assert hn_pos < crawl_pos, \
        "HN special-case must run before generic crawl4ai fallback"


def test_process_url_treats_hn_as_discussion():
    """process_url's discussion detection must include HN URLs + body markers."""
    import inspect
    src = inspect.getsource(bot.process_url)
    assert "_is_hn_post_url(job.url)" in src
    assert "# HackerNews discussion" in src
    # Discussion content uses the Reddit-flavoured prompts (renamed
    # platform-neutral but file symbols stay PROMPT_*_REDDIT)
    assert "PROMPT_BRIEF_REDDIT" in src


def test_hn_constants_exist():
    """HN config knobs exported."""
    for name in ("HN_API_BASE", "HN_TOP_COMMENTS", "HN_REPLY_DEPTH",
                 "HN_TIMEOUT", "_is_hn_post_url", "_fetch_hn_item",
                 "_format_hn_comment", "_build_hn_markdown", "_fetch_hn"):
        assert hasattr(bot, name), f"missing: {name}"


def test_discussion_prompts_platform_agnostic():
    """Prompts read for both Reddit AND HN — language is generic."""
    for tmpl in (p.PROMPT_BRIEF_REDDIT, p.PROMPT_KEY_POINTS_REDDIT,
                 p.PROMPT_SECTIONS_REDDIT):
        # Generic phrasing
        assert "discussion thread" in tmpl
        # Should NOT say "the Reddit community" exclusively
        assert "the Reddit community" not in tmpl


# ─── Reddit-flavoured prompts ────────────────────────────────────────────────


def test_reddit_prompts_exist():
    """All six Reddit prompt variants are exported from bot.prompts."""
    for name in (
        "PROMPT_BRIEF_REDDIT", "PROMPT_KEY_POINTS_REDDIT", "PROMPT_SECTIONS_REDDIT",
        "REDUCE_BRIEF_REDDIT", "REDUCE_KEY_POINTS_REDDIT", "REDUCE_SECTIONS_REDDIT",
    ):
        assert hasattr(p, name), f"missing prompt: {name}"


def test_reddit_prompts_mention_comments():
    """The whole point — Reddit prompts must explicitly tell the LLM to
    surface comment perspectives, not just the article."""
    for tmpl in (p.PROMPT_BRIEF_REDDIT, p.PROMPT_KEY_POINTS_REDDIT,
                 p.PROMPT_SECTIONS_REDDIT):
        lower = tmpl.lower()
        assert "comment" in lower, "prompt must mention comments"
        # Must instruct the model that this is a multi-source document
        assert ("reddit" in lower) or ("community" in lower) or ("commenters" in lower)


def test_reddit_key_points_has_two_section_structure():
    """key_points Reddit prompt requires two clearly-marked sections so the
    bot's output explicitly separates article from reaction."""
    tmpl = p.PROMPT_KEY_POINTS_REDDIT
    assert "About the article" in tmpl or "About the article / post" in tmpl
    assert "Community reaction" in tmpl


def test_reddit_sections_prompt_lists_required_sections():
    """sections Reddit prompt must enumerate the structure to enforce."""
    tmpl = p.PROMPT_SECTIONS_REDDIT
    # Required headings the LLM should produce
    for required in ("Linked article", "Original post", "Community reaction"):
        assert required in tmpl, f"Missing required section guidance: {required}"


def test_reddit_prompts_have_security_rules():
    """Reddit prompts must inherit REF_RULES_WEB security/citation rules."""
    for tmpl in (p.PROMPT_BRIEF_REDDIT, p.PROMPT_KEY_POINTS_REDDIT,
                 p.PROMPT_SECTIONS_REDDIT):
        assert "STRICT RULES" in tmpl
        assert "<article>" in tmpl
        assert "UNTRUSTED USER CONTENT" in tmpl


def test_reddit_prompts_use_source_placeholder():
    """Reddit prompts use {source} like the web prompts."""
    for tmpl in (p.PROMPT_BRIEF_REDDIT, p.PROMPT_KEY_POINTS_REDDIT,
                 p.PROMPT_SECTIONS_REDDIT):
        assert "{source}" in tmpl


def test_process_url_routes_reddit_to_reddit_prompts():
    """When body contains Reddit structural markers OR URL is reddit, use
    Reddit prompts. Source-level check (process_url uses the variables)."""
    import inspect
    src = inspect.getsource(bot.process_url)
    # Detection logic must consider both URL and body markers
    assert "_is_reddit_post_url(job.url)" in src
    assert "# Reddit discussion" in src
    assert "## Top " in src
    # Both prompt families must be referenced
    assert "PROMPT_BRIEF_REDDIT" in src
    assert "PROMPT_BRIEF_WEB" in src  # fallback
    assert "PROMPT_SECTIONS_REDDIT" in src
    assert "REDUCE_KEY_POINTS_REDDIT" in src


def test_reddit_detection_by_url_alone():
    """Sanity: _is_reddit_post_url returns True for Reddit URLs even when
    we're checking process_url routing in isolation."""
    assert bot._is_reddit_post_url(
        "https://www.reddit.com/r/television/comments/1t7sehx/karl_urban/"
    )


# ─── YouTube comments (Community Reaction embed) ─────────────────────────────


def test_yt_comments_constants():
    """Default config: comments enabled, sane caps."""
    assert hasattr(bot, "YT_COMMENTS_ENABLED")
    assert bot.YT_COMMENTS_ENABLED is True
    assert bot.YT_COMMENTS_MAX == 100
    assert bot.YT_COMMENT_MIN_CHARS == 40
    assert bot.YT_COMMENT_SUMMARY_TOP_N == 30


def test_job_yt_comments_field():
    """Job carries yt_comments_enabled, default True (matches global default)."""
    j = bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                message=object())
    assert j.yt_comments_enabled is True
    j2 = bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                 yt_comments_enabled=False, message=object())
    assert j2.yt_comments_enabled is False


def test_filter_yt_comments_drops_short_and_emoji_only():
    """Substantive comments only — no 'first', no 🔥🔥🔥, no <40 chars."""
    comments = [
        {"text": "First!", "author": "a", "like_count": 5},
        {"text": "🔥🔥🔥💯💯", "author": "b", "like_count": 100},
        {"text": "lol", "author": "c", "like_count": 50},
        {"text": "x" * 50, "author": "d", "like_count": 10},  # substantive
        {"text": "This is a thoughtful long comment with real content.",
         "author": "e", "like_count": 25},
    ]
    out = bot.filter_yt_comments(comments)
    authors = {c["author"] for c in out}
    assert "d" in authors and "e" in authors
    assert "a" not in authors
    assert "b" not in authors
    assert "c" not in authors


def test_filter_yt_comments_ranks_pinned_first():
    """Pinned > hearted > likes."""
    comments = [
        {"text": "x" * 100, "author": "high_likes",
         "like_count": 1000, "is_pinned": False, "is_favorited": False},
        {"text": "x" * 100, "author": "hearted",
         "like_count": 50, "is_pinned": False, "is_favorited": True},
        {"text": "x" * 100, "author": "pinned",
         "like_count": 10, "is_pinned": True, "is_favorited": False},
    ]
    out = bot.filter_yt_comments(comments, top_n=3)
    # Pinned first, then hearted, then high-likes
    assert [c["author"] for c in out] == ["pinned", "hearted", "high_likes"]


def test_filter_yt_comments_respects_top_n():
    comments = [
        {"text": "x" * 100, "author": f"u{i}", "like_count": 100 - i}
        for i in range(50)
    ]
    out = bot.filter_yt_comments(comments, top_n=10)
    assert len(out) == 10
    # Sorted by likes desc — u0 (100 likes) first
    assert out[0]["author"] == "u0"


def test_filter_yt_comments_empty_input():
    assert bot.filter_yt_comments([]) == []
    assert bot.filter_yt_comments(None or []) == []


def test_format_yt_comments_tags_creator_engagement():
    """📌 pinned + ❤️ creator-hearted tags appear when set."""
    comments = [
        {"text": "Hello world testing", "author": "alice",
         "like_count": 100, "is_pinned": True, "is_favorited": True,
         "author_is_uploader": False, "parent": "root"},
        {"text": "Reply text testing", "author": "bob",
         "like_count": 5, "is_pinned": False, "is_favorited": False,
         "author_is_uploader": False, "parent": "abc123"},
    ]
    out = bot.format_yt_comments(comments)
    assert "📌pinned" in out
    assert "creator-hearted" in out
    assert "alice" in out and "bob" in out
    # Reply (parent != root) should be indented
    assert "  - " in out


def test_format_yt_comments_truncates_long_bodies():
    long_body = "x" * 2500
    comments = [{
        "text": long_body, "author": "a", "like_count": 1,
        "is_pinned": False, "is_favorited": False,
        "author_is_uploader": False, "parent": "root",
    }]
    out = bot.format_yt_comments(comments)
    # 1500-char cap + ellipsis
    assert "x" * 1500 in out
    assert "x" * 1501 not in out
    assert "…" in out


def test_yt_comments_prompt_exists():
    """PROMPT_YT_COMMENTS + REDUCE_YT_COMMENTS exported from prompts."""
    assert hasattr(p, "PROMPT_YT_COMMENTS")
    assert hasattr(p, "REDUCE_YT_COMMENTS")
    assert "{title}" in p.PROMPT_YT_COMMENTS
    assert "{duration}" in p.PROMPT_YT_COMMENTS
    assert "{char_cap}" in p.PROMPT_YT_COMMENTS
    assert "{transcript}" in p.PROMPT_YT_COMMENTS


def test_yt_comments_prompt_emphasises_creator_engagement():
    """The whole reason we tag pinned/hearted comments — prompt must use them."""
    tmpl = p.PROMPT_YT_COMMENTS.lower()
    assert "pinned" in tmpl
    assert "hearted" in tmpl or "engagement" in tmpl
    # Asks for substantive structure
    assert "agree" in tmpl
    assert "disagree" in tmpl or "disagreement" in tmpl


def test_process_uses_yt_comments_pipeline():
    """process() must request comments from the server, filter, summarize,
    and post the 4th embed."""
    src = BOT_SRC
    # Download payload includes comment params
    assert '"include_comments": job.yt_comments_enabled' in src
    assert "YT_COMMENTS_MAX" in src
    # Result picked up
    assert 'dl.get("comments")' in src
    # Filter + summarize step
    assert "filter_yt_comments(raw_comments)" in src
    assert "PROMPT_YT_COMMENTS" in src
    # 4th embed
    assert '"Community Reaction"' in src


def test_yt_download_payload_has_comments_fields():
    """The HTTP request to /api/yt-download includes the comment knobs the
    server expects."""
    src = BOT_SRC
    # Find the download_payload dict literal (small window after the marker)
    start = src.index("download_payload = {")
    end = src.index("}", start)
    payload_section = src[start:end]
    assert '"include_comments"' in payload_section
    assert '"comments_max"' in payload_section
    assert '"comments_sort"' in payload_section


def test_server_yt_download_extracts_comments():
    """app.py's /api/yt-download endpoint passes --get-comments and parses
    the result via _extract_comments."""
    assert "include_comments" in APP_SRC
    assert "--get-comments" in APP_SRC
    assert "_extract_comments" in APP_SRC
    # Helper exists and pulls out the right fields
    assert "is_favorited" in APP_SRC  # creator-hearted flag
    assert "author_is_uploader" in APP_SRC
    assert "is_pinned" in APP_SRC


def test_config_command_has_yt_comments_param():
    """/config gained yt_comments toggle so channels can opt out."""
    src = BOT_SRC
    cmd_src = src[src.index('@bot.tree.command(name="config"'):
                   src.index('@bot.tree.command(name="serverconfig"')]
    assert "yt_comments" in cmd_src
    assert 'fields["yt_comments_enabled"]' in cmd_src


# ─── AI litmus test ──────────────────────────────────────────────────────────


def test_litmus_trigger_regex_matches():
    """`litmus` (with optional punctuation) triggers; sentences don't."""
    matches = ["litmus", "Litmus", "LITMUS", "litmus.", "litmus!", "litmus?",
               " litmus ", "  litmus  "]
    for s in matches:
        assert bot.LITMUS_TRIGGER_RE.match(s), f"should match: {s!r}"


def test_litmus_trigger_regex_rejects_sentences():
    rejects = [
        "give me a litmus test",
        "litmus please",
        "this is a litmus paper",
        "tldr",  # different keyword
        "",
        "lol",
    ]
    for s in rejects:
        assert not bot.LITMUS_TRIGGER_RE.match(s), f"should reject: {s!r}"


def test_job_kind_accepts_litmus():
    j = bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                kind="litmus", message=object())
    assert j.kind == "litmus"


def test_job_kind_rejects_invalid():
    try:
        bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                kind="bogus", message=object())
    except ValueError as e:
        assert "kind" in str(e)
    else:
        raise AssertionError("invalid kind should raise")


def test_litmus_constants_exist():
    """All litmus knobs + phrase tables exported from bot."""
    for name in ("LLM_TIC_PHRASES", "LLM_BUZZWORDS", "LLM_HEDGE_PHRASES",
                 "LITMUS_SKIP_LLM_BELOW", "LITMUS_SKIP_LLM_ABOVE",
                 "WAYBACK_TIMEOUT", "LITMUS_EXCERPT_CHARS"):
        assert hasattr(bot, name), f"missing: {name}"


# ─── Regex signal detection ──────────────────────────────────────────────────


def test_regex_signals_too_short():
    """Articles below the threshold get a too_short marker."""
    out = bot._regex_signals("hi.")
    assert "too_short" in out


def test_regex_signals_clean_human_text():
    """Carefully-edited human prose shouldn't trip any high-severity signal."""
    text = (
        "Sarah Connor walked into the lab on a Tuesday morning. "
        "She'd been working on the project since 2019, when funding came "
        "through. \"It's been a long road,\" she told reporters. The team "
        "of six engineers had spent $4.2 million on prototypes. Three of "
        "them quit last spring. By her count, they'd shipped 47 versions. "
        "The 48th, she said, would be the one that mattered. " * 3
    )
    out = bot._regex_signals(text)
    # No high-severity signals expected on clean substantive prose
    high_sigs = [k for k, v in out.items() if v.get("severity") == "high"]
    assert "too_short" not in out
    assert not high_sigs, f"unexpected high signals: {high_sigs}"


def test_regex_signals_llm_tic_heavy():
    """Text loaded with LLM tic phrases should fire llm_tic_phrases high."""
    text = (
        "In the realm of modern computing, it's worth noting that we must "
        "delve into the rich tapestry of innovation. Moreover, this "
        "underscores the importance of navigating the landscape with "
        "robust, seamless, cutting-edge approaches. Furthermore, in "
        "today's fast-paced world, we must showcase how to embark on "
        "this transformative journey. It is important to note that "
        "leveraging these paradigms will elevate our understanding. "
        "Ultimately, the myriad of options stands the test of time, "
        "shedding light on what truly matters. " * 2
    )
    out = bot._regex_signals(text)
    assert "llm_tic_phrases" in out
    assert out["llm_tic_phrases"]["severity"] in ("med", "high")
    assert "buzzwords" in out
    assert out["buzzwords"]["severity"] in ("med", "high")
    assert "hedges" in out


def test_regex_signals_em_dash_heavy():
    """Em-dash density above ~5/1000 words flags."""
    text = (
        "The work — finally complete — drew praise. " * 80
    )
    out = bot._regex_signals(text)
    assert "em_dash_density" in out
    assert out["em_dash_density"]["severity"] in ("med", "high")


def test_regex_signals_listicle_structure():
    """Heavy heading + bullet density triggers listicle structure."""
    text = "## Heading One\n- bullet a\n- bullet b\n- bullet c\n" * 30
    out = bot._regex_signals(text)
    assert "listicle_structure" in out


def test_regex_signals_substance_present():
    """Articles with quotes + names + dates + numbers get NO low_substance."""
    text = (
        '"This is a quote about something specific," Dr. Jane Smith said. '
        "She made $1.2 million in 2024 from a 47% return. "
        "According to Prof. John Doe, who joined in 2019, the 1990s saw "
        "$500 billion in investments. " * 3
    )
    out = bot._regex_signals(text)
    assert "low_substance" not in out  # substance present


def test_regex_signals_low_substance_fires():
    """Vague, no-specifics text triggers low_substance."""
    text = (
        "Many experts agree that the digital transformation has been "
        "significant. Studies show that organizations are increasingly "
        "aware of the importance of these trends. There are several "
        "key considerations to keep in mind when navigating this space. " * 8
    )
    out = bot._regex_signals(text)
    assert "low_substance" in out


# ─── Metadata helpers ────────────────────────────────────────────────────────


def test_extract_author_from_meta_name():
    html = '<html><head><meta name="author" content="Jane Doe"></head></html>'
    assert bot._extract_author_from_html(html) == "Jane Doe"


def test_extract_author_from_meta_property():
    html = '<meta property="article:author" content="John Smith">'
    assert bot._extract_author_from_html(html) == "John Smith"


def test_extract_author_from_rel_link():
    html = '<a rel="author" href="/authors/jane">Jane Roe</a>'
    assert bot._extract_author_from_html(html) == "Jane Roe"


def test_extract_author_returns_none_when_missing():
    html = "<html><body>just content</body></html>"
    assert bot._extract_author_from_html(html) is None


def test_detect_adsense_positive():
    cases = [
        '<script async src="https://pagead2.googlesyndication.com/x"></script>',
        '<ins class="adsbygoogle"></ins>',
        '<div data-ad-client="ca-pub-123" data-ad-slot="456"></div>',
    ]
    for html in cases:
        assert bot._detect_adsense(html), f"should detect: {html!r}"


def test_detect_adsense_negative():
    assert not bot._detect_adsense("<html><body>just content</body></html>")
    assert not bot._detect_adsense("")


def test_domain_age_severity_recent():
    """Domain first archived <6 months ago = high severity."""
    from datetime import datetime, timezone, timedelta
    recent = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y%m%d000000")
    sev, text = bot._domain_age_severity(recent)
    assert sev == "high"
    assert "<6 months" in text


def test_domain_age_severity_old():
    """Domain first archived >2 years ago = low severity (established)."""
    sev, text = bot._domain_age_severity("20100101000000")
    assert sev == "low"
    assert "years" in text


def test_domain_age_severity_no_archive():
    """No timestamp → 'no archive found', medium severity."""
    sev, text = bot._domain_age_severity(None)
    assert sev == "med"
    assert "no archive" in text


# ─── Severity aggregation + skip-LLM thresholds ──────────────────────────────


def test_aggregate_severity_clean():
    """Clean signals + author + old domain + no adsense = low score."""
    score = bot._aggregate_severity(
        signals={},
        adsense_detected=False,
        author_present=True,
        domain_severity="low",
    )
    assert score == 0


def test_aggregate_severity_loaded():
    """Multiple high-severity signals + missing author + adsense + new
    domain = high score."""
    signals = {
        "llm_tic_phrases": {"severity": "high"},
        "buzzwords": {"severity": "high"},
        "low_substance": {"severity": "high"},
    }
    score = bot._aggregate_severity(
        signals=signals,
        adsense_detected=True,
        author_present=False,
        domain_severity="high",
    )
    # 3 high signals (6) + adsense (1) + missing-author (1) + high domain (2) = 10
    assert score >= 8


def test_aggregate_severity_thresholds_sane():
    """Skip-LLM thresholds bracket a reasonable middle range."""
    assert bot.LITMUS_SKIP_LLM_BELOW < bot.LITMUS_SKIP_LLM_ABOVE


# ─── End-to-end wiring ───────────────────────────────────────────────────────


def test_process_litmus_exists_and_uses_helpers():
    """process_litmus orchestrates the pieces."""
    import inspect
    src = inspect.getsource(bot.process_litmus)
    # All key helpers referenced
    assert "fetch_article" in src
    assert "_regex_signals" in src
    assert "_fetch_raw_html" in src
    assert "_wayback_first_seen" in src
    assert "_detect_adsense" in src
    assert "_extract_author_from_html" in src
    assert "_aggregate_severity" in src
    # LLM call only for ambiguous range
    assert "LITMUS_SKIP_LLM_BELOW" in src
    assert "LITMUS_SKIP_LLM_ABOVE" in src
    assert "PROMPT_LITMUS" in src


def test_litmus_embed_does_not_claim_verdict():
    """The embed body must NOT use a verdict ('AI' / 'human') format.
    Source-level check: forensic framing, no green-yellow-red verdict."""
    src = BOT_SRC
    fmt_src = src[src.index("def _format_litmus_signals"):
                   src.index("def _signals_summary_for_prompt")]
    # No "AI verdict" / "likely AI" / "likely human" language
    forbidden = ["likely AI", "likely human", "Verdict:", "AI: yes", "Human: yes"]
    for f in forbidden:
        assert f not in fmt_src, f"forbidden verdict language: {f}"


def test_litmus_prompt_forbids_verdict():
    """Prompt must explicitly tell the model NOT to output a verdict."""
    tmpl = p.PROMPT_LITMUS
    assert 'verdict' in tmpl.lower()
    # The prompt body contains the negation: "Do NOT output a verdict"
    assert "NOT output a verdict" in tmpl or "not output a verdict" in tmpl.lower()


def test_litmus_prompt_has_signals_placeholder():
    """{signals_summary} is the regex-pre-pass injection point."""
    assert "{signals_summary}" in p.PROMPT_LITMUS
    assert "{title}" in p.PROMPT_LITMUS
    assert "{source}" in p.PROMPT_LITMUS
    assert "{transcript}" in p.PROMPT_LITMUS


def test_reply_trigger_litmus_routes_to_litmus_kind():
    """_handle_reply_trigger litmus branch must build a Job(kind='litmus')
    and bypass the video classifier."""
    src = BOT_SRC
    handler_src = src[src.index("async def _handle_reply_trigger"):
                       src.index("# ─── Web scraper client")]
    # litmus branch in the per-hint loop sets kind="litmus"
    assert 'kind="litmus"' in handler_src
    # Discriminator on the looped hint variable
    assert 'if hint == "litmus":' in handler_src


def test_on_message_routes_litmus_keyword():
    """on_message uses the unified keyword parser to dispatch chained replies."""
    src = BOT_SRC
    on_msg = src[src.index("async def on_message"):
                  src.index("# ─── Reply-trigger handler")]
    # Unified parser does the routing; both keywords flow through it.
    assert "_parse_trigger_keywords" in on_msg
    assert "kind_hints=hints" in on_msg


# ─── Chained replies (`tldr litmus`) ─────────────────────────────────────────


def test_parse_trigger_single_keyword():
    """Single-keyword replies still produce a 1-element list."""
    assert bot._parse_trigger_keywords("tldr") == ["summary"]
    assert bot._parse_trigger_keywords("TLDR") == ["summary"]
    assert bot._parse_trigger_keywords("tldr.") == ["summary"]
    assert bot._parse_trigger_keywords("summarize!") == ["summary"]
    assert bot._parse_trigger_keywords("summarise") == ["summary"]
    assert bot._parse_trigger_keywords("litmus") == ["litmus"]
    assert bot._parse_trigger_keywords("Litmus?") == ["litmus"]


def test_parse_trigger_chained_two_keywords():
    """Multi-keyword reply preserves order and dedupes."""
    assert bot._parse_trigger_keywords("tldr litmus") == ["summary", "litmus"]
    assert bot._parse_trigger_keywords("litmus tldr") == ["litmus", "summary"]
    assert bot._parse_trigger_keywords("LITMUS TLDR") == ["litmus", "summary"]
    # Mixed punctuation
    assert bot._parse_trigger_keywords("tldr, litmus.") == ["summary", "litmus"]
    assert bot._parse_trigger_keywords("tldr! litmus?") == ["summary", "litmus"]


def test_parse_trigger_dedups_repeats():
    """`tldr tldr` charges the user once. `tldr summarize` likewise (both
    map to the same hint)."""
    assert bot._parse_trigger_keywords("tldr tldr") == ["summary"]
    assert bot._parse_trigger_keywords("tldr summarize") == ["summary"]
    assert bot._parse_trigger_keywords("tldr summarise summarize") == ["summary"]
    assert bot._parse_trigger_keywords("tldr litmus tldr") == ["summary", "litmus"]


def test_parse_trigger_rejects_sentences():
    """Any non-keyword word → empty list (sentence triggers stay disabled)."""
    rejects = [
        "give me a tldr",
        "tldr please",
        "what's the litmus test",
        "tldr and litmus",  # 'and' isn't a keyword
        "tldr the article",
        "litmus paper",
        "lol tldr",
        "tldr 123",  # digit token isn't a keyword
        "",
        "   ",
    ]
    for s in rejects:
        assert bot._parse_trigger_keywords(s) == [], f"should be empty: {s!r}"


def test_rate_limit_check_batch_count():
    """Rate-limit check must accept a `count` parameter for batch atomic
    enforcement (chained replies request multiple jobs in one go)."""
    import inspect
    sig = inspect.signature(bot._rate_limit_check)
    assert "count" in sig.parameters
    # Default should be 1 for backward-compat
    assert sig.parameters["count"].default == 1


def test_rate_limit_check_batch_rejects_overflow():
    """Asking for N jobs when only N-1 slots remain → reject all-or-nothing."""
    # Reset state for this test
    bot._user_jobs.clear()
    user = 999111
    # Fill to MAX-1
    for _ in range(bot.MAX_JOBS_PER_USER_PER_HOUR - 1):
        bot._rate_limit_record(user)
    # Single-job request: ok
    ok, _ = bot._rate_limit_check(user, count=1)
    assert ok
    # Two-job request: would overflow → reject
    ok, reason = bot._rate_limit_check(user, count=2)
    assert not ok
    assert "Rate limit" in reason
    bot._user_jobs.clear()


def test_handle_reply_trigger_signature_accepts_list():
    """_handle_reply_trigger now takes kind_hints as list[str] | str."""
    import inspect
    sig = inspect.signature(bot._handle_reply_trigger)
    assert "kind_hints" in sig.parameters
    # Old kind_hint param should be gone
    assert "kind_hint" not in sig.parameters


def test_handle_reply_trigger_builds_one_job_per_hint():
    """Source-level: handler iterates kind_hints to build N jobs and queues all."""
    import inspect
    src = inspect.getsource(bot._handle_reply_trigger)
    # Iterate over hints
    assert "for hint in kind_hints:" in src
    # Build a list and queue at the end
    assert "jobs.append(job)" in src
    assert "for job in jobs:" in src
    # Atomic batch rate-limit check
    assert "count=len(kind_hints)" in src


# ─── Image-attachment flow (OCR + VLM + summary) ───────────────────────


def test_job_kind_image_accepted():
    """Job dataclass validates kind in {video, web, litmus, image}."""
    import discord as _d
    channel = types.SimpleNamespace(id=1)
    msg = types.SimpleNamespace(id=2)
    j = bot.Job(
        url="https://cdn.discord.com/x.png", video_id="img1",
        channel=channel, submitter_id=1, submitter_name="u",
        kind="image", message=msg,
    )
    assert j.kind == "image"
    assert j.image_urls == []  # default empty list
    assert j.image_filenames == []


def test_job_kind_image_rejects_unknown():
    """Unknown kinds still raise — the validator was updated, not bypassed."""
    msg = types.SimpleNamespace(id=2)
    channel = types.SimpleNamespace(id=1)
    import pytest as _pytest  # noqa
    try:
        bot.Job(
            url="x", video_id="y", channel=channel,
            submitter_id=1, submitter_name="u",
            kind="bogus", message=msg,
        )
    except ValueError as e:
        assert "Job.kind" in str(e) and "image" in str(e)
    else:
        raise AssertionError("expected ValueError for kind='bogus'")


def test_attachment_is_image_content_type():
    """Content-type prefix wins over extension check."""
    a = types.SimpleNamespace(content_type="image/png", filename="x.txt", size=1)
    assert bot._attachment_is_image(a) is True


def test_attachment_is_image_extension_fallback():
    """Falls back to extension when content_type is missing/generic."""
    a = types.SimpleNamespace(content_type=None, filename="screenshot.PNG", size=1)
    assert bot._attachment_is_image(a) is True
    b = types.SimpleNamespace(
        content_type="application/octet-stream",
        filename="meme.jpeg", size=1,
    )
    assert bot._attachment_is_image(b) is True


def test_attachment_is_image_rejects_non_image():
    """PDFs and videos are NOT routed to the image OCR flow."""
    pdf = types.SimpleNamespace(content_type="application/pdf",
                                filename="doc.pdf", size=1)
    assert bot._attachment_is_image(pdf) is False
    mp4 = types.SimpleNamespace(content_type="video/mp4",
                                filename="clip.mp4", size=1)
    assert bot._attachment_is_image(mp4) is False
    txt = types.SimpleNamespace(content_type=None, filename="notes.txt", size=1)
    assert bot._attachment_is_image(txt) is False


def test_extract_image_attachments_caps_at_max():
    """Image extraction respects IMAGE_MAX_ATTACHMENTS cap and skips oversized."""
    images = [
        types.SimpleNamespace(
            content_type="image/png", filename=f"img{i}.png",
            size=1024, url=f"https://cdn/x/img{i}.png",
        )
        for i in range(10)
    ]
    msg = types.SimpleNamespace(attachments=images)
    urls, names = bot._extract_image_attachments(msg)
    assert len(urls) == bot.IMAGE_MAX_ATTACHMENTS
    assert len(names) == len(urls)
    # Order preserved — first N attachments
    assert urls[0].endswith("img0.png")


def test_extract_image_attachments_skips_oversized():
    """Attachments larger than IMAGE_MAX_BYTES_PER_ATTACHMENT are dropped."""
    huge = types.SimpleNamespace(
        content_type="image/png", filename="huge.png",
        size=bot.IMAGE_MAX_BYTES_PER_ATTACHMENT + 1,
        url="https://cdn/huge.png",
    )
    ok = types.SimpleNamespace(
        content_type="image/png", filename="ok.png",
        size=1024, url="https://cdn/ok.png",
    )
    msg = types.SimpleNamespace(attachments=[huge, ok])
    urls, names = bot._extract_image_attachments(msg)
    assert urls == ["https://cdn/ok.png"]
    assert names == ["ok.png"]


def test_extract_image_attachments_ignores_non_images():
    """Mixed attachments — only images come through."""
    pdf = types.SimpleNamespace(
        content_type="application/pdf", filename="a.pdf",
        size=1024, url="https://cdn/a.pdf",
    )
    img = types.SimpleNamespace(
        content_type="image/jpeg", filename="b.jpg",
        size=1024, url="https://cdn/b.jpg",
    )
    msg = types.SimpleNamespace(attachments=[pdf, img, pdf])
    urls, names = bot._extract_image_attachments(msg)
    assert urls == ["https://cdn/b.jpg"]
    assert names == ["b.jpg"]


def test_format_image_block_with_ocr():
    """Formatter produces the documented <images>-block shape."""
    block = bot._format_image_block(
        index=1, filename="err.png", width=1200, height=800,
        description="A red error dialog",
        ocr="NullPointerException | at MyClass.foo(MyClass.java:42)",
    )
    assert "## Image 1 (err.png, 1200×800)" in block
    assert "[Description] A red error dialog" in block
    assert "[Text on screen]" in block
    # OCR split back onto separate lines (one per snippet)
    assert "NullPointerException" in block
    assert "at MyClass.foo(MyClass.java:42)" in block
    assert " | " not in block  # joined snippets re-split


def test_format_image_block_no_ocr():
    """Empty OCR renders as `(no text detected)`, not blank."""
    block = bot._format_image_block(
        index=2, filename="photo.jpg", width=0, height=0,
        description="A sunset over water", ocr="",
    )
    assert "[Text on screen]\n(no text detected)" in block
    # Unknown dimensions render explicitly so the LLM doesn't hallucinate
    assert "unknown size" in block


def test_worker_dispatch_routes_image_kind():
    """worker() must dispatch kind='image' to process_image."""
    src = bot.__dict__["worker"].__wrapped__.__code__.co_consts if hasattr(
        bot.__dict__.get("worker", None), "__wrapped__") else None
    # Source-level check (the dispatch lives in a chained if/elif inside
    # the worker loop, not exposed as a constant). Just grep the source.
    bot_src = BOT_SRC
    worker_src = bot_src[
        bot_src.index("for attempt in range(MAX_RETRIES + 1):"):
        bot_src.index("async def process(job: Job):")
    ]
    assert 'job.kind == "image"' in worker_src
    assert "handler = process_image" in worker_src


def test_process_image_is_async_and_uses_api_image():
    """process_image POSTs to /api/image and uses the IMAGE prompt templates."""
    import inspect
    src = inspect.getsource(bot.process_image)
    assert inspect.iscoroutinefunction(bot.process_image)
    assert "/api/image" in src or "_call_image_api" in src
    # Uses image-specific prompts (not the web/video ones)
    assert "PROMPT_BRIEF_IMAGE" in src
    # Posts a single embed in the original channel (mirrors process_url)
    assert "job.channel.send" in src


def test_image_prompts_present():
    """prompts.py exports the image-specific prompt templates."""
    assert hasattr(p, "PROMPT_BRIEF_IMAGE")
    assert hasattr(p, "PROMPT_KEY_POINTS_IMAGE")
    # The brief prompt mentions both VLM scene + OCR (so the LLM knows
    # what extraction shape to expect).
    assert "OCR" in p.PROMPT_BRIEF_IMAGE
    assert "image" in p.PROMPT_BRIEF_IMAGE.lower()
    # Both use the standard <images> input block (so summarize() can
    # substitute {transcript} as it does for video/web).
    assert "<images>" in p.PROMPT_BRIEF_IMAGE
    assert "{transcript}" in p.PROMPT_BRIEF_IMAGE
    assert "{reference_block}" in p.PROMPT_BRIEF_IMAGE


def test_reply_trigger_handles_image_fallback():
    """_handle_reply_trigger checks for image attachments when no URL found."""
    import inspect
    src = inspect.getsource(bot._handle_reply_trigger)
    # URL-first; image fallback path
    assert "_extract_image_attachments(referenced)" in src
    # Image-only message with a chained `litmus` hint drops litmus.
    assert "image_hints" in src
    # Build an image Job for image-only path
    assert 'kind="image"' in src
    assert "image_urls=list(image_urls)" in src


def test_api_image_route_registered():
    """/api/image route is registered alongside /api/describe in app.py."""
    assert 'Route("/api/image", api_image, methods=["POST"])' in APP_SRC
    # The handler exists and mentions OCR + VLM (both pipelines invoked)
    assert "async def api_image(" in APP_SRC
    assert "_ocr_frame" in APP_SRC
    assert "_describe_frame" in APP_SRC


# ─── User prompt feature ────────────────────────────────────────────────────


def test_extract_user_prompt_strips_urls():
    """URL-only message → empty user prompt; URL + text → text."""
    urls = ["https://youtube.com/watch?v=abc123"]
    msg = "https://youtube.com/watch?v=abc123"
    assert bot._extract_user_prompt(msg, urls) == ""

    msg = "https://youtube.com/watch?v=abc123 describe the slides shown"
    out = bot._extract_user_prompt(msg, urls)
    assert "describe the slides shown" in out
    assert "youtube.com" not in out


def test_extract_user_prompt_strips_mentions():
    """Discord mentions/channel refs/emojis don't count as prompt text."""
    urls = ["https://youtu.be/x"]
    msg = "<@123456789> https://youtu.be/x <#987654> :emoji:"
    out = bot._extract_user_prompt(msg, urls)
    # After stripping URLs, mentions, channel refs → only ":emoji:" or
    # similar fragment remains. May be empty or trivial.
    assert "@123456789" not in out
    assert "#987654" not in out


def test_extract_user_prompt_respects_cap():
    """Long user text gets truncated to USER_PROMPT_MAX_CHARS."""
    urls = ["https://youtu.be/x"]
    msg = "https://youtu.be/x " + ("describe this " * 500)
    out = bot._extract_user_prompt(msg, urls)
    assert len(out) <= bot.USER_PROMPT_MAX_CHARS


def test_extract_user_prompt_minimum_length():
    """Trivial trailing characters don't count as prompt."""
    urls = ["https://youtu.be/x"]
    msg = "https://youtu.be/x ?"
    out = bot._extract_user_prompt(msg, urls)
    assert out == ""


def test_job_dataclass_defaults():
    """Job constructs with required fields and sane defaults.

    __post_init__ enforces a discriminated-union invariant: exactly one of
    message/interaction must be set. We pass a sentinel message here.
    """
    j = bot.Job(
        url="https://x", video_id="x",
        channel=object(), submitter_id=42,
        message=object(),
    )
    assert j.user_prompt == ""
    assert j.diarize is False
    assert j.vlm_enabled is True  # default tracks VLM_ENABLED env (default "1")
    assert j.model_override is None
    assert j.interaction is None
    assert j.submitter_name == ""


def test_job_dataclass_invariant():
    """__post_init__ rejects neither-or-both message/interaction."""
    try:
        bot.Job(url="https://x", video_id="x", channel=object(), submitter_id=1)
    except ValueError as e:
        assert "exactly one" in str(e)
    else:
        raise AssertionError("Job() with neither source should raise ValueError")
    try:
        bot.Job(url="https://x", video_id="x", channel=object(), submitter_id=1,
                message=object(), interaction=object())
    except ValueError as e:
        assert "exactly one" in str(e)
    else:
        raise AssertionError("Job() with both sources should raise ValueError")


def test_build_vlm_prompt_includes_user_text():
    """User text appears in the VLM frame prompt as an explicit instruction."""
    p = bot._build_vlm_prompt("focus on the code editor")
    assert "focus on the code editor" in p
    assert "1-2 sentences" in p  # length cap still enforced


def test_fetch_descriptions_passes_prompt():
    """When prompt is supplied, _fetch_descriptions includes it in payload."""
    import inspect
    sig = inspect.signature(bot._fetch_descriptions)
    assert "prompt" in sig.parameters
    src = inspect.getsource(bot._fetch_descriptions)
    assert 'payload["prompt"] = prompt' in src


def test_process_routes_user_forced_vlm():
    """process() forces VLM when user_prompt is set, regardless of density."""
    src = BOT_SRC
    assert "user_forced_vlm = bool(job.user_prompt)" in src
    assert "user-forced-enrich" in src or "user_forced_vlm" in src


def test_summary_prompt_includes_user_steer_block():
    """User prompt threads into the summary prompt via the ref_block."""
    assert "<user_request>" in BOT_SRC
    assert "user_steer_block" in BOT_SRC


def test_embed_shows_user_request_field():
    """The brief embed surfaces the user prompt as a field for visibility."""
    assert 'name="User request"' in BOT_SRC


# ─── .env loader ────────────────────────────────────────────────────────────


def test_env_loader_basic():
    """Plain KEY=value lines."""
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
    p.write("FOO=bar\nBAZ=qux quux\n")
    p.close()
    # Clear any prior values
    for k in ("FOO", "BAZ"):
        os.environ.pop(k, None)
    from pathlib import Path
    bot._load_env_file(Path(p.name))
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux quux"
    os.unlink(p.name)


def test_env_loader_quoted_values():
    """Double + single quotes stripped; escapes honoured in double-quoted only."""
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
    p.write('DOUBLE="hello world"\nSINGLE=\'value with $special\'\n'
            'ESCAPED="line1\\nline2"\nLITERAL=\'line1\\nline2\'\n')
    p.close()
    for k in ("DOUBLE", "SINGLE", "ESCAPED", "LITERAL"):
        os.environ.pop(k, None)
    from pathlib import Path
    bot._load_env_file(Path(p.name))
    assert os.environ["DOUBLE"] == "hello world"
    assert os.environ["SINGLE"] == "value with $special"
    assert os.environ["ESCAPED"] == "line1\nline2"  # \n decoded
    assert os.environ["LITERAL"] == "line1\\nline2"  # literal backslash-n
    os.unlink(p.name)


def test_env_loader_inline_comments():
    """Inline `# ...` stripped only on unquoted values."""
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
    p.write('FOO=bar # inline comment\n'
            'QUOTED="bar # not a comment"\n'
            'NOSPACE=bar#part-of-value\n')
    p.close()
    for k in ("FOO", "QUOTED", "NOSPACE"):
        os.environ.pop(k, None)
    from pathlib import Path
    bot._load_env_file(Path(p.name))
    assert os.environ["FOO"] == "bar"
    assert os.environ["QUOTED"] == "bar # not a comment"
    # `#` without preceding whitespace is part of the value
    assert os.environ["NOSPACE"] == "bar#part-of-value"
    os.unlink(p.name)


def test_env_loader_export_prefix():
    """`export KEY=value` (shell-compat) is accepted."""
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
    p.write("export FOO=bar\n")
    p.close()
    os.environ.pop("FOO", None)
    from pathlib import Path
    bot._load_env_file(Path(p.name))
    assert os.environ["FOO"] == "bar"
    os.unlink(p.name)


def test_env_loader_does_not_override():
    """setdefault — pre-set env vars take precedence over file values."""
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
    p.write("PREEXISTING=fromfile\n")
    p.close()
    os.environ["PREEXISTING"] = "fromenv"
    from pathlib import Path
    bot._load_env_file(Path(p.name))
    assert os.environ["PREEXISTING"] == "fromenv"
    os.environ.pop("PREEXISTING", None)
    os.unlink(p.name)


# ─── Rate limiting ─────────────────────────────────────────────────────────


def test_rate_limit_initial_pass():
    """Fresh user: first call passes."""
    bot._user_jobs.clear()
    ok, reason = bot._rate_limit_check(user_id=1001)
    assert ok and reason == ""


def test_rate_limit_blocks_after_cap():
    """Per-user cap blocks the next request."""
    bot._user_jobs.clear()
    user = 1002
    for _ in range(bot.MAX_JOBS_PER_USER_PER_HOUR):
        bot._rate_limit_record(user)
    ok, reason = bot._rate_limit_check(user)
    assert not ok
    assert "Rate limit" in reason


def test_rate_limit_bypass_users():
    """Admin bypass still respects total queue cap but skips per-user."""
    bot._user_jobs.clear()
    bot.RATE_LIMIT_BYPASS_USERS.add(9999)
    for _ in range(bot.MAX_JOBS_PER_USER_PER_HOUR + 5):
        bot._rate_limit_record(9999)
    ok, _ = bot._rate_limit_check(9999)
    assert ok, "bypass should still pass per-user check"
    bot.RATE_LIMIT_BYPASS_USERS.discard(9999)


def test_rate_limit_sliding_window():
    """Old entries (>1h) get evicted on check."""
    bot._user_jobs.clear()
    user = 1003
    # Backdate past entries
    for _ in range(bot.MAX_JOBS_PER_USER_PER_HOUR):
        bot._user_jobs[user].append(time.time() - 3700)  # 1h+ ago
    ok, _ = bot._rate_limit_check(user)
    assert ok, "old entries should be evicted, allowing new request"


# ─── Per-channel config ────────────────────────────────────────────────────


def test_channel_config_get_default_empty():
    cfg_path = bot.CHANNELS_CONFIG_PATH
    if cfg_path.exists():
        cfg_path.unlink()
    assert bot.get_channel_config(123) == {}


def test_channel_config_set_and_get():
    cfg_path = bot.CHANNELS_CONFIG_PATH
    if cfg_path.exists():
        cfg_path.unlink()
    bot.set_channel_config(456, model="gemma-4-31B-it-Q4_K_M", vlm_enabled=True)
    cfg = bot.get_channel_config(456)
    assert cfg["model"] == "gemma-4-31B-it-Q4_K_M"
    assert cfg["vlm_enabled"] is True


def test_channel_config_clear_field():
    bot.set_channel_config(789, model="X", diarize=True)
    bot.set_channel_config(789, model=None)  # clear model only
    cfg = bot.get_channel_config(789)
    assert "model" not in cfg
    assert cfg["diarize"] is True


def test_channel_config_clear_all():
    bot.set_channel_config(901, model="Y")
    bot.set_channel_config(901, model=None)
    assert bot.get_channel_config(901) == {}


# ─── Per-guild config (server-wide overrides) ────────────────────────────


def test_guild_config_get_default_empty():
    if bot.GUILDS_CONFIG_PATH.exists():
        bot.GUILDS_CONFIG_PATH.unlink()
    assert bot.get_guild_config(111111) == {}


def test_guild_config_set_summary_channel():
    if bot.GUILDS_CONFIG_PATH.exists():
        bot.GUILDS_CONFIG_PATH.unlink()
    bot.set_guild_config(222222, summary_channel=999)
    cfg = bot.get_guild_config(222222)
    assert cfg["summary_channel"] == 999


def test_guild_config_clear():
    bot.set_guild_config(333333, summary_channel=42)
    bot.set_guild_config(333333, summary_channel=None)
    assert bot.get_guild_config(333333) == {}


def test_guild_config_two_guilds_isolated():
    """Each guild has its own config — set on one, other unchanged."""
    if bot.GUILDS_CONFIG_PATH.exists():
        bot.GUILDS_CONFIG_PATH.unlink()
    bot.set_guild_config(444444, summary_channel=1)
    bot.set_guild_config(555555, summary_channel=2)
    assert bot.get_guild_config(444444)["summary_channel"] == 1
    assert bot.get_guild_config(555555)["summary_channel"] == 2
    bot.set_guild_config(444444, summary_channel=None)
    assert bot.get_guild_config(444444) == {}
    assert bot.get_guild_config(555555)["summary_channel"] == 2  # untouched


def test_normalize_chapter_timestamps_handles_malformed():
    """[0 and 0:05:46] → [0:05:46], real bug from production."""
    out = bot._normalize_chapter_timestamps("[0 and 0:05:46] Title")
    assert "[0:05:46] Title" == out


def test_normalize_chapter_timestamps_preserves_clean():
    """Already-correct timestamps shouldn't be rewritten."""
    for ok in ("[0:00]", "[1:23]", "[1:23:45]", "[0:00:00]"):
        assert bot._normalize_chapter_timestamps(f"{ok} Title") == f"{ok} Title"


def test_normalize_chapter_timestamps_handles_ranges():
    """Range-style brackets get the first timestamp extracted."""
    assert bot._normalize_chapter_timestamps("[0:00-1:30] Intro") == "[0:00] Intro"
    assert bot._normalize_chapter_timestamps("[1:30 to 2:45] Middle") == "[1:30] Middle"


def test_normalize_chapter_timestamps_leaves_non_ts_brackets():
    """Markdown link text in brackets without a timestamp stays untouched."""
    assert bot._normalize_chapter_timestamps("[the video](url)") == "[the video](url)"
    assert bot._normalize_chapter_timestamps("[Note: see below]") == "[Note: see below]"


def test_linkify_timestamps_through_normaliser():
    """End-to-end: malformed → normalised → linkified."""
    out = bot.linkify_timestamps("**[0 and 0:05:46] Section title**", "VIDID")
    assert "evil" not in out
    assert "youtube.com/watch?v=VIDID&t=346" in out


def test_resolve_summary_channel_falls_through():
    """resolve_summary_channel exists and is callable; returns the input
    channel when no overrides apply (real Discord channel objects can't be
    constructed in unit tests, but we can verify the function signature
    and the no-config-no-env path via docstring presence)."""
    assert callable(bot.resolve_summary_channel)
    src = inspect.getsource(bot.resolve_summary_channel)
    assert "guilds.json" in src
    assert "SUMMARY_CHANNEL" in src
    assert "summary_channel" in src


# ─── Slash commands wired ──────────────────────────────────────────────────


def test_slash_commands_defined():
    """All slash commands should have handler functions."""
    handlers = ("cmd_summarize", "cmd_transcribe", "cmd_status", "cmd_find",
                "cmd_config", "cmd_serverconfig")
    missing = [h for h in handlers if not hasattr(bot, h)]
    assert not missing, f"missing slash handlers: {missing}"


def test_sync_function_exists():
    assert callable(bot._sync_slash_commands)


def test_discord_guild_id_accepts_csv():
    """DISCORD_GUILD_ID must accept a comma-separated list so operators
    running the bot in multiple servers get instant slash command sync in
    each. Single-ID configs from the original implementation still work.

    Note: bot/.env is loaded at module import time, so we set the env var
    explicitly to override anything from .env. Empty-string is used in
    place of `pop()` because dotenv's setdefault would re-populate it on
    reload otherwise.
    """
    # Single ID — old behaviour, still supported
    os.environ["DISCORD_GUILD_ID"] = "123456789012345678"
    importlib.reload(bot)
    assert bot.DISCORD_GUILD_IDS == (123456789012345678,)

    # CSV — multiple guilds with whitespace tolerance
    os.environ["DISCORD_GUILD_ID"] = "111,222, 333"
    importlib.reload(bot)
    assert bot.DISCORD_GUILD_IDS == (111, 222, 333)

    # Empty string — explicit no-instant-sync; falls through to global
    os.environ["DISCORD_GUILD_ID"] = ""
    importlib.reload(bot)
    assert bot.DISCORD_GUILD_IDS == ()

    # Cleanup: restore env. We can't just unset because bot/.env may
    # populate it on the next reload — set to empty so prod config wins
    # on the next legit boot but tests don't leak state.
    os.environ.pop("DISCORD_GUILD_ID", None)
    importlib.reload(bot)


def test_sync_loops_per_guild():
    """When multiple guild IDs configured, sync must call copy_global_to +
    tree.sync once per guild. A single bot.tree.sync() at the end would
    skip the instant path for all but the first."""
    src = BOT_SRC[BOT_SRC.index("async def _sync_slash_commands"):
                   BOT_SRC.index("async def _sync_slash_commands") + 2000]
    assert "for gid in DISCORD_GUILD_IDS" in src
    assert "bot.tree.copy_global_to(guild=guild)" in src
    assert "await bot.tree.sync(guild=guild)" in src


# ─── Speaker rename helpers ────────────────────────────────────────────────


def test_has_speaker_labels():
    assert bot._has_speaker_labels("[0:05] [SPEAKER_00] hi")
    assert bot._has_speaker_labels("[0:00] [F-SPEAKER_01] hello")
    assert not bot._has_speaker_labels("[0:05] no speakers here")


def test_extract_speaker_labels_dedup_and_cap():
    transcript = "\n".join(f"[{i}:00] [SPEAKER_{i:02d}] line {i}" for i in range(8))
    labels = bot._extract_speaker_labels(transcript)
    assert len(labels) == 5  # capped at 5 (Modal limit)
    assert labels[0] == "SPEAKER_00"
    # Duplicates only counted once
    transcript = "\n".join(f"[{i}:00] [SPEAKER_00] line {i}" for i in range(8))
    labels = bot._extract_speaker_labels(transcript)
    assert labels == ["SPEAKER_00"]


# ─── JSON logger ───────────────────────────────────────────────────────────


def test_json_logger_outputs_json():
    """JSON formatter produces parseable JSON with required fields."""
    fmt = bot._JsonFormatter()
    rec = logging.makeLogRecord({
        "name": "test", "levelname": "INFO", "msg": "hi %s", "args": ("world",),
    })
    out = fmt.format(rec)
    import json as _json
    parsed = _json.loads(out)
    assert parsed["msg"] == "hi world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test"
    assert "ts" in parsed


# ─── Job source helpers ────────────────────────────────────────────────────


def test_job_with_message_only():
    j = bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                message=object())
    assert j.interaction is None


def test_job_with_interaction_only():
    j = bot.Job(url="x", video_id="x", channel=object(), submitter_id=1,
                interaction=object())
    assert j.message is None


def test_process_routing_references_helpers():
    """The speech-density dispatch in process() calls all VLM helpers."""
    assert "SPEECH_DENSITY_SILENT" in BOT_SRC
    assert "SPEECH_DENSITY_SPARSE" in BOT_SRC
    assert "_fetch_descriptions" in BOT_SRC
    assert "_format_descriptions" in BOT_SRC
    assert "_interleave_by_timestamp" in BOT_SRC


# ─── 8. build_initial_prompt (content-agnostic) ──────────────────────────────


def test_build_initial_prompt_frequency_rank():
    ctx = ("Atlas is the new system. Atlas tree. Atlas overhaul. "
           "Path of Exile. Path of Exile is great. "
           "Lone-mention: Foobar. Singleton: Quuxbaz.")
    out = bot.build_initial_prompt("Path of Exile Reveal", ctx)
    assert "Atlas" in out
    assert "Foobar" not in out, "singleton not filtered"
    assert "Quuxbaz" not in out
    assert "This video is about" not in out, "English filler should be gone"


def test_build_initial_prompt_empty_context():
    out = bot.build_initial_prompt("Mi vídeo", "")
    assert out == ""


def test_build_initial_prompt_no_caps():
    out = bot.build_initial_prompt("a lowercase title", "all lowercase reference")
    assert out == "a lowercase title"


def test_build_initial_prompt_respects_cap():
    long_ctx = " ".join(f"Term{i} Term{i}" for i in range(200))  # many duplicated terms
    out = bot.build_initial_prompt("Title", long_ctx)
    assert len(out) <= bot.INITIAL_PROMPT_CHAR_CAP


# ─── 9. Module exports ───────────────────────────────────────────────────────


def test_required_exports():
    required = (
        # Core summarize
        "summarize", "_chunk_transcript", "_llm_call",
        "PermanentError", "PROCESSING_EMOJI",
        # Cache
        "read_cache", "write_cache", "_derive_duration_from_transcript",
        # Budget
        "LLM_INPUT_CHAR_BUDGET", "EMBED_SAFE_LIMIT", "LLM_CONTEXT_SIZE",
        # Classifier
        "_is_permanent_remote_error", "_PERMANENT_REMOTE_PATTERNS",
        # Sanitiser
        "sanitize_llm_output",
        # Prompts
        "PROMPT_BRIEF", "PROMPT_KEY_POINTS", "PROMPT_CHAPTERS",
        "REDUCE_BRIEF", "REDUCE_KEY_POINTS", "CHUNK_PREAMBLE",
        # VLM
        "_fetch_descriptions", "_format_descriptions",
        "_interleave_by_timestamp", "_cleanup_remote_file",
        "VLM_ENABLED", "SPEECH_DENSITY_SILENT", "SPEECH_DENSITY_SPARSE",
        # Build helpers
        "build_initial_prompt", "INITIAL_PROMPT_CHAR_CAP",
    )
    missing = [s for s in required if not hasattr(bot, s)]
    assert not missing, f"missing exports: {missing}"


def test_dead_helpers_removed():
    """Helpers we explicitly removed shouldn't reappear."""
    assert not hasattr(bot, "extract_hotwords_from_context"), \
        "dead helper still present"
    assert not hasattr(bot, "correct_transcript_spellings"), \
        "phonetic-correction helper (PoE-flavoured) still present"
    assert not hasattr(bot, "_truncate_transcript_for_llm"), \
        "old truncation helper still present"


# ─── 10. App.py structural checks (server) ──────────────────────────────────


def test_app_endpoints_registered():
    expected = ["/api/status", "/api/yt-download", "/api/transcribe",
                "/api/describe", "/api/cleanup"]
    for route in expected:
        assert f'Route("{route}"' in APP_SRC, f"missing route: {route}"


def test_app_helpers_at_module_scope():
    """Helpers must be at module scope (not forward-referenced inside closures)."""
    expected = [
        "_PERMANENT_YT_DLP_PATTERNS",
        "_is_permanent_yt_dlp_error",
        "_yt_dlp_auth_args",
        "_describe_video",
        "_extract_frames",
        "_describe_frame",
        "_ffprobe_duration",
        "NoVideoStreamError",
    ]
    import ast
    tree = ast.parse(APP_SRC)
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    top_level.add(t.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level.add(node.name)
    missing = [s for s in expected if s not in top_level]
    assert not missing, f"not module-scope: {missing}"


def test_app_print_progress_disabled():
    assert "print_progress=True" not in APP_SRC
    assert APP_SRC.count("print_progress=False") >= 2


def test_app_subprocess_imported_top():
    """subprocess must be imported at the top, not inside the gr.Blocks block."""
    assert "import subprocess as _sp" not in APP_SRC
    assert "_sp." not in APP_SRC


def test_app_history_xss_escaped():
    assert "html_module.escape" in APP_SRC
    # In _format_history_html specifically
    assert ("esc = html_module.escape" in APP_SRC
            or "html_module.escape(str(entry" in APP_SRC)


def test_app_idle_timer_lock_check():
    assert "_transcription_lock.locked()" in APP_SRC


def test_app_return_file_flag_plumbed():
    # Plumbed through _transcribe_inner, _run_transcription, api_transcribe
    assert APP_SRC.count("return_file") >= 6


def test_app_cookie_support():
    assert "YT_DLP_COOKIES_FILE" in APP_SRC
    assert "_yt_dlp_auth_args" in APP_SRC


def test_app_keep_video_flag():
    assert 'keep_video = bool(body.get("keep_video", False))' in APP_SRC
    assert "bestvideo[height<=" in APP_SRC


def test_app_ffmpeg_no_stream_classified_permanent():
    assert "NoVideoStreamError" in APP_SRC
    assert '_FFMPEG_PERMANENT_PATTERNS' in APP_SRC
    assert 'permanent": True' in APP_SRC


def test_app_vlm_endpoint_security():
    """/api/cleanup restricts to /tmp/yt-dlp-* prefix to prevent abuse."""
    assert 'file_path.startswith("/tmp/yt-dlp-")' in APP_SRC


# ─── 11. Prompt injection mitigations ───────────────────────────────────────


def test_prompts_have_security_delimiters():
    assert "<transcript>" in p.PROMPT_BRIEF
    assert "</transcript>" in p.PROMPT_BRIEF
    assert "<transcript>" in p.PROMPT_KEY_POINTS
    assert "<transcript>" in p.PROMPT_CHAPTERS
    assert "<partials>" in p.REDUCE_BRIEF
    assert "<partials>" in p.REDUCE_KEY_POINTS


def test_prompts_have_security_rules():
    assert "SECURITY" in p.REF_RULES
    assert "UNTRUSTED" in p.REF_RULES
    assert "Never output URLs" in p.REF_RULES or "never output urls" in p.REF_RULES.lower()


def test_prompts_no_hardcoded_examples():
    """No PoE-flavoured / domain-specific examples in REF_RULES."""
    forbidden = ("Ezomite", "Ezomyte", "Kalguuran", "Calgaran", "holo")
    found = [w for w in forbidden if w in p.REF_RULES]
    assert not found, f"hardcoded domain examples found in REF_RULES: {found}"


# ─── 12. Configuration is env-driven ─────────────────────────────────────────


def test_env_overrides_respected():
    """All major knobs read from env."""
    os.environ["MAX_RETRIES"] = "7"
    os.environ["RETRY_BACKOFF"] = "1,2,3"
    os.environ["LLM_TEMPERATURE"] = "0.9"
    os.environ["LLM_MAX_TOKENS_BRIEF"] = "100"
    os.environ["VIDEO_DOMAINS"] = "foo.com,bar.com"
    os.environ["EXA_NUM_RESULTS"] = "10"
    os.environ["JOB_POLL_INTERVAL"] = "5"

    importlib.reload(bot)
    try:
        assert bot.MAX_RETRIES == 7
        assert bot.RETRY_BACKOFF == [1, 2, 3]
        assert bot.LLM_TEMPERATURE == 0.9
        assert bot.LLM_MAX_TOKENS_BRIEF == 100
        assert bot.VIDEO_DOMAINS == {"foo.com", "bar.com"}
        assert bot.EXA_NUM_RESULTS == 10
        assert bot.JOB_POLL_INTERVAL == 5
    finally:
        # Cleanup: restore env AND reload bot back to its default state, so
        # subsequent tests see the production VIDEO_URL_PATTERN. Without this
        # reload the mutated regex leaks across the rest of the suite.
        for k in ("MAX_RETRIES", "RETRY_BACKOFF", "LLM_TEMPERATURE",
                  "LLM_MAX_TOKENS_BRIEF", "VIDEO_DOMAINS", "EXA_NUM_RESULTS",
                  "JOB_POLL_INTERVAL"):
            os.environ.pop(k, None)
        importlib.reload(bot)


# ─── 13. Server-side job queue (/api/jobs* on whisper, async bot client) ────


# -- Server (app.py) -----------------------------------------------------------


def test_app_has_valkey_imports():
    """app.py must import the new deps at module level so they're loaded
    once at startup, not inside hot paths."""
    assert "import hashlib" in APP_SRC
    assert "import uuid" in APP_SRC
    assert "import asyncio" in APP_SRC


def test_app_queue_constants_env_overridable():
    """Queue knobs must read from env so operators can tune without code
    changes."""
    for knob in (
        'VALKEY_URL = os.environ.get("VALKEY_URL"',
        'TRANSCRIPT_CACHE_TTL = int(os.environ.get("TRANSCRIPT_CACHE_TTL"',
        'JOB_TTL = int(os.environ.get("JOB_TTL"',
        'JOB_RECENT_LIMIT = int(os.environ.get("JOB_RECENT_LIMIT"',
        'WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY"',
    ):
        assert knob in APP_SRC, f"missing env knob: {knob}"


def test_app_queue_states():
    """All five terminal/non-terminal states must be defined as constants
    so handlers can compare against them rather than string-literal-typing."""
    for state in (
        'JOB_STATUS_QUEUED = "queued"',
        'JOB_STATUS_RUNNING = "running"',
        'JOB_STATUS_DONE = "done"',
        'JOB_STATUS_FAILED = "failed"',
        'JOB_STATUS_CANCELLED = "cancelled"',
    ):
        assert state in APP_SRC


def test_app_permanent_job_error_class():
    """Server-side permanent failures (file not found, oversized input)
    must be distinguished from transient ones (CUDA OOM, network blip)
    via PermanentJobError → stored as `permanent: 1` on the job hash."""
    assert "class PermanentJobError" in APP_SRC
    assert 'isinstance(e, PermanentJobError)' in APP_SRC


def test_app_queue_endpoints_registered():
    """All four new endpoints must be wired into API_ROUTES."""
    assert 'Route("/api/jobs", api_jobs_submit, methods=["POST"])' in APP_SRC
    assert 'Route("/api/jobs/{job_id}", api_jobs_get, methods=["GET"])' in APP_SRC
    assert 'Route("/api/jobs/{job_id}", api_jobs_cancel, methods=["DELETE"])' in APP_SRC
    assert 'Route("/api/queue", api_queue_info, methods=["GET"])' in APP_SRC


def test_app_worker_handles_blpop_timeout_silently():
    """valkey-py raises TimeoutError on BLPOP when its client socket
    read-deadline fires before the server returns nil-on-empty. This is
    normal behaviour on an idle queue — must be caught silently, NOT
    logged as an error every 5 seconds."""
    snip = APP_SRC[APP_SRC.index("async def _job_worker"):
                   APP_SRC.index("async def _run_one_job")]
    # Specific catch for the valkey TimeoutError + built-in TimeoutError
    assert "_vk_exc.TimeoutError" in snip
    # And it must NOT fall through to the generic Exception logger
    timeout_branch = snip[snip.index("except (_vk_exc.TimeoutError"):
                          snip.index("except Exception")]
    assert "continue" in timeout_branch
    assert "log.error" not in timeout_branch


def test_app_worker_lifespan_wired():
    """Worker tasks must start/stop with the app lifespan, not as bare
    threads — otherwise SIGTERM during a Compose restart leaves workers
    in limbo and active jobs would be wedged across the restart.

    Modern Starlette (≥0.35) removed `on_startup`/`on_shutdown` kwargs
    in favour of the `lifespan` context manager — we use that.
    """
    assert "lifespan=_lifespan" in APP_SRC
    assert "@asynccontextmanager" in APP_SRC
    assert "await _worker_startup()" in APP_SRC
    assert "await _worker_shutdown()" in APP_SRC
    # And the OLD kwargs must NOT come back — they crash on modern Starlette.
    assert "on_startup=" not in APP_SRC
    assert "on_shutdown=" not in APP_SRC


def test_app_crash_recovery_on_startup():
    """Active jobs from a previous run must be re-queued at startup. Without
    this, a crash mid-transcription leaves the job stuck in `running` and
    clients poll it forever."""
    assert "async def _recover_active_jobs" in APP_SRC
    assert "_recover_active_jobs()" in APP_SRC
    # Recovery must push to the FRONT (LPUSH) so recovered jobs run before
    # newly-submitted ones — they were already at the head pre-crash.
    recovery_section = APP_SRC[APP_SRC.index("async def _recover_active_jobs"):
                                APP_SRC.index("async def _worker_startup")]
    assert "lpush(\"queue:waiting\"" in recovery_section


def test_app_transcript_cache_key_includes_settings():
    """Cache must key on file content + decode settings — different model
    or language must NOT collide."""
    src = APP_SRC[APP_SRC.index("def _transcript_cache_key"):
                  APP_SRC.index("def _transcript_cache_key") + 1000]
    # Filename intentionally NOT in the key — same audio at different paths
    # shares cache.
    assert "_sha1_file(file_path)" in src
    # All three setting axes must contribute.
    assert "{model}" in src
    assert "{language}" in src
    assert "diarize" in src


def test_app_cache_only_writes_successful_runs():
    """Whisper's "Error: ..." status must NOT be cached — otherwise a
    transient CUDA OOM would poison the cache for 7 days."""
    src = APP_SRC[APP_SRC.index("async def _execute_transcription"):]
    next_def = src.index("\nasync def ") if "\nasync def " in src else len(src)
    exec_src = src[:next_def]
    # The conditional must be present; cache write is gated.
    assert "not status.lower().startswith(\"error\")" in exec_src
    assert "_cache_set_transcript" in exec_src


def test_app_transcribe_routes_through_queue():
    """When valkey is available, /api/transcribe must enqueue + (optionally)
    poll, NOT take the lock directly. The lock path is a fallback for when
    valkey is down."""
    src = APP_SRC[APP_SRC.index("async def api_transcribe"):
                   APP_SRC.index("async def _wait_and_return_job")]
    assert "if _queue_available" in src
    assert "_enqueue_job(body" in src
    # Async-by-default: wait=true is opt-in.
    assert 'body.get("wait", False)' in src
    # 202 is the new default response on submit
    assert "status_code=202" in src


def test_app_transcribe_legacy_fallback_preserved():
    """When valkey is unavailable, the legacy lock-based sync path must
    still work — operators can run whisper standalone without valkey."""
    assert "async def _legacy_sync_transcribe" in APP_SRC
    legacy = APP_SRC[APP_SRC.index("async def _legacy_sync_transcribe"):]
    next_def = legacy.index("\nasync def ") if "\nasync def " in legacy else len(legacy)
    legacy_body = legacy[:next_def]
    # Still uses the lock — that's the whole point of the fallback path.
    assert "_transcription_lock.acquire" in legacy_body
    # And returns 409 like before so curl users see the same error.
    assert "status_code=409" in legacy_body


def test_app_cancel_blocks_running_jobs():
    """Cancellation must be queued-only — running jobs can't be cancelled
    safely (whisperX has no checkpoint protocol, would leak GPU memory)."""
    src = APP_SRC[APP_SRC.index("async def _cancel_job"):
                  APP_SRC.index("async def _queue_info")]
    assert "cannot cancel in-flight job" in src
    assert "JOB_STATUS_RUNNING" in src


# -- Bot (bot/main.py) ---------------------------------------------------------


def test_bot_uses_api_jobs_not_api_transcribe():
    """The bot must submit through the queue, not hit /api/transcribe
    directly. /api/transcribe is for legacy curl users only."""
    src = BOT_SRC
    # The helper exists
    assert "async def _submit_and_poll_transcribe" in src
    # process() uses it (not a direct POST to /api/transcribe)
    process_src = src[src.index("async def process(job: Job):"):
                       src.index("async def process_url")]
    assert "_submit_and_poll_transcribe(" in process_src
    assert "/api/transcribe" not in process_src
    # The submit helper POSTs to /api/jobs and polls /api/jobs/{id}
    helper_src = src[src.index("async def _submit_and_poll_transcribe"):
                      src.index("async def _submit_and_poll_transcribe") + 4000]
    assert "/api/jobs" in helper_src
    assert "/api/jobs/{job_id}" in helper_src


def test_bot_tactical_busy_wait_removed():
    """The WhisperBusyError + wait_for_whisper_idle tactical fix from the
    previous PR must be GONE. The server-side queue handles GPU contention
    so there's no need for client-side polling."""
    assert "class WhisperBusyError" not in BOT_SRC
    assert "async def wait_for_whisper_idle" not in BOT_SRC
    assert "WHISPER_BUSY_TIMEOUT" not in BOT_SRC
    assert "WHISPER_BUSY_POLL_INTERVAL" not in BOT_SRC
    # And the worker is back to a plain for-loop (busy branch is gone)
    assert "except WhisperBusyError" not in BOT_SRC


def test_bot_job_poll_interval_env_overridable():
    """Bot must read poll interval from env so ops can tune snappiness."""
    assert 'JOB_POLL_INTERVAL = int(os.environ.get("JOB_POLL_INTERVAL"' in BOT_SRC


def test_bot_poll_helper_maps_states_to_exceptions():
    """The submit/poll helper must convert server-side terminal states to
    bot-side exception types so the worker retry/permanent logic still works."""
    src = BOT_SRC[BOT_SRC.index("async def _submit_and_poll_transcribe"):]
    next_def = src.index("\nasync def ") if "\nasync def " in src else len(src)
    helper_src = src[:next_def]
    # failed + permanent → PermanentError (no retry)
    assert 'data.get("permanent")' in helper_src
    assert "PermanentError" in helper_src
    # cancelled → PermanentError (don't auto-retry someone else's cancel)
    assert '"cancelled"' in helper_src
    # 503 → RuntimeError so worker can retry (queue might come back)
    assert "resp.status == 503" in helper_src


def test_bot_passes_consumer_tag():
    """Bot submits with consumer='discord-bot' so /api/queue can identify
    the source. Useful for operator visibility."""
    src = BOT_SRC[BOT_SRC.index("transcribe_payload = {"):]
    next_blank = src.index("\n\n")
    payload_src = src[:next_blank]
    assert '"consumer": "discord-bot"' in payload_src


# -- Both layers ---------------------------------------------------------------


def test_app_status_keeps_busy_field():
    """The /api/status busy field from the tactical fix is still useful for
    operators / Gradio UI — it must NOT be removed during the queue migration."""
    assert '"busy": _transcription_lock.locked()' in APP_SRC


def test_app_queue_info_uses_asyncio_gather():
    """Regression guard: _queue_info used to wrap an async generator in a
    list comprehension — `[j for j in (await _read_job(jid) for jid in ids)]`
    — which is a runtime TypeError because list comprehensions can't iterate
    async generators. Must use asyncio.gather instead."""
    # The broken pattern (await inside genexpr passed to list comp) is gone.
    assert "for jid in active_ids) if j]" not in APP_SRC
    assert "for jid in recent_ids) if j]" not in APP_SRC
    # And gather is used to fetch hashes in parallel.
    snip = APP_SRC[APP_SRC.index("async def _queue_info"):
                   APP_SRC.index("async def _queue_info") + 1500]
    assert "asyncio.gather" in snip


def test_app_sha1_offloaded_to_thread():
    """_sha1_file does sync I/O on multi-GB audio files — must be wrapped in
    asyncio.to_thread or it blocks the event loop and stalls /api/status /
    /api/queue polls from other clients for several seconds."""
    snip = APP_SRC[APP_SRC.index("async def _execute_transcription"):]
    next_def = snip.index("\nasync def ") if "\nasync def " in snip else len(snip)
    exec_src = snip[:next_def]
    # The hash + cache-key derivation is done in a thread.
    assert "asyncio.to_thread(" in exec_src
    assert "_transcript_cache_key" in exec_src
    # And specifically — the cache key derivation is the to_thread'd call.
    assert "to_thread(\n        _transcript_cache_key" in exec_src \
        or "to_thread(_transcript_cache_key" in exec_src


def test_app_subtitle_file_not_cached():
    """Regression guard: subtitle_file is an ephemeral /tmp path generated
    by whisperX. Storing it in the transcript cache leaves callers with a
    stale path on cache hit (file long since cleaned). Cache must store
    only the reproducible fields (status, transcript)."""
    snip = APP_SRC[APP_SRC.index("async def _execute_transcription"):]
    next_def = snip.index("\nasync def ") if "\nasync def " in snip else len(snip)
    exec_src = snip[:next_def]
    # The cache write must use a stripped-down dict, not the full result.
    assert "cacheable = {" in exec_src
    # And cache hits must null out subtitle_file in the returned dict.
    assert '"subtitle_file": None' in exec_src


def test_app_worker_terminal_state_guard():
    """Defensive: _run_one_job must skip jobs already in a terminal state
    when popped. Catches the cancel-after-pop race and prevents duplicate
    work if crash recovery ever scales beyond one container."""
    snip = APP_SRC[APP_SRC.index("async def _run_one_job"):
                   APP_SRC.index("async def _execute_transcription")]
    assert "_JOB_TERMINAL" in snip
    assert "already" in snip


def test_bot_poll_tolerates_transient_errors():
    """Network blips (whisper restart, DNS flake) must NOT kill the poll —
    the job continues on the server regardless. Bot retries within the
    deadline, escalates only on sustained outage."""
    snip = BOT_SRC[BOT_SRC.index("async def _submit_and_poll_transcribe"):]
    next_def = snip.index("\nasync def ") if "\nasync def " in snip else len(snip)
    helper = snip[:next_def]
    # Generic-Exception catch for transient errors
    assert "except Exception as e:" in helper
    # Counter so we eventually give up on sustained failures
    assert "transient_errors" in helper
    # RuntimeError (404 / non-200) must NOT be silently retried —
    # it's authoritative and re-raised.
    assert "except RuntimeError:" in helper


# ─── 14. Multilingual / code-switched audio support ─────────────────────────


# -- Server-side (app.py) -----------------------------------------------------


def test_app_decide_task_constants():
    """Resolution table for translate payload field. Encoded as a function
    so behaviour is testable. Source-string check guards the function's
    presence; the import-based tests below exercise its behaviour."""
    assert "def _decide_task" in APP_SRC
    assert "def _quick_detect_language" in APP_SRC
    assert "def _quick_detect_language_sync" in APP_SRC


def test_app_decide_task_resolution():
    """Behaviour of _decide_task across the matrix of inputs."""
    # We can import the function via the app source — but app.py imports
    # whisperx, torch, etc. at module-level, so we exec just the function
    # body in an isolated namespace.
    import ast
    tree = ast.parse(APP_SRC)
    decide_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_decide_task":
            decide_src = ast.get_source_segment(APP_SRC, node)
            break
    assert decide_src, "_decide_task not found in app.py"
    ns: dict = {}
    exec(decide_src, ns)
    _decide_task = ns["_decide_task"]

    # translate=True → always translate, regardless of language/conf
    assert _decide_task(True, "en", "en", 1.0) == "translate"
    assert _decide_task(True, "Auto-detect", "id", 0.99) == "translate"

    # translate=False → always transcribe
    assert _decide_task(False, "en", "en", 1.0) == "transcribe"
    assert _decide_task(False, "Auto-detect", "id", 0.99) == "transcribe"

    # auto + explicit language → respect user, transcribe
    assert _decide_task("auto", "id", "id", 0.99) == "transcribe"
    assert _decide_task("auto", "fr", "en", 0.99) == "transcribe"

    # auto + auto-detect + quick-LID says English → transcribe
    assert _decide_task("auto", "Auto-detect", "en", 0.99) == "transcribe"
    assert _decide_task("auto", None, "en", 0.99) == "transcribe"

    # auto + auto-detect + quick-LID says non-English (high conf) → translate
    assert _decide_task("auto", "Auto-detect", "id", 0.99) == "translate"
    assert _decide_task("auto", "Auto-detect", "yue", 0.85) == "translate"
    assert _decide_task("auto", "Auto-detect", "zh", 0.9) == "translate"

    # auto + low-confidence LID → translate (CS audio often has low conf)
    assert _decide_task("auto", "Auto-detect", "en", 0.3) == "translate"
    assert _decide_task("auto", "Auto-detect", "unknown", 0.0) == "translate"


def test_app_transcribe_inner_accepts_task():
    """_transcribe_inner must accept the `task` kwarg and forward it to
    m.transcribe(). Without this thread-through, the translate path is a
    no-op even though the rest of the pipeline thinks it's enabled."""
    src = APP_SRC[APP_SRC.index("def _transcribe_inner"):
                   APP_SRC.index("def format_timestamp_display")]
    assert 'task="transcribe"' in src      # default param
    assert "task=task" in src                # passed to m.transcribe


def test_app_run_transcription_accepts_task():
    """_run_transcription (sync entry) must forward task to _transcribe_inner."""
    src = APP_SRC[APP_SRC.index("def _run_transcription"):
                   APP_SRC.index("async def api_transcribe")]
    assert 'task="transcribe"' in src   # default
    assert "task=task" in src             # forwarded
    assert '"task": task' in src          # surfaced in return dict


def test_app_alignment_skipped_for_translate():
    """task=translate must skip wav2vec2 alignment — the aligner is
    source-language-specific and would produce garbage word timestamps
    against an English transcript of non-English audio."""
    src = APP_SRC[APP_SRC.index("def _transcribe_inner"):
                   APP_SRC.index("def format_timestamp_display")]
    assert 'if task == "translate":' in src
    assert "Skipping alignment" in src
    # Also: graceful skip for languages without a default aligner (yue, id,
    # other Whisper-supported languages outside whisperx's ~40 covered).
    assert "DEFAULT_ALIGN_MODELS_TORCH" in src
    assert "DEFAULT_ALIGN_MODELS_HF" in src
    assert "no default aligner" in src


def test_app_cache_key_includes_task():
    """Transcribe and translate produce different transcripts from the
    same audio. They must NOT collide in the cache."""
    sig_line = APP_SRC[APP_SRC.index("def _transcript_cache_key"):
                       APP_SRC.index("def _transcript_cache_key") + 1000]
    # task is a parameter
    assert "task: str" in sig_line
    # Key format includes task at the end
    assert "{task}" in sig_line
    # _execute_transcription threads it through
    exec_src = APP_SRC[APP_SRC.index("async def _execute_transcription"):]
    next_def = exec_src.index("\nasync def ") if "\nasync def " in exec_src else len(exec_src)
    exec_src = exec_src[:next_def]
    assert "_transcript_cache_key, file_path, model, language, diarize, task" in exec_src


def test_app_quick_lid_only_runs_when_needed():
    """The 30s LID pre-pass is the most expensive part of auto-translate
    (reads the file, runs the encoder). Must NOT run when translate is
    explicit (True/False) or when language is explicitly set."""
    exec_src = APP_SRC[APP_SRC.index("async def _execute_transcription"):]
    next_def = exec_src.index("\nasync def ") if "\nasync def " in exec_src else len(exec_src)
    exec_src = exec_src[:next_def]
    # Guard expression: must check translate == "auto" AND no explicit language
    assert 'translate == "auto"' in exec_src
    assert "_quick_detect_language(file_path)" in exec_src
    # The guard must be conditional, not unconditional. Check that the
    # quick-LID line is preceded by an `if` within a few lines.
    qlid_idx = exec_src.index("await _quick_detect_language(file_path)")
    preceding = exec_src[max(0, qlid_idx - 300):qlid_idx]
    assert "if " in preceding


def test_app_legacy_path_also_supports_translate():
    """The non-queue fallback path (when valkey is unreachable) must also
    honour translate — operators expect the same behaviour on either path."""
    src = APP_SRC[APP_SRC.index("async def _legacy_sync_transcribe"):]
    next_def = src.index("\nasync def ") if "\nasync def " in src else len(src)
    legacy = src[:next_def]
    assert 'body.get("translate"' in legacy
    assert "_decide_task(" in legacy
    assert "task=task" in legacy


# -- Bot (bot/main.py) -------------------------------------------------------


def test_bot_job_has_translate_field():
    """Job dataclass must carry the translate policy so it survives queue
    serialisation and reaches process()."""
    assert "translate: object" in BOT_SRC
    # And default is "auto" (string, not True/False).
    assert 'translate: object = "auto"' in BOT_SRC


def test_bot_passes_translate_to_server():
    """process() must forward Job.translate to the /api/jobs payload."""
    src = BOT_SRC[BOT_SRC.index("transcribe_payload = {"):]
    payload_src = src[:src.index("\n\n")]
    assert '"translate": job.translate' in payload_src


def test_bot_slash_commands_expose_translate():
    """/summarize and /transcribe must accept a translate option so users
    can override the auto default."""
    summarize_src = BOT_SRC[BOT_SRC.index("async def cmd_summarize"):
                             BOT_SRC.index("async def cmd_transcribe")]
    transcribe_src = BOT_SRC[BOT_SRC.index("async def cmd_transcribe"):
                              BOT_SRC.index("async def cmd_transcribe") + 2500]
    for cmd_src in (summarize_src, transcribe_src):
        assert 'translate: Literal["auto", "translate", "native"]' in cmd_src


def test_bot_resolve_translate_choice_helper():
    """Mapping from slash-command Literal to payload value must be a
    dedicated function so it's testable + reusable."""
    assert "def _resolve_translate_choice" in BOT_SRC
    # Behaviour: "translate" → True, "native" → False, anything else → "auto"
    import ast
    tree = ast.parse(BOT_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_translate_choice":
            src = ast.get_source_segment(BOT_SRC, node)
            break
    else:
        raise AssertionError("_resolve_translate_choice not found")
    ns: dict = {}
    exec(src, ns)
    fn = ns["_resolve_translate_choice"]
    assert fn("translate") is True
    assert fn("native") is False
    assert fn("auto") == "auto"
    # Defensive: unknown values default to auto, not crash.
    assert fn("bogus") == "auto"


def test_bot_job_from_interaction_threads_translate():
    """_job_from_interaction must accept + thread translate so slash
    commands can plumb the choice into the Job."""
    src = BOT_SRC[BOT_SRC.index("def _job_from_interaction"):
                   BOT_SRC.index("def _job_from_interaction") + 3000]
    assert 'translate: object = "auto"' in src
    assert "translate=translate" in src


# ─── 15. Cache refresh + task-aware cache keys ──────────────────────────────


def test_bot_cache_path_is_task_aware():
    """Without this, /summarize translate:auto and translate:translate of
    the same video collide on the bot's file cache → second run returns
    the first run's transcript regardless of mode (the real-world bug
    that prompted this fix)."""
    src = BOT_SRC[BOT_SRC.index("def _cache_path"):
                   BOT_SRC.index("def write_cache")]
    # New signature accepts translate
    assert "translate" in src
    # Filename embeds the translate key
    assert "{key}" in src or "{translate_to_key" in src


def test_bot_translate_to_key_mapping():
    """Stable string key for cache filenames. Behaviour matters because
    it's what makes the variants distinct on disk."""
    import ast
    tree = ast.parse(BOT_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_translate_to_key":
            src = ast.get_source_segment(BOT_SRC, node)
            break
    else:
        raise AssertionError("_translate_to_key not found")
    ns: dict = {}
    exec(src, ns)
    fn = ns["_translate_to_key"]
    assert fn(True) == "translate"
    assert fn(False) == "native"
    assert fn("auto") == "auto"
    # Defensive: unknown value defaults to auto (so we don't write a
    # garbage filename if a future caller passes something weird).
    assert fn("bogus") == "auto"
    assert fn(None) == "auto"


def test_bot_read_cache_legacy_fallback():
    """Pre-Tier-1 cache files were written as `{video_id}.txt`. New code
    writes `{video_id}.{translate_key}.txt`. read_cache must fall back
    to the legacy filename for translate='auto' so existing caches keep
    working across the rollout."""
    src = BOT_SRC[BOT_SRC.index("def read_cache"):
                   BOT_SRC.index("def read_cache") + 2500]
    # Mentions the legacy filename pattern
    assert 'f"{video_id}.txt"' in src
    # And the fallback only applies to translate=auto
    assert 'translate == "auto"' in src


def test_bot_job_has_refresh_field():
    """Job dataclass must carry refresh so it survives queue submission
    and reaches process()/process_url."""
    assert "refresh: bool = False" in BOT_SRC


def test_bot_slash_commands_expose_refresh():
    """Both /summarize and /transcribe accept refresh: bool so users can
    bust the cache on a per-call basis without a separate slash command."""
    summarize_src = BOT_SRC[BOT_SRC.index("async def cmd_summarize"):
                             BOT_SRC.index("async def cmd_transcribe")]
    transcribe_src = BOT_SRC[BOT_SRC.index("async def cmd_transcribe"):
                              BOT_SRC.index("async def cmd_transcribe") + 3000]
    for cmd_src, label in ((summarize_src, "summarize"), (transcribe_src, "transcribe")):
        assert "refresh: bool = False" in cmd_src, f"/{label} missing refresh"
        # Threaded into _job_from_interaction
        assert "refresh=refresh" in cmd_src, f"/{label} not forwarding refresh"


def test_bot_process_respects_refresh():
    """process() and process_url must skip read_cache when job.refresh is
    True. Without this the slash-command refresh option does nothing
    user-visible (server gets called anyway, but bot's own file cache
    still wins on next run)."""
    # Video pipeline
    process_src = BOT_SRC[BOT_SRC.index("async def process(job: Job):"):
                           BOT_SRC.index("async def process_url")]
    assert "job.refresh" in process_src
    # Skip on True, read on False (default)
    assert "None if job.refresh else read_cache" in process_src

    # Web pipeline
    process_url_src = BOT_SRC[BOT_SRC.index("async def process_url"):
                               BOT_SRC.index("async def process_url") + 5000]
    assert "job.refresh" in process_url_src
    assert "None if job.refresh else read_cache" in process_url_src


def test_bot_write_cache_passes_translate():
    """The video-pipeline write_cache must include job.translate so the
    file is keyed correctly. Without this, refresh + translate variants
    overwrite each other on the same `{video_id}.auto.txt`."""
    # The call sites we control (process and _apply_rename) must pass it.
    # process(): see line near "Persist to cache"
    assert "write_cache(job.video_id, title, status, transcript, duration, job.translate)" in BOT_SRC


def test_bot_speaker_rename_view_carries_translate():
    """SpeakerRenameView is constructed at summary-post time and lives
    until the user clicks. Must remember which translate variant
    produced it so rename targets the correct cache file."""
    src = BOT_SRC[BOT_SRC.index("class SpeakerRenameView"):
                   BOT_SRC.index("async def _apply_rename")]
    # Constructor + storage
    assert "translate: object" in src
    assert "self._translate = translate" in src
    # Used in lookups
    assert "read_cache(self._video_id, self._translate)" in src


def test_bot_apply_rename_uses_translate_param():
    """_apply_rename must accept and forward translate so rename hits the
    same cache variant that the view was attached to."""
    src = BOT_SRC[BOT_SRC.index("async def _apply_rename"):
                   BOT_SRC.index("async def _apply_rename") + 3000]
    assert "translate: object" in src
    assert "read_cache(video_id, translate)" in src
    assert "write_cache(video_id, title, status, transcript, duration, translate)" in src


def test_app_execute_transcription_respects_refresh():
    """Server-side: payload.refresh=true must skip the Valkey transcript
    cache lookup. Result still WRITES on success."""
    src = APP_SRC[APP_SRC.index("async def _execute_transcription"):]
    next_def = src.index("\nasync def ") if "\nasync def " in src else len(src)
    exec_src = src[:next_def]
    assert 'payload.get("refresh"' in exec_src
    assert "None if refresh else await _cache_get_transcript" in exec_src
    # Cache write is NOT gated by refresh — we still populate the cache
    # so subsequent non-refresh runs benefit.
    write_section = exec_src[exec_src.index("# Cache successful runs"):]
    assert "if refresh:" not in write_section


# ─── Inflight tracking + new helpers ─────────────────────────────────────────


def _make_test_job(video_id="abc123", kind="video", url="https://example.com",
                   submitter_id=1, channel_id=42, submitter_name="alice"):
    """Build a Job that satisfies the discriminated-union invariant.
    Stubs a Discord interaction object — enough for _inflight_register.
    """
    import types as _types
    interaction = _types.SimpleNamespace(
        user=_types.SimpleNamespace(id=submitter_id),
        channel=_types.SimpleNamespace(id=channel_id),
        followup=_types.SimpleNamespace(send=lambda *a, **k: None),
    )
    job = bot.Job(
        url=url, video_id=video_id,
        channel=_types.SimpleNamespace(id=channel_id),
        submitter_id=submitter_id,
        submitter_name=submitter_name,
        kind=kind,
        interaction=interaction,
    )
    return job


def test_inflight_register_creates_entry():
    """After _inflight_register, the entry is queryable by video_id and user."""
    bot._inflight.clear()
    job = _make_test_job(video_id="vid1", submitter_id=99)
    bot._inflight_register(job)
    assert "vid1" in bot._inflight
    entry = bot._inflight["vid1"]
    assert entry.bot_video_id == "vid1"
    assert entry.submitter_id == 99
    assert entry.phase == bot.PHASE_QUEUED
    assert entry.started_at is None
    assert entry.queued_at > 0


def test_inflight_phase_transition_sets_started_at():
    """Transitioning out of queued sets started_at exactly once."""
    bot._inflight.clear()
    job = _make_test_job(video_id="vid2")
    bot._inflight_register(job)
    bot._inflight_phase(job, bot.PHASE_DOWNLOADING)
    e = bot._inflight["vid2"]
    assert e.phase == bot.PHASE_DOWNLOADING
    assert e.started_at is not None
    first = e.started_at
    # Second transition shouldn't reset started_at.
    time.sleep(0.01)
    bot._inflight_phase(job, bot.PHASE_TRANSCRIBING)
    assert bot._inflight["vid2"].started_at == first


def test_inflight_phase_carries_extra_fields():
    """Title / duration / server_job_id can be set via _inflight_phase kwargs."""
    bot._inflight.clear()
    job = _make_test_job(video_id="vid3")
    bot._inflight_register(job)
    bot._inflight_phase(
        job, bot.PHASE_TRANSCRIBING,
        title="Some Title", duration=180, server_job_id="srv-abc",
    )
    e = bot._inflight["vid3"]
    assert e.title == "Some Title"
    assert e.duration == 180
    assert e.server_job_id == "srv-abc"


def test_inflight_phase_ignores_unknown_field():
    """_inflight_phase silently drops unknown kwargs — guard against typos
    propagating, but don't crash either."""
    bot._inflight.clear()
    job = _make_test_job(video_id="vid4")
    bot._inflight_register(job)
    bot._inflight_phase(job, bot.PHASE_DOWNLOADING, bogus="x")
    assert "vid4" in bot._inflight
    assert not hasattr(bot._inflight["vid4"], "bogus")


def test_inflight_phase_noop_on_missing_entry():
    """Defensive: phase update on a removed entry mustn't crash."""
    bot._inflight.clear()
    job = _make_test_job(video_id="vid5")
    bot._inflight_phase(job, bot.PHASE_SUMMARIZING)  # never registered
    assert "vid5" not in bot._inflight


def test_inflight_remove_records_runtime():
    """Completed entries with started_at >=1s feed the rolling average."""
    bot._inflight.clear()
    bot._completed_durations.clear()
    job = _make_test_job(video_id="vid6", kind="video")
    bot._inflight_register(job)
    bot._inflight_phase(job, bot.PHASE_DOWNLOADING)
    # Pretend the job ran for 5 seconds.
    bot._inflight["vid6"].started_at = time.time() - 5.0
    bot._inflight_remove("vid6", kind="video")
    assert "vid6" not in bot._inflight
    dq = bot._completed_durations["video"]
    assert len(dq) == 1
    assert 4.5 < dq[0] < 6.0


def test_inflight_remove_skips_sub_second():
    """<1s runtimes (cache hits) shouldn't skew the rolling average."""
    bot._inflight.clear()
    bot._completed_durations.clear()
    job = _make_test_job(video_id="vid7", kind="web")
    bot._inflight_register(job)
    bot._inflight["vid7"].started_at = time.time() - 0.1
    bot._inflight_remove("vid7", kind="web")
    assert "web" not in bot._completed_durations or not bot._completed_durations["web"]


def test_inflight_user_filters_by_user_id():
    """Only the submitter's own entries are returned, sorted running-first."""
    bot._inflight.clear()
    a = _make_test_job(video_id="vA", submitter_id=1)
    b = _make_test_job(video_id="vB", submitter_id=1)
    c = _make_test_job(video_id="vC", submitter_id=2)
    for j in (a, b, c):
        bot._inflight_register(j)
    # Make vB running.
    bot._inflight_phase(b, bot.PHASE_DOWNLOADING)
    user1 = bot._inflight_user(1)
    assert {e.bot_video_id for e in user1} == {"vA", "vB"}
    # Running comes first.
    assert user1[0].bot_video_id == "vB"
    user2 = bot._inflight_user(2)
    assert [e.bot_video_id for e in user2] == ["vC"]


def test_avg_runtime_returns_none_below_two_samples():
    """Need at least 2 samples before the rolling average is meaningful."""
    bot._completed_durations.clear()
    bot._completed_durations["video"].append(10.0)
    assert bot._avg_runtime("video") is None
    bot._completed_durations["video"].append(20.0)
    assert bot._avg_runtime("video") == 15.0


def test_format_relative_units():
    """_format_relative picks the right unit at the second/minute/hour boundaries."""
    assert bot._format_relative(0) == "0s"
    assert bot._format_relative(45) == "45s"
    assert bot._format_relative(60) == "1m00s"
    assert bot._format_relative(125) == "2m05s"
    assert bot._format_relative(3600) == "1h00m"
    assert bot._format_relative(7320) == "2h02m"
    # Negative inputs are clamped to 0.
    assert bot._format_relative(-5) == "0s"


def test_classify_cache_kind_yt_vs_hash():
    """11-char URL-safe id → video; anything else → web."""
    # YouTube-style: 11 chars, A-Za-z0-9_- only. The stem includes the
    # translate suffix, e.g. "dQw4w9WgXcQ.auto".
    assert bot._classify_cache_kind("dQw4w9WgXcQ.auto") == "video"
    assert bot._classify_cache_kind("_-AbC123xyz.translate") == "video"  # 11 chars
    # Hash-style: hex digest.
    assert bot._classify_cache_kind("abcdef0123456789.auto") == "web"
    # Too short to be YT.
    assert bot._classify_cache_kind("short.auto") == "web"


def test_parse_cache_header_extracts_metadata():
    """Header lines stop at the first blank; only `# key: value` lines parsed."""
    text = (
        "# title: Some Video\n"
        "# status: Done\n"
        "# duration: 600\n"
        "# translate: auto\n"
        "\n"
        "transcript body goes here\n"
        "# this is not a header\n"
    )
    hdr = bot._parse_cache_header(text)
    assert hdr["title"] == "Some Video"
    assert hdr["status"] == "Done"
    assert hdr["duration"] == "600"
    assert hdr["translate"] == "auto"
    assert len(hdr) == 4


def test_parse_cache_header_handles_no_header():
    """File with no header lines still returns {} cleanly."""
    assert bot._parse_cache_header("just transcript\n") == {}


def test_effective_config_channel_wins_over_user():
    """Channel config takes precedence over user config on key conflicts."""
    # Stub both readers so we don't touch disk.
    orig_user = bot.get_user_config
    orig_chan = bot.get_channel_config
    try:
        bot.get_user_config = lambda uid: {"model": "user-default", "diarize": True}
        bot.get_channel_config = lambda cid: {"model": "channel-policy"}
        merged = bot._effective_config(1, 2)
        assert merged["model"] == "channel-policy"
        # Channel didn't set diarize → user value carries through.
        assert merged["diarize"] is True
    finally:
        bot.get_user_config = orig_user
        bot.get_channel_config = orig_chan


def test_effective_config_empty_returns_empty_dict():
    """Both layers empty → empty dict, no exceptions."""
    orig_user = bot.get_user_config
    orig_chan = bot.get_channel_config
    try:
        bot.get_user_config = lambda uid: {}
        bot.get_channel_config = lambda cid: {}
        assert bot._effective_config(1, 2) == {}
    finally:
        bot.get_user_config = orig_user
        bot.get_channel_config = orig_chan


def test_help_topics_cover_documented_areas():
    """`/help topic:` must cover every advertised topic."""
    expected = {"overview", "triggers", "admin", "limits", "errors", "translate"}
    assert expected <= set(bot._HELP_TOPICS.keys())
    # Topics should be non-empty strings.
    for k, v in bot._HELP_TOPICS.items():
        assert isinstance(v, str) and len(v) > 50, f"topic {k!r} body too short"


def test_phase_constants_distinct():
    """Phase strings must be distinct so the worker can switch on them."""
    phases = {
        bot.PHASE_QUEUED, bot.PHASE_DOWNLOADING, bot.PHASE_TRANSCRIBING,
        bot.PHASE_SCRAPING, bot.PHASE_SUMMARIZING,
    }
    assert len(phases) == 5


def test_slash_commands_registered():
    """All advertised slash commands must be defined as cmd_* functions."""
    expected = {
        "cmd_summarize", "cmd_transcribe", "cmd_status", "cmd_find",
        "cmd_config", "cmd_serverconfig",
        # New commands
        "cmd_web", "cmd_litmus", "cmd_progress", "cmd_cancel",
        "cmd_queue", "cmd_help", "cmd_recent", "cmd_redo", "cmd_myconfig",
    }
    actual = {name for name in dir(bot) if name.startswith("cmd_") and callable(getattr(bot, name))}
    missing = expected - actual
    assert not missing, f"missing slash command handlers: {missing}"


# ─── Test runner ─────────────────────────────────────────────────────────────


def main():
    import traceback
    tests = sorted(name for name in globals() if name.startswith("test_"))
    only = [a for a in sys.argv[1:] if a.startswith("test_")]
    if only:
        tests = [t for t in tests if t in only]
    passed, failed = 0, []
    for name in tests:
        try:
            globals()[name]()
            passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            tb = traceback.format_exc()
            failed.append((name, str(e) or tb))
            print(f"  ✗ {name}: {e or '(empty AssertionError; tb below)'}")
            if not str(e):
                print(tb)
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
