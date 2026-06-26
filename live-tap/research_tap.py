#!/usr/bin/env python3
"""research_tap.py — pipe-mode inline research assistant.

Reads live transcript from stdin (one line per committed utterance, as
produced by desktop_tap.py) and provides an interactive readline prompt
for asking questions about the call. Sends the rolling transcript as
context to the whisper service /api/research endpoint and streams the
answer to stdout.

Usage
-----
# Two-terminal mode (transcript in one, research in another):
  Terminal 1:  python desktop_tap.py --loopback --out /tmp/tap.txt
  Terminal 2:  tail -f /tmp/tap.txt | python research_tap.py

# Pipe mode (combined, but you need to type while it's running):
  python desktop_tap.py --loopback | python research_tap.py

# Ask without a live feed (just the research REPL against a saved transcript):
  python research_tap.py --transcript meeting.txt

Options
-------
  --url          Whisper service base URL (default: http://localhost:7860)
  --model        LLM model override (default: server default)
  --context-words  Max words of transcript to send as context (default: 500)
  --transcript   Path to an existing transcript file (instead of stdin)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque

DEFAULT_URL = "http://localhost:7860"
DEFAULT_CONTEXT_WORDS = 500
PROMPT = "\n\033[1;36m> \033[0m"  # cyan bold prompt


# ── rolling transcript buffer ──────────────────────────────────────────────


class TranscriptBuffer:
    """Thread-safe rolling window of the most-recent transcript words."""

    def __init__(self, max_words: int = DEFAULT_CONTEXT_WORDS) -> None:
        self._lines: deque[str] = deque()
        self._lock = threading.Lock()
        self._max_words = max_words
        self._word_count = 0

    def add(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        words = len(line.split())
        with self._lock:
            self._lines.append(line)
            self._word_count += words
            # trim oldest lines to stay within word budget
            while self._word_count > self._max_words and self._lines:
                old = self._lines.popleft()
                self._word_count -= len(old.split())

    def get(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def word_count(self) -> int:
        with self._lock:
            return self._word_count


# ── /api/research streaming call ──────────────────────────────────────────


def stream_research(url: str, question: str, context: str, model: str | None = None):
    """Generator that yields text chunks from the /api/research SSE stream."""
    payload: dict = {"question": question, "context": context}
    if model:
        payload["model"] = model
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/research",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            buf = ""
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        return
                    if payload_str.startswith("[ERROR]"):
                        raise RuntimeError(payload_str[7:].strip())
                    yield payload_str.replace("\\n", "\n")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code}: {body}")


# ── stdin reader thread ────────────────────────────────────────────────────


def _read_stdin(buf: TranscriptBuffer, quiet: bool = False) -> None:
    """Background thread: drain stdin lines into the transcript buffer."""
    for line in sys.stdin:
        text = line.strip()
        if text:
            buf.add(text)
            if not quiet:
                # Re-print the transcript line so user sees what came in,
                # even if the readline prompt is displayed.
                print(f"\r\033[2K\033[0;32m[transcript]\033[0m {text}", flush=True)


def _read_file(path: str, buf: TranscriptBuffer) -> None:
    """Read an existing transcript file into the buffer (one-shot)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                buf.add(line)
    except OSError as e:
        print(f"[research] warning: cannot read {path}: {e}", file=sys.stderr)


# ── main REPL ─────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Interactive LLM research assistant fed by the live transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url", default=DEFAULT_URL,
                   help="Whisper service base URL (default: %(default)s)")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="LLM model override (default: server RESEARCH_MODEL)")
    p.add_argument("--context-words", type=int, default=DEFAULT_CONTEXT_WORDS,
                   metavar="N",
                   help="Max transcript words to send as context (default: %(default)s)")
    p.add_argument("--transcript", metavar="FILE",
                   help="Load an existing transcript file instead of reading stdin")
    p.add_argument("--quiet", action="store_true",
                   help="Don't echo transcript lines to stdout")
    args = p.parse_args()

    buf = TranscriptBuffer(args.context_words)

    # Populate from file if given; otherwise start stdin reader thread
    if args.transcript:
        _read_file(args.transcript, buf)
        print(
            f"[research] loaded {buf.word_count()} words from {args.transcript}",
            file=sys.stderr,
        )
    elif not sys.stdin.isatty():
        t = threading.Thread(target=_read_stdin, args=(buf, args.quiet), daemon=True)
        t.start()
        print(
            "[research] reading transcript from stdin. Type a question and press Enter.",
            file=sys.stderr,
        )
    else:
        print(
            "[research] no stdin pipe and no --transcript file. "
            "Run: python desktop_tap.py --loopback | python research_tap.py",
            file=sys.stderr,
        )

    print(
        f"[research] connected to {args.url} | "
        f"context window: {args.context_words} words",
        file=sys.stderr,
    )
    print(
        "[research] Type a question and press Enter. Ctrl-C / Ctrl-D to quit.",
        file=sys.stderr,
    )

    try:
        while True:
            try:
                question = input(PROMPT).strip()
            except EOFError:
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break

            ctx = buf.get()
            words = buf.word_count()
            print(
                f"\033[2m[context: {words} words]\033[0m",
                file=sys.stderr,
                flush=True,
            )

            # Stream the answer
            sys.stdout.write("\033[0;33m")  # yellow for answer
            sys.stdout.flush()
            try:
                for chunk in stream_research(args.url, question, ctx, args.model):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            except Exception as e:
                print(f"\n\033[31m[error] {e}\033[0m", flush=True)
            else:
                sys.stdout.write("\033[0m\n")
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass

    print("\n[research] bye.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
