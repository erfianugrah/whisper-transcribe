# whisper-transcribe — agent notes

Read this BEFORE running build / deploy / test commands. Discovering the
Makefile from scratch each session wastes a few hundred tool calls and
sometimes picks the wrong target (e.g. `make ship` blocking on a stuck
Docker Hub push when local-only redeploy was wanted).

## What this repo is

Two Python services + supporting infra, all in one compose stack:
- **whisper** (`app.py`) — GPU-backed whisperX transcription + VLM frame
  description + OCR (EasyOCR). HTTP API at `:7860`. Gradio UI at `/`.
- **bot** (`bot/main.py`) — Discord bot that takes URL / image messages,
  enqueues jobs to the whisper service (Valkey-backed queue), and posts
  summary embeds. Talks OpenAI-compatible LLM via `model_proxy:11434`
  on external network `llmc` (declare `external: true` in compose.yaml).

Plus: valkey (queue), crawl4ai + flaresolverr (web scraper for `tldr`
on URLs). All run locally on this WSL2 host via Docker Desktop.

There is also a **standalone CLI** (not a compose service): `live-tap/desktop_tap.py`
captures OBS / Windows-sink / mic audio via an ffmpeg subprocess and streams
16 kHz mono PCM to the `whisper-live` `/ws-stream` endpoint, printing the
transcript to stdout (one line per utterance) for piping into an LLM. It is
bot-free by design — its only dependency is the `/ws-stream` contract, and it
auto-reconnects on a whisper-live drop like the SPA Live tab and voice bot.
Run via `make live-tap` (or `make live-tap-selftest` for a hardware-free
connectivity check). Lives in `compile-check` and the regression suite
(`test_tap_*`).

**Two deploy modes:** the default (`make up`) is co-deployed with
llm-compose — it owns the external `llmc` net and serves `model_proxy`.
For transcription-only use without llm-compose, `make up-standalone`
layers `compose.standalone.yaml` (redefines `llmc` as a self-managed
bridge under a distinct name) and brings up just valkey + whisper +
whisper-live. LLM extras degrade gracefully in that mode.

## Build / deploy — Makefile is canonical

Always use `make <target>`. Never call `docker compose build` / `up`
directly without checking the equivalent target first — the Makefile
threads in build args (yt-dlp version pin), tag-with-SHA logic, and
correct restart-vs-recreate semantics.

### Common workflows

| Goal | Command |
|---|---|
| Local edit → verify still works | `make test` (lint + 361-test regression, no docker) |
| Lint only (compile + compose + bot import) | `make lint` |
| Build images + redeploy LOCAL only (no registry push) | `make build && make redeploy` |
| Build + push to Docker Hub + redeploy locally (full release) | `make ship` |
| Build + push only, no local redeploy | `make release` |
| Tail bot logs to debug a live failure | `make logs-bot` |
| Tail whisper logs (OCR, VLM, transcription) | `make logs-whisper` |
| Whisper API quick probe | `make status` |
| Bot env change | `make recreate-bot` (NOT `restart-bot` — in-place restart does NOT re-read env) |
| Whisper code change | `make build-whisper && make recreate-whisper` (compose.yaml does NOT bind-mount source despite the misleading `restart-whisper` help text) |
| Bring up WITHOUT llm-compose (transcription core only) | `make up-standalone` / tear down with `make down-standalone` |
| Capture OBS/desktop/mic audio → live transcript on stdout (no bot) | `make live-tap` (verify reachability first with `make live-tap-selftest`) |

### Footguns

- **`make ship` pushes to Docker Hub** — historically slow / sometimes
  stalls indefinitely (home upload bandwidth + accumulated layers since
  last push). If you only need the local stack updated, use
  `make build && make redeploy`. If a push stalls, `kill` it and
  redeploy locally; the `:latest` and `:<SHA>` tags are already on disk.
- **`make restart-whisper` is misleading** — the help text claims
  "picks up app.py changes via bind mount" but compose.yaml has no
  source bind mount. The only way to deploy code changes is
  build + recreate (or `make ship`).
- **`make ship` runs `lint` first** — if `make compose-check` fails
  (Docker Desktop integration dropped), the whole pipeline aborts.
  Verify Docker is up with `docker version` before invoking `make ship`.
- **Do NOT make `llmc` non-external in compose.yaml to "fix" standalone.**
  A non-external compose net that finds a pre-existing `llmc` (created
  by llm-compose) aborts startup with an "incorrect label" error. The
  standalone overlay sidesteps this by giving the bridge a *distinct*
  name (`whisper-standalone-llmc`); llm-compose stays untouched so
  co-deploy keeps working. Use `make up-standalone`, never hand-edit
  the network's `external:` flag.

## Test discipline

`tests/test_regression.py` is a single 4700+ line file. Run via
`python3 tests/test_regression.py` (or `make test`). It stubs aiohttp
+ discord and verifies behaviour at the function level — no Docker
needed. 361/361 should pass before any commit.

When adding code:
- New code path → add a regression test in the appropriate `─── Section ───`
  block in test_regression.py.
- Prompt template changes → add a `test_<feature>_prompts_*` that asserts
  the template imports cleanly + has the expected placeholders +
  security rules (see `test_image_prompts_present` for the pattern).
- New `/api/<endpoint>` → add a `test_api_<x>_route_registered` that
  greps APP_SRC for the `Route(...)` line.

## Project conventions

- **Job dataclass discriminator**: `kind ∈ {video, web, litmus, image}`.
  Adding a new kind requires: validator update in `__post_init__`,
  `_RetrySpec` fields if anything needs to survive a Retry-button click,
  worker dispatch in the `for attempt in range(MAX_RETRIES + 1)` loop,
  a `process_<kind>()` handler, and reply-trigger or auto-trigger wiring.
- **Reply triggers**: `_parse_trigger_keywords` requires the message
  body to consist ONLY of trigger words (`tldr`/`summarize`/`summarise`/
  `litmus`). User-prompt steering doesn't flow through reply triggers
  by design — only the `on_message` URL-detection path supports it.
- **Image attachments**: routed via `_handle_reply_trigger`'s
  image-attachment fallback when no URL is found. `litmus` on images
  is rejected with a friendly error.
- **LLM prompts**: live in `bot/prompts.py`. Brief = single-paragraph;
  KeyPoints = bulleted structured; each prompt block must include
  `{transcript}` + `{reference_block}` placeholders + a security
  preamble that calls the input UNTRUSTED USER CONTENT and forbids
  link invention.
- **VLM data URLs**: `_describe_frame()` hard-codes `image/jpeg` in the
  data URL prefix. Any new path that feeds the VLM MUST pre-normalise
  to JPEG via PIL (see `api_image` in app.py for the pattern). Failure
  mode is `VLM HTTP 400: Failed to load image or audio file`.

## Commits

- Solo dev → direct commit on `main`, no PRs.
- GPG-signed required (`commit.gpgsign=true` is global). If gpg-agent
  hangs, `timeout 15 git commit -S` retry usually works.
- Subject convention: `<area>: <imperative>`. Areas seen in history:
  `bot:`, `whisper:`, `bot+whisper:`, `compose:`, `docs:`, `fix:`,
  `cache:`, `tests:`. Use the narrowest area that fits.
- Real UTF-8 in commit messages (em-dash, ellipsis, arrows, etc.).
  For messages with special chars, write to a file via heredoc with
  REAL characters typed in, then `git commit -F /tmp/msg.txt`. Never
  embed `\uXXXX` escape sequences — bash doesn't expand them.

## What ALREADY exists (don't reinvent)

- A `retry-on-LLM-offline` view (`RetryJobsView`) that survives bot
  restarts via `_job_to_retry_spec`. New kinds need to round-trip
  their data through `_RetrySpec`.
- A processing-emoji lifecycle: ⏳ queued → 🎧/📰/🖼️ fetching →
  🧠 summarising → ✅ done. `PROCESSING_EMOJI` tuple covers all kinds
  for cleanup. Add a new emoji constant + entry when adding a kind.
- A `summarize()` helper with map-reduce + adaptive halving on context
  overflow. Feed it: input body, prompt template, max_tokens, and any
  `{...}` placeholders the template expects via kwargs.
- `send_long_embed(channel, title, body, color)` chunks across embeds
  when content exceeds the 4096-char per-embed cap.
- `resolve_summary_channel(channel)` — applies the optional
  detail-channel split routing (header in original, detail embeds
  in a separate channel).

## Docker Desktop on WSL2 — known fragility

The host runs Docker Desktop with WSL integration. When Docker Desktop
restarts on the Windows side, `/mnt/wsl/docker-desktop/` unmounts and
`docker` commands fail with "The command 'docker' could not be found
in this WSL 2 distro." The containers continue running inside the
Docker VM but the CLI is dead until Docker Desktop is restarted from
the Windows tray. Not a code issue, not recoverable from inside WSL.
