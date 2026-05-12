# Design: server-side job queue for whisper

Status: **Implemented** in the same PR that introduced this doc. Kept here as
the architectural reference for future contributors. The "Open questions"
and "Migration" sections describe historical decisions, not pending work.

The tactical client-side busy-wait described as the predecessor approach
(commit before the queue) has been removed — see `git log -p bot/main.py`
for the WhisperBusyError / wait_for_whisper_idle removal.

## Problem

`/api/transcribe` is synchronous and single-locked (`_transcription_lock`).
Concurrent submissions get HTTP 409. Every consumer (Discord bot, Gradio
UI, MCP server, ad-hoc curl) reimplements busy-wait logic, usually
incorrectly:

- Bot before this PR: 4 attempts × ~130s backoff → false failures on jobs
  >2 minutes.
- MCP: `job_id` + `wait_job` already correct, but only because the MCP
  server polls aggressively — bypasses Discord's queue entirely.
- Gradio UI: blocks the user's tab on a 409.

We patched the bot client-side (`WhisperBusyError` + `wait_for_whisper_idle`).
Long-term that's a band-aid: every new consumer has to learn the dance.
The right place to queue is the server.

## Goals

1. **Single async contract.** Submit returns a job_id immediately.
   Caller polls or subscribes for status/result. Same shape for every
   consumer.
2. **Survives whisper restarts.** A reboot mid-transcription must not
   lose the queue. Persistence in Valkey.
3. **Visibility.** `/api/queue` returns depth, ETA, current job, recent
   history. Operators (and the Discord bot's `/status` slash command)
   can show users where they are in line.
4. **Backwards compat (optional).** Keep `/api/transcribe` synchronous
   behaviour for short jobs to avoid breaking existing curl scripts.
   Best-effort — primary path is async.
5. **Cache layer (bonus).** Shared transcript cache in Valkey so the
   bot, MCP, and Gradio UI all skip re-transcription of the same video.

## Non-goals

- Multi-GPU scheduling. We have one GPU; the queue serialises around
  one worker. (Future: trivially extendable to N workers when we add
  GPUs.)
- Priority queues across tenants. Pure FIFO. If we need priority later,
  add a separate fast lane with a small concurrent budget.
- Cancellation of in-flight jobs. Pause/resume on the running job is
  expensive (no checkpointing in whisperx). Cancel only removes
  not-yet-started jobs from the queue. In-flight cancel = "let it
  finish, then ignore result" — caller-side.
- Auth / rate limiting per consumer. Out of scope; current threat
  model is internal-only.

## Architecture

```
                  ┌──────────────────────────────────────────┐
                  │              Valkey (port 6379)          │
                  │  ┌────────────┐  ┌────────────────────┐  │
                  │  │ queue:waiting│  │ jobs:{id} (hash)   │  │
                  │  │   (LIST)   │  │ {status,result,...}│  │
                  │  └────────────┘  └────────────────────┘  │
                  │  ┌────────────┐  ┌────────────────────┐  │
                  │  │ jobs:active│  │ transcripts:{hash} │  │
                  │  │   (SET)    │  │ (STRING, shared $) │  │
                  │  └────────────┘  └────────────────────┘  │
                  └──────────────────────────────────────────┘
                            ▲                  ▲
                            │                  │
   ┌─────────┐    submit    │                  │    BLPOP / acquire
   │ bot/MCP │──────────────┘                  └─────────┐
   │ Gradio  │                                           │
   │ curl    │  poll /api/jobs/{id}            ┌─────────┴─────────┐
   └─────────┘ ◄────────────────────────────── │  whisper worker   │
                                               │  (one per GPU)    │
                                               │  app.py:           │
                                               │  _run_transcription│
                                               └───────────────────┘
```

### Components

1. **Valkey** — new service in `compose.yaml`. ~10MB RAM, no persistence
   beyond AOF for crash recovery. `valkey:8-alpine`.
2. **API surface** in `app.py`:
   - `POST /api/transcribe` — keeps current sync behaviour BUT also
     accepts `?async=1` (or `async: true` in body) → returns 202 +
     `{job_id}`. Async-by-default in a future major bump.
   - `POST /api/jobs` — explicit async submit. Returns 202 + `{job_id}`.
   - `GET  /api/jobs/{id}` — current status + result if complete.
   - `DELETE /api/jobs/{id}` — cancel if queued (not in-flight).
   - `GET  /api/queue` — `{depth, active, recent: [...]}`.
3. **Worker loop** in `app.py` — async task started in `lifespan`:
   `while True: job = BLPOP("queue:waiting"); run; update jobs:{id}`.
4. **Client libs** — bot/MCP both switch to the new endpoints. Helpers
   in each consumer mirror MCP's `wait_job(job_id)` pattern.

## Schema (Valkey)

```
queue:waiting              LIST  job_ids in FIFO order
jobs:active                SET   job_ids currently running (size = N workers; 1 today)
jobs:{job_id}              HASH  {
                                   status: queued|running|done|failed|cancelled,
                                   submitted_at, started_at, completed_at,
                                   payload: {file_path, model, language, ...},  # original /api/transcribe body
                                   result:  <json transcript+status+subtitle>,  # on done
                                   error:   <string + permanent: bool>,         # on failed
                                   consumer: <free-form tag, e.g. "discord-bot">,
                                   eta_seconds: <int|null>,
                                 }
jobs:recent                LIST  bounded to last 100 job_ids (LPUSH + LTRIM)
transcripts:{sha1(file)}   STRING shared transcript cache (TTL: TRANSCRIPT_CACHE_TTL, default 7d)
```

TTL on `jobs:{job_id}`: 1h after `completed_at`. Enough for clients to
fetch the result; expires automatically.

## API contracts

### POST /api/jobs (new, async)

```json
// request — same body as /api/transcribe today
{
  "file_path": "/tmp/yt-dlp-abc/video.wav",
  "model": "turbo",
  "language": "Auto-detect",
  "diarize": false,
  "initial_prompt": "...",
  "consumer": "discord-bot"   // optional, for visibility/debug
}

// response
HTTP 202 Accepted
{
  "job_id": "wbx_a1b2c3d4",
  "status": "queued",
  "position": 3,
  "eta_seconds": 1840
}
```

### GET /api/jobs/{job_id}

```json
// queued
{ "status": "queued", "position": 2, "eta_seconds": 920 }

// running
{
  "status": "running",
  "started_at": "2026-05-12T13:57:01Z",
  "progress": null   // future: % done from whisperx callbacks
}

// done
{
  "status": "done",
  "result": { "status": "Done -- 4321 segments", "transcript": "...", "subtitle_file": null },
  "completed_at": "2026-05-12T14:42:18Z"
}

// failed
{
  "status": "failed",
  "error": "CUDA OOM",
  "permanent": false,
  "completed_at": "..."
}
```

### DELETE /api/jobs/{job_id}

- Queued → removed from `queue:waiting`, marked cancelled. 200.
- Running → 409 with `{error: "cannot cancel in-flight job"}`. Caller
  can fire-and-forget on their side.
- Already terminal → 404.

### GET /api/queue

```json
{
  "depth": 4,
  "active": [{ "job_id": "wbx_...", "started_at": "...", "consumer": "discord-bot" }],
  "recent": [ /* last 20 terminal jobs */ ]
}
```

### POST /api/transcribe (legacy, sync)

- Default behaviour unchanged for compat: 200 on completion, 409 if
  busy. Internally just synthesises a job, blocks until done, returns
  the result.
- Optional `async=true` flag → forwards to `/api/jobs` 202 response.
- Deprecated in v3 (after one minor cycle of overlap).

## Client adoption

### Discord bot

- Replace the `WhisperBusyError` + `wait_for_whisper_idle` machinery
  with a `submit_job → poll_until_done` pair.
- Existing per-job retry loop stays (network errors, LLM hiccups). Only
  the GPU-contention branch goes away.
- ⏳ reaction stays during `queued` state, swaps to 🎧 on `running`.
- Discord `/status` slash command surfaces `/api/queue` depth + your
  position in line.

### MCP server

- `whisper_yt_transcribe` already returns `job_id` for >50s jobs. Wire
  the server-side job_id straight through — no more polling the
  synchronous endpoint. `wait_job` becomes a thin proxy over
  `GET /api/jobs/{id}`.

### Gradio UI

- Show "Position X in queue (~Y min)" while waiting. Today it just
  blocks with a spinner.
- Cancel button for queued state.

### Shared transcript cache

- Worker hashes `file_path` content (sha1, cheap on already-downloaded
  WAV) before running.
- Cache hit → skip whisper, return cached `{transcript, status}`.
- Cache miss → run, store on success.
- Discord bot's `CACHE_DIR` file cache becomes redundant for video-id
  hits but useful for offline development. Keep both, server cache wins
  on hit.

## Migration

Three phases, each independently shippable:

1. **Server queue, sync endpoint preserved.** Land Valkey + `/api/jobs*`.
   Existing consumers keep using `/api/transcribe` (which now internally
   enqueues + blocks). Zero behaviour change for callers; we get
   persistence and visibility immediately.

2. **Bot switches to async.** Bot uses `/api/jobs` + polling. Remove
   `WhisperBusyError` + `wait_for_whisper_idle` (the tactical fix from
   this PR). MCP server too — its `job_id` plumbing collapses to a thin
   passthrough.

3. **Async-by-default for `/api/transcribe`.** Document deprecation,
   keep sync path for one minor version, remove. Bonus: drop the
   `_transcription_lock` entirely — worker is the only thing that takes
   the GPU.

## Risks

| Risk | Mitigation |
|------|------------|
| Valkey adds a dep | Tiny (~10MB), used in many of our other repos, well-understood. Alpine image. |
| Job persistence across whisper restarts → ambiguous | On boot, worker scans `jobs:active`; any in-flight from a previous run gets re-queued (status flips to `queued`). Single-worker single-GPU = safe to retry. |
| File paths in jobs become stale | Jobs reference `/tmp/yt-dlp-*` files; bot/UI clean those up. On worker pickup, if file missing → fail with `permanent: true`. Add a "submitted file fingerprint" (size + mtime) to detect mid-flight tampering. |
| Cache poisoning | Cache key = sha1 of file contents (not filename). Pre-image attack would require a hash collision; not in our threat model. |
| Out-of-order submit/poll on slow networks | Job IDs are monotonic + opaque. Status endpoint is idempotent. |
| Operator runs `make down -v` | Wipes Valkey. Document this. Recovery: jobs lost but new submissions work. AOF persistence makes container restarts safe; only `-v` (volume delete) is destructive. |

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Valkey + `/api/jobs*` + sync passthrough + tests | 1–2 days |
| 2 | Bot + MCP migration + remove tactical fix | 1 day |
| 3 | Async-default flag + deprecation + cache | 1 day |

Total: **~3–4 days** of focused work.

## Open questions

1. **Persistence vs. ephemeral?** AOF gives durability. RDB snapshots
   cheaper. Default to AOF for safety; document tuning knob.
2. **Cache key includes model + language?** Yes — `sha1(file) +
   model + language + diarize` so different decode settings don't
   collide.
3. **Per-consumer rate limit?** Not now. Add `consumer` tag for
   observability; rate-limit in a future PR if abused.
4. **Webhooks instead of polling?** Nice but adds complexity (callback
   URLs, retry semantics, security). Polling is fine for our consumer
   set.
5. **Multiple GPUs?** Trivially: `WORKER_CONCURRENCY=N` env var,
   N concurrent BLPOP workers. Out of scope for v1.

## Decision needed

Before starting phase 1:
- Confirm Valkey (not Redis) — matches our other deployments.
- Confirm we want to keep `/api/transcribe` sync path for compat
  during phase 1 (vs. cutting over hard).
- Confirm cache TTL default (proposed: 7 days).
