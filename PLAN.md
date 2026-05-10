# PLAN

Forward-looking work tracker. Each item lists a verified scope, open
questions, effort estimate, and risk. Don't pick one up until the **Open
questions** section is resolved — that's how we avoid rebuilding things.

Status legend:
- `[idea]`     — proposed, not yet investigated
- `[verified]` — preconditions checked, ready to design
- `[blocked]`  — waiting on a dependency or decision
- `[active]`   — being implemented now
- `[done]`     — landed; will be moved out of this file
- `[dropped]`  — decided not to do; kept for the rationale

---

## A. Slash commands  `[verified]`

**Goal**: Replace (or supplement) the current "watch every message for URLs"
flow with explicit Discord slash commands. Cleaner UX, native arg validation,
better discoverability.

**Verified preconditions**:
- `discord.py 2.7.1` is installed in the bot (`bot/requirements.txt`).
- `discord.app_commands` and `discord.ui` modules are available
  (verified via `docker exec ... python3 -c 'import discord; ...'`).
- The Discord application needs the `applications.commands` scope. Have
  to confirm that's already on the user's bot invite URL — if not, the
  user has to re-invite the bot. **Open question.**

**Design sketch**:
```
/summarize <url> [prompt: str] [model: str] [vlm: bool]
/transcribe <url> [language: str] [diarize: bool]
/status
/cancel <video_id>           # graceful cancel of a queued job
```
- Add a `discord.app_commands.CommandTree` on `bot.tree`.
- Register commands via `@bot.tree.command(name=…, description=…)`.
- `tree.sync()` on `on_ready` (one-time per guild; can scope to a single
  test guild during dev to avoid Discord's 1-hour propagation cap on
  global commands).
- The current `on_message` URL-listener stays for backward compat. Tag
  jobs as `source: "slash" | "message"` so we can tell them apart in
  logs and eventually deprecate the implicit path.

**Open questions**:
- **Coexistence vs. cutover**: keep both forever, deprecate URL-listener
  on a date, or remove it now? If keeping both, do slash commands
  trigger the same `Job` queue or get separate handling?
- **Guild scope**: dev with one test guild's commands (instant sync) or
  global (1h propagation, but works in DMs and across all servers)?
- **Permission model**: any user can run them, or restrict to certain
  roles? `app_commands.checks.has_permissions` or a role allowlist?

**Effort**: ~2-3 h (signature rewrite + command tree wiring + smoke test).
**Risk**: low. Failure mode is "commands don't show up in the picker"
which is recoverable via re-sync.
**Test impact**: regression suite needs ~3 new tests for command
registration + arg parsing.

---

## B. Speaker rename UI  `[verified]`

**Goal**: When diarization is enabled, let users rename `SPEAKER_00 →
Alice` from inside Discord and have the embed update.

**Verified preconditions**:
- Whisper service already supports diarization end-to-end
  (`load_diarization()`, pyannote models, HF_TOKEN gate at
  `app.py:106-108`). Already loaded; user has HF_TOKEN set per logs.
- Whisper service exposes a Gradio rename UI for the same purpose
  (`_apply_speaker_renames` in `app.py`); we can mirror that logic.
- Bot does NOT currently send `diarize: true` to `/api/transcribe`
  (see `transcribe_payload` in `bot/main.py:545`).
- `discord.ui` (View / Button / Modal / Select) is available in 2.7.1.

**Design sketch**:
1. Add `DIARIZE` env (default off) and per-job opt-in via slash command
   `/transcribe ... diarize:true`. When set, bot passes `diarize: true`
   on `/api/transcribe`.
2. Embed gains a "Rename speakers" `discord.ui.Button` when the response
   contains speaker labels.
3. Click → `discord.ui.Modal` with one text field per detected speaker
   ("SPEAKER_00 → ", "SPEAKER_01 → ").
4. Submit → re-summarize with renamed speakers (or just patch the
   transcript and re-render the embeds — cheaper since LLM calls
   already ran).
5. Persist rename map in `bot-cache` keyed by video_id so subsequent
   cache hits keep the names.

**Open questions**:
- **Cost vs. value**: diarization adds ~30s and the embed gets cluttered
  for >2 speakers. Worth it as opt-in only? Default off is sane.
- **Rerun strategy**: easier to re-do summary with renamed labels (one
  extra LLM round) vs. patch the existing embed text in-place. Patching
  is cheaper and avoids LLM drift but visually requires editing the
  Discord message (allowed, doable).
- **Permission model**: anyone in the channel can rename, or only the
  original requester? The latter is safer but more state to track.

**Effort**: ~4-6 h (diarize integration + Button + Modal + rerun path
+ cache update). **Risk**: medium — interactive UI has more failure
modes than fire-and-forget commands.
**Dependency**: Easier on top of A (slash commands). The Button can be
attached to embeds posted via either path, but a clean `/transcribe`
slash command makes the diarize opt-in obvious.

---

## C. Observability metrics  `[blocked]`

**Goal**: Job counts, latencies, retry rates, VLM-fallback rate, model
swap counts. Enough signal to know "is the bot healthy?" without tailing
logs.

**Verified preconditions**:
- **No existing observability stack on the host**. `docker ps` shows no
  grafana/prometheus/loki containers; `~/llm-compose` has none either.
- `prometheus_client` is a small (~50 KB) stdlib-only-ish dep.

**Why this is blocked**:
The metrics endpoint itself is ~30 lines, but without a scraper +
dashboard it's just a `/metrics` page nobody reads. Three options to
unblock:

  1. **Skip metrics, write structured logs**: drop in
     `python-json-logger`, log per-event JSON, scrape with a CLI grep
     when needed. Lowest cost, zero infra.
  2. **Stand up minimal Prometheus + Grafana** as a separate compose
     project (`~/observability/`?). Reusable for `lockstep`,
     `composer`, `llm-compose`, etc. Real value but ~half-day setup.
  3. **Use an existing hosted service** (Grafana Cloud free tier,
     Better Stack, etc.). Lowest infra cost, more credentials to
     manage.

**Open questions**:
- **Which path?** (1 / 2 / 3). Picking #1 first and graduating to #2
  later is reasonable.
- **Scope of metrics**:
  - Job counts (by status: success/permanent-fail/transient-retry)
  - Per-stage latency: download / transcribe / VLM / summarize
  - Cache hit rate
  - VLM fallback rate (visual-only, hybrid, user-forced)
  - Model swap count (proxy already logs this; could expose)
  - Discord post latency

**Effort**: 30 min for option 1, ~half-day for option 2.
**Risk**: low for option 1, medium for option 2 (compose project +
volumes + alerting setup).

---

## D. Per-channel preset config  `[idea]`

**Goal**: Different Discord channels use different LLM presets. E.g.
`#serious-podcasts` uses gemma-4-31B for richer summaries; `#bot-spam`
uses a 4B model.

**Verified preconditions**:
- Bot currently uses one global `LLM_MODEL` env var.
- `bot-cache` named volume can hold a `channels.json` config file —
  already writable as uid 1000.

**Design sketch**:
- New file `bot-cache/channels.json`:
  ```json
  {
    "123456789": { "model": "gemma-4-31B-it-Q4_K_M", "vlm_enabled": true },
    "987654321": { "model": "Qwen3.5-4B-Q8_0",       "vlm_enabled": false }
  }
  ```
- Slash command `/config model <preset>` (admin-gated) writes the entry.
- `process()` looks up `channels.json[message.channel.id]` and falls
  back to env defaults.

**Open questions**:
- **Source of truth**: one JSON file vs. SQLite. JSON is fine for tens
  of channels; SQLite for hundreds + audit log of changes.
- **Permission model**: who can configure? Channel-permission `Manage
  Channel`? A role allowlist? A static "admin user IDs" env?
- **Concurrency**: bot only reads/writes from one process, so a simple
  `threading.Lock` around the JSON file works. No need for SQLite just
  for that.

**Effort**: ~3-4 h (config schema + load/save + slash command + dispatch
in process()). **Risk**: medium — adds a config surface that must be
documented and maintained.
**Dependency**: A (slash commands) if config is set via slash.

---

## E. Per-user rate limiting  `[idea]`

**Goal**: A single Discord user can't DoS the bot by spamming URLs.
Each video download is ≤ 500 MB; without limits, malicious or bored
users can chew bandwidth + GPU time.

**Verified preconditions**:
- Bot has no rate limiting today.
- Worker queue is shared across all users (`asyncio.Queue`).

**Design sketch**:
- In-memory rate limit: `defaultdict[user_id, deque[timestamp]]`.
- Reject a job if user has more than `MAX_JOBS_PER_USER_PER_HOUR` in
  the last 60 minutes.
- React `🚫` to the offending message instead of queueing it.
- Configurable via env: `MAX_JOBS_PER_USER_PER_HOUR=5`,
  `RATE_LIMIT_BYPASS_USERS=…` (admins).

**Open questions**:
- **Window size**: per-hour, per-day, sliding window?
- **Total queue cap** independent of per-user (in case multiple users
  collectively flood)? Probably yes, e.g. 20 jobs queued max.
- **Persistence**: do we care about counting across bot restarts?
  Probably not; an attacker rebooting the bot to reset their counter
  is a different problem.

**Effort**: ~1-2 h. **Risk**: low.

---

## F. Web frontend for past summaries  `[idea]`

**Goal**: Browse past transcripts/summaries from a webpage. Useful for
finding "that video about X we summarised three months ago".

**Verified preconditions**:
- Bot writes transcripts to `bot-cache:/app/cache/<video_id>.txt`.
- Whisper service has a Gradio UI at `:7860/` already (live).
- No existing web UI for the bot's outputs.

**Design sketch**: probably overkill. Two simpler alternatives:
1. **Search slash command**: `/find <keywords>` → grep through
   `bot-cache`, return matching titles + jump links to the original
   Discord posts.
2. **Static HTML index**: nightly cron generates `index.html` from
   `bot-cache/` with searchable list. Served by Gradio's static
   handler or a tiny nginx.

**Open questions**:
- Is there demand? If you've never re-looked-up a past summary, drop.
- Do we even have the original Discord message link? Yes — bot could
  store it in the cache file header.

**Effort**: ~1 h for option 1, ~3 h for option 2.
**Risk**: low. Drop if not actually needed.

---

## G. App.py monolith refactor  `[blocked]`

**Goal**: Split `app.py` (currently ~2500 lines) into logical modules.

**Status**: Deliberately deferred. Refactoring without a feature touching
the affected code creates two-source-of-truth problems (we tried it once
with `whisper_app/` and deleted it). Re-evaluate when adding a feature
that meaningfully changes one of these areas.

**When to revisit**:
- Adding a real-time progress streaming endpoint touches the
  transcription pipeline → split out `transcription.py`.
- Adding a second VLM mode (e.g. video-level instead of frame-level)
  → split out `vlm.py`.
- Migrating Gradio to a non-Gradio frontend → split out `ui.py` first.

---

## Decision queue

When picking next, walk this list top-down. The order reflects impact
and unblocking order, not arbitrary priority.

1. **A. Slash commands** — unblocks B, D. Lowest risk per hour.
2. **B. Speaker rename UI** — meaningful UX feature, depends on A.
3. **D. Per-channel preset config** — useful when you have varied content
   types in different channels. Depends on A.
4. **E. Per-user rate limiting** — security; do once you have any
   non-trusted users with bot access.
5. **C. Observability** — only after the above generate enough volume
   that "is it healthy?" stops being a tail-the-logs question.
6. **F. Web frontend** — drop unless demand emerges.
