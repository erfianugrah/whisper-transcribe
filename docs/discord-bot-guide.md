# Discord bot guide

Setup and usage for the **TL;DW Bot** — transcribes videos posted in
Discord, summarises them via a local LLM, posts the result back as
embeds.

---

## Quick reference (most common usage)

| You do… | Bot does… |
|---|---|
| Paste any video URL in a watched channel | Transcribes + summarises automatically (3 embeds: brief, key points, chapters; 4th Community Reaction for YouTube). |
| `<URL> describe what's on the slides` | Forces frame-level VLM enrichment + steers summary toward your ask. |
| Reply `tldr` (or `summarize`) to a message containing a URL | **Web URL flow** — scrapes the article and posts brief + key points + sections. Works for any non-video URL. **Reddit + HackerNews** get a structured fetch (linked article + post + top comments). If the URL IS a video, falls through to the video pipeline. |
| Reply `litmus` to a message containing a URL | **AI litmus test** — surfaces stylistic + metadata signals (LLM-tic phrases, em-dash density, hedge usage, generic buzzwords, listicle structure, domain age via Wayback, AdSense detection, author byline). Forensic report, no verdict. |
| Reply `tldr litmus` (or any keyword combo) to a message | **Chained reply** — fires both flows in one go. Order in your reply preserves order of execution; duplicates dedupe; rate-limit charges per fired job. |
| `/summarize url:<URL> prompt:<text>` | Slash equivalent of paste-with-text (cleaner UX, arg validation, `model:` autocompletes from the LLM proxy). |
| `/transcribe url:<URL> diarize:true prompt:<text>` | Adds speaker labels. Embed grows a 🏷️ Rename speakers button. `prompt:` symmetric with `/summarize`. |
| `/web url:<URL>` | Slash equivalent of the `tldr` reply — summarise any article. |
| `/litmus url:<URL>` | Slash equivalent of the `litmus` reply — AI litmus test. |
| `/progress` | Your in-flight jobs with phase + elapsed + ETA. Far richer than the reaction emojis. |
| `/cancel job:<pick from list>` | Cancel one of YOUR queued or transcribing jobs. Autocompletes from your live entries. |
| `/queue` | Server-wide view of everything in the bot queue right now. |
| `/find query:<keywords> kind:<video\|web> since_days:<N>` | Search past transcripts with optional filters. |
| `/recent kind:<video\|web> limit:<N>` | Last N cached summaries (newest first). |
| `/redo video_id:<id>` | Re-run a cached YouTube job with different translate / model. Autocompletes from cache. |
| `/transcribe-join` | Join your voice channel and live-transcribe it (per-speaker, attributed, timestamped) into a thread. Needs `VOICE_TRANSCRIBE_ENABLED=1`. |
| `/transcribe-leave` | Stop live transcription, leave the channel, archive the thread. |
| `/myconfig model:<name> diarize:true` | Your personal defaults — applied in any channel. |
| `/help topic:<overview\|triggers\|admin\|limits\|errors\|translate>` | Ephemeral help cards. |
| `/status verbose:true` | Queue depth, service health, plus per-job phase/elapsed when verbose. |
| `/config …` | Per-channel: model / VLM / diarize / yt_comments defaults (needs Manage Channel). |
| `/serverconfig …` | Per-server: where Key Points + Chapters land (needs Manage Server). |

---

## Initial setup

### 1. Create a Discord application + bot

1. Go to https://discord.com/developers/applications → **New Application**.
2. **Bot** tab → reset/copy the token. Put it in `bot/.env` as
   `DISCORD_TOKEN=…`.
3. **Bot** tab → enable the **Message Content Intent** (required for
   the URL-listening flow). Server Members and Presence intents are
   not needed.
4. **OAuth2 → URL Generator**:
   - **Scopes**: check **`bot`** AND **`applications.commands`**
     ← critical — without the second one, slash commands won't show up.
   - **Bot permissions**: at minimum `Send Messages`, `Embed Links`,
     `Add Reactions`, `Read Message History`. For diarize rename UI:
     `Use External Emojis`. For per-channel `/config`: nothing extra
     (the command itself checks the user's `Manage Channel` perm).
5. Copy the generated URL, open it, pick your server. Authorise.

### 2. Set bot env vars

In `bot/.env` (see `bot/.env.example` for full options):

```dotenv
DISCORD_TOKEN=...                  # required
WHISPER_API_URL=http://whisper:7860
LLM_API_URL=http://model_proxy:11434/v1
LLM_MODEL=Qwen3.5-4B-Q8_0          # any model on llm-compose proxy
EXA_API_KEY=...                    # optional — drives terminology hints

# Rate limiting (sane defaults)
MAX_JOBS_PER_USER_PER_HOUR=20
MAX_QUEUE_SIZE=40
RATE_LIMIT_BYPASS_USERS=          # CSV of Discord user IDs

# Slash command sync mode:
#   unset → global commands (works in DMs + all guilds; ~1h propagation)
#   set   → guild-scoped (instant sync for one test server during dev)
DISCORD_GUILD_ID=                 # numeric guild ID

# Optional structured logging (pipe to your log aggregator)
LOG_JSON=0                         # 1=JSON lines; 0=human-readable
LOG_LEVEL=INFO

# Optional: restrict to specific channel IDs (CSV); empty = all channels
ALLOWED_CHANNELS=
```

### 3. Start the stack

```bash
cd ~/whisper-transcribe
make ship   # build + push + recreate
```

First start: bot logs `Slash commands synced …` once it's ready. If
`DISCORD_GUILD_ID` is set, your slash commands appear in that guild
within seconds. If unset, allow up to 1 hour for global commands to
propagate to all your servers (Discord-side cache).

---

## How users interact

### Path A — Just paste a URL

```
@user: https://www.youtube.com/watch?v=abc123
🤖    [reaction: ⏳ → 🎧 → 🧠 → ✅]
🤖    [embed: TL;DW + Key Points + Chapters + Community Reaction]
```

For YouTube videos, a **4th embed (Community Reaction)** summarises top
comments — what viewers broadly agree on, where they disagree, and which
comments the creator engaged with (pinned / hearted / replied). On by
default; channels can opt out with `/config yt_comments:false`. Comments
are skipped on cache hits (the cache pre-dates the feature).

The bot watches every message in allowed channels. **Auto-trigger only
fires for URLs whose shape clearly identifies a video** — YouTube
watch/shorts/live links, Twitch VODs and clips, Vimeo numeric IDs,
TikTok video paths, v.redd.it, Dailymotion, Rumble, Odysee, Bilibili,
SoundCloud, etc.

Text posts on video-hosting domains (e.g. a Reddit text post, a
non-video tweet, an Instagram profile page) **don't** auto-trigger,
even though those domains are in `VIDEO_DOMAINS`. This avoids spamming
the channel with "yt-dlp can't handle this URL" errors every time
someone shares a Reddit article link. To get a summary of a text post,
use Path C below (`tldr` reply).

### Path B — Paste a URL + steering text

```
@user: https://www.youtube.com/watch?v=abc123
       Pay attention to the slides; what frameworks are mentioned?
🤖    [embed includes "User request" field showing your ask]
```

Any non-trivial text alongside the URL forces VLM enrichment (the bot
extracts video frames and asks a vision-language model to describe
them) AND steers the summary LLM toward your request.

### Path C — Web URL summary (`tldr` reply)

Reply to a message containing any non-video URL with **`tldr`** or
**`summarize`** (case-insensitive, optional trailing punctuation):

```
@user1: This article finally explains the new release: https://example.com/post
@user2: tldr
🤖    [reaction: ⏳ → 📰 → 🧠 → ✅ on @user2's "tldr" message]
🤖    [embed: TL;DR + Key Points + Sections]
```

What counts as a trigger:
- The reply body must be ONLY the keyword (`tldr`, `summarize`, `summarise`)
  with optional trailing `.` or `!`. Sentences like "give me a tldr of this"
  intentionally don't trigger.
- The replied-to message must contain a URL. The bot picks the first one
  and ignores Discord-internal links (channel/message links).

What it does:
- **Clear video URL** (YouTube watch link, Twitch VOD, Vimeo, etc.) →
  routes through the existing video pipeline. Cache hit if already
  summarised.
- **Anything else** → routes to the web pipeline. Scrapes via Crawl4AI
  (Playwright + readability), falls back to FlareSolverr if Cloudflare
  blocks it. Output: brief paragraph, key-points list, sections list
  (semantic headings, no timestamps). Hard CAPTCHA / Turnstile pages
  still fail — the bot will say so.

**Routing fallback** (the screenshot-in-the-bug-report case): if a URL
got mis-classified as video and yt-dlp returns "Unsupported URL" or
"No video found", the bot automatically retries as a web job. So a
Reddit URL that turns out to be a link post to an article will end up
summarised as the article. Genuine video failures (private, age-gated,
geo-blocked) don't fall through — they fail cleanly because the
article version would be just as restricted.

Cache: scraped articles are cached by URL hash for 24h
(`CACHE_TTL` default). Re-replying `tldr` to the same URL inside that
window reuses the cached scrape and just regenerates summaries.

### Path D — AI litmus test (`litmus` reply)

Reply to a message containing any URL with **`litmus`** (or `litmus.` /
`litmus?`):

```
@user1: this article smells like AI to me: https://example.com/post
@user2: litmus
🤖    [reaction: ⏳ → 📰 → 🧠 → ✅ on @user2's "litmus" message]
🤖    [embed: 🔍 Litmus: <article title>]
```

What it does:
1. Scrapes the article (same path as `tldr` — Crawl4AI / FlareSolverr,
   Reddit + HackerNews structured fetch).
2. Runs a regex pre-pass over the text for stylistic markers — LLM-tic
   phrases (`delve into`, `tapestry of`, `navigate the landscape`, …),
   em-dash density (LLMs over-use them), hedge phrases (`it's worth
   noting`), generic buzzwords (`robust`, `seamless`, `cutting-edge`),
   listicle structure (heading + bullet density), substance markers
   (presence of quotes, named individuals, specific dates / numbers).
3. Fetches metadata in parallel — domain age via the Wayback Machine
   (recently-registered domains pumping out content are a strong
   AI-content tell), author byline detection (`<meta name=author>`,
   `rel=author`), AdSense / DoubleClick markers (cheap content-mill
   signal).
4. Aggregates signals into a severity score. **Skips the LLM call** if
   the score is clearly clean OR clearly LLM-style — only the ambiguous
   middle range gets a qualitative LLM read.
5. Posts a forensic embed: signals list with severity dots
   (🟢 typical-human / 🟡 elevated / 🔴 beyond typical-human), an
   optional qualitative read, and an explicit caveat about detection
   unreliability.

**No verdict by design.** AI detection is fundamentally unreliable —
false positives are common on careful technical writing, and lightly-
edited LLM output evades easy classification. The bot describes what
it sees and lets you decide.

**Trigger requires the bare keyword** — `litmus`, `Litmus`, `litmus.`,
`litmus?`, etc. Sentences like "give me a litmus test of this" don't
trigger; that's by design (avoids accidental fires).

### Chained reply (`tldr litmus` etc.)

Reply with multiple keywords in any order to fire both flows from a
single reply:

```
@user1: this looks AI-generated to me: https://example.com/article
@user2: tldr litmus
🤖    [⏳ → 📰 → 🧠 → ✅] (summary embeds)
🤖    [⏳ → 📰 → 🧠 → ✅] (litmus embed)
```

Rules:
- Body must be ENTIRELY composed of recognised keywords (any combination
  of `tldr` / `summarize` / `summarise` / `litmus`) plus optional
  punctuation. Any extra word — including `and` — disables the trigger.
- Order preserves dispatch order. `tldr litmus` runs summary first;
  `litmus tldr` runs litmus first. The second job waits in queue while
  the first runs (single GPU, sequential worker).
- Duplicates dedupe: `tldr tldr` and `tldr summarize` both fire one
  summary job (the user gets charged for one slot, not two).
- Rate-limit + queue-cap are checked atomically. If you have one slot
  left and reply `tldr litmus`, both are rejected — no partial fires.

### Path E — Slash commands (recommended for explicit options)

```
/summarize url:<URL> prompt:<text> model:<override>
```

- `url` — required.
- `prompt` — optional steering, same effect as Path B.
- `model` — override `LLM_MODEL` for this run. **Autocompletes** from the
  LLM proxy's `/v1/models` endpoint (cached for 5 minutes).

```
/transcribe url:<URL> diarize:true prompt:<text>
```

- Same flow as `/summarize` but with speaker diarization enabled.
- `prompt` is symmetric with `/summarize` — forces VLM, steers summary.
- Emoji on the brief embed: 🏷️ **Rename speakers** button.
- Click → modal with a text field per detected speaker → submit →
  bot re-runs the brief summary with your names baked in.

```
/web url:<URL> prompt:<text>
```

Slash equivalent of the `tldr` reply trigger. Pass any non-video URL —
the bot scrapes via Crawl4AI / FlareSolverr / Reddit structured-fetch
and produces brief + key points + sections embeds.

```
/litmus url:<URL>
```

Slash equivalent of the `litmus` reply trigger. Forensic AI-writing
signals on any article. Always runs the page-fetch path (no video
transcription) even on video URLs.

```
/progress
```

Your in-flight jobs with phase + elapsed time + ETA. Replaces guessing
what 🎧 means — surfaces title (post-download), current phase
(downloading / transcribing / scraping / summarising), elapsed wall
time, and an ETA when computable. Ephemeral, only you see it.

```
/cancel job:<autocomplete from your active jobs>
```

Cancel one of your own queued or running jobs. Behaviour by phase:

- **queued** (still in bot queue) — soft cancel, drops cleanly.
- **transcribing** with whisper-side queue position — forwards `DELETE
  /api/jobs/{id}`; whisper cancels if still queued server-side.
- **transcribing in flight** — whisperX has no safe interrupt point.
  Bot tells you to wait.
- **downloading / scraping / summarising** — also non-interruptible
  today.

```
/queue
```

Server-wide queue listing — every active and queued job with phase,
elapsed time, and submitter. Useful for "is the worker stuck?" or
"why hasn't mine started yet?".

```
/find query:<keywords> kind:<video|web|any> since_days:<N> limit:<N>
```

Searches the bot's transcript cache (case-insensitive substring).
Filters:
- `kind` — video, web, or any (default any).
- `since_days` — limit to the last N days.
- `limit` — 1-25 results (default 10).

```
/recent kind:<any|video|web> limit:<N>
```

Newest cached summaries first. Same filtering as `/find` but without a
keyword. Use this to glance at what's been summarised recently.

```
/redo video_id:<autocomplete>
```

Re-run a cached YouTube job with different `translate` / `model` /
`prompt` without retyping the URL. Defaults `refresh:true` (the common
case is "the model got something wrong, try again"); pass
`refresh:false` to re-summarise the existing cached transcript without
re-downloading.

```
/help topic:<overview|triggers|admin|limits|errors|translate>
```

In-Discord ephemeral help cards. `overview` (default) lists every
slash command; the named topics drill into specifics.

```
/status verbose:true
```

Shows queue size, your usage in the last hour, whisper service health,
active VLM model. With `verbose:true`, also lists the bot queue's
contents (titles, kinds, phases, elapsed).

---

## Voice-channel live transcription

Live-transcribe an active voice call into a text thread. Off by default —
set `VOICE_TRANSCRIBE_ENABLED=1` (and the `discord-ext-voice-recv`
extension + libopus must be present in the bot image) for the two slash
commands below to register. Discord's mandatory DAVE end-to-end voice
encryption is handled transparently by discord.py 2.7.1+.

```
/transcribe-join
```

Run it **while you are in a voice channel**. The bot joins that channel,
creates a dedicated public thread (🎙️ `<channel> — <date> <time>`), posts
a consent notice, and streams the call to whisper-live live:

- **One stream per speaker.** Each talker gets their own whisper-live
  slot, so lines are attributed by display name and timestamped as they
  land in the thread.
- Audio is resampled (48 kHz stereo → 16 kHz mono) on the receive thread
  and converted to text live — **nothing is stored**.
- Capacity is bounded: `VOICE_MAX_SPEAKERS` simultaneous streams (each
  consumes one `LIVE_MAX_STREAMS` slot on whisper-live). A speaker's
  stream is closed after `VOICE_SPEAKER_IDLE_S` seconds of silence and
  re-opens on their next utterance.
- Where the thread is created is governed by `VOICE_TRANSCRIPT_CHANNEL_ID`
  (`0`/unset = the channel the command was invoked in). If the bot lacks
  thread permissions it posts in the channel directly.

Consent notice posted on join:

> 🔴 **This voice channel is now being transcribed.** Audio is converted
> to text live and not stored. Leave the channel if you do not consent.

```
/transcribe-leave
```

Flushes any buffered audio, closes every per-speaker stream (freeing the
whisper-live slots), disconnects from the voice channel, posts
*“— transcription ended —”*, and archives the thread so it drops out of
the active list.

**Tuning latency.** Snappiness is governed by the bot-side flush cadence
(`VOICE_SEND_INTERVAL`, `VOICE_MAX_SILENCE_S`) and the whisper-live
streaming knobs (`LIVE_PROCESS_INTERVAL`, `LIVE_MIN_CHUNK_S`,
`LIVE_TAIL_SILENCE_S`, `LIVE_BEAM_SIZE`). See the Environment Variables
tables in the [README](../README.md#environment-variables). These are
tuned low in `compose.yaml` for live output.

**Verify it freed capacity** after `/transcribe-leave`: whisper-live
`GET /health` should report `active_streams` back to `0`.

---

## Per-user defaults (`/myconfig`)

Want the bot to default to a particular model or always diarize for you
without retyping each time? Set per-user defaults — they apply in every
channel where you use the bot.

```
/myconfig model:gemma-4-31B-it-Q4_K_M    # your default model
/myconfig diarize:true                    # default to speaker diarization on /transcribe
/myconfig show:true                       # print your current overrides
/myconfig clear:true                      # wipe all your overrides
/myconfig model:                          # empty value clears that one field
```

**Precedence** (most specific → least):

1. Explicit slash argument (`/summarize model:<x>` wins for that run).
2. Per-channel config (`/config model:<y>` — channel admin's policy).
3. Per-user config (`/myconfig model:<z>` — your personal preference).
4. Env default (`LLM_MODEL`).

Channel config outranks user config on purpose: a channel admin's
policy ("this channel summarises long-form videos with the 31B model")
should beat a personal preference. Use `/myconfig` for things that
follow you around (your usual model, "I always want speaker labels"),
not policy.

Stored under `bot-cache/users.json`.

## Channel-specific config

Want a serious-discussion channel to use `gemma-4-31B`, but a casual
one to stay on the small default? Run `/config` in the channel you
want to customise:

```
/config model:gemma-4-31B-it-Q4_K_M     # use this preset for /summarize in this channel
/config diarize:true                     # enable diarization by default here
/config vlm:false                        # disable VLM fallback in this channel
/config yt_comments:false                # skip the Community Reaction embed for YT videos
/config show:true                        # print current config without changing it
```

`/config` requires the **Manage Channel** permission. Settings persist
across bot restarts (stored in `bot-cache/channels.json`).

To clear an individual setting:

```
/config model:                           # empty value clears the override
```

## Server-wide config

Some settings only make sense at the server level — most importantly,
where the detailed Key Points + Chapters embeds should land.

```
/serverconfig summary_channel:#bot-summaries   # detail embeds go here
/serverconfig show:true                          # print current server config
/serverconfig clear:true                         # revert to global defaults
```

`/serverconfig` requires the **Manage Server** permission. Settings
persist across bot restarts (stored in `bot-cache/guilds.json`).

When set, the bot keeps the brief TL;DW embed in the originating channel
(with a "Full breakdown →" jump link) but posts Key Points and Chapters
to the chosen summary channel. The bot must have **Send Messages** +
**Embed Links** in that channel — `/serverconfig` checks before saving.

**Precedence** (most specific wins):

1. Per-guild `/serverconfig summary_channel` — server-wide.
2. Global `SUMMARY_CHANNEL` env var — every server uses the same channel.
3. None — Key Points / Chapters post in the same channel as the brief.

Multi-server example: each server has its own summaries archive without
leaking detail embeds across servers.

```
Server A:  /serverconfig summary_channel:#archive-A
Server B:  /serverconfig summary_channel:#archive-B
Server C:  (no command run — keeps everything in-channel)
```

---

## Rate limits

Each user is capped at `MAX_JOBS_PER_USER_PER_HOUR` (default **20**)
sliding-window per hour. Hitting the cap reacts 🚫 to the offending
message and posts a clarifying reply with the retry timer.

The bot also enforces a global queue cap (`MAX_QUEUE_SIZE`, default
**40**) — once that fills, new jobs are rejected with a "queue full"
error.

**Chained replies count as N jobs**: replying `tldr litmus` to a message
charges 2 slots against your per-hour cap (atomic — either both queue
or neither does). Same applies to multiple URLs posted in one message.

To bypass per-user limits for trusted users (yourself, mods),
add their Discord user IDs to `RATE_LIMIT_BYPASS_USERS`:

```dotenv
RATE_LIMIT_BYPASS_USERS=123456789012345678,234567890123456789
```

Bypass users still respect the queue cap — no admin can crash the bot
either.

To get a Discord user ID: enable Developer Mode in Discord settings,
right-click the user → **Copy User ID**.

---

## What happens when things go wrong

| Bot reaction | Means | What to do |
|---|---|---|
| ⏳ | Job queued, waiting for the worker. | Wait. |
| 🎧 | Downloading audio (video flow). | Wait (~10s for short videos, minutes for long). |
| 📰 | Scraping article (web flow). | Wait (~5-30s; longer when FlareSolverr fallback kicks in). |
| 🧠 | Content acquired, now summarising. | Wait. |
| ✅ | Done — embed has been posted. | — |
| 🚫 | Rate limit hit. | Wait the time the bot mentions, or ask an admin to add you to bypass. |
| ❌ | Permanent failure — won't retry. | Check the failure message; usually it's content the bot can't process (private video, no media, geo-blocked, hard CAPTCHA, etc.). |

For ages-restricted videos: see `bot/.env.example` →
`YT_DLP_COOKIES_FILE`. You'll need a cookies.txt from a logged-in
YouTube account mounted into the whisper container.

---

## Diagnostics

```bash
# What's the bot doing right now?
make logs-bot

# Whisper service health (queue, GPU, vision model status)
make status            # whisper /api/status
curl -s http://localhost:7860/api/status | jq

# Scraper services (web URL flow)
make logs-scraper      # both crawl4ai + flaresolverr
make logs-crawl4ai
make logs-flaresolverr
make status-scraper    # probe both /health endpoints from inside the bot

# What slash commands are registered?
# (Discord client → type / in any channel → bot picker shows them.
#  If nothing shows up, the OAuth2 invite URL likely missed
#  applications.commands scope — re-invite with the right URL.)

# Force-resync slash commands (e.g. after editing definitions).
# Slash commands sync at startup; restart the bot:
make restart-bot

# Recreate scraper containers (e.g. after a hung Chromium instance):
make restart-scraper
```

---

## Logs

Default logs are human-readable:

```
2026-05-10 10:32:41,872 INFO     tldw: Queued abc123 from user#1234 (channel=987654)
```

For machine-parseable / aggregator-friendly output, set
`LOG_JSON=1` in `bot/.env`. Each log line becomes a single JSON
object:

```json
{"ts": "2026-05-10T10:32:41.872+00:00", "level": "INFO", "logger": "tldw",
 "msg": "Queued abc123 from user#1234 (channel=987654)"}
```

`LOG_LEVEL` accepts standard Python levels (`DEBUG`, `INFO`, `WARNING`,
`ERROR`).

---

## When to use which path

| Scenario | Recommended |
|---|---|
| Just pasted a video, want a summary | URL paste (Path A) |
| Want the bot to focus on something specific | URL + text (Path B) or `/summarize prompt:` |
| Saw an interesting article someone posted | Reply `tldr` or `/web url:` |
| Wondering if an article is AI-generated / AI-spam | Reply `litmus` or `/litmus url:` |
| Want explicit control over args (model, diarize) | `/summarize`, `/transcribe` (Path E) |
| "What's the bot doing with my job?" | `/progress` (phase + ETA per job) |
| Queued wrong URL by accident | `/cancel job:` |
| "Is the queue stuck?" | `/queue` or `/status verbose:true` |
| Need to find a past summary | `/find query:` (with filters) |
| Glance at what's been summarised recently | `/recent` |
| Want to re-run a YouTube summary with diff opts | `/redo video_id:` (autocompletes from cache) |
| Always want a particular model / diarize | `/myconfig` (once) |
| Channel has different needs from defaults | `/config` (one-time, admin) |
| New to the bot, need a tour | `/help topic:overview` |
| Diarized output with renaming | `/transcribe diarize:true` then 🏷️ Rename speakers |
| Live-transcribe an ongoing voice call | `/transcribe-join` (then `/transcribe-leave` to stop) |

The auto-paste, reply-trigger, and slash paths all share the same job
queue, rate limits, and cache. None of them replace each other — pick
whichever matches your context. Casual users keep pasting and replying
`tldr`; power users use slash for explicit control.
