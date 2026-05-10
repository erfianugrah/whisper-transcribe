# Discord bot guide

Setup and usage for the **TL;DW Bot** — transcribes videos posted in
Discord, summarises them via a local LLM, posts the result back as
embeds.

---

## Quick reference (most common usage)

| You do… | Bot does… |
|---|---|
| Paste any video URL in a watched channel | Transcribes + summarises automatically (3 embeds: brief, key points, chapters). |
| `<URL> describe what's on the slides` | Forces frame-level VLM enrichment + steers summary toward your ask. |
| `/summarize url:<URL> prompt:<text>` | Same as above via slash command (cleaner UX, arg validation). |
| `/transcribe url:<URL> diarize:true` | Adds speaker labels. Embed grows a 🏷️ Rename speakers button. |
| `/find query:<keywords>` | Searches your past transcripts for matching content. |
| `/status` | Shows queue depth, your rate-limit usage, whisper service health. |

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
MAX_JOBS_PER_USER_PER_HOUR=5
MAX_QUEUE_SIZE=20
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
🤖    [embed: TL;DW + Key Points + Chapters]
```

The bot watches every message in allowed channels. Anything yt-dlp
supports (YouTube, Twitch, Vimeo, Twitter/X, Instagram, Reddit, Rumble,
Odysee, Kick, Bilibili, SoundCloud, Dailymotion, …) triggers a job.

### Path B — Paste a URL + steering text

```
@user: https://www.youtube.com/watch?v=abc123
       Pay attention to the slides; what frameworks are mentioned?
🤖    [embed includes "User request" field showing your ask]
```

Any non-trivial text alongside the URL forces VLM enrichment (the bot
extracts video frames and asks a vision-language model to describe
them) AND steers the summary LLM toward your request.

### Path C — Slash commands (recommended for explicit options)

```
/summarize url:<URL> prompt:<text> model:<override>
```

- `url` — required.
- `prompt` — optional steering, same effect as Path B.
- `model` — override `LLM_MODEL` for this run (advanced; pick from
  `model_proxy`'s `/v1/models`).

```
/transcribe url:<URL> diarize:true
```

- Same flow as `/summarize` but with speaker diarization enabled.
- Emoji on the brief embed: 🏷️ **Rename speakers** button.
- Click → modal with a text field per detected speaker → submit →
  bot re-runs the brief summary with your names baked in.

```
/find query:<keywords>
```

Searches the bot's transcript cache (case-insensitive substring) and
returns up to 10 matches with clickable links.

```
/status
```

Shows queue size, your usage in the last hour, whisper service health,
active VLM model.

---

## Channel-specific config

Want a serious-discussion channel to use `gemma-4-31B`, but a casual
one to stay on the small default? Run `/config` in the channel you
want to customise:

```
/config model:gemma-4-31B-it-Q4_K_M     # use this preset for /summarize in this channel
/config diarize:true                     # enable diarization by default here
/config vlm:false                        # disable VLM fallback in this channel
/config show:true                        # print current config without changing it
```

`/config` requires the **Manage Channel** permission. Settings persist
across bot restarts (stored in `bot-cache/channels.json`).

To clear an individual setting:

```
/config model:                           # empty value clears the override
```

---

## Rate limits

Each user is capped at `MAX_JOBS_PER_USER_PER_HOUR` (default **5**)
sliding-window per hour. Hitting the cap reacts 🚫 to the offending
message and posts a clarifying reply with the retry timer.

The bot also enforces a global queue cap (`MAX_QUEUE_SIZE`, default
**20**) — once that fills, new jobs are rejected with a "queue full"
error.

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
| 🎧 | Currently downloading audio + transcribing. | Wait (~10s for short videos, minutes for long). |
| 🧠 | Transcribed, now summarising. | Wait. |
| ✅ | Done — embed has been posted. | — |
| 🚫 | Rate limit hit. | Wait the time the bot mentions, or ask an admin to add you to bypass. |
| ❌ | Permanent failure — won't retry. | Check the failure message; usually it's content the bot can't process (private video, no media, geo-blocked, etc.). |

For ages-restricted videos: see `bot/.env.example` →
`YT_DLP_COOKIES_FILE`. You'll need a cookies.txt from a logged-in
YouTube account mounted into the whisper container.

---

## Diagnostics

```bash
# What's the bot doing right now?
make logs-bot

# Whisper service health (queue, GPU, vision model status)
curl -s http://localhost:7860/api/status | jq

# What slash commands are registered?
# (Discord client → type / in any channel → bot picker shows them.
#  If nothing shows up, the OAuth2 invite URL likely missed
#  applications.commands scope — re-invite with the right URL.)

# Force-resync slash commands (e.g. after editing definitions).
# Slash commands sync at startup; restart the bot:
make restart-bot
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
| Want explicit control over args (model, diarize) | `/summarize`, `/transcribe` (Path C) |
| Need to find a past summary | `/find` |
| Channel has different needs from defaults | `/config` (one-time) |
| Diarized output with renaming | `/transcribe diarize:true` then 🏷️ Rename speakers |

The URL-listening path stays around indefinitely — slash commands
don't replace it, they augment it. Casual users keep pasting; power
users use slash. Both share the same job queue, rate limits, and
caching.
