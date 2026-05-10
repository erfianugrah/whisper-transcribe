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
| `/summarize url:<URL> prompt:<text>` | Slash equivalent of paste-with-text (cleaner UX, arg validation). |
| `/transcribe url:<URL> diarize:true` | Adds speaker labels. Embed grows a 🏷️ Rename speakers button. |
| `/find query:<keywords>` | Searches your past transcripts for matching content. |
| `/status` | Shows queue depth, your rate-limit usage, whisper service health. |
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
| Saw an interesting article someone posted | Reply `tldr` (Path C) |
| Wondering if an article is AI-generated / AI-spam | Reply `litmus` (Path D) |
| Want explicit control over args (model, diarize) | `/summarize`, `/transcribe` (Path E) |
| Need to find a past summary | `/find` |
| Channel has different needs from defaults | `/config` (one-time) |
| Diarized output with renaming | `/transcribe diarize:true` then 🏷️ Rename speakers |

The auto-paste, reply-trigger, and slash paths all share the same job
queue, rate limits, and cache. None of them replace each other — pick
whichever matches your context. Casual users keep pasting and replying
`tldr`; power users use slash for explicit control.
