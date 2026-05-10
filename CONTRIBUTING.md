# Contributing to whisper-transcribe

Thanks for considering a contribution. This is a personal hobbyist project,
but PRs are welcome — especially for bug fixes, additional video platforms
(yt-dlp adapters), better prompt templates, and operational improvements.

## What's in scope

✅ Welcome:
- Bug fixes with a clear repro path
- New URL-flow scrapers (Reddit-style structured fetches for other
  community sites — Hacker News, Lemmy, etc.)
- Better default prompts (model-agnostic, content-agnostic)
- Additional VLM providers / models for the silent-video flow
- Performance: parallelism, batching, smarter caching
- Documentation, README clarifications, examples
- Tests for any of the above

❓ Discuss before building:
- Major refactors (open an issue first; this codebase deliberately ships
  some duplication where it preserves clarity over abstraction)
- New top-level features (e.g. additional reply-trigger keywords) —
  worth aligning on the UX shape
- Cloud-API integrations (OpenAI, Anthropic) — fine in principle but the
  project is local-first by design

❌ Out of scope:
- Hosted-service offerings, multi-tenant deployments, SaaS infrastructure
- Closed-source models / proprietary backends
- Discord-specific features that don't generalise (the bot's reply-trigger
  pattern is reusable; "show this user's last 10 deleted messages" is not)

## Development setup

Standard local-dev:

```bash
git clone https://github.com/erfianugrah/whisper-transcribe.git
cd whisper-transcribe
cp .env.example .env             # add HF_TOKEN if testing diarization
cp bot/.env.example bot/.env     # add DISCORD_TOKEN if testing bot
make build && make up
```

For live editing without rebuilds, use the dev overlay:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d
docker compose restart whisper bot   # picks up Python changes
```

## Tests

Run the regression suite without Docker (stubs out aiohttp / discord):

```bash
make test
```

There are >200 tests; PRs touching `bot/main.py`, `bot/prompts.py`, or
`app.py` should add coverage. Tests live in `tests/test_regression.py`
and follow the existing pattern (no pytest required — plain `assert`
statements in functions named `test_*`).

For changes touching compose files:

```bash
make compose-check
```

For changes touching the bot's import surface:

```bash
make bot-import-check
```

## Code style

- Python: stdlib-first; new dependencies need a clear justification (the
  bot image is intentionally small).
- Existing comments are extensive and explain *why*, not *what* — match
  that level of detail when adding non-obvious logic.
- No AI attribution in commit messages, comments, or PR bodies.
- Single-word variable names where unambiguous; multi-word where clarity
  helps. Prefer early returns over deep nesting.
- Async functions: use `asyncio.to_thread` over `loop.run_in_executor`.
- Type hints on public functions; `from __future__ import annotations` is
  fine if it helps avoid forward-reference gymnastics.

## Commit messages

Match the existing style (run `git log --oneline -20` to see):

```
Bot: short title — what changed

Bullet body if the change deserves explanation. Focus on WHY the change
exists, not what files were touched. Include a "Tests" line when test
coverage changes.
```

For multi-area changes, lead with the most affected component
(`Bot:`, `Whisper:`, `Compose:`, `Docs:`).

## Pull request workflow

1. Fork, create a feature branch off `main`.
2. Run `make test` + `make compose-check` before pushing.
3. Open a PR with the same shape as the existing commit messages.
4. Expect review focused on: does it match the codebase's existing
   patterns, does it preserve the local-first architecture, does it
   ship with tests.

## Reporting bugs

Useful issue includes:
- Bot version (`git rev-parse --short HEAD`)
- LLM endpoint type (Ollama / llama.cpp / llm-compose / OpenAI / etc.)
- Relevant log excerpts (`make logs-bot` or `make logs-whisper`)
- Reproduction steps — ideally a specific URL that triggers the bug

Logs are JSON-structured when `LOG_JSON=1` is set in `bot/.env`.

## Security

If you find a vulnerability — particularly in the prompt-injection
defenses (`sanitize_llm_output`, `REF_RULES`, `_PERMANENT_REMOTE_PATTERNS`),
please open a private security advisory on GitHub rather than a public
issue.

## License

By contributing, you agree your changes ship under the [MIT License](LICENSE)
that covers the rest of the project.
