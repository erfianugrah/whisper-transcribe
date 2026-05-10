# whisper-transcribe — GPU transcription service + Discord TL;DW bot
#
# Quick start:
#   make build          # build both images
#   make up             # start whisper + bot
#   make logs           # tail all logs
#   make push           # push both images to Docker Hub

REGISTRY      := erfianugrah
WHISPER_IMAGE := $(REGISTRY)/whisper-transcribe
BOT_IMAGE     := $(REGISTRY)/whisper-transcribe-bot
COMPOSE       := docker compose
GIT_SHA       := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

# Latest yt-dlp version from PyPI, fetched at make-invocation time. The
# Dockerfile's volatile yt-dlp layer is keyed on this — when PyPI has a new
# release the layer rebuilds (correct: we want the new version); when there's
# no release the layer cache-hits (correct: nothing to do). Falls back to a
# pinned floor if the network call fails so offline builds still work.
YT_DLP_VERSION := $(shell curl -s --max-time 5 https://pypi.org/pypi/yt-dlp/json 2>/dev/null \
	| python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null \
	|| echo "2026.3.17")

.DEFAULT_GOAL := help

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─── Build ────────────────────────────────────────────────────────────────────

.PHONY: build build-whisper build-bot
build: build-whisper build-bot ## Build both images

build-whisper: ## Build whisper service image (auto-bumps yt-dlp to latest PyPI release)
	@echo "Building whisper with yt-dlp $(YT_DLP_VERSION)"
	$(COMPOSE) build --build-arg YT_DLP_VERSION=$(YT_DLP_VERSION) whisper

build-bot: ## Build bot image
	$(COMPOSE) build bot

# ─── Lifecycle ────────────────────────────────────────────────────────────────

.PHONY: up up-whisper up-bot down restart restart-whisper restart-bot
up: ## Start whisper + bot (detached)
	$(COMPOSE) up -d

up-whisper: ## Start only whisper
	$(COMPOSE) up -d whisper

up-bot: ## Start only bot
	$(COMPOSE) up -d bot

down: ## Stop and remove containers
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) restart

restart-whisper: ## Restart whisper only (picks up app.py changes via bind mount)
	$(COMPOSE) restart whisper

restart-bot: ## Restart bot only (requires rebuild for code changes)
	$(COMPOSE) restart bot

# ─── Logs ─────────────────────────────────────────────────────────────────────

.PHONY: logs logs-whisper logs-bot
logs: ## Tail all logs (-f)
	$(COMPOSE) logs -f --tail 50

logs-whisper: ## Tail whisper logs
	$(COMPOSE) logs -f --tail 50 whisper

logs-bot: ## Tail bot logs
	$(COMPOSE) logs -f --tail 50 bot

# ─── Shell / debug ────────────────────────────────────────────────────────────

.PHONY: shell-whisper shell-bot status ps
shell-whisper: ## Exec bash in whisper container
	$(COMPOSE) exec whisper bash

shell-bot: ## Exec bash in bot container (sh — slim image has no bash)
	$(COMPOSE) exec bot sh

status: ## Show whisper /api/status JSON
	@curl -s http://localhost:7860/api/status | python3 -m json.tool

ps: ## Show running containers
	$(COMPOSE) ps

# ─── Push (Docker Hub) ────────────────────────────────────────────────────────

.PHONY: push push-whisper push-bot tag
push: push-whisper push-bot ## Tag with git SHA + push both :latest and :SHA

push-whisper: build-whisper ## Push whisper image
	docker tag $(WHISPER_IMAGE):latest $(WHISPER_IMAGE):$(GIT_SHA)
	docker push $(WHISPER_IMAGE):latest
	docker push $(WHISPER_IMAGE):$(GIT_SHA)

push-bot: build-bot ## Push bot image
	docker tag $(BOT_IMAGE):latest $(BOT_IMAGE):$(GIT_SHA)
	docker push $(BOT_IMAGE):latest
	docker push $(BOT_IMAGE):$(GIT_SHA)

tag: ## Tag both images as VERSION=x.y.z (also pushes if PUSH=1)
	@test -n "$(VERSION)" || (echo "Usage: make tag VERSION=x.y.z [PUSH=1]" && exit 1)
	docker tag $(WHISPER_IMAGE):latest $(WHISPER_IMAGE):$(VERSION)
	docker tag $(BOT_IMAGE):latest $(BOT_IMAGE):$(VERSION)
	@if [ "$(PUSH)" = "1" ]; then \
	  docker push $(WHISPER_IMAGE):$(VERSION) && \
	  docker push $(BOT_IMAGE):$(VERSION); \
	fi

# ─── Release (lint → build → push → redeploy) ────────────────────────────────

.PHONY: release ship redeploy
release: lint build push ## Lint + build both + push both (no redeploy)
	@echo ""
	@echo "Released $(GIT_SHA): $(WHISPER_IMAGE) and $(BOT_IMAGE)"
	@echo "Run 'make redeploy' to recreate local containers from the new images."

ship: release redeploy ## Lint + build + push + recreate local containers (full cycle)

redeploy: ## Recreate local containers from current :latest images
	$(COMPOSE) up -d --force-recreate
	@echo ""
	@echo "Containers recreated. Tail logs with: make logs"

# ─── Lint / verify ────────────────────────────────────────────────────────────

.PHONY: lint compile-check compose-check ruff bot-import-check test
lint: compile-check compose-check bot-import-check ## Run all static checks (no rebuild needed)

test: lint ## Lint + full E2E regression suite (no docker required)
	@python3 tests/test_regression.py

compile-check: ## ast.parse + py_compile (catches syntax + bytecode errors)
	@python3 -m py_compile app.py bot/main.py bot/prompts.py
	@echo "  py_compile: app.py, bot/main.py, bot/prompts.py OK"
	@python3 -c "import ast; [ast.parse(open(p).read()) for p in ['app.py','bot/main.py','bot/prompts.py']]"
	@echo "  ast.parse OK"

compose-check: ## Validate compose YAML (prod + dev overlay)
	@$(COMPOSE) config -q && echo "  compose.yaml OK"
	@$(COMPOSE) -f compose.yaml -f compose.dev.yaml config -q && echo "  compose.dev.yaml overlay OK"

bot-import-check: ## Import bot main module under stubbed deps + verify exports
	@python3 -c "$$BOT_IMPORT_CHECK"

ruff: ## Optional: run ruff if installed (pip install ruff)
	@command -v ruff >/dev/null 2>&1 && ruff check app.py bot/ || echo "  ruff not installed (pip install ruff)"

# Inline import-check script. Stubs out aiohttp/discord so we can import the
# bot module without network dependencies, then verifies the public symbols
# exist and the dead code stays gone.
define BOT_IMPORT_CHECK
import os, sys, types, tempfile
os.environ['DISCORD_TOKEN'] = 'lint-stub'
os.environ['CACHE_DIR'] = tempfile.mkdtemp(prefix='lint-')
sys.path.insert(0, 'bot')
for name in ('aiohttp',):
    sys.modules[name] = types.ModuleType(name)
sys.modules['aiohttp'].ClientSession = object
sys.modules['aiohttp'].ClientTimeout = lambda **k: None
discord = types.ModuleType('discord')
discord.Intents = type('I', (), {'default': staticmethod(lambda: types.SimpleNamespace(message_content=False))})
discord.HTTPException = Exception
for n in ('Embed', 'Message', 'TextChannel'): setattr(discord, n, object)
sys.modules['discord'] = discord
sys.modules['discord.ext'] = types.ModuleType('discord.ext')
commands = types.ModuleType('discord.ext.commands')
class B:
    def __init__(s,*a,**k): pass
    def event(s,fn): return fn
commands.Bot = B
sys.modules['discord.ext.commands'] = commands
import main
required = ('summarize', '_chunk_transcript', '_llm_call', 'PermanentError',
            'PROCESSING_EMOJI', 'read_cache', 'write_cache',
            'LLM_INPUT_CHAR_BUDGET', 'EMBED_SAFE_LIMIT',
            '_is_permanent_remote_error', 'PROMPT_BRIEF', 'PROMPT_KEY_POINTS',
            'PROMPT_CHAPTERS', 'REDUCE_BRIEF', 'REDUCE_KEY_POINTS',
            'CHUNK_PREAMBLE')
missing = [s for s in required if not hasattr(main, s)]
assert not missing, f'missing exports: {missing}'
assert not hasattr(main, 'extract_hotwords_from_context'), 'dead helper still present'
print(f'  bot import + exports OK ({len(required)} symbols)')
endef
export BOT_IMPORT_CHECK

# ─── Cleanup ──────────────────────────────────────────────────────────────────

.PHONY: clean clean-cache prune
clean: ## Remove containers + named volumes (KEEPS images)
	$(COMPOSE) down -v

clean-cache: ## Clear bot transcript cache volume
	docker volume rm -f whisper-transcribe_bot-cache

prune: ## Remove dangling images + build cache (system-wide)
	docker image prune -f
	docker builder prune -f
