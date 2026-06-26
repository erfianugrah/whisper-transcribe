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
GIT_SHA       := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

# Two compose handles:
# Compose binding. The llm-compose overlay was removed in May 2026 — its
# network (`llmc`) is now declared `external` in compose.yaml directly, so
# `docker compose up` always lands the bot + whisper on the right network
# and reaches `model_proxy` by hostname. Bring up llm-compose first for the
# co-deployed default; for a standalone stack (no llm-compose) use the
# `*-standalone` targets, which swap in compose.standalone.yaml.
COMPOSE := docker compose
COMPOSE_RUNTIME := docker compose
# Standalone overlay: redefines `llmc` as a self-managed bridge so the stack
# comes up WITHOUT llm-compose running (see compose.standalone.yaml). Used by
# the `*-standalone` targets only; the default targets keep the co-deployed
# behaviour (llmc external, reaches model_proxy by hostname).
COMPOSE_STANDALONE := docker compose -f compose.yaml -f compose.standalone.yaml

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

.PHONY: build build-whisper build-bot build-live
build: build-whisper build-live build-bot ## Build all service images

build-whisper: ## Build whisper service image (auto-bumps yt-dlp to latest PyPI release)
	@echo "Building whisper with yt-dlp $(YT_DLP_VERSION)"
	$(COMPOSE_RUNTIME) build --build-arg YT_DLP_VERSION=$(YT_DLP_VERSION) whisper

build-live: ## Build whisper-live streaming sidecar image
	$(COMPOSE_RUNTIME) build whisper-live

build-bot: ## Build bot image
	$(COMPOSE_RUNTIME) build bot

# ─── Lifecycle ────────────────────────────────────────────────────────────────

.PHONY: up up-whisper up-bot up-standalone down down-standalone restart restart-whisper restart-bot
up: ## Start whisper + bot (detached) — co-deployed, needs llm-compose's llmc net
	$(COMPOSE_RUNTIME) up -d

up-whisper: ## Start only whisper
	$(COMPOSE_RUNTIME) up -d whisper

up-bot: ## Start only bot
	$(COMPOSE_RUNTIME) up -d bot

up-standalone: ## Start transcription core WITHOUT llm-compose (valkey+whisper+whisper-live)
	$(COMPOSE_STANDALONE) up -d valkey whisper whisper-live

live-tap: ## Stream OBS/desktop/mic audio to whisper-live, print transcript (see live-tap/README.md). Override with ARGS="--device '...'"
	@python3 live-tap/desktop_tap.py $(ARGS)

live-tap-selftest: ## Verify the live-tap can reach whisper-live (5s sine tone, no audio hardware)
	@python3 live-tap/desktop_tap.py --self-test

research-tap: ## Interactive LLM research REPL fed by the live transcript. Pipe: desktop_tap.py --loopback | make research-tap
	@python3 live-tap/research_tap.py $(ARGS)

down: ## Stop and remove containers
	$(COMPOSE_RUNTIME) down

down-standalone: ## Stop the standalone stack (removes the self-managed llmc bridge)
	$(COMPOSE_STANDALONE) down

restart: ## Restart all services
	$(COMPOSE_RUNTIME) restart

restart-whisper: ## Restart whisper only (picks up app.py changes via bind mount)
	$(COMPOSE_RUNTIME) restart whisper

restart-bot: ## Restart bot only (in-place; does NOT re-read bot/.env — use recreate-bot for env changes)
	$(COMPOSE_RUNTIME) restart bot

restart-scraper: ## Restart crawl4ai + flaresolverr together (e.g. after a hung browser)
	$(COMPOSE_RUNTIME) restart crawl4ai flaresolverr

.PHONY: recreate-bot recreate-whisper recreate-scraper restart-scraper
recreate-bot: ## Tear down + recreate bot container (re-reads bot/.env, refreshes image)
	$(COMPOSE_RUNTIME) up -d --force-recreate bot

recreate-whisper: ## Tear down + recreate whisper container (re-reads .env)
	$(COMPOSE_RUNTIME) up -d --force-recreate whisper

recreate-scraper: ## Tear down + recreate scraper services (refreshes images, clears state)
	$(COMPOSE_RUNTIME) up -d --force-recreate crawl4ai flaresolverr

# ─── Logs ─────────────────────────────────────────────────────────────────────

.PHONY: logs logs-whisper logs-bot logs-crawl4ai logs-flaresolverr logs-scraper
logs: ## Tail all logs (-f)
	$(COMPOSE_RUNTIME) logs -f --tail 50

logs-whisper: ## Tail whisper logs
	$(COMPOSE_RUNTIME) logs -f --tail 50 whisper

logs-bot: ## Tail bot logs
	$(COMPOSE_RUNTIME) logs -f --tail 50 bot

logs-crawl4ai: ## Tail crawl4ai (primary scraper) logs
	$(COMPOSE_RUNTIME) logs -f --tail 50 crawl4ai

logs-flaresolverr: ## Tail flaresolverr (CF-challenge fallback) logs
	$(COMPOSE_RUNTIME) logs -f --tail 50 flaresolverr

logs-scraper: ## Tail BOTH scraper services together
	$(COMPOSE_RUNTIME) logs -f --tail 50 crawl4ai flaresolverr

# ─── Shell / debug ────────────────────────────────────────────────────────────

.PHONY: shell-whisper shell-bot status status-scraper ps
shell-whisper: ## Exec bash in whisper container
	$(COMPOSE_RUNTIME) exec whisper bash

shell-bot: ## Exec bash in bot container (sh — slim image has no bash)
	$(COMPOSE_RUNTIME) exec bot sh

status: ## Show whisper /api/status JSON
	@curl -s http://localhost:7860/api/status | python3 -m json.tool

status-scraper: ## Probe crawl4ai /health + flaresolverr / from inside the bot container
	@echo "── crawl4ai ──"
	@$(COMPOSE_RUNTIME) exec -T bot python3 -c "import urllib.request,json; r=urllib.request.urlopen('http://crawl4ai:11235/health',timeout=3); print(json.dumps(json.loads(r.read()),indent=2))" || echo "crawl4ai: unreachable"
	@echo ""
	@echo "── flaresolverr ──"
	@$(COMPOSE_RUNTIME) exec -T bot python3 -c "import urllib.request,json; r=urllib.request.urlopen('http://flaresolverr:8191/',timeout=3); print(json.dumps(json.loads(r.read()),indent=2))" || echo "flaresolverr: unreachable"

ps: ## Show running containers
	$(COMPOSE_RUNTIME) ps

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
	$(COMPOSE_RUNTIME) up -d --force-recreate
	@echo ""
	@echo "Containers recreated. Tail logs with: make logs"

# ─── Lint / verify ────────────────────────────────────────────────────────────

.PHONY: lint compile-check compose-check ruff bot-import-check test
lint: compile-check compose-check bot-import-check ## Run all static checks (no rebuild needed)

test: lint ## Lint + full E2E regression suite (no docker required)
	@python3 tests/test_regression.py

compile-check: ## ast.parse + py_compile (catches syntax + bytecode errors)
	@python3 -m py_compile app.py bot/main.py bot/prompts.py live-tap/desktop_tap.py live-tap/research_tap.py
	@echo "  py_compile: app.py, bot/main.py, bot/prompts.py, live-tap/desktop_tap.py, live-tap/research_tap.py OK"
	@python3 -c "import ast; [ast.parse(open(p).read()) for p in ['app.py','bot/main.py','bot/prompts.py','live-tap/desktop_tap.py','live-tap/research_tap.py']]"
	@echo "  ast.parse OK"

compose-check: ## Validate compose YAML (prod + dev + standalone overlays)
	@$(COMPOSE) config -q && echo "  compose.yaml OK"
	@$(COMPOSE) -f compose.yaml -f compose.dev.yaml config -q && echo "  compose.dev.yaml overlay OK"
	@$(COMPOSE_STANDALONE) config -q && echo "  compose.standalone.yaml overlay OK"

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
discord.Interaction = object
discord.ButtonStyle = types.SimpleNamespace(secondary='secondary', primary='primary', danger='danger', success='success')
discord.Object = lambda **k: None
for n in ('Embed', 'Message', 'TextChannel'): setattr(discord, n, object)
sys.modules['discord'] = discord
# discord.app_commands stub
app_commands = types.ModuleType('discord.app_commands')
app_commands.CommandTree = type('CT', (), {'__init__': lambda s,*a,**k: None})
def _passthrough_decorator(*a, **k):
    def _wrap(fn): return fn
    return _wrap
app_commands.command = _passthrough_decorator
app_commands.describe = _passthrough_decorator
app_commands.autocomplete = _passthrough_decorator
app_commands.Choice = type('Choice', (), {'__init__': lambda s,name='',value='': setattr(s,'name',name) or setattr(s,'value',value)})
class _R:
    def __class_getitem__(cls, item): return int
app_commands.Range = _R
sys.modules['discord.app_commands'] = app_commands
discord.app_commands = app_commands
# discord.ui stub
ui = types.ModuleType('discord.ui')
ui.View = type('View', (), {'__init__': lambda s,*a,**k: None})
ui.Modal = type('Modal', (), {
    '__init__': lambda s,*a,**k: None,
    '__init_subclass__': classmethod(lambda cls, **k: None),
    'add_item': lambda s,i: None,
})
ui.Button = type('Button', (), {})
ui.TextInput = type('TextInput', (), {'__init__': lambda s,*a,**k: None})
ui.Select = type('Select', (), {})
def _ui_button_decorator(*a, **k):
    def _wrap(fn): return fn
    return _wrap
ui.button = _ui_button_decorator
sys.modules['discord.ui'] = ui
discord.ui = ui
sys.modules['discord.ext'] = types.ModuleType('discord.ext')
commands = types.ModuleType('discord.ext.commands')
class B:
    def __init__(s,*a,**k):
        s.tree = types.SimpleNamespace(
            command=_passthrough_decorator,
            sync=lambda **kw: None,
            copy_global_to=lambda **kw: None,
        )
    def event(s,fn): return fn
commands.Bot = B
sys.modules['discord.ext.commands'] = commands
import main
required = ('summarize', '_chunk_transcript', '_llm_call', 'PermanentError',
            'PROCESSING_EMOJI', 'read_cache', 'write_cache',
            'LLM_INPUT_CHAR_BUDGET', 'EMBED_SAFE_LIMIT',
            '_is_permanent_remote_error', 'PROMPT_BRIEF', 'PROMPT_KEY_POINTS',
            'PROMPT_CHAPTERS', 'REDUCE_BRIEF', 'REDUCE_KEY_POINTS',
            'CHUNK_PREAMBLE',
            # Web URL summary flow (added with the "tldr" reply trigger)
            'process_url', 'fetch_article', '_fetch_via_crawl4ai',
            '_fetch_via_flaresolverr', '_handle_reply_trigger',
            '_extract_first_url', '_hash_url', '_is_video_url',
            '_looks_like_cf_challenge', 'REPLY_TRIGGER_RE',
            'PROMPT_BRIEF_WEB', 'PROMPT_KEY_POINTS_WEB', 'PROMPT_SECTIONS',
            'REDUCE_BRIEF_WEB', 'REDUCE_KEY_POINTS_WEB', 'REDUCE_SECTIONS',
            'SCRAPER_API', 'FLARESOLVERR_API', 'SCRAPER_TIMEOUT')
missing = [s for s in required if not hasattr(main, s)]
assert not missing, f'missing exports: {missing}'
assert not hasattr(main, 'extract_hotwords_from_context'), 'dead helper still present'
print(f'  bot import + exports OK ({len(required)} symbols)')
endef
export BOT_IMPORT_CHECK

# ─── Migration (one-shot, for upgrades from pre-non-root images) ─────────────

.PHONY: migrate-from-root
migrate-from-root: ## ONE-TIME: chown stale root-owned volumes after non-root switch
	@echo "Stopping containers so we can mutate the volumes safely..."
	@$(COMPOSE_RUNTIME) stop || true
	@echo ""
	@echo "Chowning model-cache (HF models) to uid 1000..."
	@docker volume inspect whisper-transcribe_model-cache >/dev/null 2>&1 && \
	    docker run --rm -v whisper-transcribe_model-cache:/c alpine chown -R 1000:1000 /c \
	    || echo "  (model-cache volume not found — skip)"
	@echo "Chowning bot-cache (transcript cache) to uid 1000..."
	@docker volume inspect whisper-transcribe_bot-cache >/dev/null 2>&1 && \
	    docker run --rm -v whisper-transcribe_bot-cache:/c alpine chown -R 1000:1000 /c \
	    || echo "  (bot-cache volume not found — skip)"
	@echo "Importing legacy ./uploads/history.json into the uploads named volume (if present)..."
	@if [ -f ./uploads/history.json ]; then \
	    $(COMPOSE_RUNTIME) up -d --no-recreate whisper >/dev/null 2>&1 || true; \
	    docker compose cp ./uploads/history.json whisper:/data/history.json && \
	        docker compose exec -u 0 whisper chown 1000:1000 /data/history.json && \
	        echo "  imported ./uploads/history.json"; \
	  else \
	    echo "  (no legacy ./uploads/history.json — skip)"; \
	  fi
	@echo ""
	@echo "Migration complete. Bring stack back up with: make up"
	@echo "Fresh installs of this repo do NOT need this command — named volumes"
	@echo "in compose.yaml are auto-initialised with the image's uid 1000 ownership."

# ─── Cleanup ──────────────────────────────────────────────────────────────────

.PHONY: clean clean-cache prune
clean: ## Remove containers + named volumes (KEEPS images)
	$(COMPOSE_RUNTIME) down -v

clean-cache: ## Clear bot transcript cache volume
	docker volume rm -f whisper-transcribe_bot-cache

prune: ## Remove dangling images + build cache (system-wide)
	docker image prune -f
	docker builder prune -f
