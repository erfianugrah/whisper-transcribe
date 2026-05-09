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

.DEFAULT_GOAL := help

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─── Build ────────────────────────────────────────────────────────────────────

.PHONY: build build-whisper build-bot
build: build-whisper build-bot ## Build both images

build-whisper: ## Build whisper service image
	$(COMPOSE) build whisper

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

# ─── Lint / verify ────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Syntax-check Python sources
	python3 -c "import ast; ast.parse(open('app.py').read())" && echo "app.py OK"
	python3 -c "import ast; ast.parse(open('bot/main.py').read())" && echo "bot/main.py OK"

# ─── Cleanup ──────────────────────────────────────────────────────────────────

.PHONY: clean clean-cache prune
clean: ## Remove containers + named volumes (KEEPS images)
	$(COMPOSE) down -v

clean-cache: ## Clear bot transcript cache volume
	docker volume rm -f whisper-transcribe_bot-cache

prune: ## Remove dangling images + build cache (system-wide)
	docker image prune -f
	docker builder prune -f
