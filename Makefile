SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

IMAGE_CPU ?= silero-tts-service:cpu
IMAGE_CUDA ?= silero-tts-service:cuda

.PHONY: help venv setup install-runtime format format-check lint typecheck test check pre-commit audit-deps build build-cuda run clean

help:
	@echo "Targets:"
	@echo "  make setup            Create .venv and install test/dev dependencies"
	@echo "  make install-runtime  Install production dependencies into .venv"
	@echo "  make format           Auto-format code with Ruff"
	@echo "  make format-check     Check formatting with Ruff"
	@echo "  make lint             Run Ruff lint checks"
	@echo "  make typecheck        Run mypy type checks"
	@echo "  make test             Run pytest with mock TTS engine"
	@echo "  make check            Run format-check + lint + typecheck + test"
	@echo "  make audit-deps       Run pip-audit for requirements files"
	@echo "  make build            Build CPU Docker image"
	@echo "  make build-cuda       Build CUDA Docker image"
	@echo "  make run              Run CPU Docker image"
	@echo "  make clean            Remove caches"

venv:
	test -d "$(VENV)" || $(PYTHON) -m venv "$(VENV)"
	$(VENV_PYTHON) -m pip install --upgrade pip

setup: venv
	$(PIP) install -r requirements_test.txt -r requirements_dev.txt

install-runtime: venv
	$(PIP) install -r requirements.txt

format:
	$(VENV_PYTHON) -m ruff format .
	$(VENV_PYTHON) -m ruff check . --fix

format-check:
	$(VENV_PYTHON) -m ruff format --check .

lint:
	$(VENV_PYTHON) -m ruff check .

typecheck:
	$(VENV_PYTHON) -m mypy

test:
	TTS_ENGINE=mock TMPDIR=/tmp $(VENV_PYTHON) -m pytest

check: format-check lint typecheck test

pre-commit:
	$(VENV)/bin/pre-commit run --all-files

audit-deps:
	$(VENV_PYTHON) -m pip_audit -r requirements.txt
	$(VENV_PYTHON) -m pip_audit -r requirements_test.txt
	$(VENV_PYTHON) -m pip_audit -r requirements_docker.txt

build:
	docker build -f dockerfile -t $(IMAGE_CPU) .

build-cuda:
	docker build -f Dockerfile.cuda -t $(IMAGE_CUDA) .

run:
	docker run --rm -p 9898:9898 \
		-e SILERO_DEVICE=auto \
		-e NUMBER_OF_THREADS=4 \
		--name tts_silero \
		$(IMAGE_CPU)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
