# Silero TTS Service

[![CI](https://github.com/dexogen/silero-tts-api/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dexogen/silero-tts-api/actions/workflows/ci.yml)
[![Docker](https://github.com/dexogen/silero-tts-api/actions/workflows/docker-ghcr.yml/badge.svg?branch=main)](https://github.com/dexogen/silero-tts-api/actions/workflows/docker-ghcr.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GHCR](https://img.shields.io/badge/GHCR-ghcr.io%2Fdexogen%2Fsilero--tts--api-2ea44f?logo=github)](https://ghcr.io/dexogen/silero-tts-api)

OpenAI-compatible TTS сервис на базе Silero v5 для Open WebUI и любых клиентов, ожидающих OpenAI Audio API.

## Что делает сервис
- Поднимает HTTP API в формате OpenAI (`/v1/models`, `/v1/audio/speech`)
- Поддерживает Silero модели RU/UKR с алиасами `tts-1` и `tts-1-hd`
- Работает на `cpu` или `cuda` через переменную `SILERO_DEVICE`
- Отдает аудио потоково в форматах `wav`, `mp3`, `opus`, `flac`, `pcm`

## Публичные эндпоинты
- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`

Legacy-эндпоинты (`/process`, `/voices`, `/settings`, `/clear_cache`) удалены и не поддерживаются.

## Быстрый старт

### 1. Локальная разработка и quality checks
```bash
make setup
make check
```

### 2. Запуск реального TTS движка (локально)
```bash
cp .env.example .env
make install-runtime
.venv/bin/python main.py
```

Проверка:
```bash
curl -s http://localhost:9898/health
```

## Переменные окружения
- `SILERO_DEVICE`: `auto` (default), `cpu`, `cuda`
- `NUMBER_OF_THREADS`: число потоков для `torch.set_num_threads()` (default `4`)
- `TTS_ENGINE`: `mock` для тестов/CI (по умолчанию используется реальный движок)

Шаблон находится в `.env.example`.

## API модель/голоса

### Модели (`GET /v1/models`)
- `tts-1` (алиас `silero-tts-v5-ru`)
- `tts-1-hd` (алиас `silero-tts-v5-ru`)
- `silero-tts-v5-ru`
- `silero-tts-v5-ukr`

### Голоса
- RU (`tts-1`, `tts-1-hd`, `silero-tts-v5-ru`): `aidar`, `baya`, `kseniya`, `xenia`, `eugene`, `random`
- UKR (`silero-tts-v5-ukr`): `mykyta`/`ukr_mykyta`, `kateryna`, `lada`, `oleksa`, `tetiana`, `random`

### Формат запроса `/v1/audio/speech`
Обязательные поля:
- `model`
- `voice`
- `input`

Опциональные поля:
- `response_format`: `wav` (default), `mp3`, `opus`, `flac`, `pcm`
- `sample_rate`: `8000`, `24000`, `48000` (default `48000`)
- `bitrate`: `64`, `96`, `128`, `192` (для `mp3`/`opus`)
- `speed`: `0.25..4.0` (default `1.0`)

## Примеры API

Список моделей:
```bash
curl -s http://localhost:9898/v1/models
```

Синтез WAV:
```bash
curl -X POST http://localhost:9898/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "voice": "xenia",
    "input": "Привет! Это тест синтеза речи."
  }' \
  --output out.wav
```

Синтез MP3:
```bash
curl -X POST http://localhost:9898/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "silero-tts-v5-ru",
    "voice": "aidar",
    "input": "Проверка mp3 формата.",
    "response_format": "mp3",
    "sample_rate": 24000,
    "bitrate": 128,
    "speed": 1.1
  }' \
  --output out.mp3
```

## Docker

CPU:
```bash
docker build -f dockerfile -t silero-tts-api:cpu .
docker run --rm -p 9898:9898 \
  -e SILERO_DEVICE=cpu \
  -e NUMBER_OF_THREADS=4 \
  silero-tts-api:cpu
```

CUDA:
```bash
docker build -f Dockerfile.cuda -t silero-tts-api:cuda .
docker run --rm -p 9898:9898 --gpus all \
  -e SILERO_DEVICE=cuda \
  -e NUMBER_OF_THREADS=4 \
  silero-tts-api:cuda
```

Compose:
```bash
docker compose up --build
```

GPU-профиль:
```bash
docker compose --profile gpu up --build
```

## Команды разработки (`Makefile`)
- `make setup`: создать `.venv` и установить test/dev зависимости
- `make install-runtime`: установить runtime зависимости
- `make format`: авто-форматирование и авто-фикс ruff
- `make lint`: линт ruff
- `make typecheck`: mypy
- `make test`: pytest (`TTS_ENGINE=mock`)
- `make check`: `format-check + lint + typecheck + test`
- `make audit-deps`: `pip-audit` по runtime/test/docker зависимостям
- `make build`: сборка CPU образа
- `make build-cuda`: сборка CUDA образа

## Smoke test
```bash
bash scripts/smoke_openai_tts.sh
```

С переопределением URL:
```bash
BASE_URL=http://127.0.0.1:9898 bash scripts/smoke_openai_tts.sh
```

## Open WebUI
Рекомендуемые настройки:
- `Engine`: `OpenAI`
- `Base URL`: `http://host:9898/v1`
- `API Key`: `none` (или пусто)
- `Model`: `tts-1` или `silero-tts-v5-ru`
- `Voice`: `xenia`, `aidar`, `baya`, `kseniya`, `eugene`

## CI/CD
- `CI` (`.github/workflows/ci.yml`): `make check` на Python 3.10/3.12 + Docker build smoke
- `Docker` (`.github/workflows/docker-ghcr.yml`): сборка/публикация образов в GHCR

## Архитектурная документация
- `docs/repo-structure.md`
- `docs/refactor-plan.md`
- `docs/audit-report.md`
- `docs/adr/`

## Лицензия
MIT (`LICENSE`)
