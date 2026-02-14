# ADR 0001: OpenAI-Compatible HTTP Contract

## Status
Accepted

## Context
Сервис используется как TTS backend для Open WebUI и других клиентов, которые ожидают OpenAI-подобный API.

## Decision
- Публичный контракт ограничен следующими endpoint:
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/audio/speech`
- Ошибки возвращаются в OpenAI-style JSON формате:
  - `error.message`
  - `error.type`
  - `error.param`
  - `error.code`
- Legacy endpoint не поддерживаются.

## Consequences
- Интеграции остаются совместимыми без клиентских адаптеров.
- Внутренние рефакторинги допускаются, пока сохраняется контракт API.
