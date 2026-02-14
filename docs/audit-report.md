# Audit Report

## Executive Summary
1. Введен единый воспроизводимый контур качества: `make setup` и `make check`.
2. Добавлены и зафиксированы инструменты качества: `ruff`, `mypy`, `pytest`, `pre-commit`.
3. CI усилен: quality gates (lint/typecheck/test) и обязательный Docker build smoke.
4. Обновлены и зафиксированы зависимости в `requirements*.txt` (runtime/test/docker).
5. Устранены известные уязвимости в зависимостях (по результатам `pip-audit`).
6. Упрощен logging-конфиг (убрана лишняя зависимость от `pydantic.BaseModel`).
7. Снижена сложность API handler через декомпозицию валидации payload.
8. Удален устаревший ключ `version` из `docker-compose.yml`.
9. Обновлена инженерная документация: `README`, `CONTRIBUTING`, ADR, структура репозитория.
10. Зафиксированы legacy-находки и план их безопасной верификации.

## Findings Table
| category | severity | file/path | evidence | recommendation | effort | risk |
|---|---|---|---|---|---|---|
| Reproducibility | High | `Makefile` | Ранее не было единого reproducible workflow локально | Использовать `make setup` и `make check` как стандарт | S | Low |
| Quality gates | High | `.github/workflows/ci.yml` | Ранее CI запускал только pytest | Держать обязательными `format/lint/typecheck/test + docker build smoke` | S | Low |
| Dependency security | High | `requirements.txt` | Старый стек имел CVE по `fastapi/starlette` (подтверждено `pip-audit`) | Обновлены pinned версии (`fastapi 0.129.0`, `starlette 0.52.1` transitively) | M | Medium |
| Dependency hygiene | Medium | `requirements.txt` | `pytest` был runtime dependency | Убрано из runtime, оставлено в test/dev контурах | S | Low |
| Compatibility | Medium | `app/logger.py` | Конфиг на `pydantic.BaseModel` ломался при pydantic v2 | Переведен на простой dict-конфиг `LOG_CONFIG` | S | Low |
| Code complexity | Medium | `app/handlers.py` | `create_audio_speech` имел высокую сложность (C901) | Декомпозиция парсинга/валидации в отдельные функции | M | Low |
| Compose config | Low | `docker-compose.yml` | `docker compose` предупреждал, что `version` obsolete | Удален устаревший ключ | S | Low |
| Documentation | Medium | `README.md` | Команды и процесс разработки были не полностью согласованы с текущим состоянием | README переписан под единый workflow и troubleshooting | M | Low |
| Governance | Low | `.github/CODEOWNERS` | Не было формализованного ownership | Добавлен `CODEOWNERS` | S | Low |

## Legacy & Dead Code

### Safe to remove now
- `pytest` в runtime (`requirements.txt`) удален.
- `version` в `docker-compose.yml` удален как obsolete.

### Needs verification
- `docs/ClientConf.png`
- `docs/MopidyConfig.png`
- `docs/RebootHa.png`
- `docs/RhasspyAssistantConfig.png`
- `docs/RhasspyConfig.png`
- `docs/TtsSay.png`

Причина:
- Файлы не используются в текущем README/коде, но могут быть нужны для внешней вики/старых интеграционных гайдов.

Способ верификации:
1. Проверить внешние ссылки на эти изображения (Wiki/README в других репозиториях).
2. Если ссылок нет в течение 1 релиза, удалить их в отдельном PR.

### Keep
- `app/tts_engine.py` как монолитный модуль пока сохранен из-за тесной связности runtime логики и чувствительности к regressions.
- `dockerfile` (нижний регистр) сохранен для обратной совместимости с существующими workflow и скриптами.

## Build & CI

### Было
- Локально: частично воспроизводимый запуск.
- CI: только `pytest` на Python 3.10.
- Docker build и качество кода не были единым обязательным gate.

### Стало
- Локально:
  - `make setup`
  - `make check`
  - `make audit-deps`
  - `make install-runtime`
  - `make build` / `make build-cuda`
- CI (`.github/workflows/ci.yml`):
  - matrix Python 3.10/3.12
  - `make check` (format/lint/typecheck/test)
  - `docker build -f dockerfile ...` smoke
- Security:
  - `pip-audit -r requirements*.txt` -> `No known vulnerabilities found`

### Проверки, выполненные в аудите
- `make check` passed
- `docker build -f dockerfile -t silero-tts-api:ci .` passed
- `curl http://127.0.0.1:29898/health` в запущенном контейнере passed
- `pip-audit` по runtime/test/docker/dev зависимостям passed
