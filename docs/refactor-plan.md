# Refactor Plan

## Milestone 1: Baseline and Reproducibility
- Инвентаризация репозитория и фиксация текущего happy path
- Проверка локальной воспроизводимости test pipeline
- Выявление несовместимостей зависимостей по версиям Python

Зависимости:
- Нет

Риски:
- Ложное ощущение стабильности из-за локального окружения

Rollback:
- Возврат только документационных изменений

## Milestone 2: Quality Toolchain
- Введение `ruff` (format/lint), `mypy`, `pytest` как обязательных checks
- Единые команды через `Makefile`: `setup`, `format`, `lint`, `typecheck`, `test`, `check`
- Добавление `pyproject.toml` и `requirements_dev.txt`

Зависимости:
- Milestone 1

Риски:
- Фолс-позитивы линтера
- Падения в typecheck на неаннотированных участках

Rollback:
- Откат конфигурации quality tools без изменения runtime API

## Milestone 3: CI Hardening
- CI quality matrix для Python 3.10/3.12
- Добавление Docker build smoke как обязательного quality gate
- Согласование локальных и CI команд (`make check`)

Зависимости:
- Milestone 2

Риски:
- Рост времени CI

Rollback:
- Временное отключение build-smoke job при сохранении quality job

## Milestone 4: Dependency and Security Hygiene
- Пинning зависимостей для test/runtime/docker контуров
- Обновление web-стека до актуальных версий
- `pip-audit` проверка на уязвимости

Зависимости:
- Milestone 2

Риски:
- Несовместимость старого кода с новыми версиями библиотек

Rollback:
- Частичный rollback отдельных пакетов, если ломается контракт API

## Milestone 5: Documentation and Governance
- Обновление `README.md`
- Добавление `CONTRIBUTING.md`, `CODEOWNERS`, ADR
- Формирование `docs/audit-report.md` и `docs/repo-structure.md`

Зависимости:
- Все предыдущие milestone

Риски:
- Неполное покрытие edge-case сценариев в документации

Rollback:
- Документация откатывается независимо от runtime
