# Contributing

## Требования
- Python `3.10+`
- Docker (для проверки контейнерной сборки)
- GNU `make`

## Быстрый старт для разработчика
```bash
make setup
make check
```

Для запуска реального движка:
```bash
cp .env.example .env
make install-runtime
.venv/bin/python main.py
```

## Стандарты кода
- Форматирование и линт: `ruff`
- Типизация: `mypy`
- Тесты: `pytest`
- Локальный quality gate: `make check`

Перед PR:
```bash
make check
make audit-deps
```

## Pre-commit
```bash
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files
```

## Ветки и PR
- Ветка от `main` с осмысленным именем (`feat/...`, `fix/...`, `chore/...`)
- Один PR = одна логическая задача
- PR должен проходить CI (`lint`, `typecheck`, `test`, `docker build smoke`)
- Если меняется поведение API, обновляйте README и тесты в том же PR

## Безопасность
- Не коммитьте секреты (`.env`, токены, ключи, приватные URL)
- Все новые переменные окружения добавляйте в `.env.example`
