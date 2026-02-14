# Repository Structure

## Целевая структура
```text
.
├── app/                      # backend приложение
│   ├── handlers.py           # HTTP слой (валидация/маршрутизация)
│   ├── tts_engine.py         # доменная логика TTS
│   └── logger.py             # logging config
├── tests/                    # unit/integration тесты API и нормализации текста
├── scripts/                  # ручные и smoke-скрипты
├── docs/                     # документация и отчеты
│   └── adr/                  # архитектурные решения
├── configs/                  # дополнительные инфраструктурные конфиги
├── .github/workflows/        # CI/CD
├── main.py                   # входная точка приложения
├── Makefile                  # единый интерфейс команд
├── pyproject.toml            # lint/typecheck/test конфигурация
├── requirements*.txt         # зависимости по контурам
└── docker-compose.yml        # локальная оркестрация
```

## Правила нейминга
- Python-модули: `snake_case.py`
- Тесты: `tests/test_*.py`
- Скрипты: `scripts/*.sh` с явным shebang
- Docker: `dockerfile` (CPU), `Dockerfile.cuda` (GPU)
- Документы решений: `docs/adr/NNNN-<topic>.md`

## Правила размещения
- HTTP/контрактные изменения: `app/handlers.py` + тесты в `tests/`
- Бизнес-логика синтеза: `app/tts_engine.py`
- Изменения процессов разработки: `Makefile`, `pyproject.toml`, `.pre-commit-config.yaml`
- Процессы CI: `.github/workflows/`
- Отчеты и аудит: `docs/`

## Мотивация
- Прозрачная граница между API слоем и TTS логикой.
- Воспроизводимые команды через единый `Makefile`.
- Документация и архитектурные решения находятся рядом с кодом и версионируются вместе с ним.
