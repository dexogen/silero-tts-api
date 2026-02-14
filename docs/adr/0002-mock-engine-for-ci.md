# ADR 0002: Mock TTS Engine for Tests and CI

## Status
Accepted

## Context
Полноценный Silero runtime требует тяжелых зависимостей и загрузки моделей, что делает unit/integration тесты медленными и нестабильными в CI.

## Decision
- Для тестового контура используется `TTS_ENGINE=mock`.
- `make test` и CI запускают pytest только в mock режиме.
- Runtime с реальным Silero проверяется отдельным Docker build smoke и ручным e2e тестом.

## Consequences
- CI становится быстрым и воспроизводимым.
- Проверка качества API и валидации параметров не зависит от внешних модельных артефактов.
- Нужны периодические ручные e2e проверки реального движка на релизных ветках.
