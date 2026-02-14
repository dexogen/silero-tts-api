# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-15

### Added
- Unified developer workflow via `Makefile` (`setup`, `check`, `audit-deps`, `build`).
- Quality toolchain: `ruff`, `mypy`, `pytest`, `pre-commit`.
- Architecture and governance docs: ADRs, audit report, refactor plan, repo structure, contributing guide.
- CI quality gates with Python matrix (3.10/3.12) and Docker build smoke.

### Changed
- Updated dependency stack to current secure versions.
- Refactored request validation in API handlers to reduce complexity and improve maintainability.
- Switched logging configuration to a plain dict-based config compatible with Pydantic v2.
- Updated repository and GHCR links to `dexogen/silero-tts-api`.

### Security
- Added dependency vulnerability checks (`pip-audit`) for runtime, test, and docker requirements.

[1.0.0]: https://github.com/dexogen/silero-tts-api/releases/tag/v1.0.0
