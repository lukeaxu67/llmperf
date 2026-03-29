# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LLMPerf is a YAML-driven benchmarking toolkit for LLM providers. It supports multi-provider performance evaluation, cost tracking, continuous monitoring, and deep analysis (stability, health, rate limiting, etc.).

## Commands

```bash
# Install (src layout, editable)
pip install -e .

# Run all tests (--maxfail=1 by default via pytest.ini)
pytest

# Run a single test file
pytest tests/test_rate_limiter.py -v

# Run a single test function
pytest tests/test_rate_limiter.py::test_qps_mode -v

# CLI entry points
llmperf-run -c template/config.yaml --pricing-file pricings/prices.yaml
llmperf-analyze --type stability --config analysis.yaml
llmperf-web

# Frontend (in frontend/ directory)
npm install && npm run dev   # dev server
npm run build                # production build
```

## Architecture

### Source Layout

- `src/llmperf/` - Python package (setuptools src layout)
- `frontend/` - React 18 + TypeScript + Ant Design + Vite
- `tests/` - pytest suite (asyncio_mode=auto, multiprocessing spawn)
- `template/` - example YAML task configs
- `pricings/` - pricing table YAMLs
- `resource/` - example datasets

### Execution Flow

`RunConfig (YAML)` -> `RunManager (runner.py)` -> `ProcessManager` -> per-executor processes -> `Executor` (ThreadPoolExecutor for concurrency) -> `Provider` (LLM API call) -> `RunRecord` -> `Storage` (SQLite)

- **RunManager** (`runner.py`): top-level orchestrator. Loads config, dataset, pricing; creates ProcessManager; registers run in DB.
- **ProcessManager** (`executors/process_manager.py`): builds executor dependency DAG from `after` fields, spawns one process per executor, manages lifecycle.
- **Executor** (`executors/base.py`): iterates dataset rows with rate limiting, runs provider calls concurrently via ThreadPoolExecutor, writes RunRecords to storage.
- **Provider** (`providers/base.py`): normalizes messages, calls LLM API, returns unified RunRecord with timing/token metrics.

### Registry Pattern (DoubleRegistry)

All extensible components use a `(type, impl)` two-key registry in `utils/registry.py`:
- Providers: e.g. `("openai", "chat")`, `("mock", "chat")`
- Executors: e.g. `("openai", "chat")`
- Dataset sources: e.g. `("jsonl", "demo")`, `("generator", "simple_questions")`
- Exporters, notification channels: same pattern
- Analysis types use a single-key registry (type -> class)

New providers/executors/sources are registered via decorators and auto-discovered on import.

### Config Model

Pydantic v2 models in `config/models.py`. Key hierarchy: `RunConfig` -> `DatasetConfig` (source + iterator) -> `ExecutorConfig[]` (id, type, impl, concurrency, model, param, rate, after).

### Dataset Pipeline

`DatasetSource` (JSONL/CSV/HTTP/Generator) -> `DatasetIterator` (thread-safe via RLock, applies `MutationChain`) -> test cases consumed by executors. Termination boundaries (max_rounds, max_total_seconds) are enforced by the executor.

### Web Layer

FastAPI app (`web/main.py`) with routers under `web/routers/` and service layer under `web/services/`. WebSocket at `/ws/` for real-time task progress. Frontend communicates via REST + WebSocket.

### Database

SQLite at `~/llmperf/db/perf.sqlite` (override via `db_path` in YAML config). Three main tables: `runs` (task metadata), `executions` (per-call records with timing/tokens/cost), `pricing_history`.

## Key Conventions

- Python >= 3.12 required
- Pydantic v2 for all config and data models
- Rate limiting supports two modes: `qps` (queries/sec) or `interval_seconds` (fixed interval)
- Pricing is in per-million-token units; price snapshots are captured at execution time
- Executor `after` field defines DAG dependencies between executors in the same task
- Tests use pytest-asyncio with `asyncio_mode = auto`; multiprocessing uses `spawn` method
- The env var `EVALUATE_DB_PATH` controls the DB path in tests
