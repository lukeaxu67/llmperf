from __future__ import annotations

import tempfile
import time

import pytest

from llmperf.config.models import ExecutorConfig
from llmperf.datasets.dataset_iterators import DatasetIterator
from llmperf.datasets.types import Message, TestCase
from llmperf.executors.base import BaseExecutor
from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage
from llmperf.utils.rate_limiter import RateLimiter


def _case(case_id: str) -> TestCase:
    return TestCase(
        id=case_id,
        messages=[Message(role="user", content=f"hello {case_id}")],
        metadata={},
    )


def test_rate_limiter_rejects_conflicting_config() -> None:
    with pytest.raises(ValueError, match="only one of qps or interval_seconds"):
        RateLimiter(qps=1.0, interval_seconds=1.0)

    with pytest.raises(ValueError, match="interval_seconds must be > 0"):
        RateLimiter(interval_seconds=0.0)

    with pytest.raises(ValueError, match="qps must be > 0"):
        RateLimiter(qps=0.0)


class _TimestampExecutor(BaseExecutor):
    def process_row(self, run_id: str, row: TestCase, price=None) -> RunRecord:  # type: ignore[override]
        record = RunRecord(
            run_id=run_id,
            executor_id=self.config.id,
            dataset_row_id=row.id,
            provider="mock",
            model="mock",
            status=200,
            info="{}",
        )
        record.extra = {"start_ts": time.monotonic()}
        return record


def test_executor_rate_limits_dispatch() -> None:
    rows = [_case("a"), _case("b"), _case("c")]
    dataset = DatasetIterator(rows)
    cfg = ExecutorConfig.model_validate(
        {
            "id": "e1",
            "name": "e1",
            "type": "mock",
            "model": "mock",
            "impl": "chat",
            "concurrency": 10,
            "rate": {"interval_seconds": 0.05},
        }
    )
    executor = _TimestampExecutor(cfg)

    with tempfile.TemporaryDirectory() as td:
        storage = Storage(f"{td}/perf.sqlite")
        try:
            records = executor.run(
                "run1",
                dataset,
                storage,
                max_rows=3,
            )
        finally:
            storage.close()

    starts = sorted(float(r.extra["start_ts"]) for r in records)
    assert len(starts) == 3
    assert starts[1] - starts[0] >= 0.03
    assert starts[2] - starts[1] >= 0.03

