from __future__ import annotations

import tempfile

from llmperf.config.models import ExecutorConfig
from llmperf.datasets.dataset_iterators import DatasetIterator
from llmperf.datasets.types import Message, TestCase
from llmperf.executors.base import BaseExecutor
from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage


def _case(case_id: str) -> TestCase:
    return TestCase(
        id=case_id,
        messages=[Message(role="user", content=f"hello {case_id}")],
        metadata={},
    )


def test_dataset_iterator_emits_round_and_case_indices() -> None:
    it = DatasetIterator([_case("a"), _case("b")])

    first = next(it)
    second = next(it)
    third = next(it)

    assert first.metadata and first.metadata["base_id"] == "a"
    assert first.metadata["pass_index"] == 1
    assert first.metadata["case_index"] == 0

    assert second.metadata and second.metadata["base_id"] == "b"
    assert second.metadata["pass_index"] == 1
    assert second.metadata["case_index"] == 1

    assert third.metadata and third.metadata["base_id"] == "a"
    assert third.metadata["pass_index"] == 2
    assert third.metadata["case_index"] == 0


class _NoopExecutor(BaseExecutor):
    def process_row(self, run_id: str, row: TestCase, price=None) -> RunRecord:  # type: ignore[override]
        return RunRecord(
            run_id=run_id,
            executor_id=self.config.id,
            dataset_row_id=row.id,
            provider="mock",
            model="mock",
            status=200,
            info="{}",
        )


def test_executor_is_single_stop_authority_for_round_limit() -> None:
    rows = [_case("a"), _case("b")]
    dataset = DatasetIterator(rows)
    cfg = ExecutorConfig(id="e1", name="e1", type="mock", model="mock", impl="chat")
    executor = _NoopExecutor(cfg)

    with tempfile.TemporaryDirectory() as td:
        storage = Storage(f"{td}/perf.sqlite")
        try:
            records = executor.run(
                "run1",
                dataset,
                storage,
                max_rounds=2,
                rows_per_round=len(rows),
            )
        finally:
            storage.close()

    assert len(records) == 4


def test_executor_is_single_stop_authority_for_time_limit() -> None:
    rows = [_case("a"), _case("b")]
    dataset = DatasetIterator(rows)
    cfg = ExecutorConfig(id="e1", name="e1", type="mock", model="mock", impl="chat")
    executor = _NoopExecutor(cfg)

    with tempfile.TemporaryDirectory() as td:
        storage = Storage(f"{td}/perf.sqlite")
        try:
            records = executor.run(
                "run1",
                dataset,
                storage,
                max_total_seconds=0.0,
            )
        finally:
            storage.close()

    assert records == []


def test_round_limit_cooperates_with_max_rows() -> None:
    rows = [_case("a"), _case("b")]
    dataset = DatasetIterator(rows)
    cfg = ExecutorConfig(id="e1", name="e1", type="mock", model="mock", impl="chat")
    executor = _NoopExecutor(cfg)

    with tempfile.TemporaryDirectory() as td:
        storage = Storage(f"{td}/perf.sqlite")
        try:
            records = executor.run(
                "run1",
                dataset,
                storage,
                max_rows=3,
                max_rounds=2,
                rows_per_round=len(rows),
            )
        finally:
            storage.close()

    assert len(records) == 3
