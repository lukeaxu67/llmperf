from __future__ import annotations

from llmperf.records.model import RunRecord
from llmperf.utils.counter import get_word_num


def test_run_record_stream_granularity_metrics_use_effective_payload_frames():
    record = RunRecord(
        run_id="run-1",
        executor_id="exec-1",
        dataset_row_id="row-1",
        provider="mock",
        model="mock-model",
        status=200,
        qtokens=100,
        atokens=12,
        ctokens=20,
        reasoning=["first reasoning", ""],
        content=["", "second content"],
        reasoning_times=[1000, 1100, 1200],
        content_times=[1000, 1150, 1300],
        usage={
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    )

    assert record.tokens_per_frame == 6
    assert record.first_frame_chars == get_word_num("first reasoning")
    assert record.cache_ratio == 0.2


def test_run_record_stream_granularity_metrics_are_zero_without_payload_frames():
    record = RunRecord(
        run_id="run-1",
        executor_id="exec-1",
        dataset_row_id="row-1",
        provider="mock",
        model="mock-model",
        status=200,
        qtokens=0,
        atokens=12,
        reasoning=[""],
        content=[],
    )

    assert record.tokens_per_frame == 0
    assert record.first_frame_chars == 0
