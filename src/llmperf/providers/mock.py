from __future__ import annotations

import random
import time

from ..records.model import RunRecord, now_ms
from ..utils.counter import get_word_num
from .base import BaseProvider, ProviderRequest, register_provider
from .streaming import StreamAccumulator


class MockProvider(BaseProvider):
    """Deterministic provider for pipeline debugging."""

    RESPONSES = [
        "这是一个测试回答。",
        "Mock response generated for benchmarking.",
        "Pipeline OK ✓",
    ]

    def invoke(self, request: ProviderRequest) -> RunRecord:
        seed = hash((request.dataset_row_id, request.executor_id)) & 0xFFFF
        random.seed(seed)

        response_text = random.choice(self.RESPONSES)
        record = RunRecord(
            run_id=request.run_id,
            executor_id=request.executor_id,
            dataset_row_id=request.dataset_row_id,
            provider="mock",
            model=request.model or "mock-001",
        )

        accumulator = StreamAccumulator(record, debug=request.options.get("stream_debug"))

        # 使用开始时间
        base_time = now_ms()

        for idx, chunk in enumerate(response_text.split()):
            # 模拟每个chunk之间的延迟（增加时间戳差异）
            chunk_time = base_time + (idx * 100) + 10  # 每个chunk间隔100ms
            accumulator.append_content(chunk, ts_ms=chunk_time, seq=idx)

        prompt_text = "\n".join([msg.get("content", "") for msg in request.messages])
        record.qtokens = get_word_num(prompt_text)
        record.atokens = get_word_num(response_text)
        record.usage = {"prompt_tokens": record.qtokens, "completion_tokens": record.atokens}
        record.status = 200
        record.request_params = dict(request.options or {})
        accumulator.finalize(success=True)
        return record


register_provider("mock", "default")(MockProvider)
