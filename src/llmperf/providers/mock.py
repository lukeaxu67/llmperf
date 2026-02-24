from __future__ import annotations

import random
import time

from ..records.model import RunRecord, now_ms
from ..utils.counter import get_word_num
from .base import BaseProvider, ProviderRequest, register_provider
from .streaming import StreamAccumulator


class MockProvider(BaseProvider):
    """Mock provider for testing and debugging.

    This provider simulates LLM responses with configurable performance
    characteristics to test different model behaviors without making
    actual API calls.

    Configuration options (via request.options):
    - mock_ttft_ms: Time to first token in milliseconds (default: 100)
    - mock_tokens_per_chunk: Tokens per streaming chunk (default: 5)
    - mock_chunks: Number of chunks to generate (default: 5)
    - mock_chunk_interval_ms: Interval between chunks in milliseconds (default: 50)
    """

    # Response templates that vary by model type
    RESPONSE_TEMPLATES = {
        "gpt-4": [
            "根据您的问题，我来详细解释一下这个概念。首先，我们需要理解其基本原理。",
            "这是一个很好的问题。让我从几个方面来分析：第一，理论基础；第二，实际应用。",
            "关于这个话题，我认为需要考虑多个维度。让我逐一说明。",
        ],
        "claude": [
            "我很乐意为您解答这个问题。让我们先理清关键概念，然后深入探讨。",
            "这个问题很有意思。我的理解是，我们需要从多个角度来分析。",
            "让我来帮您分析这个问题。首先，核心要点如下。",
        ],
        "gemini": [
            "根据我的分析，这个问题的答案涉及几个方面。",
            "这是一个复杂的问题，我来为您提供详细的解答。",
            "基于您的问题，我建议从以下几个角度来考虑。",
        ],
        "default": [
            "这是一个测试回答，用于模拟模型响应。",
            "Mock response generated for benchmarking and testing.",
            "Pipeline OK - all systems functioning properly.",
        ]
    }

    # Mock pricing for different models (CNY per million tokens)
    # Format: (input_price, output_price)
    MOCK_PRICES = {
        # GPT-4 high cost
        "gpt-4": (210.0, 420.0),
        "gpt-4-turbo": (70.0, 210.0),
        "gpt-4o": (105.0, 315.0),

        # Claude medium cost
        "claude": (105.0, 525.0),
        "claude-3": (105.0, 525.0),
        "claude-3-opus": (140.0, 700.0),
        "claude-3-sonnet": (105.0, 525.0),
        "claude-3-haiku": (17.5, 87.5),

        # Gemini lower cost
        "gemini": (70.0, 210.0),
        "gemini-pro": (70.0, 210.0),
        "gemini-ultra": (175.0, 525.0),

        # Lite low cost
        "lite-model": (7.0, 21.0),
        "lite": (7.0, 21.0),
        "lite-001": (7.0, 21.0),

        # Slow high cost (simulates slower, more expensive model)
        "slow-model": (280.0, 560.0),
        "slow": (280.0, 560.0),
        "slow-001": (280.0, 560.0),

        # Default pricing
        "default": (35.0, 105.0),
    }

    # Backward compatibility alias
    MODEL_PRICING = MOCK_PRICES

    def invoke(self, request: ProviderRequest) -> RunRecord:
        """Generate a mock response with configurable timing characteristics."""
        seed = hash((request.dataset_row_id, request.executor_id)) & 0xFFFF
        random.seed(seed)

        # Get mock parameters from options
        options = request.options or {}
        ttft_ms = options.get("mock_ttft_ms", 100)
        tokens_per_chunk = options.get("mock_tokens_per_chunk", 5)
        chunks_count = options.get("mock_chunks", 5)
        chunk_interval_ms = options.get("mock_chunk_interval_ms", 50)

        # Select response based on model type
        model_key = request.model or "default"
        response_templates = self._get_response_templates(model_key)
        response_text = random.choice(response_templates)

        # Create record
        record = RunRecord(
            run_id=request.run_id,
            executor_id=request.executor_id,
            dataset_row_id=request.dataset_row_id,
            provider="mock",
            model=request.model or "mock-001",
        )

        accumulator = StreamAccumulator(record, debug=options.get("stream_debug"))

        # Simulate streaming response with realistic timing
        base_time = now_ms()

        # Split response into chunks
        words = response_text.split()
        total_words = len(words)

        for idx in range(chunks_count):
            # Calculate chunk content
            start_idx = idx * (total_words // chunks_count)
            end_idx = start_idx + tokens_per_chunk if idx < chunks_count - 1 else total_words
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words) if chunk_words else "."

            # Calculate timing to simulate realistic streaming
            if idx == 0:
                # First chunk arrives after TTFT
                chunk_time = base_time + ttft_ms
            else:
                # Subsequent chunks arrive at intervals
                chunk_time = base_time + ttft_ms + (idx * chunk_interval_ms)

            accumulator.append_content(chunk_text, ts_ms=int(chunk_time), seq=idx)

        # Calculate token counts
        prompt_text = "\n".join([msg.get("content", "") for msg in request.messages])
        record.qtokens = get_word_num(prompt_text)
        record.atokens = get_word_num(response_text)
        record.usage = {
            "prompt_tokens": record.qtokens,
            "completion_tokens": record.atokens
        }

        # Calculate mock cost based on model pricing
        input_price, output_price = self._get_model_pricing(request.model or "default")
        record.prompt_cost = (record.qtokens * input_price) / 1_000_000
        record.completion_cost = (record.atokens * output_price) / 1_000_000
        record.total_cost = record.prompt_cost + record.completion_cost
        record.currency = "CNY"
        record.input_price_snapshot = input_price
        record.output_price_snapshot = output_price

        record.status = 200
        record.request_params = dict(options)

        accumulator.finalize(success=True)
        return record

    def _get_response_templates(self, model: str) -> list[str]:
        """Get appropriate response templates based on model name."""
        model_lower = model.lower()

        # Match model type to templates
        if "gpt-4" in model_lower or "gpt4" in model_lower:
            return self.RESPONSE_TEMPLATES["gpt-4"]
        elif "claude" in model_lower:
            return self.RESPONSE_TEMPLATES["claude"]
        elif "gemini" in model_lower:
            return self.RESPONSE_TEMPLATES["gemini"]
        else:
            return self.RESPONSE_TEMPLATES["default"]

    def _get_model_pricing(self, model: str) -> tuple[float, float]:
        """Get mock pricing for a model (input_price, output_price) in CNY per million tokens."""
        model_lower = model.lower()

        # Check for gpt-4 variants
        if "gpt-4o" in model_lower:
            return self.MODEL_PRICING["gpt-4o"]
        elif "turbo" in model_lower:
            return self.MODEL_PRICING["gpt-4-turbo"]
        elif "gpt-4" in model_lower or "gpt4" in model_lower:
            return self.MODEL_PRICING["gpt-4"]

        # Check for claude variants
        elif "opus" in model_lower:
            return self.MODEL_PRICING["claude-3-opus"]
        elif "sonnet" in model_lower:
            return self.MODEL_PRICING["claude-3-sonnet"]
        elif "haiku" in model_lower:
            return self.MODEL_PRICING["claude-3-haiku"]
        elif "claude" in model_lower:
            return self.MODEL_PRICING["claude"]

        # Check for gemini variants
        elif "ultra" in model_lower:
            return self.MODEL_PRICING["gemini-ultra"]
        elif "pro" in model_lower:
            return self.MODEL_PRICING["gemini-pro"]
        elif "gemini" in model_lower:
            return self.MODEL_PRICING["gemini"]

        # Check for special mock models (support both old and new naming)
        elif "slow" in model_lower:
            return self.MODEL_PRICING["slow"]
        elif "lite" in model_lower:
            return self.MODEL_PRICING["lite"]

        # Default pricing
        else:
            return self.MODEL_PRICING["default"]


register_provider("mock", "default")(MockProvider)
register_provider("mock", "chat")(MockProvider)
