from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from openai import APIStatusError, OpenAI

from ..records.model import RunRecord, now_ms
from .base import BaseProvider, ProviderRequest, normalize_messages, register_provider
from .streaming import StreamAccumulator

logger = logging.getLogger(__name__)


def _shuffle_prefix(text: str, prefix_len: int = 10) -> str:
    if not text:
        return text
    n = min(prefix_len, len(text))
    prefix_chars = list(text[: n])
    if n > 1:
        random.shuffle(prefix_chars)
    return "".join(prefix_chars) + text[n:]


def _message_payload(messages: List[Dict[str, str]], mode: str):
    normalized = normalize_messages(messages)
    if mode == "random":
        randomized: List[Dict[str, str]] = []
        for msg in normalized:
            randomized.append({"role": msg["role"], "content": _shuffle_prefix(msg["content"])})
        return randomized
    if mode == "response":
        return normalized
    return normalized


class OpenAIChatProvider(BaseProvider):
    """Base provider for OpenAI-compatible Chat Completions APIs."""

    env_prefix: Optional[str] = None

    def _prefix(self) -> str:
        return (self.env_prefix or self.provider_name).upper()

    def build_client(self, options: Dict[str, Any]) -> OpenAI:
        prefix = self._prefix()
        base_url = options.get("api_url") or os.environ.get(f"{prefix}_BASE_URL")
        api_key = options.get("api_key") or os.environ.get(f"{prefix}_API_KEY")
        timeout = options.get("timeout", 60)
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def handle_special_chunk(self, chunk, record: RunRecord) -> bool:
        code = getattr(chunk, "code", 0)
        if code not in (0, None):
            record.status = int(code)
            record.info = json.dumps({"desc": getattr(chunk, "message", ""), "sid": getattr(chunk, "sid", "")})
            logger.error(
                "[%s] provider error code=%s message=%s sid=%s",
                self.provider_name,
                code,
                getattr(chunk, "message", ""),
                getattr(chunk, "sid", ""),
            )
            return True
        return False

    def on_delta(self, accumulator: StreamAccumulator, delta, ts: int):
        reasoning_content = getattr(delta, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content:
            accumulator.append_reasoning(reasoning_content, ts_ms=ts)
        content = getattr(delta, "content", "")
        if isinstance(content, str) and content:
            accumulator.append_content(content, ts_ms=ts)

    def update_usage(self, record: RunRecord, usage):
        if not usage:
            return
        record.qtokens = getattr(usage, "prompt_tokens", record.qtokens)
        record.atokens = getattr(usage, "completion_tokens", record.atokens)
        record.ctokens = (
            getattr(getattr(usage, "prompt_tokens_details", object()), "cached_tokens", 0)
            or getattr(usage, "prompt_cache_hit_tokens", 0)
        )
        usage_dict = getattr(usage, "to_dict", None)
        record.usage = usage_dict() if callable(usage_dict) else json.loads(usage.model_dump_json())

    def invoke(self, request: ProviderRequest) -> RunRecord:
        client = self.build_client(request.options)
        messages_payload = _message_payload(request.messages, request.options.get("messages_mode", "standard"))
        extra_body = request.options.get("extra_body")
        stream_options = request.options.get("stream_options", {"include_usage": True})
        record = RunRecord(
            run_id=request.run_id,
            executor_id=request.executor_id,
            dataset_row_id=request.dataset_row_id,
            provider=request.provider,
            model=request.model,
        )
        accumulator = StreamAccumulator(record, debug=request.options.get("stream_debug"))
        stream_failed = False
        final_ts = None
        try:
            stream = client.chat.completions.create(
                model=request.model,
                messages=messages_payload,
                stream=True,
                stream_options=stream_options,
                extra_body=extra_body,
                timeout=request.options.get("timeout", 300),
            )
            for chunk in stream:
                if self.handle_special_chunk(chunk, record):
                    stream_failed = True
                    break
                if not chunk.choices:
                    self.update_usage(record, getattr(chunk, "usage", None))
                    continue
                delta = chunk.choices[0].delta
                ts = now_ms()
                self.on_delta(accumulator, delta, ts)
                self.update_usage(record, getattr(chunk, "usage", None))
            final_ts = now_ms()
        except Exception as exc:
            if isinstance(exc, APIStatusError):
                record.status = getattr(exc, "status_code", -1)
                logger.error(
                    "[%s] APIStatusError status=%s message=%s",
                    self.provider_name,
                    getattr(exc, "status_code", "-"),
                    getattr(exc, "message", str(exc)),
                )
            else:
                record.status = -1
                logger.error("[%s] provider exception: %s", self.provider_name, exc)
            record.info = json.dumps({"desc": str(exc)})
            stream_failed = True
            final_ts = final_ts or now_ms()
        finally:
            accumulator.finalize(success=not stream_failed, final_ts=final_ts)

        if not record.reasoning and not record.content:
            record.status = -1
            record.info = json.dumps({"desc": "no content"})
        else:
            record.status = record.status or 200
        return record


class SparkChatProvider(OpenAIChatProvider):
    env_prefix = "IFLYTEK"

    def handle_special_chunk(self, chunk, record: RunRecord) -> bool:
        handled = super().handle_special_chunk(chunk, record)
        if handled and record.status != 0:
            return True
        # Spark 同帧返回 usage
        if getattr(chunk, "usage", None):
            self.update_usage(record, chunk.usage)
        return False


class HuoshanChatProvider(OpenAIChatProvider):
    env_prefix = "HUOSHAN"


class DefaultVendorProvider(OpenAIChatProvider):
    pass


PROVIDER_MAPPING = {
    "openai": DefaultVendorProvider,
    "qianwen": DefaultVendorProvider,
    "zhipu": DefaultVendorProvider,
    "deepseek": DefaultVendorProvider,
    "spark": SparkChatProvider,
    "hunyuan": DefaultVendorProvider,
    "huoshan": HuoshanChatProvider,
    "moonshot": DefaultVendorProvider,
}


for provider, cls in PROVIDER_MAPPING.items():
    register_provider(provider, "default")(cls)
