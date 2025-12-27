from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from openai import APIStatusError, OpenAI

from ..records.model import RunRecord, now_ms
from .base import BaseProvider, ProviderRequest, normalize_messages, register_provider
from .streaming import StreamAccumulator

logger = logging.getLogger(__name__)


def _event_to_dict(event: Any) -> Dict[str, Any]:
    try:
        if hasattr(event, "model_dump") and callable(event.model_dump):
            return event.model_dump()
    except Exception:
        pass
    try:
        if hasattr(event, "to_dict") and callable(event.to_dict):
            return event.to_dict()
    except Exception:
        pass
    if isinstance(event, dict):
        return event
    try:
        if hasattr(event, "json") and callable(event.json):
            return json.loads(event.json())
    except Exception:
        pass
    try:
        return json.loads(str(event))
    except Exception:
        return {"type": None}


class ResponsesProvider(BaseProvider):
    def __init__(self, provider_name: str):
        super().__init__(provider_name)

    def _client(self, options: Dict[str, Any]) -> OpenAI:
        base_url = options.get("api_url") or os.environ.get(f"{self.provider_name.upper()}_BASE_URL")
        api_key = options.get("api_key") or os.environ.get(f"{self.provider_name.upper()}_API_KEY")
        return OpenAI(api_key=api_key, base_url=base_url, timeout=options.get("timeout", 60))

    def invoke(self, request: ProviderRequest) -> RunRecord:
        client = self._client(request.options)
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

        extra_body = request.options.get("extra_body")
        try:
            stream = client.responses.create(
                model=request.model,
                input=normalize_messages(request.messages),
                stream=True,
                extra_body=extra_body,
                timeout=request.options.get("timeout", 300),
            )
            items: Dict[str, Dict[str, Any]] = {}
            for event in stream:
                e = _event_to_dict(event)
                etype = e.get("type")
                ts = now_ms()
                match etype:
                    case "response.output_item.added":
                        item = e.get("item") or {}
                        item_id = item.get("id")
                        if item_id:
                            items[item_id] = {
                                "item_id": item_id,
                                "type": item.get("type"),
                                "role": item.get("role"),
                                "output_index": e.get("output_index"),
                                "start_ms": ts,
                                "start_seq": e.get("sequence_number"),
                                "status": item.get("status"),
                            }
                    case "response.output_item.done":
                        item = e.get("item") or {}
                        item_id = item.get("id")
                        if item_id and item_id in items:
                            seg = items[item_id]
                            seg["end_ms"] = ts
                            seg["end_seq"] = e.get("sequence_number")
                            seg["status"] = item.get("status") or seg.get("status")
                            seg["duration_ms"] = ts - seg["start_ms"]
                    case "response.output_text.delta":
                        delta = e.get("delta") or ""
                        if delta:
                            accumulator.append_content(delta, ts_ms=ts, seq=e.get("sequence_number"))
                    case "response.reasoning_summary_text.delta":
                        delta = e.get("delta") or ""
                        if delta:
                            accumulator.append_reasoning(delta, ts_ms=ts, seq=e.get("sequence_number"))
                    case "response.completed":
                        resp = e.get("response") or {}
                        usage = resp.get("usage") or {}
                        record.usage = usage
                        record.qtokens = usage.get("input_tokens", 0)
                        record.atokens = usage.get("output_tokens", 0)
                        record.ctokens = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                        final_ts = ts
                    case "response.failed" | "response.incomplete":
                        resp = e.get("response") or {}
                        usage = resp.get("usage") or {}
                        if usage:
                            record.usage = usage
                        stream_failed = True
                        final_ts = ts
                    case _:
                        pass

            record.output_items = sorted(items.values(), key=lambda x: (x.get("output_index", 0), x.get("start_ms", 0)))
            final_ts = final_ts or now_ms()
        except Exception as exc:
            if isinstance(exc, APIStatusError):
                logger.error(
                    "[%s-response] APIStatusError status=%s message=%s",
                    self.provider_name,
                    getattr(exc, "status_code", "-"),
                    getattr(exc, "message", str(exc)),
                )
            else:
                logger.error("[%s-response] provider exception: %s", self.provider_name, exc)
            record.status = getattr(exc, "status_code", -1) if isinstance(exc, APIStatusError) else -1
            record.info = json.dumps({"desc": str(exc)})
            stream_failed = True
            final_ts = final_ts or now_ms()
        finally:
            accumulator.finalize(success=not stream_failed, final_ts=final_ts)

        if stream_failed or (not record.reasoning and not record.content):
            record.status = record.status or -1
        else:
            record.status = 200
        return record


for provider in ["huoshan_response", "responses"]:
    register_provider(provider, "response")(ResponsesProvider)
