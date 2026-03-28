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

_INTERNAL_OPTION_KEYS = {
    "api_url",
    "api_key",
    "timeout",
    "default_headers",
    "messages_mode",
    "stream_debug",
    "extra_body",
    "extra_headers",
    "extra_header",
}


def _safe_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return str(value)


def _serialize_exception(exc: Exception) -> str:
    payload: Dict[str, Any] = {
        "error_type": type(exc).__name__,
        "desc": getattr(exc, "message", None) or str(exc),
    }
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        payload["status_code"] = status_code
    request_id = getattr(exc, "request_id", None)
    if request_id:
        payload["request_id"] = request_id
    body = _safe_json_value(getattr(exc, "body", None))
    if body not in (None, "", {}):
        payload["body"] = body
    response = getattr(exc, "response", None)
    if response is not None:
        response_body = _safe_json_value(getattr(response, "json", None))
        if callable(getattr(response, "json", None)):
            try:
                response_body = response.json()
            except Exception:
                response_body = None
        if response_body not in (None, "", {}):
            payload["response"] = response_body
    return json.dumps(payload, ensure_ascii=False)


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
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=options.get("timeout", 60),
            default_headers=options.get("default_headers"),
        )

    def invoke(self, request: ProviderRequest) -> RunRecord:
        client = self._client(request.options)
        extra_body = request.options.get("extra_body")
        extra_headers = request.options.get("extra_headers") or request.options.get("extra_header")
        forwarded_options = {
            key: value
            for key, value in request.options.items()
            if key not in _INTERNAL_OPTION_KEYS
        }
        record = RunRecord(
            run_id=request.run_id,
            executor_id=request.executor_id,
            dataset_row_id=request.dataset_row_id,
            provider=request.provider,
            model=request.model,
            request_params={
                **forwarded_options,
                "extra_body": extra_body or {},
                "extra_headers": extra_headers or {},
            },
        )
        accumulator = StreamAccumulator(record, debug=request.options.get("stream_debug"))
        stream_failed = False
        final_ts = None
        try:
            stream = client.responses.create(
                model=request.model,
                input=normalize_messages(request.messages),
                stream=True,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=request.options.get("timeout", 300),
                **forwarded_options,
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
            record.info = _serialize_exception(exc)
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
