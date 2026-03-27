"""OpenAI-compatible mock endpoints for request inspection and tests."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(prefix="/mock/v1", tags=["mock-openai"])


@dataclass
class CapturedRequest:
    request_id: str
    path: str
    method: str
    headers: dict[str, str]
    body: dict[str, Any]
    created_at: int


class MockRequestRecorder:
    def __init__(self):
        self._items: list[CapturedRequest] = []

    def add(self, *, path: str, method: str, headers: dict[str, str], body: dict[str, Any]) -> CapturedRequest:
        item = CapturedRequest(
            request_id=uuid.uuid4().hex,
            path=path,
            method=method,
            headers=headers,
            body=body,
            created_at=int(time.time() * 1000),
        )
        self._items.append(item)
        if len(self._items) > 100:
            self._items = self._items[-100:]
        return item

    def latest(self) -> CapturedRequest | None:
        return self._items[-1] if self._items else None

    def clear(self) -> None:
        self._items.clear()

    def items(self) -> list[CapturedRequest]:
        return list(self._items)


_recorder = MockRequestRecorder()


def get_mock_request_recorder() -> MockRequestRecorder:
    return _recorder


async def _capture_request(request: Request) -> CapturedRequest:
    try:
        body = await request.json()
    except Exception:
        body = {}
    headers = {key.lower(): value for key, value in request.headers.items()}
    return _recorder.add(
        path=request.url.path,
        method=request.method,
        headers=headers,
        body=body if isinstance(body, dict) else {"raw": body},
    )


def _last_user_message(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    return content
    response_input = body.get("input")
    if isinstance(response_input, list):
        for item in reversed(response_input):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    return content
    return "ping"


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


@router.delete("/requests")
async def clear_requests():
    _recorder.clear()
    return {"ok": True}


@router.get("/requests")
async def list_requests():
    return {"items": [asdict(item) for item in _recorder.items()]}


@router.get("/requests/latest")
async def latest_request():
    item = _recorder.latest()
    if not item:
        return JSONResponse(status_code=404, content={"detail": "No captured requests"})
    return asdict(item)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    captured = await _capture_request(request)
    body = captured.body
    model = str(body.get("model", "mock-model"))
    prompt = _last_user_message(body)
    reply_text = f"mock:{prompt[:80]}"
    prompt_tokens = _token_count(prompt)
    completion_tokens = _token_count(reply_text)

    if body.get("stream", False):
        created = int(time.time())

        def event_stream():
            chunks = [
                {
                    "id": f"chatcmpl-{captured.request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                },
                {
                    "id": f"chatcmpl-{captured.request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": reply_text}, "finish_reason": None}],
                },
                {
                    "id": f"chatcmpl-{captured.request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
            ]
            for item in chunks:
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-{captured.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@router.post("/responses")
async def responses(request: Request):
    captured = await _capture_request(request)
    body = captured.body
    model = str(body.get("model", "mock-model"))
    prompt = _last_user_message(body)
    reply_text = f"mock:{prompt[:80]}"
    prompt_tokens = _token_count(prompt)
    completion_tokens = _token_count(reply_text)

    if body.get("stream", False):
        response_id = f"resp-{captured.request_id}"

        def event_stream():
            events = [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "sequence_number": 1,
                    "item": {
                        "id": f"item-{captured.request_id}",
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 2,
                    "delta": reply_text,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "sequence_number": 3,
                    "item": {
                        "id": f"item-{captured.request_id}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                    },
                },
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": {
                        "id": response_id,
                        "model": model,
                        "usage": {
                            "input_tokens": prompt_tokens,
                            "output_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    },
                },
            ]
            for item in events:
                yield f"event: {item['type']}\n"
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": f"resp-{captured.request_id}",
        "object": "response",
        "model": model,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": reply_text}]}],
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
