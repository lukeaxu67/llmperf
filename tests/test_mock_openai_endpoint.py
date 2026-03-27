from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from urllib import request as urllib_request

import uvicorn
from fastapi.testclient import TestClient

from llmperf.providers.base import ProviderRequest
from llmperf.providers.openai_chat import OpenAIChatProvider
from llmperf.providers.response_api import ResponsesProvider
from llmperf.web.main import create_app


class LiveServer:
    def __init__(self):
        self.app = create_app()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with urllib_request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1):
                    return f"http://127.0.0.1:{self.port}"
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("live server did not start in time")

    def __exit__(self, exc_type, exc, tb):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _json_request(url: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url, data=data, headers=headers, method=method)
    with urllib_request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_mock_chat_endpoint_captures_headers_body_and_forwarded_params():
    with LiveServer() as base_url:
        _json_request(f"{base_url}/mock/v1/requests", method="DELETE")

        provider = OpenAIChatProvider("openai")
        record = provider.invoke(
            ProviderRequest(
                run_id="run-1",
                executor_id="exec-1",
                dataset_row_id="row-1",
                provider="openai",
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello mock"}],
                options={
                    "api_url": f"{base_url}/mock/v1",
                    "api_key": "sk-test",
                    "extra_headers": {"x-trace-id": "trace-123"},
                    "extra_body": {"reasoning": {"effort": "low"}},
                    "temperature": 0.2,
                    "max_tokens": 64,
                },
            )
        )

        assert record.status == 200
        captured = _json_request(f"{base_url}/mock/v1/requests/latest")
        assert captured["headers"]["authorization"] == "Bearer sk-test"
        assert captured["headers"]["x-trace-id"] == "trace-123"
        assert captured["body"]["temperature"] == 0.2
        assert captured["body"]["max_tokens"] == 64
        assert captured["body"]["reasoning"] == {"effort": "low"}
        assert captured["body"]["stream"] is True


def test_mock_responses_endpoint_captures_headers_body_and_forwarded_params():
    with LiveServer() as base_url:
        _json_request(f"{base_url}/mock/v1/requests", method="DELETE")

        provider = ResponsesProvider("responses")
        record = provider.invoke(
            ProviderRequest(
                run_id="run-2",
                executor_id="exec-2",
                dataset_row_id="row-2",
                provider="responses",
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "inspect response api"}],
                options={
                    "api_url": f"{base_url}/mock/v1",
                    "api_key": "sk-response",
                    "extra_headers": {"x-env": "test"},
                    "extra_body": {"metadata": {"suite": "responses"}},
                    "temperature": 0.3,
                },
            )
        )

        assert record.status == 200
        captured = _json_request(f"{base_url}/mock/v1/requests/latest")
        assert captured["headers"]["authorization"] == "Bearer sk-response"
        assert captured["headers"]["x-env"] == "test"
        assert captured["body"]["temperature"] == 0.3
        assert captured["body"]["metadata"] == {"suite": "responses"}
        assert captured["body"]["stream"] is True


def test_test_run_uses_mock_openai_endpoint(tmp_path: Path):
    dataset_path = tmp_path / "mock.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "1", "messages": [{"role": "user", "content": "ping from test-run"}]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with LiveServer() as mock_base:
        _json_request(f"{mock_base}/mock/v1/requests", method="DELETE")

        config_content = f"""
info: "Mock OpenAI Test"
dataset:
  source:
    type: "jsonl"
    name: "mock"
    config:
      path: "{dataset_path.as_posix()}"
executors:
  - id: "openai-chat"
    name: "OpenAI Chat"
    type: "openai"
    impl: "chat"
    concurrency: 1
    model: "gpt-4o-mini"
    api_url: "{mock_base}/mock/v1"
    api_key: "sk-task"
    param:
      extra_headers:
        x-source: "test-run"
      extra_body:
        metadata:
          case: "task-test-run"
      temperature: 0.4
"""

        with TestClient(create_app()) as client:
            response = client.post(
                "/api/tasks/test-run",
                json={"config_content": config_content},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        captured = _json_request(f"{mock_base}/mock/v1/requests/latest")
        assert captured["headers"]["authorization"] == "Bearer sk-task"
        assert captured["headers"]["x-source"] == "test-run"
        assert captured["body"]["metadata"] == {"case": "task-test-run"}
        assert captured["body"]["temperature"] == 0.4
