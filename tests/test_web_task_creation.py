from __future__ import annotations

import time

from fastapi.testclient import TestClient

from llmperf.providers.base import ProviderRequest
from llmperf.providers.openai_chat import OpenAIChatProvider
from llmperf.records.model import RunRecord
from llmperf.web.main import create_app, get_task_service


LEGACY_WEB_CONFIG = """
info: "Web Task"
dataset:
  source:
    type: "jsonl"
    name: "test_3"
    config:
      path: "resource/test_3.jsonl"
  iterator:
    mutation_chain: ["identity"]
    max_rounds: 1
executors:
  - id: "mock-001"
    name: "Mock Executor"
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "mock-model"
"""


def test_test_run_resolves_legacy_web_dataset_path():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/tasks/test-run",
            json={"config_content": LEGACY_WEB_CONFIG},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["response"]


def test_create_task_normalizes_dataset_path_for_web_tasks():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/tasks",
            json={
                "config_content": LEGACY_WEB_CONFIG,
                "auto_start": False,
            },
        )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    stored_config = get_task_service()._config_contents[run_id]
    assert "data/datasets/test_3.jsonl" in stored_config.replace("\\", "/").lower()


def test_openai_chat_provider_forwards_extra_headers(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            delta = type("Delta", (), {"reasoning_content": None, "content": "hello"})()
            choice = type("Choice", (), {"delta": delta})()
            chunk = type("Chunk", (), {"choices": [choice]})()
            return [chunk]

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setattr(OpenAIChatProvider, "build_client", lambda self, options: FakeClient())

    provider = OpenAIChatProvider("openai")
    record = provider.invoke(
        ProviderRequest(
            run_id="run-1",
            executor_id="exec-1",
            dataset_row_id="row-1",
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            options={
                "extra_header": {"x-foo": "bar"},
                "extra_body": {"reasoning": {"effort": "low"}},
            },
        )
    )

    assert captured["extra_headers"] == {"x-foo": "bar"}
    assert captured["extra_body"] == {"reasoning": {"effort": "low"}}
    assert record.status == 200


def test_test_run_returns_actionable_error_details(monkeypatch):
    from llmperf.providers import base as provider_base

    class FakeProvider:
        def invoke(self, request):
            return RunRecord(
                run_id=request.run_id,
                executor_id=request.executor_id,
                dataset_row_id=request.dataset_row_id,
                provider=request.provider,
                model=request.model,
                status=401,
                info='{"error_type":"AuthenticationError","desc":"Invalid API key","request_id":"req_123"}',
            )

    monkeypatch.setattr(provider_base, "create_provider", lambda *args, **kwargs: FakeProvider())

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/tasks/test-run",
            json={"config_content": LEGACY_WEB_CONFIG},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert len(body["results"]) == 1
    assert body["results"][0]["status_code"] == 401
    assert body["results"][0]["error_type"] == "AuthenticationError"
    assert "状态码: 401" in body["results"][0]["error"]
    assert "错误信息: Invalid API key" in body["results"][0]["error"]
    assert "请求ID: req_123" in body["results"][0]["error"]


def test_openai_chat_provider_preserves_exception_details_when_no_content(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("Model access denied.")

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setattr(OpenAIChatProvider, "build_client", lambda self, options: FakeClient())

    provider = OpenAIChatProvider("qianwen")
    record = provider.invoke(
        ProviderRequest(
            run_id="run-2",
            executor_id="exec-2",
            dataset_row_id="row-2",
            provider="qianwen",
            model="qwen3.5-plus",
            messages=[{"role": "user", "content": "ping"}],
            options={},
        )
    )

    assert record.status == -1
    assert "Model access denied." in record.info
    assert "no content" not in record.info


def test_openai_chat_provider_does_not_forward_model_twice(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            delta = type("Delta", (), {"reasoning_content": None, "content": "ok"})()
            choice = type("Choice", (), {"delta": delta})()
            chunk = type("Chunk", (), {"choices": [choice]})()
            return [chunk]

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setattr(OpenAIChatProvider, "build_client", lambda self, options: FakeClient())

    provider = OpenAIChatProvider("openai")
    provider.invoke(
        ProviderRequest(
            run_id="run-3",
            executor_id="exec-3",
            dataset_row_id="row-3",
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            options={"model": "gpt-4o-mini", "temperature": 0.1},
        )
    )

    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0.1


def test_test_run_executes_executors_in_parallel(monkeypatch):
    from llmperf.providers import base as provider_base

    config_content = """
info: "Parallel Test"
dataset:
  source:
    type: "jsonl"
    name: "test_3"
    config:
      path: "resource/test_3.jsonl"
  iterator:
    mutation_chain: ["identity"]
    max_rounds: 1
executors:
  - id: "exec-a"
    name: "Executor A"
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "mock-a"
  - id: "exec-b"
    name: "Executor B"
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "mock-b"
"""

    class SlowProvider:
        def invoke(self, request):
            time.sleep(0.35)
            return RunRecord(
                run_id=request.run_id,
                executor_id=request.executor_id,
                dataset_row_id=request.dataset_row_id,
                provider=request.provider,
                model=request.model,
                status=200,
                content=["ok"],
                action_times=[1, 2, 3],
            )

    monkeypatch.setattr(provider_base, "create_provider", lambda *args, **kwargs: SlowProvider())

    started = time.perf_counter()
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/tasks/test-run",
            json={"config_content": config_content},
        )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert len(body["results"]) == 2
    assert elapsed < 0.65
