from __future__ import annotations

from fastapi.testclient import TestClient

from llmperf.providers.base import ProviderRequest
from llmperf.providers.openai_chat import OpenAIChatProvider
from llmperf.web.main import create_app, get_task_service


LEGACY_WEB_CONFIG = """
info: "Web Task"
dataset:
  source:
    type: "jsonl"
    name: "test_2"
    config:
      path: "resource/test_2.jsonl"
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
    assert "data/datasets/test_2.jsonl" in stored_config.replace("\\", "/").lower()


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
