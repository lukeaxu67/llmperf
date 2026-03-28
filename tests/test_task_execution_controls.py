from __future__ import annotations

import threading
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from llmperf.config.models import ExecutorConfig
from llmperf.datasets.types import Message, TestCase
from llmperf.executors.openai_chat import OpenAIChatExecutor
from llmperf.executors.process_manager import ProcessManager, TaskCancelledError
from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage
from llmperf.web.main import create_app
from llmperf.web import main as web_main


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


def _reset_services() -> None:
    web_main._task_service = None
    web_main._analysis_service = None


def test_create_task_persists_scheduled_snapshot_and_pricing(tmp_path, monkeypatch):
    db_path = tmp_path / "scheduled.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    with TestClient(create_app()) as client:
        pricing_response = client.post(
            "/api/pricing",
            json={
                "provider": "mock",
                "model": "mock-model",
                "input_price": 2.0,
                "output_price": 8.0,
            },
        )
        assert pricing_response.status_code == 200

        scheduled_at = (datetime.now() + timedelta(minutes=5)).isoformat()
        response = client.post(
            "/api/tasks",
            json={
                "config_content": LEGACY_WEB_CONFIG,
                "auto_start": False,
                "task_type": "benchmark",
                "scheduled_at": scheduled_at,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "scheduled"

        run_id = body["run_id"]
        config_response = client.get(f"/api/tasks/{run_id}/config")
        assert config_response.status_code == 200
        config_content = config_response.json()["config_content"]
        assert "pricing:" in config_content
        assert "provider: mock" in config_content
        assert "model: mock-model" in config_content

        tasks_response = client.get("/api/tasks")
        total_cost_response = client.get("/api/pricing/cost/total")
        assert tasks_response.status_code == 200
        assert total_cost_response.status_code == 200
        assert tasks_response.json()["total"] == total_cost_response.json()["run_count"] == 1


def test_rerun_endpoint_reuses_existing_config(tmp_path, monkeypatch):
    db_path = tmp_path / "rerun.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    with TestClient(create_app()) as client:
        create_response = client.post(
            "/api/tasks",
            json={
                "config_content": LEGACY_WEB_CONFIG,
                "auto_start": False,
            },
        )
        assert create_response.status_code == 200
        source_run_id = create_response.json()["run_id"]

        rerun_response = client.post(
            f"/api/tasks/{source_run_id}/rerun",
            json={"auto_start": False},
        )
        assert rerun_response.status_code == 200
        rerun_run_id = rerun_response.json()["run_id"]
        assert rerun_run_id != source_run_id

        source_config = client.get(f"/api/tasks/{source_run_id}/config").json()["config_content"]
        rerun_config = client.get(f"/api/tasks/{rerun_run_id}/config").json()["config_content"]
        assert source_config == rerun_config


def test_start_endpoint_starts_scheduled_task_immediately(tmp_path, monkeypatch):
    db_path = tmp_path / "start.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    with TestClient(create_app()) as client:
        scheduled_at = (datetime.now() + timedelta(minutes=10)).isoformat()
        create_response = client.post(
            "/api/tasks",
            json={
                "config_content": LEGACY_WEB_CONFIG,
                "auto_start": False,
                "scheduled_at": scheduled_at,
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        service = web_main.get_task_service()
        calls: list[str] = []

        def fake_run_task(target_run_id: str) -> None:
            calls.append(target_run_id)

        monkeypatch.setattr(service, "run_task", fake_run_task)

        start_response = client.post(f"/api/tasks/{run_id}/start")
        assert start_response.status_code == 200
        assert start_response.json()["message"] == "Task start requested"
        assert calls == [run_id]

        task_response = client.get(f"/api/tasks/{run_id}")
        assert task_response.status_code == 200
        assert task_response.json()["status"] == "pending"


def test_openai_chat_executor_uses_runtime_pricing_when_catalog_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "pricing.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))

    storage = Storage(str(db_path))
    storage.add_pricing(
        provider="openai",
        model="gpt-4o-mini",
        input_price=1.5,
        output_price=6.0,
    )

    executor = OpenAIChatExecutor(
        ExecutorConfig(
            id="exec-1",
            name="OpenAI Executor",
            type="openai",
            impl="chat",
            model="gpt-4o-mini",
            param={},
        )
    )

    class FakeProvider:
        def invoke(self, request):
            return RunRecord(
                run_id=request.run_id,
                executor_id=request.executor_id,
                dataset_row_id=request.dataset_row_id,
                provider=request.provider,
                model=request.model,
                status=200,
                qtokens=2000,
                atokens=500,
                ctokens=0,
                action_times=[1, 2, 3],
                content=["ok"],
            )

    executor.provider = FakeProvider()
    record = executor.process_row(
        "run-1",
        TestCase(id="row-1", messages=[Message(role="user", content="ping")]),
        price=None,
    )

    assert record.total_cost > 0
    assert record.input_price_snapshot == 1.5
    assert record.output_price_snapshot == 6.0


def test_process_manager_raises_on_cancellation(monkeypatch):
    cancel_event = threading.Event()
    cancel_event.set()

    manager = ProcessManager(
        run_id="run-cancel",
        config=type(
            "Cfg",
            (),
            {
                "pricing": [],
                "executors": [
                    ExecutorConfig(
                        id="exec-1",
                        name="Executor 1",
                        type="mock",
                        impl="chat",
                        model="mock-model",
                        param={},
                    )
                ],
                "multiprocess": None,
            },
        )(),
        rows=[],
        iterator_steps=["identity"],
        cancel_event=cancel_event,
    )

    try:
        manager.run_all()
    except TaskCancelledError:
        pass
    else:
        raise AssertionError("Expected TaskCancelledError")


def test_process_manager_raises_when_subprocess_fails(monkeypatch):
    class FakeProcess:
        def __init__(self, *args, name=None, **kwargs):
            self.name = name or "fake-process"
            self.exitcode = 1

        def start(self):
            return None

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    monkeypatch.setattr("llmperf.executors.process_manager.mp.Process", FakeProcess)

    manager = ProcessManager(
        run_id="run-fail",
        config=type(
            "Cfg",
            (),
            {
                "pricing": [],
                "executors": [
                    ExecutorConfig(
                        id="exec-1",
                        name="Executor 1",
                        type="mock",
                        impl="chat",
                        model="mock-model",
                        param={},
                    )
                ],
                "multiprocess": None,
            },
        )(),
        rows=[],
        iterator_steps=["identity"],
    )

    try:
        manager.run_all()
    except RuntimeError as exc:
        assert "Executor subprocesses failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
