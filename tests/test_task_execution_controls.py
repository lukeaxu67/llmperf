from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from llmperf.config.models import ExecutorConfig
from llmperf.datasets.types import Message, TestCase
from llmperf.executors.base import BaseExecutor
from llmperf.executors.openai_chat import OpenAIChatExecutor
from llmperf.executors.process_manager import ProcessManager, TaskCancelledError
from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage
from llmperf.web.main import create_app
from llmperf.web import main as web_main
from llmperf.web.services.run_config_service import load_run_config_content
from llmperf.web.services.task_service import TaskService, TaskStatus


LEGACY_WEB_CONFIG = """
info: "Web Task"
dataset:
  source:
    type: "jsonl"
    name: "test"
    config:
      path: "resource/test.jsonl"
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


CHAIN_WEB_CONFIG = """
info: "Chained Task"
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
    after: ["exec-a"]
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


def test_recover_endpoint_reuses_same_run_id(tmp_path, monkeypatch):
    db_path = tmp_path / "recover.sqlite"
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
        run_id = create_response.json()["run_id"]

        cancel_response = client.post(f"/api/tasks/{run_id}/cancel")
        assert cancel_response.status_code == 200

        service = web_main.get_task_service()
        calls: list[str] = []

        def fake_run_task(target_run_id: str) -> None:
            calls.append(target_run_id)

        monkeypatch.setattr(service, "run_task", fake_run_task)

        recover_response = client.post(f"/api/tasks/{run_id}/recover")
        assert recover_response.status_code == 200
        assert recover_response.json()["run_id"] == run_id
        assert recover_response.json()["message"] == "Task recovery started"
        assert calls == [run_id]


def test_batch_progress_endpoint_returns_multiple_task_snapshots(tmp_path, monkeypatch):
    db_path = tmp_path / "progress-batch.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    with TestClient(create_app()) as client:
        first = client.post("/api/tasks", json={"config_content": LEGACY_WEB_CONFIG, "auto_start": False})
        second = client.post("/api/tasks", json={"config_content": LEGACY_WEB_CONFIG, "auto_start": False})
        assert first.status_code == 200
        assert second.status_code == 200

        first_run_id = first.json()["run_id"]
        second_run_id = second.json()["run_id"]

        storage = Storage(str(db_path))
        storage.insert_record(
            RunRecord(
                run_id=first_run_id,
                executor_id="mock-001",
                dataset_row_id="row-1",
                provider="mock",
                model="mock-model",
                status=200,
                qtokens=10,
                atokens=20,
                total_cost=0.12,
                currency="CNY",
                action_times=[1, 2, 3],
                content=["ok"],
            )
        )

        response = client.post(
            "/api/tasks/progress/batch",
            json={"run_ids": [first_run_id, second_run_id]},
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert first_run_id in items
        assert second_run_id in items
        assert items[first_run_id]["completed"] == 1


def test_task_errors_endpoint_returns_parsed_error_details(tmp_path, monkeypatch):
    db_path = tmp_path / "task-errors.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    with TestClient(create_app()) as client:
        create_response = client.post(
            "/api/tasks",
            json={"config_content": LEGACY_WEB_CONFIG, "auto_start": False},
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        storage = Storage(str(db_path))
        storage.insert_record(
            RunRecord(
                run_id=run_id,
                executor_id="mock-001",
                dataset_row_id="row-err",
                provider="mock",
                model="mock-model",
                status=403,
                info='{"error_type":"Model.AccessDenied","message":"Model access denied."}',
            )
        )

        response = client.get(f"/api/tasks/{run_id}/errors")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["errors"][0]["error"]["status_code"] == 403
        assert "Model access denied." in body["errors"][0]["error"]["error_message"]


def test_rename_task_updates_persisted_name(tmp_path, monkeypatch):
    db_path = tmp_path / "rename.sqlite"
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
        run_id = create_response.json()["run_id"]

        rename_response = client.put(
            f"/api/tasks/{run_id}/name",
            json={"task_name": "Renamed Task"},
        )
        assert rename_response.status_code == 200

        task_response = client.get(f"/api/tasks/{run_id}")
        assert task_response.status_code == 200
        assert task_response.json()["task_name"] == "Renamed Task"


def test_rename_task_rejects_running_task(tmp_path, monkeypatch):
    db_path = tmp_path / "rename-running.sqlite"
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
        run_id = create_response.json()["run_id"]

        service = web_main.get_task_service()
        task_info = service.get_task(run_id)
        assert task_info is not None
        task_info.status = TaskStatus.RUNNING

        rename_response = client.put(
            f"/api/tasks/{run_id}/name",
            json={"task_name": "Should Fail"},
        )
        assert rename_response.status_code == 400
        assert "cannot be changed" in rename_response.json()["detail"]


def test_executor_resume_skips_completed_dataset_rows(tmp_path):
    db_path = tmp_path / "resume-skip.sqlite"
    storage = Storage(str(db_path))

    class DummyExecutor(BaseExecutor):
        def __init__(self, config):
            super().__init__(config)
            self.processed_ids: list[str] = []

        def process_row(self, run_id, row, price=None):
            self.processed_ids.append(row.id)
            return RunRecord(
                run_id=run_id,
                executor_id=self.config.id,
                dataset_row_id=row.id,
                provider="mock",
                model="mock-model",
                status=200,
                qtokens=10,
                atokens=10,
                action_times=[1, 2, 3],
                content=["ok"],
            )

    executor = DummyExecutor(
        ExecutorConfig(
            id="exec-1",
            name="Executor 1",
            type="mock",
            impl="chat",
            model="mock-model",
            param={},
        )
    )

    storage.insert_record(
        RunRecord(
            run_id="run-1",
            executor_id="exec-1",
            dataset_row_id="row-1",
            provider="mock",
            model="mock-model",
            status=200,
        )
    )

    rows = [
        TestCase(id="row-1", messages=[Message(role="user", content="a")]),
        TestCase(id="row-2", messages=[Message(role="user", content="b")]),
    ]
    executor.run("run-1", rows, storage)

    assert executor.processed_ids == ["row-2"]


def test_task_service_restores_running_tasks_after_restart(tmp_path, monkeypatch):
    db_path = tmp_path / "restore-running.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    config, normalized_content = load_run_config_content(LEGACY_WEB_CONFIG)
    storage = Storage(str(db_path))
    storage.register_run(
        "run-restored",
        config,
        config_path="",
        config_content=normalized_content,
        task_type="benchmark",
        status="running",
    )
    storage.update_run_status("run-restored", status="running", started_at=int(time.time()))

    calls: list[str] = []

    def fake_run_task(self, run_id: str) -> None:
        calls.append(run_id)

    monkeypatch.setattr(TaskService, "run_task", fake_run_task)

    service = TaskService()
    assert "run-restored" in service._tasks

    deadline = time.time() + 2
    while time.time() < deadline and calls != ["run-restored"]:
        time.sleep(0.05)

    assert calls == ["run-restored"]


def test_storage_infers_legacy_run_statuses_from_timestamps(tmp_path):
    db_path = tmp_path / "legacy-status.sqlite"
    storage = Storage(str(db_path))
    config, normalized_content = load_run_config_content(LEGACY_WEB_CONFIG)

    storage.register_run(
        "run-legacy-completed",
        config,
        config_path="",
        config_content=normalized_content,
        task_type="benchmark",
        status="pending",
    )
    storage.update_run_status(
        "run-legacy-completed",
        status="pending",
        started_at=int(time.time()) - 30,
        completed_at=int(time.time()) - 5,
        error_message="",
    )

    storage.register_run(
        "run-legacy-running",
        config,
        config_path="",
        config_content=normalized_content,
        task_type="benchmark",
        status="pending",
    )
    storage.update_run_status(
        "run-legacy-running",
        status="pending",
        started_at=int(time.time()) - 20,
        completed_at=0,
        error_message="",
    )

    assert storage.get_run("run-legacy-completed")["status"] == "completed"
    assert storage.get_run("run-legacy-running")["status"] == "running"
    assert storage.count_runs("completed") == 1
    assert storage.count_runs("running") == 1


def test_task_service_prefers_config_snapshot_over_stale_config_path(tmp_path, monkeypatch):
    db_path = tmp_path / "stale-config.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    config, normalized_content = load_run_config_content(LEGACY_WEB_CONFIG)
    storage = Storage(str(db_path))
    storage.register_run(
        "run-stale-config",
        config,
        config_path="/tmp/should-not-be-used.yaml",
        config_content=normalized_content,
        task_type="benchmark",
        status="completed",
    )
    storage.update_run_status(
        "run-stale-config",
        status="completed",
        started_at=int(time.time()) - 5,
        completed_at=int(time.time()),
    )

    service = TaskService()
    loaded = service._load_run_config("run-stale-config")
    assert loaded is not None
    assert loaded.dataset.source.name == "test"


def test_create_task_from_config_path_persists_self_contained_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "config-path.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    config_path = tmp_path / "task.yaml"
    config_path.write_text(LEGACY_WEB_CONFIG, encoding="utf-8")

    service = TaskService()
    task = service.create_task(config_path=str(config_path))
    snapshot = service._storage.get_run(task.run_id)

    assert snapshot is not None
    assert snapshot["config_content"]
    assert snapshot["config_path"] == ""
    assert task.config_path is None


def test_task_service_marks_topology_end_node_completed_when_all_executors_completed(tmp_path, monkeypatch):
    db_path = tmp_path / "end-node.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    _reset_services()

    config, _ = load_run_config_content(CHAIN_WEB_CONFIG)
    service = TaskService()
    topology = service._build_topology(
        config,
        [
            {"id": "exec-a", "name": "Executor A", "status": "completed"},
            {"id": "exec-b", "name": "Executor B", "status": "completed"},
        ],
    )

    end_node = next(node for node in topology["nodes"] if node["id"] == "__end__")
    assert end_node["status"] == "completed"


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
