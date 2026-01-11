from __future__ import annotations

import json
import tempfile
from pathlib import Path

from llmperf.analysis import create_analysis
from llmperf.config.models import RunConfig
from llmperf.datasets.types import Message, TestCase
from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage


def _write_jsonl(path: Path, cases: list[TestCase]) -> None:
    path.write_text(
        "\n".join([json.dumps(c.model_dump(), ensure_ascii=False) for c in cases]) + "\n",
        encoding="utf-8",
    )


def test_history_analysis_summarizes_runs_and_models() -> None:
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "perf.sqlite")
        storage = Storage(db_path)
        try:
            run_id = "run1"
            cfg_path = Path(td) / "task.yaml"
            cfg_path.write_text(
                "info: test\n"
                "dataset:\n"
                "  source:\n"
                "    type: jsonl\n"
                "    name: demo\n"
                "    config:\n"
                "      path: dataset.jsonl\n"
                "executors:\n"
                "  - id: e1\n"
                "    name: e1\n"
                "    type: mock\n"
                "    model: mock\n",
                encoding="utf-8",
            )
            cfg = RunConfig.model_validate(json.loads(json.dumps({"info": "test", "dataset": {"source": {"type": "jsonl", "name": "demo", "config": {"path": "dataset.jsonl"}}}, "executors": [{"id": "e1", "name": "e1", "type": "mock", "model": "mock"}]})))
            storage.register_run(
                run_id,
                cfg,
                config_path=str(cfg_path),
                config_content=cfg_path.read_text(encoding="utf-8"),
                pricing_path=None,
                pricing_content="",
            )
            ok = RunRecord(run_id=run_id, executor_id="e1", dataset_row_id="a", provider="mock", model="mock", status=200)
            ok.total_cost = 1.5
            err = RunRecord(run_id=run_id, executor_id="e1", dataset_row_id="b", provider="mock", model="mock", status=500)
            err.total_cost = 0.0
            storage.insert_record(ok)
            storage.insert_record(err)
        finally:
            storage.close()

        analysis = create_analysis("history", {"query": {"db_path": db_path}})
        out = analysis.run()
        assert out["count"] == 1
        assert out["items"][0]["run_id"] == "run1"
        models = out["items"][0]["models"]
        assert models[0]["count"] == 2
        assert models[0]["success_count"] == 1
        assert models[0]["failure_count"] == 1
        assert models[0]["total_cost"] == 1.5


def test_export_analysis_appends_assistant_message() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        db_path = str(td_path / "perf.sqlite")
        dataset_path = td_path / "dataset.jsonl"
        cfg_path = td_path / "task.yaml"

        cases = [
            TestCase(id="a", messages=[Message(role="user", content="hi")], metadata={}),
        ]
        _write_jsonl(dataset_path, cases)

        cfg_path.write_text(
            "info: test\n"
            "dataset:\n"
            "  source:\n"
            "    type: jsonl\n"
            "    name: demo\n"
            "    config:\n"
            "      path: dataset.jsonl\n"
            "      encoding: utf-8\n"
            "executors:\n"
            "  - id: e1\n"
            "    name: e1\n"
            "    type: mock\n"
            "    model: mock\n",
            encoding="utf-8",
        )

        storage = Storage(db_path)
        try:
            cfg = RunConfig.model_validate(
                {
                    "info": "test",
                    "dataset": {"source": {"type": "jsonl", "name": "demo", "config": {"path": "dataset.jsonl"}}},
                    "executors": [{"id": "e1", "name": "e1", "type": "mock", "model": "mock"}],
                }
            )
            storage.register_run(
                "run1",
                cfg,
                config_path=str(cfg_path),
                config_content=cfg_path.read_text(encoding="utf-8"),
                pricing_path=None,
                pricing_content="",
            )
            rec = RunRecord(run_id="run1", executor_id="e1", dataset_row_id="a", provider="mock", model="mock", status=200)
            rec.content = ["hello from model"]
            storage.insert_record(rec)
        finally:
            storage.close()

        analysis = create_analysis("export", {"run_id": "run1", "output_dir": td, "query": {"db_path": db_path}})
        out = analysis.run()
        out_path = Path(out["output_path"])
        lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        exported = TestCase.model_validate_json(lines[0])
        assert exported.messages[-1].role == "assistant"
        assert "hello from model" in exported.messages[-1].content
        assert exported.metadata and exported.metadata["executor_id"] == "e1"
