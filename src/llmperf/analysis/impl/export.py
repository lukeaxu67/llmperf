from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from ..analysis_registry import register_analysis
from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery

from llmperf.datasets.types import Message, TestCase
from llmperf.records.db import RunORM


def _resolve_path(value: str, *, base_dir: Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _load_run_config(config_content: str) -> dict:
    data = yaml.safe_load(config_content or "") or {}
    if not isinstance(data, dict):
        raise ValueError("run config snapshot must be a YAML mapping")
    return data


def _extract_jsonl_dataset_path(run: RunORM, *, override_path: str | None) -> Path:
    if override_path:
        return Path(override_path).expanduser().resolve()

    config_content = (run.config_content or "").strip()
    if not config_content:
        raise ValueError("run config snapshot is missing; specify dataset_path in export config")

    cfg = _load_run_config(config_content)
    dataset = cfg.get("dataset") or {}
    source = (dataset.get("source") or {}) if isinstance(dataset, dict) else {}
    source_type = source.get("type")
    if source_type != "jsonl":
        raise ValueError(
            "export currently requires dataset.source.type=jsonl; specify dataset_path to override"
        )
    source_cfg = source.get("config") or {}
    if not isinstance(source_cfg, dict):
        raise ValueError("dataset.source.config must be a mapping")
    path_value = source_cfg.get("path")
    if not path_value:
        raise ValueError("dataset.source.config.path missing from run config snapshot")

    base_dir = None
    try:
        if run.config_path:
            base_dir = Path(run.config_path).expanduser().resolve().parent
    except Exception:
        base_dir = None

    return _resolve_path(str(path_value), base_dir=base_dir)


def _assistant_text(record) -> str:
    content = "".join(getattr(record, "content", []) or [])
    if content.strip():
        return content
    reasoning = "".join(getattr(record, "reasoning", []) or [])
    if reasoning.strip():
        return reasoning
    info = str(getattr(record, "info", "") or "").strip()
    status = getattr(record, "status", None)
    if info:
        return f"[ERROR status={status}] {info}"
    return f"[EMPTY status={status}]"


@register_analysis("export")
class ExportAnalysis(BaseAnalysis["ExportAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        run_id: str = Field(min_length=1)
        output_dir: str = "."
        output_path: Optional[str] = None
        dataset_path: Optional[str] = None
        include_failed: bool = True
        executor_ids: List[str] = Field(default_factory=list)
        query: RecordQuery = Field(default_factory=RecordQuery)

    def run(self) -> Dict[str, Any]:
        storage = self.config.query.storage()
        try:
            with storage.db.session() as session:
                run = session.query(RunORM).filter(RunORM.id == self.config.run_id).first()
                if not run:
                    raise ValueError(f"run not found: {self.config.run_id}")

            dataset_path = _extract_jsonl_dataset_path(run, override_path=self.config.dataset_path)
            base_cases = [
                TestCase.model_validate_json(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            by_id: Dict[str, TestCase] = {c.id: c for c in base_cases}

            out_path: Path
            if self.config.output_path:
                out_path = Path(self.config.output_path).expanduser()
                if not out_path.is_absolute():
                    out_path = Path(self.config.output_dir).expanduser().resolve() / out_path
            else:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                out_path = Path(self.config.output_dir).expanduser().resolve() / f"export-{self.config.run_id}-{ts}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            records = list(storage.fetch_run_records(self.config.run_id))
            if self.config.executor_ids:
                allow = set(self.config.executor_ids)
                records = [r for r in records if r.executor_id in allow]
            if not self.config.include_failed:
                records = [r for r in records if int(r.status) == 200]

            exported = 0
            missing_base = 0

            with out_path.open("w", encoding="utf-8") as handle:
                for rec in records:
                    exec_mode = (rec.extra or {}).get("execution_mode")
                    if not isinstance(exec_mode, dict):
                        exec_mode = {}
                    base_id = str(exec_mode.get("base_id") or rec.dataset_row_id)
                    base = by_id.get(base_id)
                    if not base:
                        missing_base += 1
                        continue

                    pass_index = exec_mode.get("pass_index")
                    suffix = f"{rec.executor_id}"
                    if isinstance(pass_index, int):
                        suffix = f"{suffix}__pass{pass_index}"
                    case_id = f"{base_id}__{suffix}"

                    out_case = base.model_copy(deep=True)
                    out_case.id = case_id
                    out_case.messages = list(out_case.messages) + [
                        Message(role="assistant", content=_assistant_text(rec))
                    ]

                    meta = dict(out_case.metadata or {})
                    meta.update(
                        {
                            "run_id": rec.run_id,
                            "executor_id": rec.executor_id,
                            "provider": rec.provider,
                            "model": rec.model,
                            "status": int(rec.status),
                            "request_params": rec.request_params,
                            "cost": {
                                "prompt_cost": float(rec.prompt_cost),
                                "completion_cost": float(rec.completion_cost),
                                "cache_cost": float(rec.cache_cost),
                                "total_cost": float(rec.total_cost),
                                "currency": rec.currency,
                            },
                            "execution_mode": exec_mode,
                        }
                    )
                    out_case.metadata = meta

                    handle.write(json.dumps(out_case.model_dump(), ensure_ascii=False) + "\n")
                    exported += 1

            return {
                "output_path": str(out_path),
                "dataset_path": str(dataset_path),
                "exported": exported,
                "missing_base_cases": missing_base,
                "filtered_records": len(records),
            }
        finally:
            storage.close()
