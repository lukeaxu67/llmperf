from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from ..analysis_registry import register_analysis
from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery
from ..utils import _params_key

from llmperf.records.db import ExecutionORM, RunORM
from llmperf.records.model import RunRecord


def _run_time_range(storage, run_id: str) -> tuple[int, int]:
    with storage.db.session() as session:
        run = session.query(RunORM).filter(RunORM.id == run_id).first()
        if not run:
            raise ValueError(f"run not found: {run_id}")
        start_ts = int(run.created_at or 0)
        last = (
            session.query(ExecutionORM.created_at)
            .filter(ExecutionORM.run_id == run_id)
            .order_by(ExecutionORM.created_at.desc())
            .first()
        )
        end_ts = int(last[0]) if last else start_ts
        return start_ts, end_ts


def _group_key(record: RunRecord, group_by_request_params: bool) -> Tuple[str, str, str]:
    params = _params_key(record.request_params) if group_by_request_params else ""
    return (record.provider, record.model, params)


@register_analysis("history")
class HistoryAnalysis(BaseAnalysis["HistoryAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        query: RecordQuery = Field(default_factory=RecordQuery)
        group_by_request_params: bool = True

    def run(self) -> Dict[str, Any]:
        storage = self.config.query.storage()
        try:
            with storage.db.session() as session:
                runs = session.query(RunORM).order_by(RunORM.created_at.desc()).all()
                if self.config.query.run_ids:
                    allow = set(self.config.query.run_ids)
                    runs = [r for r in runs if r.id in allow]
                if self.config.query.start_ts is not None:
                    runs = [r for r in runs if int(r.created_at) >= int(self.config.query.start_ts)]
                if self.config.query.end_ts is not None:
                    runs = [r for r in runs if int(r.created_at) <= int(self.config.query.end_ts)]

            items: List[Dict[str, Any]] = []
            for run in runs:
                records = list(storage.fetch_run_records(run.id))
                if self.config.query.provider:
                    records = [r for r in records if r.provider == self.config.query.provider]
                if self.config.query.model:
                    records = [r for r in records if r.model == self.config.query.model]

                grouped: Dict[Tuple[str, str, str], List[RunRecord]] = defaultdict(list)
                for rec in records:
                    grouped[_group_key(rec, self.config.group_by_request_params)].append(rec)

                start_ts, end_ts = _run_time_range(storage, run.id)

                models: List[Dict[str, Any]] = []
                for (provider, model, params_json), recs in grouped.items():
                    count = len(recs)
                    success = len([r for r in recs if int(r.status) == 200])
                    failure = count - success
                    currency = recs[0].currency if recs else ""
                    models.append(
                        {
                            "provider": provider,
                            "model": model,
                            "request_params": params_json,
                            "count": count,
                            "success_count": success,
                            "failure_count": failure,
                            "prompt_cost": float(sum(r.prompt_cost for r in recs)),
                            "completion_cost": float(sum(r.completion_cost for r in recs)),
                            "cache_cost": float(sum(r.cache_cost for r in recs)),
                            "total_cost": float(sum(r.total_cost for r in recs)),
                            "currency": currency,
                        }
                    )

                items.append(
                    {
                        "run_id": run.id,
                        "info": run.info,
                        "created_at": int(run.created_at or 0),
                        "start_time": datetime.fromtimestamp(start_ts).isoformat(sep=" ", timespec="seconds"),
                        "end_time": datetime.fromtimestamp(end_ts).isoformat(sep=" ", timespec="seconds"),
                        "duration_seconds": max(end_ts - start_ts, 0),
                        "models": sorted(models, key=lambda x: (x["provider"], x["model"], x["request_params"])),
                    }
                )

            return {"count": len(items), "items": items}
        finally:
            storage.close()

