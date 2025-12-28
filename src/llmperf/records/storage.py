from __future__ import annotations

import threading
import time
from typing import Iterable, Optional

from .db import Database, ExecutionORM, RunORM, dumps, loads

from llmperf.config.models import RunConfig
from llmperf.records.model import RunRecord


class Storage:
    def __init__(self, db_path: str = "data.db"):
        self.db = Database(db_path)
        self._lock = threading.Lock()

    def register_run(
        self,
        run_id: str,
        cfg: RunConfig,
        *,
        config_path: str,
        config_content: str,
        pricing_path: str | None,
        pricing_content: str,
    ) -> None:
        with self.db.session() as session:
            session.merge(
                RunORM(
                    id=run_id,
                    task_type="run",
                    info=cfg.info,
                    created_at=int(time.time()),
                    config_path=config_path,
                    config_content=config_content,
                    pricing_path=pricing_path or "",
                    pricing_content=pricing_content,
                )
            )
            session.commit()

    def insert_record(self, record: RunRecord) -> None:
        payload = ExecutionORM(
            run_id=record.run_id,
            executor_id=record.executor_id,
            dataset_row_id=record.dataset_row_id,
            provider=record.provider,
            model=record.model,
            status=record.status,
            info=record.info,
            qtokens=record.qtokens,
            atokens=record.atokens,
            ctokens=record.ctokens,
            prompt_cost=record.prompt_cost,
            completion_cost=record.completion_cost,
            cache_cost=record.cache_cost,
            total_cost=record.total_cost,
            currency=record.currency,
            usage_json=dumps(record.usage),
            request_params_json=dumps(record.request_params),
            action_times=dumps(record.action_times),
            reasoning_times=dumps(record.reasoning_times),
            content_times=dumps(record.content_times),
            reasoning_json=dumps(record.reasoning),
            content_json=dumps(record.content),
            extra_json=dumps(record.extra),
            created_at=int(time.time()),
        )
        with self._lock:
            with self.db.session() as session:
                session.add(payload)
                session.commit()

    def fetch_run_records(self, run_id: str) -> Iterable[RunRecord]:
        with self.db.session() as session:
            rows = (
                session.query(ExecutionORM).filter(ExecutionORM.run_id == run_id).order_by(ExecutionORM.id.asc()).all()
            )
            for row in rows:
                yield RunRecord(
                    run_id=row.run_id,
                    executor_id=row.executor_id,
                    dataset_row_id=row.dataset_row_id,
                    provider=getattr(row, "provider", ""),
                    model=row.model,
                    status=row.status,
                    info=row.info,
                    qtokens=row.qtokens,
                    atokens=row.atokens,
                    ctokens=row.ctokens,
                    prompt_cost=row.prompt_cost,
                    completion_cost=row.completion_cost,
                    cache_cost=getattr(row, "cache_cost", 0.0),
                    total_cost=row.total_cost,
                    currency=row.currency,
                    usage=loads(row.usage_json, {}),
                    request_params=loads(getattr(row, "request_params_json", None), {}),
                    action_times=loads(row.action_times, []),
                    reasoning_times=loads(row.reasoning_times, []),
                    content_times=loads(row.content_times, []),
                    reasoning=loads(row.reasoning_json, []),
                    content=loads(row.content_json, []),
                    extra=loads(row.extra_json, {}),
                )

    def query_records(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> Iterable[RunRecord]:
        with self.db.session() as session:
            query = session.query(ExecutionORM)
            if provider:
                query = query.filter(ExecutionORM.provider == provider)
            if model:
                query = query.filter(ExecutionORM.model == model)
            if start_ts:
                query = query.filter(ExecutionORM.created_at >= start_ts)
            if end_ts:
                query = query.filter(ExecutionORM.created_at <= end_ts)
            query = query.order_by(ExecutionORM.created_at.desc())
            for row in query.all():
                yield RunRecord(
                    run_id=row.run_id,
                    executor_id=row.executor_id,
                    dataset_row_id=row.dataset_row_id,
                    provider=getattr(row, "provider", ""),
                    model=row.model,
                    status=row.status,
                    info=row.info,
                    qtokens=row.qtokens,
                    atokens=row.atokens,
                    ctokens=row.ctokens,
                    prompt_cost=row.prompt_cost,
                    completion_cost=row.completion_cost,
                    cache_cost=getattr(row, "cache_cost", 0.0),
                    total_cost=row.total_cost,
                    currency=row.currency,
                    usage=loads(row.usage_json, {}),
                    request_params=loads(getattr(row, "request_params_json", None), {}),
                    action_times=loads(row.action_times, []),
                    reasoning_times=loads(row.reasoning_times, []),
                    content_times=loads(row.content_times, []),
                    reasoning=loads(row.reasoning_json, []),
                    content=loads(row.content_json, []),
                    extra=loads(row.extra_json, {}),
                )
