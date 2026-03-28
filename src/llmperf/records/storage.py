from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .db import Database, ExecutionORM, PricingHistoryORM, RunORM, dumps, loads

from llmperf.config.models import RunConfig
from llmperf.records.model import RunRecord


@dataclass
class PricingRecord:
    """Pricing record for a provider/model at a specific time."""
    id: int
    provider: str
    model: str
    input_price: float  # CNY per million tokens
    output_price: float  # CNY per million tokens
    cache_read_price: float = 0.0
    cache_write_price: float = 0.0
    effective_at: int = 0
    created_at: int = 0
    note: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "cache_read_price": self.cache_read_price,
            "cache_write_price": self.cache_write_price,
            "effective_at": self.effective_at,
            "created_at": self.created_at,
            "note": self.note,
        }


class Storage:
    def __init__(self, db_path: str = "data.db"):
        self.db = Database(db_path)
        self._lock = threading.Lock()

    def close(self) -> None:
        self.db.close()

    def register_run(
        self,
        run_id: str,
        cfg: RunConfig,
        *,
        config_path: str,
        config_content: str,
        task_type: str = "benchmark",
        status: str = "pending",
        scheduled_at: int = 0,
        pricing_path: str | None = None,
        pricing_content: str = "",
    ) -> None:
        # pricing_path and pricing_content are kept for backward compatibility
        # but are no longer stored in the database
        with self.db.session() as session:
            run = session.query(RunORM).filter(RunORM.id == run_id).first()
            now = int(time.time())
            if not run:
                run = RunORM(
                    id=run_id,
                    task_type=task_type,
                    status=status,
                    info=cfg.info,
                    created_at=now,
                    scheduled_at=scheduled_at,
                    config_path=config_path,
                    config_content=config_content,
                )
                session.add(run)
            else:
                run.task_type = task_type or run.task_type
                run.status = status or run.status
                run.info = cfg.info
                run.scheduled_at = scheduled_at
                run.config_path = config_path
                run.config_content = config_content
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
            input_price_snapshot=getattr(record, "input_price_snapshot", 0.0),
            output_price_snapshot=getattr(record, "output_price_snapshot", 0.0),
            cache_price_snapshot=getattr(record, "cache_price_snapshot", 0.0),
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
                    input_price_snapshot=getattr(row, "input_price_snapshot", 0.0),
                    output_price_snapshot=getattr(row, "output_price_snapshot", 0.0),
                    cache_price_snapshot=getattr(row, "cache_price_snapshot", 0.0),
                    usage=loads(row.usage_json, {}),
                    request_params=loads(getattr(row, "request_params_json", None), {}),
                    action_times=loads(row.action_times, []),
                    reasoning_times=loads(row.reasoning_times, []),
                    content_times=loads(row.content_times, []),
                    reasoning=loads(row.reasoning_json, []),
                    content=loads(row.content_json, []),
                    extra=loads(row.extra_json, {}),
                    created_at=row.created_at,
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
                    input_price_snapshot=getattr(row, "input_price_snapshot", 0.0),
                    output_price_snapshot=getattr(row, "output_price_snapshot", 0.0),
                    cache_price_snapshot=getattr(row, "cache_price_snapshot", 0.0),
                    usage=loads(row.usage_json, {}),
                    request_params=loads(getattr(row, "request_params_json", None), {}),
                    action_times=loads(row.action_times, []),
                    reasoning_times=loads(row.reasoning_times, []),
                    content_times=loads(row.content_times, []),
                    reasoning=loads(row.reasoning_json, []),
                    content=loads(row.content_json, []),
                    extra=loads(row.extra_json, {}),
                    created_at=row.created_at,
                )

    def list_runs(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> List[dict]:
        """List all runs from the database.

        Args:
            limit: Maximum number of runs to return.
            offset: Pagination offset.
            status: Optional status filter.

        Returns:
            List of run dictionaries.
        """
        with self.db.session() as session:
            query = session.query(RunORM)
            if status:
                query = query.filter(RunORM.status == status)
            rows = query.order_by(RunORM.created_at.desc()).offset(offset).limit(limit).all()
            return [
                {
                    "run_id": row.id,
                    "task_type": row.task_type,
                    "status": getattr(row, "status", "pending"),
                    "info": row.info,
                    "created_at": row.created_at,
                    "started_at": getattr(row, "started_at", 0),
                    "scheduled_at": getattr(row, "scheduled_at", 0),
                    "completed_at": getattr(row, "completed_at", 0),
                    "error_message": getattr(row, "error_message", ""),
                    "config_path": row.config_path,
                    "config_content": getattr(row, "config_content", ""),
                    "total_cost": getattr(row, "total_cost", 0.0),
                    "currency": getattr(row, "currency", "CNY"),
                }
                for row in rows
            ]

    def count_runs(self, status: str | None = None) -> int:
        with self.db.session() as session:
            query = session.query(RunORM)
            if status:
                query = query.filter(RunORM.status == status)
            return query.count()

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a single run snapshot."""
        with self.db.session() as session:
            row = session.query(RunORM).filter(RunORM.id == run_id).first()
            if not row:
                return None
            return {
                "run_id": row.id,
                "task_type": row.task_type,
                "status": getattr(row, "status", "pending"),
                "info": row.info,
                "created_at": row.created_at,
                "started_at": getattr(row, "started_at", 0),
                "scheduled_at": getattr(row, "scheduled_at", 0),
                "completed_at": getattr(row, "completed_at", 0),
                "error_message": getattr(row, "error_message", ""),
                "config_path": row.config_path,
                "config_content": getattr(row, "config_content", ""),
                "total_cost": getattr(row, "total_cost", 0.0),
                "currency": getattr(row, "currency", "CNY"),
            }

    def update_run_cost(self, run_id: str, total_cost: float, currency: str = "CNY") -> None:
        """Update the total cost for a run."""
        with self.db.session() as session:
            run = session.query(RunORM).filter(RunORM.id == run_id).first()
            if run:
                run.total_cost = total_cost
                run.currency = currency
                session.commit()

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        started_at: Optional[int] = None,
        completed_at: Optional[int] = None,
        scheduled_at: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with self.db.session() as session:
            run = session.query(RunORM).filter(RunORM.id == run_id).first()
            if not run:
                return
            run.status = status
            if started_at is not None:
                run.started_at = started_at
            if completed_at is not None:
                run.completed_at = completed_at
            if scheduled_at is not None:
                run.scheduled_at = scheduled_at
            if error_message is not None:
                run.error_message = error_message
            session.commit()

    def mark_run_completed(self, run_id: str, completed_at: Optional[int] = None) -> None:
        """Record run completion timestamp."""
        with self.db.session() as session:
            run = session.query(RunORM).filter(RunORM.id == run_id).first()
            if run:
                run.completed_at = completed_at or int(time.time())
                session.commit()

    def delete_run(self, run_id: str) -> bool:
        with self.db.session() as session:
            run = session.query(RunORM).filter(RunORM.id == run_id).first()
            if not run:
                return False
            session.query(ExecutionORM).filter(ExecutionORM.run_id == run_id).delete()
            session.delete(run)
            session.commit()
            return True

    def get_total_cost(self) -> dict:
        """Get total cost across all runs."""
        with self.db.session() as session:
            # Sum from runs table
            runs = session.query(RunORM).all()
            total_cost = sum(getattr(r, "total_cost", 0.0) for r in runs)
            run_count = session.query(RunORM).count()

            # Also calculate from executions for accuracy
            from sqlalchemy import func
            exec_total = session.query(func.sum(ExecutionORM.total_cost)).scalar() or 0.0

            return {
                "total_cost": max(total_cost, exec_total),
                "run_count": run_count,
                "currency": "CNY",
            }

    # ==================== Pricing History Methods ====================

    @staticmethod
    def _normalize_pricing_provider(provider: str) -> str:
        return str(provider or "").strip().lower()

    @staticmethod
    def _normalize_pricing_model(model: str) -> str:
        return str(model or "").strip()

    def add_pricing(
        self,
        provider: str,
        model: str,
        input_price: float,
        output_price: float,
        cache_read_price: float = 0.0,
        cache_write_price: float = 0.0,
        effective_at: Optional[int] = None,
        note: str = "",
    ) -> PricingRecord:
        """Add a new pricing record."""
        provider = self._normalize_pricing_provider(provider)
        model = self._normalize_pricing_model(model)
        if effective_at is None:
            effective_at = int(time.time())

        with self.db.session() as session:
            orm = PricingHistoryORM(
                provider=provider,
                model=model,
                input_price=input_price,
                output_price=output_price,
                cache_read_price=cache_read_price,
                cache_write_price=cache_write_price,
                effective_at=effective_at,
                created_at=int(time.time()),
                note=note,
            )
            session.add(orm)
            session.commit()
            return PricingRecord(
                id=orm.id,
                provider=orm.provider,
                model=orm.model,
                input_price=orm.input_price,
                output_price=orm.output_price,
                cache_read_price=orm.cache_read_price,
                cache_write_price=orm.cache_write_price,
                effective_at=orm.effective_at,
                created_at=orm.created_at,
                note=orm.note,
            )

    def list_pricing(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100,
    ) -> List[PricingRecord]:
        """List pricing records, optionally filtered by provider/model."""
        with self.db.session() as session:
            from sqlalchemy import func

            query = session.query(PricingHistoryORM)
            if provider:
                normalized_provider = self._normalize_pricing_provider(provider)
                query = query.filter(func.lower(PricingHistoryORM.provider) == normalized_provider)
            if model:
                normalized_model = self._normalize_pricing_model(model).lower()
                query = query.filter(func.lower(PricingHistoryORM.model) == normalized_model)
            query = query.order_by(PricingHistoryORM.effective_at.desc()).limit(limit)

            return [
                PricingRecord(
                    id=row.id,
                    provider=row.provider,
                    model=row.model,
                    input_price=row.input_price,
                    output_price=row.output_price,
                    cache_read_price=row.cache_read_price,
                    cache_write_price=row.cache_write_price,
                    effective_at=row.effective_at,
                    created_at=row.created_at,
                    note=row.note,
                )
                for row in query.all()
            ]

    def get_pricing(self, pricing_id: int) -> Optional[PricingRecord]:
        """Get a specific pricing record."""
        with self.db.session() as session:
            row = session.query(PricingHistoryORM).filter(PricingHistoryORM.id == pricing_id).first()
            if not row:
                return None
            return PricingRecord(
                id=row.id,
                provider=row.provider,
                model=row.model,
                input_price=row.input_price,
                output_price=row.output_price,
                cache_read_price=row.cache_read_price,
                cache_write_price=row.cache_write_price,
                effective_at=row.effective_at,
                created_at=row.created_at,
                note=row.note,
            )

    def delete_pricing(self, pricing_id: int) -> bool:
        """Delete a pricing record."""
        with self.db.session() as session:
            row = session.query(PricingHistoryORM).filter(PricingHistoryORM.id == pricing_id).first()
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True

    def get_pricing_at_time(
        self,
        provider: str,
        model: str,
        timestamp: int,
    ) -> Optional[PricingRecord]:
        """Get the effective pricing for a provider/model at a specific time."""
        with self.db.session() as session:
            from sqlalchemy import func

            normalized_provider = self._normalize_pricing_provider(provider)
            normalized_model = self._normalize_pricing_model(model).lower()
            row = (
                session.query(PricingHistoryORM)
                .filter(func.lower(PricingHistoryORM.provider) == normalized_provider)
                .filter(func.lower(PricingHistoryORM.model) == normalized_model)
                .filter(PricingHistoryORM.effective_at <= timestamp)
                .order_by(PricingHistoryORM.effective_at.desc())
                .first()
            )
            if not row:
                return None
            return PricingRecord(
                id=row.id,
                provider=row.provider,
                model=row.model,
                input_price=row.input_price,
                output_price=row.output_price,
                cache_read_price=row.cache_read_price,
                cache_write_price=row.cache_write_price,
                effective_at=row.effective_at,
                created_at=row.created_at,
                note=row.note,
            )

    def get_providers_models(self) -> List[dict]:
        """Get list of unique provider/model combinations with latest pricing."""
        with self.db.session() as session:
            from sqlalchemy import func

            # Get latest pricing for each provider/model
            subquery = (
                session.query(
                    PricingHistoryORM.provider,
                    PricingHistoryORM.model,
                    func.max(PricingHistoryORM.effective_at).label("max_effective"),
                )
                .group_by(PricingHistoryORM.provider, PricingHistoryORM.model)
                .subquery()
            )

            results = (
                session.query(PricingHistoryORM)
                .join(
                    subquery,
                    (PricingHistoryORM.provider == subquery.c.provider)
                    & (PricingHistoryORM.model == subquery.c.model)
                    & (PricingHistoryORM.effective_at == subquery.c.max_effective),
                )
                .all()
            )

            return [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "input_price": r.input_price,
                    "output_price": r.output_price,
                    "effective_at": r.effective_at,
                }
                for r in results
            ]

    def get_pricing_history_by_provider(
        self,
        provider: Optional[str] = None,
        days: int = 30,
    ) -> List[dict]:
        """Get pricing history for charts, optionally filtered by provider."""
        cutoff = int(time.time()) - (days * 24 * 3600)

        with self.db.session() as session:
            query = session.query(PricingHistoryORM).filter(
                PricingHistoryORM.effective_at >= cutoff
            )
            if provider:
                query = query.filter(PricingHistoryORM.provider == provider)
            query = query.order_by(PricingHistoryORM.effective_at.asc())

            return [
                {
                    "id": r.id,
                    "provider": r.provider,
                    "model": r.model,
                    "input_price": r.input_price,
                    "output_price": r.output_price,
                    "effective_at": r.effective_at,
                    "note": r.note,
                }
                for r in query.all()
            ]

    def get_cost_summary_by_provider(self, days: int = 30) -> List[dict]:
        """Get cost summary grouped by provider and model."""
        cutoff = int(time.time()) - (days * 24 * 3600)

        with self.db.session() as session:
            from sqlalchemy import func

            results = (
                session.query(
                    ExecutionORM.provider,
                    ExecutionORM.model,
                    func.count(ExecutionORM.id).label("request_count"),
                    func.sum(ExecutionORM.qtokens).label("total_input_tokens"),
                    func.sum(ExecutionORM.atokens).label("total_output_tokens"),
                    func.sum(ExecutionORM.total_cost).label("total_cost"),
                )
                .filter(ExecutionORM.created_at >= cutoff)
                .filter(ExecutionORM.status == 200)
                .group_by(ExecutionORM.provider, ExecutionORM.model)
                .all()
            )

            return [
                {
                    "provider": r.provider or "unknown",
                    "model": r.model,
                    "request_count": r.request_count,
                    "total_input_tokens": r.total_input_tokens or 0,
                    "total_output_tokens": r.total_output_tokens or 0,
                    "total_cost": r.total_cost or 0.0,
                }
                for r in results
            ]
