from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sqlalchemy.exc import OperationalError

from .db import Database, ExecutionORM, PricingHistoryORM, RunORM, dumps, loads

from llmperf.config.models import RunConfig
from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _normalize_run_status(
        raw_status: str | None,
        *,
        started_at: int = 0,
        scheduled_at: int = 0,
        completed_at: int = 0,
        error_message: str = "",
    ) -> str:
        known_statuses = {
            "scheduled",
            "pending",
            "running",
            "paused",
            "completed",
            "failed",
            "cancelled",
        }
        status = str(raw_status or "").strip().lower()
        if status not in known_statuses:
            status = "pending"

        if status != "pending":
            return status

        error_text = str(error_message or "").strip().lower()
        if completed_at:
            if "cancel" in error_text or "取消" in error_text:
                return "cancelled"
            if error_text:
                return "failed"
            return "completed"
        if started_at:
            return "running"
        if scheduled_at and scheduled_at > int(time.time()):
            return "scheduled"
        return "pending"

    def _serialize_run(self, row: RunORM) -> dict:
        started_at = int(getattr(row, "started_at", 0) or 0)
        scheduled_at = int(getattr(row, "scheduled_at", 0) or 0)
        completed_at = int(getattr(row, "completed_at", 0) or 0)
        error_message = getattr(row, "error_message", "") or ""
        return {
            "run_id": row.id,
            "task_type": row.task_type,
            "status": getattr(row, "normalized_status", None) or self._normalize_run_status(
                getattr(row, "status", "pending"),
                started_at=started_at,
                scheduled_at=scheduled_at,
                completed_at=completed_at,
                error_message=error_message,
            ),
            "info": row.info,
            "created_at": row.created_at,
            "started_at": started_at,
            "scheduled_at": scheduled_at,
            "completed_at": completed_at,
            "error_message": error_message,
            "config_path": row.config_path,
            "config_content": getattr(row, "config_content", ""),
            "total_cost": getattr(row, "total_cost", 0.0),
            "currency": getattr(row, "currency", "CNY"),
        }

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
                normalized_status = self._normalize_run_status(
                    status,
                    started_at=0,
                    scheduled_at=scheduled_at,
                    completed_at=0,
                    error_message="",
                )
                run = RunORM(
                    id=run_id,
                    task_type=task_type,
                    status=status,
                    normalized_status=normalized_status,
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
                run.normalized_status = self._normalize_run_status(
                    run.status,
                    started_at=int(getattr(run, "started_at", 0) or 0),
                    scheduled_at=int(getattr(run, "scheduled_at", 0) or 0),
                    completed_at=int(getattr(run, "completed_at", 0) or 0),
                    error_message=getattr(run, "error_message", "") or "",
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
        # Retry logic for SQLITE_BUSY errors
        max_retries = 5
        try:
            max_retries = int(os.getenv("DB_WRITE_RETRY", "5"))
        except (ValueError, TypeError):
            max_retries = 5
        retry_sleep = 0.2
        try:
            retry_sleep = float(os.getenv("DB_WRITE_RETRY_SLEEP", "0.2"))
        except (ValueError, TypeError):
            retry_sleep = 0.2

        with self._lock:
            attempt = 0
            while True:
                try:
                    with self.db.session() as session:
                        session.add(payload)
                        session.commit()
                    break
                except OperationalError as e:
                    msg = str(e).lower()
                    if "database is locked" in msg or "database table is locked" in msg:
                        attempt += 1
                        if attempt > max_retries:
                            logger.error(
                                "DB insert failed after %d retries: %s",
                                max_retries,
                                e,
                            )
                            raise
                        logger.debug(
                            "Database locked, retrying (%d/%d): %s",
                            attempt,
                            max_retries,
                            e,
                        )
                        time.sleep(max(0.0, retry_sleep) * attempt)
                        continue
                    raise

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

    def get_completed_dataset_row_ids(self, run_id: str, executor_id: str) -> set[str]:
        with self.db.session() as session:
            rows = (
                session.query(ExecutionORM.dataset_row_id)
                .filter(ExecutionORM.run_id == run_id)
                .filter(ExecutionORM.executor_id == executor_id)
                .distinct()
                .all()
            )
            return {str(row[0]) for row in rows if row and row[0]}

    def get_executor_completion_counts(self, run_id: str) -> dict[str, int]:
        with self.db.session() as session:
            from sqlalchemy import func

            rows = (
                session.query(
                    ExecutionORM.executor_id,
                    func.count(func.distinct(ExecutionORM.dataset_row_id)),
                )
                .filter(ExecutionORM.run_id == run_id)
                .group_by(ExecutionORM.executor_id)
                .all()
            )
            return {str(executor_id): int(count or 0) for executor_id, count in rows if executor_id}

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
        with self.db.session() as session:
            query = session.query(RunORM).order_by(RunORM.created_at.desc())
            if status:
                normalized_status = self._normalize_run_status(status)
                query = query.filter(RunORM.normalized_status == normalized_status)
            rows = query.offset(offset).limit(limit).all()
            return [self._serialize_run(row) for row in rows]

    def count_runs(self, status: str | None = None) -> int:
        from sqlalchemy import func
        with self.db.session() as session:
            query = session.query(func.count(RunORM.id))
            if status:
                normalized_status = self._normalize_run_status(status)
                query = query.filter(RunORM.normalized_status == normalized_status)
            return query.scalar() or 0

    def get_run_counts(self, run_id: str) -> dict:
        """Aggregate completion stats for a run using SQL, no record deserialization."""
        from sqlalchemy import func, case
        with self.db.session() as session:
            row = (
                session.query(
                    func.count(ExecutionORM.id).label("total"),
                    func.sum(case((ExecutionORM.status == 200, 1), else_=0)).label("success_count"),
                    func.sum(ExecutionORM.total_cost).label("total_cost"),
                    func.min(ExecutionORM.currency).label("currency"),
                )
                .filter(ExecutionORM.run_id == run_id)
                .one()
            )
            total = int(row.total or 0)
            success = int(row.success_count or 0)
            return {
                "total": total,
                "success_count": success,
                "error_count": total - success,
                "total_cost": float(row.total_cost or 0.0),
                "currency": row.currency or "CNY",
            }

    def get_run_counts_by_executor(self, run_id: str) -> list[dict]:
        """Per-executor aggregate stats using SQL GROUP BY."""
        from sqlalchemy import func, case
        with self.db.session() as session:
            rows = (
                session.query(
                    ExecutionORM.executor_id,
                    func.count(ExecutionORM.id).label("total"),
                    func.sum(case((ExecutionORM.status == 200, 1), else_=0)).label("success_count"),
                    func.sum(ExecutionORM.total_cost).label("total_cost"),
                    func.min(ExecutionORM.currency).label("currency"),
                )
                .filter(ExecutionORM.run_id == run_id)
                .group_by(ExecutionORM.executor_id)
                .all()
            )
            result = []
            for r in rows:
                total = int(r.total or 0)
                success = int(r.success_count or 0)
                result.append({
                    "executor_id": r.executor_id or "",
                    "total": total,
                    "success_count": success,
                    "error_count": total - success,
                    "total_cost": float(r.total_cost or 0.0),
                    "currency": r.currency or "CNY",
                })
            return result

    def fetch_run_errors(
        self,
        run_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch failed execution records (status != 200) with minimal columns."""
        with self.db.session() as session:
            rows = (
                session.query(
                    ExecutionORM.id,
                    ExecutionORM.executor_id,
                    ExecutionORM.status,
                    ExecutionORM.info,
                    ExecutionORM.created_at,
                    ExecutionORM.model,
                    ExecutionORM.provider,
                    ExecutionORM.dataset_row_id,
                )
                .filter(ExecutionORM.run_id == run_id)
                .filter(ExecutionORM.status != 200)
                .order_by(ExecutionORM.id.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            result = []
            for r in rows:
                info = {}
                if r.info:
                    try:
                        info = json.loads(r.info)
                    except Exception:
                        info = {"raw": r.info}
                result.append({
                    "id": r.id,
                    "executor_id": r.executor_id or "",
                    "status": r.status,
                    "error_type": info.get("error_type") or info.get("type", ""),
                    "error_message": info.get("error") or info.get("message") or info.get("msg", ""),
                    "model": r.model or "",
                    "provider": r.provider or "",
                    "dataset_row_id": r.dataset_row_id or "",
                    "created_at": r.created_at,
                })
            return result

    def count_run_errors(self, run_id: str) -> int:
        """Count failed execution records for a run."""
        from sqlalchemy import func
        with self.db.session() as session:
            return (
                session.query(func.count(ExecutionORM.id))
                .filter(ExecutionORM.run_id == run_id)
                .filter(ExecutionORM.status != 200)
                .scalar() or 0
            )

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a single run snapshot."""
        with self.db.session() as session:
            row = session.query(RunORM).filter(RunORM.id == run_id).first()
            if not row:
                return None
            return self._serialize_run(row)

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
            run.normalized_status = self._normalize_run_status(
                status,
                started_at=int(getattr(run, "started_at", 0) or 0),
                scheduled_at=int(getattr(run, "scheduled_at", 0) or 0),
                completed_at=int(getattr(run, "completed_at", 0) or 0),
                error_message=getattr(run, "error_message", "") or "",
            )
            session.commit()

    def update_run_info(self, run_id: str, info: str) -> bool:
        with self.db.session() as session:
            run = session.query(RunORM).filter(RunORM.id == run_id).first()
            if not run:
                return False
            run.info = info
            session.commit()
            return True

    def update_run_config_snapshot(
        self,
        run_id: str,
        *,
        cfg: RunConfig,
        config_content: str,
        config_path: str = "",
    ) -> bool:
        with self.db.session() as session:
            run = session.query(RunORM).filter(RunORM.id == run_id).first()
            if not run:
                return False
            run.info = cfg.info
            run.config_content = config_content
            run.config_path = config_path
            session.commit()
            return True

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
