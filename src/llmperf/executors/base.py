from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Dict, Iterable, List, Optional

from ..config.models import ExecutorConfig
from ..datasets.types import TestCase as DatasetRow
from ..pricing.catalog import PriceCatalog
from ..records.model import RunRecord
from ..records.storage import Storage
from ..utils.rate_limiter import RateLimiter
from ..utils.registry import DoubleRegistry, registration_decorator

logger = logging.getLogger(__name__)


class BaseExecutor:
    def __init__(self, config: ExecutorConfig):
        self.config = config

    def prepare(self, *args, **kwargs) -> None:
        """Hook for one-time setup per run."""

    def build_messages(self, row: DatasetRow) -> List[dict]:
        return [message.model_dump() for message in row.messages]

    def process_row(self, run_id: str, row: DatasetRow, price: PriceCatalog | None = None) -> RunRecord:
        raise NotImplementedError

    def run(
        self,
        run_id: str,
        dataset: Iterable[DatasetRow],
        storage: Storage,
        price: PriceCatalog | None = None,
        deadline_ts: float | None = None,
        max_rows: int | None = None,
        max_total_seconds: float | None = None,
        max_rounds: int | None = None,
        rows_per_round: int | None = None,
        exec_meta: Dict[str, object] | None = None,
    ) -> List[RunRecord]:
        rate_cfg = self.config.rate
        limiter = RateLimiter(
            qps=(rate_cfg.qps if rate_cfg else None),
            interval_seconds=(rate_cfg.interval_seconds if rate_cfg else None),
        )

        start_ts = time.time()

        if max_total_seconds is not None:
            seconds_deadline = start_ts + float(max_total_seconds)
            if deadline_ts is None:
                deadline_ts = seconds_deadline
            else:
                deadline_ts = min(deadline_ts, seconds_deadline)

        if max_rounds is not None:
            if rows_per_round is None:
                raise ValueError("rows_per_round is required when max_rounds is set")
            rounds_max_rows = int(rows_per_round) * int(max_rounds)
            if max_rows is None:
                max_rows = rounds_max_rows
            else:
                max_rows = min(int(max_rows), rounds_max_rows)

        total_rows: Optional[int]
        if hasattr(dataset, "__len__"):
            try:
                total_rows = len(dataset)  # type: ignore[arg-type]
            except TypeError:
                total_rows = None
        else:
            total_rows = None
        if total_rows is None and max_rows is not None:
            total_rows = int(max_rows)
        if total_rows is not None:
            logger.info("Executor %s starting with %d rows", self.config.id, total_rows)
        else:
            logger.info("Executor %s starting", self.config.id)

        records: List[RunRecord] = []
        max_workers = max(1, self.config.concurrency)
        meta_base = dict(exec_meta or {})

        next_progress = 0.1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_meta: Dict[concurrent.futures.Future[RunRecord], Dict[str, object]] = {}

            dispatched = 0
            completed = 0
            row_index = 0
            iterator = iter(dataset)

            def _can_dispatch_more() -> bool:
                if max_rows is not None and dispatched >= max_rows:
                    return False
                if deadline_ts is not None and time.time() >= deadline_ts:
                    return False
                return True

            def _submit_next() -> bool:
                nonlocal dispatched, row_index
                if not _can_dispatch_more():
                    return False
                try:
                    row = next(iterator)
                except StopIteration:
                    return False
                limiter.acquire()
                if not _can_dispatch_more():
                    return False
                row_meta = dict(meta_base)
                row_meta["row_index"] = row_index
                row_index += 1
                metadata = row.metadata or {}
                if "pass_index" in metadata:
                    row_meta["pass_index"] = metadata["pass_index"]
                if "base_id" in metadata:
                    row_meta["base_id"] = metadata["base_id"]
                fut = executor.submit(self.process_row, run_id, row, price)
                future_meta[fut] = row_meta
                dispatched += 1
                return True

            for _ in range(max_workers):
                if not _submit_next():
                    break

            while future_meta:
                done, _pending = concurrent.futures.wait(
                    set(future_meta.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    meta = future_meta.pop(fut, None) or {}
                    record = fut.result()
                    if meta:
                        extra = dict(record.extra or {})
                        exec_info = extra.get("execution_mode")
                        if not isinstance(exec_info, dict):
                            exec_info = {}
                        exec_info.update(meta)
                        extra["execution_mode"] = exec_info
                        record.extra = extra
                    records.append(record)
                    storage.insert_record(record)
                    completed += 1

                    if total_rows and total_rows > 0:
                        fraction = completed / total_rows
                        if fraction >= next_progress:
                            elapsed = time.time() - start_ts
                            remaining = total_rows - completed
                            eta = (elapsed / completed * remaining) if completed else 0.0
                            logger.info(
                                "Executor %s progress: %d/%d (%.0f%%), elapsed=%.1fs, eta=%.1fs",
                                self.config.id,
                                completed,
                                total_rows,
                                fraction * 100.0,
                                elapsed,
                                eta,
                            )
                            next_progress += 0.1

                    if _can_dispatch_more():
                        _submit_next()
        return records


executor_registry: DoubleRegistry[BaseExecutor] = DoubleRegistry()


def register_executor(type_name: str, impl: str = "default", **metadata):
    return registration_decorator(executor_registry, type_name, impl, **metadata)


def create_executor(config: ExecutorConfig) -> BaseExecutor:
    executor_cls = executor_registry.get(config.type, config.impl)
    return executor_cls(config)
