from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..config.models import ExecutorConfig, MultiprocessConfig, PricingEntry, RunConfig
from ..datasets.types import TestCase as DatasetRow
from ..datasets.dataset_iterators import DatasetIterator
from ..datasets.mutation_chain import MutationChain
from ..pricing.catalog import PriceCatalog
from ..records.storage import Storage
from .base import create_executor

logger = logging.getLogger(__name__)


def _run_executor_in_subprocess(
    cfg_data: dict,
    run_id: str,
    rows: List[DatasetRow],
    iterator_steps: List[str],
    max_total_seconds: Optional[float],
    max_rounds: Optional[int],
    pricing_data: List[dict],
    db_path: str,
    deadline_ts: Optional[float],
    max_rows: Optional[int],
    exec_meta: Optional[Dict[str, object]],
) -> None:
    """
    Subprocess entrypoint for running a single executor over the dataset.
    Reconstructs configuration and uses the regular executor + storage pipeline.
    """
    config = ExecutorConfig.model_validate(cfg_data)
    pricing_entries = [PricingEntry.model_validate(p) for p in pricing_data]
    price_catalog = PriceCatalog(pricing_entries)
    storage = Storage(db_path)
    executor = create_executor(config)
    logger.info("[child %s] starting executor %s", run_id, config.id)
    mutation_chain = MutationChain(iterator_steps or ["identity"])
    iterator = DatasetIterator(
        rows,
        mutation_chain=mutation_chain,
    )
    executor.run(
        run_id,
        iterator,
        storage,
        price_catalog,
        deadline_ts=deadline_ts,
        max_rows=max_rows,
        max_total_seconds=max_total_seconds,
        max_rounds=max_rounds,
        rows_per_round=len(rows),
        exec_meta=exec_meta,
    )
    logger.info("[child %s] finished executor %s", run_id, config.id)


@dataclass
class ProcessManager:
    """Manage multi-process execution of executors for a single run."""

    run_id: str
    config: RunConfig
    rows: Sequence[DatasetRow]
    iterator_steps: List[str]
    max_total_seconds: Optional[float] = None
    max_rounds: Optional[int] = None
    db_path: str = "data.db"
    deadline_ts: Optional[float] = None
    max_rows: Optional[int] = None
    exec_meta: Optional[Dict[str, object]] = None

    def _mp_config(self) -> MultiprocessConfig:
        return self.config.multiprocess or MultiprocessConfig(per_executor=True)

    def run_all(self) -> None:
        """
        Execute all configured executors, respecting their dependency DAG,
        using separate worker processes when multiprocess is enabled.

        Fairness:
        - Ready executors are launched in batches up to max_workers.
        - Executors in the same batch run concurrently in separate processes.
        - Batches are computed repeatedly until all executors complete.
        """
        mp_cfg = self._mp_config()
        if not mp_cfg.per_executor:
            raise ValueError("ProcessManager only supports per-executor multiprocess mode.")

        pricing_data = [entry.model_dump() for entry in self.config.pricing]
        base_rows = list(self.rows)

        pending: Dict[str, ExecutorConfig] = {cfg.id: cfg for cfg in self.config.executors}
        completed: set[str] = set()

        max_workers = mp_cfg.max_workers or max(1, mp.cpu_count())
        logger.info(
            "Starting multi-process execution for run %s with max_workers=%d",
            self.run_id,
            max_workers,
        )

        processes: List[mp.Process] = []
        try:
            while pending:
                ready_ids = [
                    exec_id
                    for exec_id, cfg in pending.items()
                    if all(dep in completed for dep in cfg.after)
                ]
                if not ready_ids:
                    raise RuntimeError(
                        "Executor dependency cycle detected in multi-process runner"
                    )

                # Simple round-robin fairness: run ready executors in batches.
                batch_ids = ready_ids[:max_workers]
                processes = []
                for exec_id in batch_ids:
                    cfg = pending.pop(exec_id)
                    rows_copy = list(base_rows)
                    p = mp.Process(
                        target=_run_executor_in_subprocess,
                        args=(
                            cfg.model_dump(),
                            self.run_id,
                            rows_copy,
                            list(self.iterator_steps),
                            self.max_total_seconds,
                            self.max_rounds,
                            pricing_data,
                            self.db_path,
                            self.deadline_ts,
                            self.max_rows,
                            self.exec_meta,
                        ),
                        name=f"executor-{exec_id}",
                    )
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()
                    if p.exitcode not in (0, None):
                        logger.error(
                            "Executor process %s exited with code %s",
                            p.name,
                            p.exitcode,
                        )

                completed.update(batch_ids)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received; terminating executor processes...")
            for p in processes:
                try:
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    continue
            for p in processes:
                try:
                    p.join(timeout=2)
                except Exception:
                    continue
            raise
