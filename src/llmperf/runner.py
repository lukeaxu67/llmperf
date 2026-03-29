from __future__ import annotations

import logging
import threading
import uuid
from typing import Dict, Optional

from llmperf.config.loader import load_config
from llmperf.config.models import RunConfig
from llmperf.config.runtime import load_runtime_config
from llmperf.datasets.types import TestCase as DatasetRow
from llmperf.datasets.dataset_source_registry import load_dataset
from llmperf.executors.process_manager import ProcessManager
from llmperf.pricing.catalog import PriceCatalog
from llmperf.pricing.loader import read_text as read_text_file
from llmperf.records.storage import Storage

logger = logging.getLogger(__name__)


class RunManager:
    def __init__(
        self,
        config: RunConfig,
        *,
        config_path: str = "",
        config_content: str | None = None,
        pricing_path: Optional[str] = None,
        run_id: str | None = None,
        register_run: bool = True,
        cancel_event: threading.Event | None = None,
    ):
        self.config = config
        self.config_path = config_path
        self.run_id = run_id or uuid.uuid4().hex
        self.cancel_event = cancel_event
        self.dataset = load_dataset(
            config.dataset.source.type,
            name=(config.dataset.source.name or "dataset"),
            config=(config.dataset.source.config or {}),
        )

        self.price_catalog = PriceCatalog(list(config.pricing))

        if config_content is not None:
            self.config_content = config_content
        elif config_path:
            self.config_content = read_text_file(config_path)
        else:
            self.config_content = ""
        # pricing_path is kept for backward compatibility but no longer used
        # Prices are now fetched automatically from pricing_history table

        runtime = load_runtime_config()
        self.db_path = config.db_path or str(runtime.db_path)
        self.storage = Storage(self.db_path)
        if register_run:
            self.storage.register_run(
                self.run_id,
                config,
                config_path=config_path,
                config_content=self.config_content,
                task_type="benchmark",
                status="running",
            )

    def run(self) -> str:
        rows: list[DatasetRow] = list(self.dataset)

        iterator_cfg = self.config.dataset.iterator or None
        mutation_chain_steps = (iterator_cfg.mutation_chain or ["identity"]) if iterator_cfg else ["identity"]
        max_total_seconds = iterator_cfg.max_total_seconds if iterator_cfg else None
        max_rounds = iterator_cfg.max_rounds if iterator_cfg else None
        if max_rounds is None:
            max_rounds = 1

        logger.info("Loaded dataset with %d cases", len(self.dataset))

        exec_meta: Dict[str, object] = {}

        manager = ProcessManager(
            self.run_id,
            self.config,
            rows,
            iterator_steps=mutation_chain_steps,
            max_total_seconds=max_total_seconds,
            max_rounds=max_rounds,
            db_path=self.db_path,
            deadline_ts=None,
            max_rows=None,
            exec_meta=exec_meta,
            cancel_event=self.cancel_event,
        )
        manager.run_all()

        from .analysis import create_analysis

        export = create_analysis(
            "summary",
            {
                "run_id": self.run_id,
                "task_name": (self.config.info or "run"),
                "output_dir": ".",
                "query": {"db_path": self.db_path},
            },
        )
        export.run()
        return self.run_id


def execute_from_yaml(
    config_path: str,
    *,
    pricing_path: Optional[str] = None,  # Kept for backward compatibility, no longer used
    config: Optional[RunConfig] = None,
    run_id: str | None = None,
) -> str:
    cfg = config or load_config(config_path)
    manager = RunManager(cfg, config_path=config_path, pricing_path=None, run_id=run_id)
    try:
        return manager.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (Ctrl-C).")
        raise
