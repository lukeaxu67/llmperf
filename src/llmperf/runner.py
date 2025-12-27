from __future__ import annotations

import logging
import time
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
        config_path: str,
        pricing_path: Optional[str],
        run_id: str | None = None,
    ):
        self.config = config
        self.config_path = config_path
        self.run_id = run_id or uuid.uuid4().hex
        self.dataset = load_dataset(
            config.dataset.source.type,
            name=(config.dataset.source.name or "dataset"),
            config=(config.dataset.source.config or {}),
        )

        self.price_catalog = PriceCatalog(list(config.pricing))

        self.config_content = read_text_file(config_path)
        self.pricing_file_path = pricing_path
        if pricing_path:
            self.pricing_content = read_text_file(pricing_path)
        elif config.pricing:
            import yaml

            self.pricing_content = yaml.safe_dump(
                {"pricing": [entry.model_dump() for entry in config.pricing]},
                allow_unicode=True,
                sort_keys=False,
            )
        else:
            self.pricing_content = ""

        runtime = load_runtime_config()
        self.db_path = config.db_path or str(runtime.db_path)
        self.storage = Storage(self.db_path)
        self.storage.register_run(
            self.run_id,
            config,
            config_path=config_path,
            config_content=self.config_content,
            pricing_path=self.pricing_file_path,
            pricing_content=self.pricing_content,
        )

    def run(self) -> str:
        rows: list[DatasetRow] = list(self.dataset)

        iterator_cfg = self.config.dataset.iterator or None
        mutation_chain_steps = (
            (iterator_cfg.mutation_chain or ["identity"]) if iterator_cfg else ["identity"]
        )
        max_total_seconds = iterator_cfg.max_total_seconds if iterator_cfg else None
        max_rounds = iterator_cfg.max_rounds if iterator_cfg else None

        logger.info(
            "Loaded dataset with %d cases",
            len(self.dataset),
        )

        exec_meta: Dict[str, object] = {}

        # Multi-process path: delegate executor execution to ProcessManager when enabled.
        manager = ProcessManager(
            self.run_id,
            self.config,
            rows,
            iterator_steps=mutation_chain_steps,
            iterator_max_total_seconds=max_total_seconds,
            iterator_max_rounds=max_rounds,
            db_path=self.db_path,
            deadline_ts=None,
            max_rows=None,
            exec_meta=exec_meta,
        )
        manager.run_all()

        from .analysis import create_analysis

        export = create_analysis(
            "excel",
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
    pricing_path: Optional[str] = None,
    config: Optional[RunConfig] = None,
    run_id: str | None = None,
) -> str:
    cfg = config or load_config(config_path)
    manager = RunManager(cfg, config_path=config_path, pricing_path=pricing_path, run_id=run_id)
    try:
        return manager.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user (Ctrl-C).")
        raise
