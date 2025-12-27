from __future__ import annotations

import os
from typing import Any, Dict

from ..config.models import ExecutorConfig
from ..datasets.types import TestCase as DatasetRow
from ..pricing.catalog import PriceCatalog
from ..providers.base import ProviderRequest, create_provider
from ..records.model import RunRecord
from ..utils.backoff import BackoffConfig, should_retry_status, sleep_with_backoff
from .base import BaseExecutor, register_executor


def _resolve_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name.upper(), default)


class OpenAIChatExecutor(BaseExecutor):
    def __init__(self, config: ExecutorConfig):
        super().__init__(config)
        provider_name = config.param.get("provider_name", config.type)
        impl = config.param.get("provider_impl", config.impl)
        self.provider = create_provider(provider_name, impl, provider_name)

    def process_row(self, run_id: str, row: DatasetRow, price: PriceCatalog | None = None) -> RunRecord:
        dataset_row_id = row.id
        messages = self.build_messages(row)
        request_params = dict(self.config.param or {})
        options = dict(request_params)
        options["api_key"] = self.config.api_key or _resolve_env(f"{self.config.type}_api_key")
        options["api_url"] = self.config.api_url or _resolve_env(f"{self.config.type}_base_url")
        options["model"] = self.config.model
        request = ProviderRequest(
            run_id=run_id,
            executor_id=self.config.id,
            dataset_row_id=dataset_row_id,
            provider=self.config.type,
            model=self.config.model or "",
            messages=messages,
            options=options,
        )
        backoff_cfg = BackoffConfig()
        retries_meta: list[dict[str, Any]] = []

        attempt = 1
        while True:
            record = self.provider.invoke(request)
            if not should_retry_status(record.status) or attempt >= backoff_cfg.max_attempts:
                break
            delay = sleep_with_backoff(attempt, backoff_cfg)
            retries_meta.append(
                {
                    "attempt": attempt,
                    "status": record.status,
                    "delay_ms": int(delay * 1000),
                }
            )
            attempt += 1

        if retries_meta:
            extra = dict(record.extra or {})
            prev = extra.get("retries")
            if isinstance(prev, list):
                retries_meta = prev + retries_meta
            extra["retries"] = retries_meta
            record.extra = extra

        record.request_params = dict(request_params)

        if price:
            prompt_cost, completion_cost, cache_cost, total_cost, currency = price.compute_cost(
                record.provider,
                record.model,
                record.qtokens,
                record.atokens,
                record.ctokens,
            )
            record.prompt_cost = prompt_cost
            record.completion_cost = completion_cost
            record.cache_cost = cache_cost
            record.total_cost = total_cost
            record.currency = currency
        return record


for provider in [
    "openai",
    "qianwen",
    "zhipu",
    "deepseek",
    "spark",
    "hunyuan",
    "huoshan",
    "moonshot",
    "mock",
]:
    register_executor(provider, "chat")(OpenAIChatExecutor)
    register_executor(provider, "default")(OpenAIChatExecutor)
