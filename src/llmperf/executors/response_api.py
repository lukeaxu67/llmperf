from __future__ import annotations

from ..config.models import ExecutorConfig
from ..datasets.types import TestCase as DatasetRow
from ..pricing.catalog import PriceCatalog
from ..providers.base import ProviderRequest, create_provider
from ..records.model import RunRecord
from .base import BaseExecutor, register_executor


class ResponseExecutor(BaseExecutor):
    def __init__(self, config: ExecutorConfig):
        super().__init__(config)
        provider_name = config.param.get("provider_name", config.type)
        impl = config.param.get("provider_impl", config.impl)
        self.provider = create_provider(provider_name, impl, provider_name)

    def process_row(self, run_id: str, row: DatasetRow, price: PriceCatalog | None = None) -> RunRecord:
        messages = self.build_messages(row)
        request_params = dict(self.config.param or {})
        request = ProviderRequest(
            run_id=run_id,
            executor_id=self.config.id,
            dataset_row_id=row.id,
            provider=self.config.type,
            model=self.config.model or "",
            messages=messages,
            options=dict(request_params),
        )
        record = self.provider.invoke(request)
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


register_executor("huoshan_response", "response")(ResponseExecutor)
register_executor("responses", "response")(ResponseExecutor)
