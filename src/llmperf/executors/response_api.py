from __future__ import annotations

from ..config.models import ExecutorConfig
from ..datasets.types import TestCase as DatasetRow
from ..pricing.catalog import PriceCatalog
from ..providers.base import ProviderRequest, create_provider
from ..records.model import RunRecord
from ..web.services.pricing_service import PricingService
from .base import BaseExecutor, register_executor


class ResponseExecutor(BaseExecutor):
    def __init__(self, config: ExecutorConfig):
        super().__init__(config)
        provider_name = config.param.get("provider_name", config.type)
        impl = config.param.get("provider_impl", config.impl)
        self.provider = create_provider(provider_name, impl, provider_name)
        self.pricing_service = PricingService()

    def process_row(self, run_id: str, row: DatasetRow, price: PriceCatalog | None = None) -> RunRecord:
        messages = self.build_messages(row)
        request_params = dict(self.config.param or {})
        options = dict(request_params)
        options["api_key"] = self.config.api_key
        options["api_url"] = self.config.api_url
        request = ProviderRequest(
            run_id=run_id,
            executor_id=self.config.id,
            dataset_row_id=row.id,
            provider=self.config.type,
            model=self.config.model or "",
            messages=messages,
            options=options,
        )
        record = self.provider.invoke(request)
        record.request_params = dict(request_params)
        price_item = price.get(record.provider, record.model) if price else None
        if price_item:
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
            record.input_price_snapshot = price_item.input_price_per_1k * 1000.0
            record.output_price_snapshot = price_item.output_price_per_1k * 1000.0
            record.cache_price_snapshot = price_item.cache_input_price_per_1k * 1000.0
        else:
            current_price = self.pricing_service.get_current_price(record.provider, record.model)
            has_runtime_price = any(
                value > 0
                for value in (
                    current_price.input_price,
                    current_price.output_price,
                    current_price.cache_read_price,
                )
            )
            if has_runtime_price:
                prompt_cost, completion_cost, cache_cost, total_cost, currency = self.pricing_service.compute_cost(
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
                record.input_price_snapshot = current_price.input_price
                record.output_price_snapshot = current_price.output_price
                record.cache_price_snapshot = current_price.cache_read_price
        return record


register_executor("huoshan_response", "response")(ResponseExecutor)
register_executor("responses", "response")(ResponseExecutor)
