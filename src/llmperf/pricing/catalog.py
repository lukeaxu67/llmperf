from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..config.models import PricingEntry


@dataclass
class PriceItem:
    provider: str
    model: str
    unit_scale: int
    input_price_per_1k: float
    output_price_per_1k: float
    cache_input_price_per_1k: float
    cache_output_price_per_1k: float
    currency: str = "CNY"


class PriceCatalog:
    def __init__(self, entries: list[PricingEntry] | None = None):
        self._items: Dict[tuple[str, str], PriceItem] = {}
        if entries:
            for entry in entries:
                self.register(entry)

    def register(self, entry: PricingEntry) -> None:
        key = (entry.provider, entry.model)
        unit_scale = 1_000 if entry.unit == "per_1k" else 1_000_000
        divisor = unit_scale / 1_000.0
        input_price_per_1k = entry.input_price / divisor if divisor else entry.input_price
        output_price_per_1k = entry.output_price / divisor if divisor else entry.output_price
        cache_input_price_per_1k = input_price_per_1k * entry.cache_input_discount
        cache_output_price_per_1k = output_price_per_1k * entry.cache_output_discount
        self._items[key] = PriceItem(
            provider=entry.provider,
            model=entry.model,
            unit_scale=unit_scale,
            input_price_per_1k=input_price_per_1k,
            output_price_per_1k=output_price_per_1k,
            cache_input_price_per_1k=cache_input_price_per_1k,
            cache_output_price_per_1k=cache_output_price_per_1k,
            currency=entry.currency,
        )

    def get(self, provider: str, model: str) -> Optional[PriceItem]:
        return self._items.get((provider, model))

    def compute_cost(
        self,
        provider: str,
        model: str,
        qtokens: int,
        atokens: int,
        ctokens: int = 0,
    ) -> tuple[float, float, float, float, str]:
        item = self.get(provider, model)
        if not item:
            return 0.0, 0.0, 0.0, 0.0, "CNY"
        safe_qtokens = max(qtokens or 0, 0)
        safe_atokens = max(atokens or 0, 0)
        safe_ctokens = max(min(ctokens or 0, safe_qtokens), 0)
        prompt_tokens = max(safe_qtokens - safe_ctokens, 0)
        prompt_cost = (prompt_tokens / 1000.0) * item.input_price_per_1k
        completion_cost = (safe_atokens / 1000.0) * item.output_price_per_1k
        cache_cost = (safe_ctokens / 1000.0) * item.cache_input_price_per_1k
        total = prompt_cost + completion_cost + cache_cost
        return prompt_cost, completion_cost, cache_cost, total, item.currency
