from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..pricing.catalog import PriceCatalog
from ..records.model import RunRecord
from .base import BaseAnalysis, Query, register_analysis


def _params_key(params: dict) -> str:
    import json

    return json.dumps(params or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@register_analysis("cost")
class CostAnalysis(BaseAnalysis["CostAnalysis.Config"]):
    class Config(BaseModel):
        query: Query = Field(default_factory=Query)
        price_catalog_path: Optional[str] = None
        group_by_request_params: bool = True

    def run(self) -> Dict[str, Any]:
        storage = self.config.query.storage()
        records = list(
            storage.query_records(
                provider=self.config.query.provider,
                model=self.config.query.model,
                start_ts=self.config.query.start_ts,
                end_ts=self.config.query.end_ts,
            )
        )
        if self.config.query.run_ids:
            allow = set(self.config.query.run_ids)
            records = [r for r in records if r.run_id in allow]

        price_catalog = self._price_catalog(path=self.config.price_catalog_path)
        if price_catalog is not None:
            for r in records:
                prompt_cost, completion_cost, cache_cost, total_cost, currency = price_catalog.compute_cost(
                    r.provider, r.model, r.qtokens, r.atokens, r.ctokens
                )
                r.prompt_cost = prompt_cost
                r.completion_cost = completion_cost
                r.cache_cost = cache_cost
                r.total_cost = total_cost
                r.currency = currency

        grouped: Dict[str, list[RunRecord]] = defaultdict(list)
        for r in records:
            key = f"{r.provider}.{r.model}"
            if self.config.group_by_request_params:
                key = f"{key}:{_params_key(r.request_params)}"
            grouped[key].append(r)

        out: Dict[str, Any] = {"items": {}}
        for key, items in grouped.items():
            out["items"][key] = {
                "count": len(items),
                "prompt_cost": sum(r.prompt_cost for r in items),
                "completion_cost": sum(r.completion_cost for r in items),
                "cache_cost": sum(r.cache_cost for r in items),
                "total_cost": sum(r.total_cost for r in items),
                "currency": (items[0].currency if items else ""),
            }
        return out
