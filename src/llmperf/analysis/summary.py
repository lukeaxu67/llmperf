from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

from pydantic import BaseModel, Field

from ..records.model import RunRecord
from .base import BaseAnalysis, Query, register_analysis


def _params_key(params: dict) -> str:
    import json

    return json.dumps(params or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[int(k)])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


@register_analysis("summary")
class SummaryAnalysis(BaseAnalysis["SummaryAnalysis.Config"]):
    class Config(BaseModel):
        query: Query = Field(default_factory=Query)
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

        grouped: Dict[str, list[RunRecord]] = defaultdict(list)
        for r in records:
            key = f"{r.provider}.{r.model}"
            if self.config.group_by_request_params:
                key = f"{key}:{_params_key(r.request_params)}"
            grouped[key].append(r)

        out: Dict[str, Any] = {"items": {}}
        for key, items in grouped.items():
            first_resp = [float(r.first_resp_time) for r in items if r.first_resp_time]
            out["items"][key] = {
                "count": len(items),
                "avg_first_resp_ms": (sum(first_resp) / len(first_resp)) if first_resp else 0.0,
                "p95_first_resp_ms": _percentile(first_resp, 0.95) if first_resp else 0.0,
                "avg_char_per_second": (
                    sum(float(r.char_per_second) for r in items if r.char_per_second)
                    / max(len([r for r in items if r.char_per_second]), 1)
                ),
                "avg_token_throughput": (
                    sum(float(r.token_throughput) for r in items if r.token_throughput)
                    / max(len([r for r in items if r.token_throughput]), 1)
                ),
            }
        return out
