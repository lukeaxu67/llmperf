from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

from pydantic import BaseModel, Field

from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery
from ..analysis_registry import register_analysis
from ..utils import _percentile, _params_key

from llmperf.records.model import RunRecord


@register_analysis("cross")
class CrossAnalysis(BaseAnalysis["CrossAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        query: RecordQuery = Field(default_factory=RecordQuery)
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
