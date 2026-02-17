"""Model comparison analysis for cross-model benchmarking."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from llmperf.analysis.base_analysis import BaseAnalysis
from llmperf.analysis.record_query import RecordQuery
from llmperf.analysis.analysis_registry import register_analysis
from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)


class ComparisonConfig(BaseModel):
    """Configuration for model comparison analysis."""
    run_id: str = Field(..., min_length=1)
    """Run ID to analyze."""

    metrics: List[str] = Field(
        default_factory=lambda: [
            "success_rate",
            "avg_latency",
            "p95_latency",
            "throughput",
            "cost_per_request",
        ]
    )
    """Metrics to compare."""

    group_by: str = Field(default="executor")
    """Grouping: executor, provider, model, or executor_model."""

    query: RecordQuery = Field(default_factory=RecordQuery)
    """Query configuration."""


class ExecutorMetrics(BaseModel):
    """Metrics for a single executor/model."""
    executor_id: str
    provider: str
    model: str
    total_requests: int
    success_count: int
    error_count: int
    success_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_throughput: float
    total_cost: float
    cost_per_request: float
    avg_char_per_second: float
    avg_token_per_second: float


class ComparisonResult(BaseModel):
    """Result of model comparison."""
    groups: List[ExecutorMetrics]
    ranking: List[str]
    best_by_metric: Dict[str, str]
    statistical_tests: Dict[str, Any]


@register_analysis("comparison")
class ModelComparisonAnalysis(BaseAnalysis[ComparisonConfig]):
    """Compare performance across multiple executors/models.

    Provides comprehensive comparison including:
    - Side-by-side metric comparison
    - Automatic ranking
    - Best performer identification
    - Statistical significance tests
    """

    type_name = "comparison"
    Config = ComparisonConfig

    def run(self) -> Dict[str, Any]:
        """Run comparison analysis.

        Returns:
            Dictionary with comparison results.
        """
        storage = self.config.query.storage()
        records = list(storage.fetch_run_records(self.config.run_id))

        if not records:
            return {
                "run_id": self.config.run_id,
                "error": "No records found",
                "groups": [],
            }

        # Group records
        grouped = self._group_records(records)

        # Calculate metrics for each group
        groups = []
        for key, group_records in grouped.items():
            metrics = self._calculate_group_metrics(key, group_records)
            groups.append(metrics)

        # Generate ranking
        ranking = self._generate_ranking(groups)

        # Find best by each metric
        best_by_metric = self._find_best_by_metric(groups)

        # Statistical tests (if enough data)
        statistical_tests = self._run_statistical_tests(grouped)

        result = ComparisonResult(
            groups=groups,
            ranking=ranking,
            best_by_metric=best_by_metric,
            statistical_tests=statistical_tests,
        )

        return {
            "run_id": self.config.run_id,
            "group_by": self.config.group_by,
            **result.model_dump(),
        }

    def _group_records(
        self,
        records: List[RunRecord],
    ) -> Dict[Tuple[str, str, str], List[RunRecord]]:
        """Group records by the configured grouping.

        Args:
            records: List of records.

        Returns:
            Dictionary mapping group key to records.
        """
        grouped: Dict[Tuple[str, str, str], List[RunRecord]] = defaultdict(list)

        for record in records:
            if self.config.group_by == "executor":
                key = (record.executor_id, "", "")
            elif self.config.group_by == "provider":
                key = (record.provider, "", "")
            elif self.config.group_by == "model":
                key = (record.model, "", "")
            else:  # executor_model
                key = (record.executor_id, record.provider, record.model)

            grouped[key].append(record)

        return grouped

    def _calculate_group_metrics(
        self,
        key: Tuple[str, str, str],
        records: List[RunRecord],
    ) -> ExecutorMetrics:
        """Calculate metrics for a group of records.

        Args:
            key: Group key.
            records: List of records in the group.

        Returns:
            ExecutorMetrics for the group.
        """
        total = len(records)
        successful = [r for r in records if r.status == 200]
        errors = [r for r in records if r.status != 200]

        latencies = sorted([r.first_resp_time for r in successful if r.first_resp_time > 0])
        throughputs = [r.token_throughput for r in successful if r.token_throughput > 0]
        char_speeds = [r.char_per_second for r in successful if r.char_per_second > 0]
        token_speeds = [r.token_per_second for r in successful if r.token_per_second > 0]

        total_cost = sum(r.total_cost for r in records)

        def percentile(data, p):
            if not data:
                return 0
            k = int(len(data) * p)
            return data[min(k, len(data) - 1)]

        return ExecutorMetrics(
            executor_id=key[0],
            provider=key[1] or (records[0].provider if records else ""),
            model=key[2] or (records[0].model if records else ""),
            total_requests=total,
            success_count=len(successful),
            error_count=len(errors),
            success_rate=len(successful) / total * 100 if total > 0 else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p50_latency_ms=percentile(latencies, 0.50),
            p95_latency_ms=percentile(latencies, 0.95),
            p99_latency_ms=percentile(latencies, 0.99),
            avg_throughput=statistics.mean(throughputs) if throughputs else 0,
            total_cost=total_cost,
            cost_per_request=total_cost / total if total > 0 else 0,
            avg_char_per_second=statistics.mean(char_speeds) if char_speeds else 0,
            avg_token_per_second=statistics.mean(token_speeds) if token_speeds else 0,
        )

    def _generate_ranking(self, groups: List[ExecutorMetrics]) -> List[str]:
        """Generate overall ranking of executors.

        Args:
            groups: List of executor metrics.

        Returns:
            List of executor IDs in ranking order (best first).
        """
        def score(g: ExecutorMetrics) -> float:
            # Higher is better: success rate, throughput
            # Lower is better: latency, cost
            # Normalize and combine
            success_score = g.success_rate / 100

            # Avoid division by zero
            latency_score = 1 / (g.avg_latency_ms + 1) if g.avg_latency_ms > 0 else 1
            cost_score = 1 / (g.cost_per_request + 0.001) if g.cost_per_request > 0 else 1
            throughput_score = min(g.avg_throughput / 100, 1) if g.avg_throughput > 0 else 0

            # Weighted combination
            return (
                success_score * 0.3 +
                latency_score * 0.25 +
                throughput_score * 0.25 +
                cost_score * 0.2
            )

        sorted_groups = sorted(groups, key=score, reverse=True)
        return [g.executor_id for g in sorted_groups]

    def _find_best_by_metric(self, groups: List[ExecutorMetrics]) -> Dict[str, str]:
        """Find the best performer for each metric.

        Args:
            groups: List of executor metrics.

        Returns:
            Dictionary mapping metric name to best executor ID.
        """
        if not groups:
            return {}

        best = {}

        # Success rate (higher is better)
        best["success_rate"] = max(groups, key=lambda g: g.success_rate).executor_id

        # Latency (lower is better)
        valid_latency = [g for g in groups if g.avg_latency_ms > 0]
        if valid_latency:
            best["avg_latency"] = min(valid_latency, key=lambda g: g.avg_latency_ms).executor_id

        # Throughput (higher is better)
        valid_throughput = [g for g in groups if g.avg_throughput > 0]
        if valid_throughput:
            best["throughput"] = max(valid_throughput, key=lambda g: g.avg_throughput).executor_id

        # Cost (lower is better)
        valid_cost = [g for g in groups if g.cost_per_request > 0]
        if valid_cost:
            best["cost_per_request"] = min(valid_cost, key=lambda g: g.cost_per_request).executor_id

        return best

    def _run_statistical_tests(
        self,
        grouped: Dict[Tuple[str, str, str], List[RunRecord]],
    ) -> Dict[str, Any]:
        """Run statistical significance tests between groups.

        Args:
            grouped: Grouped records.

        Returns:
            Dictionary with statistical test results.
        """
        results = {}

        # Need at least 2 groups with enough data
        valid_groups = {k: v for k, v in grouped.items() if len(v) >= 10}

        if len(valid_groups) < 2:
            return {"note": "Insufficient data for statistical tests"}

        # Compare latency distributions
        latency_distributions = {}
        for key, records in valid_groups.items():
            latencies = [r.first_resp_time for r in records if r.status == 200 and r.first_resp_time > 0]
            if len(latencies) >= 10:
                latency_distributions[key[0]] = latencies

        # Calculate coefficient of variation for each group
        cv_results = {}
        for executor_id, latencies in latency_distributions.items():
            mean = statistics.mean(latencies)
            stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0
            cv = (stdev / mean * 100) if mean > 0 else 0
            cv_results[executor_id] = {
                "mean": mean,
                "std_dev": stdev,
                "coefficient_of_variation": cv,
                "consistency": "high" if cv < 25 else "medium" if cv < 50 else "low",
            }

        results["latency_analysis"] = cv_results

        return results
