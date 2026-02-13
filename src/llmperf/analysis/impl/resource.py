"""
Resource behavior analysis for LLM performance monitoring.

Analyzes TTFT vs generation rate correlation, request autocorrelation,
worker assignment patterns, and queue vs processing time inference.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery
from ..analysis_registry import register_analysis
from ..statistics import (
    correlation,
    autocorrelation,
    percentile,
    mean,
    std_dev,
    AutocorrelationStats,
)

from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage


@register_analysis("resource")
class ResourceAnalysis(BaseAnalysis["ResourceAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        query: RecordQuery = Field(default_factory=RecordQuery)

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

        # Apply max_duration_hours limit
        if self.config.query.max_duration_hours is not None:
            records = self._apply_duration_limit(records, self.config.query.max_duration_hours)

        if not records:
            return {"error": "No records found for the given query"}

        # Sort by created_at for time-series analysis
        records.sort(key=lambda r: r.created_at)

        # Group by provider/model
        grouped: Dict[Tuple[str, str], List[RunRecord]] = defaultdict(list)
        for rec in records:
            grouped[(rec.provider, rec.model)].append(rec)

        results = {
            "by_provider_model": {},
        }

        for (provider, model), recs in grouped.items():
            results["by_provider_model"][f"{provider}.{model}"] = self._analyze_group(
                provider, model, recs
            )

        return results

    def _analyze_group(
        self, provider: str, model: str, records: List[RunRecord]
    ) -> Dict[str, Any]:
        """Analyze resource behavior for a group of records."""

        # Extract metrics for successful requests only
        successful = [r for r in records if r.status == 200 and r.is_valid()]

        ttft_values = [float(r.first_resp_time) for r in successful if r.first_resp_time > 0]
        session_time_values = [float(r.session_time) for r in successful if r.session_time > 0]
        throughput_values = [float(r.token_throughput) for r in successful if r.token_throughput > 0]

        # 1. TTFT vs Generation Rate Correlation
        ttft_throughput_corr = self._calculate_ttft_throughput_correlation(successful)

        # 2. Request Autocorrelation
        ttft_autocorr = AutocorrelationStats.from_values(ttft_values) if ttft_values else None
        throughput_autocorr = AutocorrelationStats.from_values(throughput_values) if throughput_values else None

        # 3. TTFT vs Total Time Correlation (queue inference)
        ttft_total_corr = correlation(ttft_values, session_time_values) if ttft_values and session_time_values else 0

        # 4. High TTFT Analysis
        high_ttft_threshold = percentile(ttft_values, 0.75) if ttft_values else 0
        high_ttft_throughput = [t for r, t in zip(successful, throughput_values)
                                if float(r.first_resp_time) > high_ttft_threshold]

        normal_ttft_throughput = [t for r, t in zip(successful, throughput_values)
                                  if float(r.first_resp_time) <= high_ttft_threshold]

        # 5. Sequential Pattern Detection
        sequential_stats = self._analyze_sequential_patterns(ttft_values)

        return {
            "total_requests": len(records),
            "successful_requests": len(successful),

            # TTFT vs Throughput Correlation
            "ttft_throughput_correlation": {
                "correlation": ttft_throughput_corr,
                "interpretation": self._interpret_correlation(ttft_throughput_corr, "ttft_throughput"),
            },

            # Autocorrelation
            "ttft_autocorrelation": {
                "lag1": ttft_autocorr.lag1 if ttft_autocorr else 0,
                "lag5": ttft_autocorr.lag5 if ttft_autocorr else 0,
                "lag10": ttft_autocorr.lag10 if ttft_autocorr else 0,
                "lag30": ttft_autocorr.lag30 if ttft_autocorr else 0,
                "interpretation": self._interpret_autocorrelation(ttft_autocorr),
            },

            "throughput_autocorrelation": {
                "lag1": throughput_autocorr.lag1 if throughput_autocorr else 0,
                "lag5": throughput_autocorr.lag5 if throughput_autocorr else 0,
                "lag10": throughput_autocorr.lag10 if throughput_autocorr else 0,
                "interpretation": self._interpret_autocorrelation(throughput_autocorr),
            },

            # Queue vs Processing Inference
            "ttft_total_time_correlation": {
                "correlation": ttft_total_corr,
                "interpretation": self._interpret_ttft_total_correlation(ttft_total_corr),
            },

            # High TTFT Analysis
            "high_ttft_analysis": {
                "threshold": high_ttft_threshold,
                "high_ttft_avg_throughput": mean(high_ttft_throughput) if high_ttft_throughput else 0,
                "normal_ttft_avg_throughput": mean(normal_ttft_throughput) if normal_ttft_throughput else 0,
                "throughput_ratio": (
                    mean(high_ttft_throughput) / mean(normal_ttft_throughput)
                    if high_ttft_throughput and normal_ttft_throughput and mean(normal_ttft_throughput) > 0
                    else 0
                ),
                "interpretation": self._interpret_high_ttft_analysis(
                    high_ttft_throughput, normal_ttft_throughput
                ),
            },

            # Sequential Patterns
            "sequential_patterns": sequential_stats,

            # Overall Insights
            "insights": self._generate_insights(
                ttft_throughput_corr,
                ttft_autocorr,
                ttft_total_corr,
                sequential_stats,
            ),
        }

    def _calculate_ttft_throughput_correlation(self, records: List[RunRecord]) -> float:
        """Calculate correlation between TTFT and throughput."""
        if len(records) < 3:
            return 0.0

        ttft = [float(r.first_resp_time) for r in records if r.first_resp_time > 0]
        throughput = [float(r.token_throughput) for r in records if r.token_throughput > 0]

        if len(ttft) != len(throughput) or len(ttft) < 3:
            return 0.0

        return correlation(ttft, throughput)

    def _interpret_correlation(self, corr: float, metric_type: str) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(corr)

        if abs_corr < 0.2:
            strength = "Very weak or no correlation"
        elif abs_corr < 0.4:
            strength = "Weak correlation"
        elif abs_corr < 0.6:
            strength = "Moderate correlation"
        elif abs_corr < 0.8:
            strength = "Strong correlation"
        else:
            strength = "Very strong correlation"

        if metric_type == "ttft_throughput":
            if corr < -0.3:
                return f"{strength}. High TTFT correlates with LOW throughput. Suggests: Load-related degradation, GPU resource contention."
            elif corr > 0.3:
                return f"{strength}. High TTFT correlates with HIGH throughput. Suggests: Batch processing, queue waiting time."
            else:
                return f"{strength}. TTFT and throughput appear independent. Suggests: Different bottlenecks or dynamic allocation."
        return strength

    def _interpret_autocorrelation(self, autocorr: AutocorrelationStats | None) -> str:
        """Interpret autocorrelation results."""
        if not autocorr:
            return "Insufficient data"

        lag1 = autocorr.lag1
        lag5 = autocorr.lag5

        if abs(lag1) > 0.5:
            if lag1 > 0:
                return "Strong positive autocorrelation at lag 1. Suggests: Fixed worker assignment, stateful routing."
            else:
                return "Strong negative autocorrelation at lag 1. Suggests: Round-robin or alternating resources."
        elif abs(lag1) > 0.2:
            return f"Moderate autocorrelation (lag1={lag1:.2f}). Some temporal dependency in performance."
        else:
            return f"Low autocorrelation (lag1={lag1:.2f}). Requests are largely independent. Good load balancing."

    def _interpret_ttft_total_correlation(self, corr: float) -> str:
        """Interpret TTFT vs Total Time correlation."""
        if corr > 0.8:
            return "Very high correlation. TTFT dominates total time. Suggests: Minimal generation time variation, queue effect minimal."
        elif corr > 0.5:
            return "High correlation. TTFT significantly impacts total time. Suggests: First-token time is the main bottleneck."
        elif corr < 0.3:
            return "Low correlation. Generation time varies independently. Suggests: Variable output length or GPU batch scheduling effects."
        else:
            return "Moderate correlation. Both TTFT and generation time contribute to total latency."

    def _interpret_high_ttft_analysis(
        self, high_ttft_tp: List[float], normal_ttft_tp: List[float]
    ) -> str:
        """Interpret high TTFT throughput analysis."""
        if not high_ttft_tp or not normal_ttft_tp:
            return "Insufficient data"

        high_mean = mean(high_ttft_tp)
        normal_mean = mean(normal_ttft_tp)

        if high_mean > normal_mean * 1.2:
            return "High TTFT + high throughput. Suggests: GPU batch processing, queue waiting time."
        elif high_mean < normal_mean * 0.8:
            return "High TTFT + low throughput. Suggests: Resource contention, throttling."
        else:
            return "High TTFT but similar throughput. Suggests: Cold start or initialization overhead."

    def _analyze_sequential_patterns(self, values: List[float]) -> Dict[str, Any]:
        """Analyze sequential patterns in the time series."""
        if len(values) < 10:
            return {"note": "Insufficient data for sequential analysis"}

        # Find runs of consecutive high/low values
        median = percentile(values, 0.5)
        high_threshold = median * 1.2
        low_threshold = median * 0.8

        current_run = 0
        runs = []

        for v in values:
            if v > high_threshold:
                if current_run <= 0:
                    runs.append(('high', 1))
                else:
                    runs[-1] = ('high', runs[-1][1] + 1)
                current_run = 1
            elif v < low_threshold:
                if current_run >= 0:
                    runs.append(('low', 1))
                else:
                    runs[-1] = ('low', runs[-1][1] + 1)
                current_run = -1
            else:
                current_run = 0

        # Calculate statistics
        high_runs = [length for typ, length in runs if typ == 'high']
        low_runs = [length for typ, length in runs if typ == 'low']

        return {
            "high_value_runs": len(high_runs),
            "avg_high_run_length": sum(high_runs) / len(high_runs) if high_runs else 0,
            "max_high_run_length": max(high_runs) if high_runs else 0,
            "low_value_runs": len(low_runs),
            "avg_low_run_length": sum(low_runs) / len(low_runs) if low_runs else 0,
            "max_low_run_length": max(low_runs) if low_runs else 0,
        }

    def _generate_insights(
        self,
        ttft_throughput_corr: float,
        ttft_autocorr: AutocorrelationStats | None,
        ttft_total_corr: float,
        sequential_stats: Dict[str, Any],
    ) -> List[str]:
        """Generate resource behavior insights."""
        insights = []

        # Correlation insights
        if abs(ttft_throughput_corr) > 0.4:
            insights.append(f"TTFT-Throughput correlation: {ttft_throughput_corr:.2f}")

        # Autocorrelation insights
        if ttft_autocorr and abs(ttft_autocorr.lag1) > 0.3:
            if ttft_autocorr.lag1 > 0:
                insights.append("Positive autocorrelation detected - requests may be routed to same worker")
            else:
                insights.append("Negative autocorrelation detected - possible round-robin load balancing")

        # Sequential pattern insights
        if sequential_stats.get('max_high_run_length', 0) > 5:
            insights.append(f"Detected runs of up to {sequential_stats['max_high_run_length']} consecutive high-latency requests")

        # Queue vs processing
        if ttft_total_corr > 0.7:
            insights.append("TTFT dominates total time - first token is the main bottleneck")
        elif ttft_total_corr < 0.4:
            insights.append("Generation time varies independently - GPU batch scheduling effects detected")

        return insights
