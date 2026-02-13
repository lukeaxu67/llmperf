"""
Stability and volatility analysis for LLM performance monitoring.

Analyzes latency distribution, time-of-day patterns, sliding window stability,
and extreme request detection.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from datetime import datetime

from pydantic import BaseModel, Field

from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery
from ..analysis_registry import register_analysis
from ..statistics import (
    percentile,
    mean,
    std_dev,
    outlier_count,
    outlier_ratio,
    DistributionSummary,
)
from ..timeseries import (
    group_by_time_segment,
    compare_day_night,
    sliding_window_analysis,
    calculate_volatility_series,
    find_anomaly_windows,
    get_hour_of_day,
)

from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage


@register_analysis("stability")
class StabilityAnalysis(BaseAnalysis["StabilityAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        query: RecordQuery = Field(default_factory=RecordQuery)
        segment_size: int = Field(default=6, description="Hours per time segment (default: 6)")
        window_minutes: int = Field(default=30, description="Sliding window size in minutes")
        step_minutes: int = Field(default=5, description="Step size for sliding window")
        outlier_threshold: float = Field(default=2.0, description="Multiplier for P95 outlier detection")
        z_score_threshold: float = Field(default=2.0, description="Z-score threshold for anomaly detection")

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

        # Apply max_duration_hours limit (default 24 hours)
        if self.config.query.max_duration_hours is not None and records:
            records = self._apply_duration_limit(records)

        if not records:
            return {"error": "No records found for the given query"}

        # Group by provider/model
        grouped: Dict[Tuple[str, str], List[RunRecord]] = defaultdict(list)
        for rec in records:
            grouped[(rec.provider, rec.model)].append(rec)

        # Add duration info to results
        timestamps = [r.created_at for r in records]
        duration_hours = (max(timestamps) - min(timestamps)) / (1000 * 3600) if timestamps else 0

        results = {
            "config": {
                "segment_size": self.config.segment_size,
                "window_minutes": self.config.window_minutes,
                "step_minutes": self.config.step_minutes,
                "outlier_threshold": self.config.outlier_threshold,
                "z_score_threshold": self.config.z_score_threshold,
                "max_duration_hours": self.config.query.max_duration_hours,
            },
            "duration_hours": duration_hours,
            "total_records": len(records),
            "by_provider_model": {},
        }

        for (provider, model), recs in grouped.items():
            results["by_provider_model"][f"{provider}.{model}"] = self._analyze_group(
                provider, model, recs
            )

        return results

    def _apply_duration_limit(self, records: List[RunRecord]) -> List[RunRecord]:
        """Apply max duration limit to records, keeping the most recent N hours."""
        if not records:
            return records

        # Sort by timestamp
        sorted_records = sorted(records, key=lambda r: r.created_at)

        # Get the latest timestamp
        latest_ts = sorted_records[-1].created_at

        # Calculate cutoff timestamp
        max_duration_ms = int(self.config.query.max_duration_hours * 3600 * 1000)
        cutoff_ts = latest_ts - max_duration_ms

        # Filter records after cutoff
        filtered = [r for r in sorted_records if r.created_at >= cutoff_ts]

        return filtered

    def _analyze_group(
        self, provider: str, model: str, records: List[RunRecord]
    ) -> Dict[str, Any]:
        """Analyze a group of records for the same provider/model."""

        # Extract metrics
        timestamps = [rec.created_at for rec in records]
        ttft_values = [float(rec.first_resp_time) for rec in records if rec.first_resp_time > 0]
        total_time_values = [float(rec.last_resp_time) for rec in records if rec.last_resp_time > 0]
        throughput_values = [float(rec.token_throughput) for rec in records if rec.token_throughput > 0]
        output_lengths = [int(rec.atokens) for rec in records if rec.atokens > 0]
        error_flags = [int(rec.status) != 200 for rec in records]

        # 1. TTFT Distribution
        ttft_dist = DistributionSummary.from_values(ttft_values, self.config.outlier_threshold)

        # 2. Time Segmentation (0-6, 6-12, 12-18, 18-24)
        segments = group_by_time_segment(
            timestamps,
            ttft_values,
            total_time_values,
            throughput_values,
            error_flags,
            output_lengths,
            self.config.segment_size,
        )

        # 3. Day/Night Comparison
        day_stats, night_stats = compare_day_night(timestamps, ttft_values)

        # 4. Sliding Window Analysis
        window_stats = sliding_window_analysis(
            timestamps,
            ttft_values,
            self.config.window_minutes,
            self.config.step_minutes,
        )

        # 5. Volatility Analysis
        volatility = calculate_volatility_series(window_stats) if window_stats else []

        # 6. Anomaly Detection
        anomalies = find_anomaly_windows(
            window_stats,
            self.config.z_score_threshold,
        )

        # 7. Error Rate Analysis
        error_count = sum(error_flags)
        error_rate = error_count / len(records) if records else 0

        # 8. Output Length Distribution
        output_dist = DistributionSummary.from_values(
            [float(l) for l in output_lengths],
            self.config.outlier_threshold,
        )

        return {
            "total_requests": len(records),
            "successful_requests": len(records) - error_count,
            "error_rate": error_rate,

            # TTFT Distribution
            "ttft_distribution": {
                "count": ttft_dist.count,
                "mean": ttft_dist.mean,
                "std": ttft_dist.std,
                "cv": ttft_dist.cv,
                "min": ttft_dist.min,
                "max": ttft_dist.max,
                "p50": ttft_dist.p50,
                "p90": ttft_dist.p90,
                "p95": ttft_dist.p95,
                "p99": ttft_dist.p99,
                "outlier_count": ttft_dist.outlier_count,
                "outlier_ratio": ttft_dist.outlier_ratio,
            },

            # Time Segments
            "time_segments": {
                name: {
                    "start_hour": seg.start_hour,
                    "end_hour": seg.end_hour,
                    "count": seg.count,
                    "mean_ttft": seg.mean_ttft,
                    "mean_total_time": seg.mean_total_time,
                    "mean_throughput": seg.mean_throughput,
                    "error_rate": seg.error_rate,
                    "mean_output_length": seg.mean_output_length,
                }
                for name, seg in segments.items()
            },

            # Day/Night Comparison
            "day_night_comparison": {
                "day": {
                    "hours": "06-18",
                    **day_stats,
                },
                "night": {
                    "hours": "18-06",
                    **night_stats,
                },
                "day_night_ratio": {
                    "ttft_ratio": day_stats['mean'] / night_stats['mean'] if night_stats['mean'] > 0 else 0,
                    "throughput_ratio": (mean(throughput_values[:len(throughput_values)//2]) /
                                      max(mean(throughput_values[len(throughput_values)//2:]), 0.001))
                                      if len(throughput_values) > 1 else 0,
                }
            },

            # Sliding Window
            "sliding_window_summary": {
                "window_count": len(window_stats),
                "volatility_spikes": sum(1 for v in volatility if v.get('is_spike', False)),
                "anomaly_count": len(anomalies),
            },

            # Output Length
            "output_length_distribution": {
                "count": output_dist.count,
                "mean": output_dist.mean,
                "min": output_dist.min,
                "max": output_dist.max,
                "p50": output_dist.p50,
                "p95": output_dist.p95,
            },

            # Insights
            "insights": self._generate_insights(
                ttft_dist, segments, day_stats, night_stats, volatility, anomalies
            ),
        }

    def _generate_insights(
        self,
        ttft_dist: DistributionSummary,
        segments: Dict[str, "TimeSegment"],
        day_stats: Dict[str, float],
        night_stats: Dict[str, float],
        volatility: List[Dict],
        anomalies: List[Dict],
    ) -> List[str]:
        """Generate human-readable insights from the analysis."""
        insights = []

        # CV insight
        if ttft_dist.cv > 0.5:
            insights.append(f"High volatility detected (CV={ttft_dist.cv:.2f}), response times vary significantly")
        elif ttft_dist.cv < 0.2:
            insights.append(f"Stable performance (CV={ttft_dist.cv:.2f}), consistent response times")

        # Outlier insight
        if ttft_dist.outlier_ratio > 0.05:
            insights.append(f"High outlier ratio ({ttft_dist.outlier_ratio*100:.1f}%), may indicate throttling or resource issues")

        # Day/Night insight
        if day_stats['count'] > 0 and night_stats['count'] > 0:
            ratio = day_stats['mean'] / night_stats['mean'] if night_stats['mean'] > 0 else 1
            if ratio > 1.3:
                insights.append(f"Day time is {ratio:.1f}x slower than night, possible congestion during business hours")
            elif ratio < 0.7:
                insights.append(f"Night time is {1/ratio:.1f}x slower than day, possible maintenance windows")

        # Volatility insight
        spike_count = sum(1 for v in volatility if v.get('is_spike', False))
        if spike_count > 0:
            insights.append(f"Detected {spike_count} volatility spikes, possible resource contention or hot-swapping")

        # Anomaly insight
        if anomalies:
            insights.append(f"Found {len(anomalies)} anomalous time windows, may indicate scheduling changes")

        # Time segment insights
        if len(segments) >= 2:
            segment_means = [(s.mean_ttft, s.name) for s in segments.values() if s.count > 0]
            if segment_means:
                max_seg = max(segment_means, key=lambda x: x[0])
                min_seg = min(segment_means, key=lambda x: x[0])
                ratio = max_seg[0] / min_seg[0] if min_seg[0] > 0 else 1
                if ratio > 2:
                    insights.append(f"Segment {max_seg[1]} is {ratio:.1f}x slower than {min_seg[1]}, strong temporal pattern")

        return insights
