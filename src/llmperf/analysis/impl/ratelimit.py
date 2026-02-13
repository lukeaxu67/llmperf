"""
Rate limiting detection analysis for LLM performance monitoring.

Detects minute-level periodic patterns, hourly patterns, fixed-interval slowdowns,
request count-based throttling, and scheduling patterns.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

from pydantic import BaseModel, Field

from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery
from ..analysis_registry import register_analysis
from ..statistics import percentile, mean, std_dev, variance
from ..timeseries import detect_periodicity, sliding_window_analysis

from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage


@register_analysis("ratelimit")
class RatelimitAnalysis(BaseAnalysis["RatelimitAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        query: RecordQuery = Field(default_factory=RecordQuery)
        minute_bins: int = Field(default=60, description="Number of minute bins to analyze")
        hour_bins: int = Field(default=24, description="Number of hour bins to analyze")

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

        # Sort by timestamp
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
        """Analyze rate limiting patterns for a group of records."""

        # Extract metrics
        timestamps = [r.created_at for r in records]
        ttft_values = [float(r.first_resp_time) for r in records if r.first_resp_time > 0]
        status_codes = [r.status for r in records]

        # 1. Minute-level pattern analysis
        minute_patterns = self._analyze_minute_patterns(timestamps, ttft_values)

        # 2. Hourly pattern analysis
        hourly_patterns = self._analyze_hourly_patterns(timestamps, ttft_values)

        # 3. Request interval analysis
        interval_patterns = self._analyze_request_intervals(timestamps, ttft_values)

        # 4. Periodicity detection
        periodicity = {
            "1_min": detect_periodicity(timestamps, ttft_values, period_minutes=1),
            "5_min": detect_periodicity(timestamps, ttft_values, period_minutes=5),
            "10_min": detect_periodicity(timestamps, ttft_values, period_minutes=10),
            "15_min": detect_periodicity(timestamps, ttft_values, period_minutes=15),
            "60_min": detect_periodicity(timestamps, ttft_values, period_minutes=60),
        }

        # 5. Fixed interval slowdown detection
        fixed_intervals = self._detect_fixed_interval_slowdowns(timestamps, ttft_values)

        # 6. Request count-based throttling detection
        count_throttling = self._detect_count_based_throttling(records)

        return {
            "total_requests": len(records),

            # Minute patterns
            "minute_patterns": minute_patterns,

            # Hourly patterns
            "hourly_patterns": hourly_patterns,

            # Interval patterns
            "interval_patterns": interval_patterns,

            # Periodicity
            "periodicity": periodicity,

            # Fixed intervals
            "fixed_interval_slowdowns": fixed_intervals,

            # Count-based throttling
            "count_based_throttling": count_throttling,

            # Insights
            "insights": self._generate_insights(
                minute_patterns,
                hourly_patterns,
                periodicity,
                fixed_intervals,
                count_throttling,
            ),
        }

    def _analyze_minute_patterns(
        self, timestamps: List[int], ttft_values: List[float]
    ) -> Dict[str, Any]:
        """Analyze patterns by minute of hour."""
        if len(timestamps) != len(ttft_values):
            # Align arrays
            min_len = min(len(timestamps), len(ttft_values))
            timestamps = timestamps[:min_len]
            ttft_values = ttft_values[:min_len]

        # Group by minute of hour
        minute_bins: Dict[int, List[float]] = defaultdict(list)

        for ts, ttft in zip(timestamps, ttft_values):
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            minute = dt.minute
            minute_bins[minute].append(ttft)

        # Calculate stats for each minute
        minute_stats = {}
        for minute in range(60):
            if minute in minute_bins and minute_bins[minute]:
                minute_stats[minute] = {
                    "mean": mean(minute_bins[minute]),
                    "count": len(minute_bins[minute]),
                }

        # Find rhythmic patterns (e.g., every 5 minutes, every 10 minutes)
        rhythmic_patterns = self._find_rhythmic_patterns(minute_stats)

        return {
            "by_minute": minute_stats,
            "rhythmic_patterns": rhythmic_patterns,
        }

    def _find_rhythmic_patterns(self, minute_stats: Dict[int, Dict]) -> List[Dict[str, Any]]:
        """Find patterns that repeat at regular intervals."""
        if not minute_stats:
            return []

        means = {minute: stats['mean'] for minute, stats in minute_stats.items()}

        patterns = []

        # Check for patterns at different intervals
        for interval in [5, 10, 15, 20, 30]:
            if interval > 60:
                continue

            # Check if the pattern at this interval is consistent
            interval_values = []
            for offset in range(interval):
                if offset in means:
                    interval_values.append((offset, means[offset]))

            if len(interval_values) >= 2:
                # Calculate variance of means at this interval
                interval_means = [m for _, m in interval_values]
                interval_var = variance(interval_means)

                # Overall variance
                all_means = list(means.values())
                overall_var = variance(all_means)

                # If variance at this interval is significantly different from overall
                if overall_var > 0:
                    pattern_strength = 1 - (interval_var / overall_var)
                    if pattern_strength > 0.3:
                        patterns.append({
                            "interval_minutes": interval,
                            "pattern_strength": pattern_strength,
                            "interpretation": self._interpret_interval_pattern(interval, pattern_strength),
                        })

        return sorted(patterns, key=lambda x: x['pattern_strength'], reverse=True)

    def _interpret_interval_pattern(self, interval: int, strength: float) -> str:
        """Interpret interval pattern."""
        if strength > 0.6:
            return f"Strong {interval}-minute rhythmic pattern detected. Possible internal scheduling."
        elif strength > 0.4:
            return f"Moderate {interval}-minute pattern detected."
        else:
            return f"Weak {interval}-minute pattern detected."

    def _analyze_hourly_patterns(
        self, timestamps: List[int], ttft_values: List[float]
    ) -> Dict[str, Any]:
        """Analyze patterns by hour of day."""
        if len(timestamps) != len(ttft_values):
            min_len = min(len(timestamps), len(ttft_values))
            timestamps = timestamps[:min_len]
            ttft_values = ttft_values[:min_len]

        # Group by hour of day
        hour_bins: Dict[int, List[float]] = defaultdict(list)

        for ts, ttft in zip(timestamps, ttft_values):
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            hour = dt.hour
            hour_bins[hour].append(ttft)

        # Calculate stats for each hour
        hour_stats = {}
        for hour in range(24):
            if hour in hour_bins and hour_bins[hour]:
                hour_stats[hour] = {
                    "mean": mean(hour_bins[hour]),
                    "std": std_dev(hour_bins[hour]) if len(hour_bins[hour]) > 1 else 0,
                    "count": len(hour_bins[hour]),
                }

        # Find worst hours
        if hour_stats:
            sorted_hours = sorted(hour_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
            worst_hours = sorted_hours[:3]
            best_hours = sorted_hours[-3:]
        else:
            worst_hours = []
            best_hours = []

        return {
            "by_hour": hour_stats,
            "slowest_hours": [{"hour": h, **stats} for h, stats in worst_hours],
            "fastest_hours": [{"hour": h, **stats} for h, stats in best_hours],
        }

    def _analyze_request_intervals(
        self, timestamps: List[int], ttft_values: List[float]
    ) -> Dict[str, Any]:
        """Analyze patterns in request intervals."""
        if len(timestamps) < 2:
            return {"note": "Insufficient data for interval analysis"}

        # Calculate intervals between consecutive requests
        intervals = []
        for i in range(1, len(timestamps)):
            interval_s = (timestamps[i] - timestamps[i-1]) / 1000
            intervals.append(interval_s)

        # Align intervals with TTFT (interval i corresponds to request i+1's TTFT)
        aligned_ttft = ttft_values[1:] if len(ttft_values) > 1 else []

        if not intervals or not aligned_ttft or len(intervals) != len(aligned_ttft):
            return {"note": "Unable to align intervals with TTFT"}

        # Check if shorter intervals correlate with higher TTFT (queue buildup)
        from ..statistics import correlation

        interval_ttft_corr = correlation(intervals, aligned_ttft)

        # Group intervals into buckets
        short_intervals = [ttft for interval, ttft in zip(intervals, aligned_ttft) if interval < 60]
        medium_intervals = [ttft for interval, ttft in zip(intervals, aligned_ttft) if 60 <= interval < 120]
        long_intervals = [ttft for interval, ttft in zip(intervals, aligned_ttft) if interval >= 120]

        return {
            "mean_interval_s": mean(intervals),
            "interval_ttft_correlation": interval_ttft_corr,
            "short_interval_ttft": mean(short_intervals) if short_intervals else 0,
            "medium_interval_ttft": mean(medium_intervals) if medium_intervals else 0,
            "long_interval_ttft": mean(long_intervals) if long_intervals else 0,
            "interpretation": self._interpret_interval_correlation(interval_ttft_corr),
        }

    def _interpret_interval_correlation(self, corr: float) -> str:
        """Interpret interval-TTFT correlation."""
        if corr < -0.3:
            return "Negative correlation: Shorter intervals lead to higher TTFT. Possible queue buildup."
        elif corr > 0.3:
            return "Positive correlation: Shorter intervals lead to lower TTFT. Possible warm cache."
        else:
            return "No significant correlation between interval and TTFT."

    def _detect_fixed_interval_slowdowns(
        self, timestamps: List[int], ttft_values: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect slowdowns that occur at fixed intervals (e.g., every xx:00)."""
        if len(timestamps) != len(ttft_values) or len(timestamps) < 10:
            return []

        # Calculate average TTFT for each minute of hour
        minute_ttft: Dict[int, List[float]] = defaultdict(list)
        for ts, ttft in zip(timestamps, ttft_values):
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            minute = dt.minute
            minute_ttft[minute].append(ttft)

        minute_means = {m: mean(vals) for m, vals in minute_ttft.items()}

        if len(minute_means) < 5:
            return []

        # Find minutes that are consistently slower
        overall_mean = mean(list(minute_means.values()))
        overall_std = std_dev(list(minute_means.values()))

        slowdown_minutes = []
        for minute, mean_ttft in minute_means.items():
            z_score = (mean_ttft - overall_mean) / overall_std if overall_std > 0 else 0
            if z_score > 1.5:  # More than 1.5 std above mean
                slowdown_minutes.append({
                    "minute": minute,
                    "mean_ttft": mean_ttft,
                    "z_score": z_score,
                    "slowdown_factor": mean_ttft / overall_mean if overall_mean > 0 else 1,
                })

        return sorted(slowdown_minutes, key=lambda x: x['slowdown_factor'], reverse=True)

    def _detect_count_based_throttling(
        self, records: List[RunRecord]
    ) -> Dict[str, Any]:
        """Detect patterns that suggest count-based throttling."""
        if len(records) < 10:
            return {"note": "Insufficient data for count-based throttling detection"}

        # Check if errors cluster after certain request counts
        # Look for patterns like: N successful, then 1 error, repeat

        error_positions = [i for i, r in enumerate(records) if r.status != 200]

        if len(error_positions) < 3:
            return {"note": "Not enough errors to detect pattern"}

        # Calculate gaps between errors
        error_gaps = [error_positions[i] - error_positions[i-1] for i in range(1, len(error_positions))]

        # Check if gaps are consistent (suggesting quota-based throttling)
        if len(error_gaps) >= 3:
            from ..statistics import coefficient_of_variation

            gap_cv = coefficient_of_variation(error_gaps)
            mean_gap = mean(error_gaps)

            if gap_cv < 0.3:  # Low variation in gaps
                return {
                    "pattern_detected": True,
                    "pattern_type": "consistent_gap",
                    "mean_gap_requests": mean_gap,
                    "gap_cv": gap_cv,
                    "interpretation": f"Errors occur approximately every {mean_gap:.0f} requests. Possible quota-based throttling.",
                }
            else:
                return {
                    "pattern_detected": False,
                    "gap_cv": gap_cv,
                    "interpretation": "Error gaps are inconsistent, likely not quota-based throttling.",
                }

        return {"note": "Insufficient error data for pattern detection"}

    def _generate_insights(
        self,
        minute_patterns: Dict[str, Any],
        hourly_patterns: Dict[str, Any],
        periodicity: Dict[str, Any],
        fixed_intervals: List[Dict],
        count_throttling: Dict[str, Any],
    ) -> List[str]:
        """Generate rate limiting insights."""
        insights = []

        # Rhythmic patterns
        rhythmic = minute_patterns.get('rhythmic_patterns', [])
        if rhythmic:
            top_pattern = rhythmic[0]
            if top_pattern['pattern_strength'] > 0.5:
                insights.append(f"Detected {top_pattern['interval_minutes']}-minute rhythmic pattern - possible internal scheduling")

        # Periodicity
        for period, result in periodicity.items():
            if result.get('pattern_strength', 0) > 0.4:
                insights.append(f"Strong periodicity detected at {period} intervals")

        # Fixed interval slowdowns
        if fixed_intervals:
            worst = fixed_intervals[0]
            insights.append(f"Consistent slowdown at minute {worst['minute']}:00 - {worst['slowdown_factor']:.1f}x slower")

        # Count-based throttling
        if count_throttling.get('pattern_detected'):
            insights.append(f"Count-based throttling pattern detected: errors every ~{count_throttling['mean_gap_requests']:.0f} requests")

        # Hourly patterns
        slowest = hourly_patterns.get('slowest_hours', [])
        if slowest:
            worst_hour = slowest[0]
            insights.append(f"Slowest hour: {worst_hour['hour']}:00 - {worst_hour['mean']:.0f}ms average TTFT")

        return insights
