"""Time series analysis for performance metrics."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from llmperf.analysis.base_analysis import BaseAnalysis
from llmperf.analysis.record_query import RecordQuery
from llmperf.analysis.analysis_registry import register_analysis
from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)


class TimeSeriesConfig(BaseModel):
    """Configuration for time series analysis."""
    run_id: str = Field(..., min_length=1)
    """Run ID to analyze."""

    metrics: List[str] = Field(
        default_factory=lambda: ["latency", "throughput", "cost", "error_rate"]
    )
    """Metrics to analyze."""

    interval: str = Field(default="1m")
    """Aggregation interval: 1m, 5m, 15m, 1h, 1d."""

    query: RecordQuery = Field(default_factory=RecordQuery)
    """Query configuration."""


class TimeSeriesPoint(BaseModel):
    """A single point in time series."""
    timestamp: datetime
    value: float
    count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    std_dev: Optional[float] = None


class MetricTimeSeries(BaseModel):
    """Time series data for a single metric."""
    metric: str
    interval: str
    data: List[TimeSeriesPoint]
    total_count: int
    avg_value: float
    trend: str = "stable"  # stable, increasing, decreasing


@register_analysis("timeseries")
class TimeSeriesAnalysis(BaseAnalysis[TimeSeriesConfig]):
    """Time series analysis for performance metrics.

    Analyzes how metrics change over time during a run,
    identifying trends and patterns.
    """

    type_name = "timeseries"
    Config = TimeSeriesConfig

    def run(self) -> Dict[str, Any]:
        """Run time series analysis.

        Returns:
            Dictionary with time series data for each metric.
        """
        storage = self.config.query.storage()
        records = list(storage.fetch_run_records(self.config.run_id))

        if not records:
            return {
                "run_id": self.config.run_id,
                "error": "No records found",
                "metrics": {},
            }

        # Parse interval
        interval_seconds = self._parse_interval(self.config.interval)

        # Analyze each metric
        results = {}
        for metric in self.config.metrics:
            ts_data = self._analyze_metric(records, metric, interval_seconds)
            results[metric] = ts_data.model_dump()

        return {
            "run_id": self.config.run_id,
            "interval": self.config.interval,
            "metrics": results,
        }

    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds.

        Args:
            interval: Interval string (e.g., "1m", "5m", "1h").

        Returns:
            Interval in seconds.
        """
        if interval.endswith("s"):
            return int(interval[:-1])
        elif interval.endswith("m"):
            return int(interval[:-1]) * 60
        elif interval.endswith("h"):
            return int(interval[:-1]) * 3600
        elif interval.endswith("d"):
            return int(interval[:-1]) * 86400
        else:
            return 60  # Default to 1 minute

    def _analyze_metric(
        self,
        records: List[RunRecord],
        metric: str,
        interval_seconds: int,
    ) -> MetricTimeSeries:
        """Analyze a single metric over time.

        Args:
            records: List of records.
            metric: Metric name.
            interval_seconds: Interval in seconds.

        Returns:
            MetricTimeSeries with analysis results.
        """
        # Extract values with timestamps
        data_points: List[Tuple[int, float]] = []

        for record in records:
            ts = record.created_at
            value = self._get_metric_value(record, metric)
            if value is not None:
                data_points.append((ts, value))

        if not data_points:
            return MetricTimeSeries(
                metric=metric,
                interval=self.config.interval,
                data=[],
                total_count=0,
                avg_value=0.0,
                trend="stable",
            )

        # Group by time bucket
        min_ts = min(ts for ts, _ in data_points)
        buckets: Dict[int, List[float]] = defaultdict(list)

        for ts, value in data_points:
            bucket = (ts - min_ts) // interval_seconds
            buckets[bucket].append(value)

        # Calculate statistics for each bucket
        points = []
        all_values = []

        for bucket_idx in sorted(buckets.keys()):
            values = buckets[bucket_idx]
            all_values.extend(values)

            ts = min_ts + bucket_idx * interval_seconds
            point = TimeSeriesPoint(
                timestamp=datetime.fromtimestamp(ts),
                value=statistics.mean(values),
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            )
            points.append(point)

        # Calculate trend
        trend = self._calculate_trend(points)

        return MetricTimeSeries(
            metric=metric,
            interval=self.config.interval,
            data=points,
            total_count=len(data_points),
            avg_value=statistics.mean(all_values) if all_values else 0.0,
            trend=trend,
        )

    def _get_metric_value(self, record: RunRecord, metric: str) -> Optional[float]:
        """Get metric value from record.

        Args:
            record: The record.
            metric: Metric name.

        Returns:
            Metric value, or None if not applicable.
        """
        if metric == "latency":
            return float(record.first_resp_time) if record.first_resp_time > 0 else None
        elif metric == "throughput":
            return float(record.token_throughput) if record.token_throughput > 0 else None
        elif metric == "cost":
            return float(record.total_cost)
        elif metric == "error_rate":
            return 1.0 if record.status != 200 else 0.0
        elif metric == "char_per_second":
            return float(record.char_per_second) if record.char_per_second > 0 else None
        elif metric == "token_per_second":
            return float(record.token_per_second) if record.token_per_second > 0 else None
        else:
            return None

    def _calculate_trend(self, points: List[TimeSeriesPoint]) -> str:
        """Calculate trend direction from time series points.

        Args:
            points: List of time series points.

        Returns:
            Trend string: "increasing", "decreasing", or "stable".
        """
        if len(points) < 3:
            return "stable"

        # Use linear regression slope
        n = len(points)
        x_sum = sum(range(n))
        y_sum = sum(p.value for p in points)
        xy_sum = sum(i * points[i].value for i in range(n))
        xx_sum = sum(i * i for i in range(n))

        slope = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum)

        # Determine trend based on slope
        avg_value = y_sum / n
        relative_slope = abs(slope) / avg_value if avg_value > 0 else 0

        if relative_slope < 0.01:  # Less than 1% change
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
