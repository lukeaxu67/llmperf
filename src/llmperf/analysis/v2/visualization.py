"""Visualization data generator for charts and graphs."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)


class ChartDataPoint(BaseModel):
    """A single data point for a chart."""
    label: str
    value: float
    extra: Dict[str, Any] = Field(default_factory=dict)


class ChartSeries(BaseModel):
    """A series of data for a chart."""
    name: str
    data: List[ChartDataPoint]
    color: Optional[str] = None


class ChartData(BaseModel):
    """Complete chart data structure."""
    chart_type: str
    title: str
    x_axis_label: str = ""
    y_axis_label: str = ""
    series: List[ChartSeries]
    options: Dict[str, Any] = Field(default_factory=dict)


class ChartDataGenerator:
    """Generate chart-ready data from records.

    This class transforms record data into formats suitable
    for various charting libraries (ECharts, Chart.js, etc.).
    """

    # Default color palette
    COLORS = [
        "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
        "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#48b8d0",
    ]

    def __init__(self, records: List[RunRecord]):
        """Initialize generator with records.

        Args:
            records: List of records to generate charts from.
        """
        self.records = records

    def generate_latency_distribution(self) -> ChartData:
        """Generate latency distribution histogram data.

        Returns:
            ChartData for histogram.
        """
        latencies = [r.first_resp_time for r in self.records if r.first_resp_time > 0]

        if not latencies:
            return ChartData(
                chart_type="histogram",
                title="Latency Distribution",
                x_axis_label="Latency (ms)",
                y_axis_label="Count",
                series=[],
            )

        # Create histogram bins
        min_val = min(latencies)
        max_val = max(latencies)
        bin_count = 20
        bin_width = (max_val - min_val) / bin_count if max_val > min_val else 1

        bins = defaultdict(int)
        for lat in latencies:
            bin_idx = min(int((lat - min_val) / bin_width), bin_count - 1)
            bin_start = min_val + bin_idx * bin_width
            bin_label = f"{bin_start:.0f}-{bin_start + bin_width:.0f}"
            bins[bin_label] += 1

        data_points = [
            ChartDataPoint(label=label, value=count)
            for label, count in sorted(bins.items(), key=lambda x: float(x[0].split("-")[0]))
        ]

        return ChartData(
            chart_type="bar",
            title="Latency Distribution",
            x_axis_label="Latency (ms)",
            y_axis_label="Count",
            series=[ChartSeries(
                name="Requests",
                data=data_points,
                color=self.COLORS[0],
            )],
            options={
                "bins": bin_count,
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
            },
        )

    def generate_success_rate_by_executor(self) -> ChartData:
        """Generate success rate pie chart by executor.

        Returns:
            ChartData for pie chart.
        """
        by_executor: Dict[str, Tuple[int, int]] = {}

        for r in self.records:
            if r.executor_id not in by_executor:
                by_executor[r.executor_id] = [0, 0]
            by_executor[r.executor_id][0] += 1
            if r.status == 200:
                by_executor[r.executor_id][1] += 1

        data_points = []
        for i, (executor_id, (total, success)) in enumerate(by_executor.items()):
            data_points.append(ChartDataPoint(
                label=executor_id,
                value=success / total * 100 if total > 0 else 0,
                extra={"total": total, "success": success},
            ))

        return ChartData(
            chart_type="pie",
            title="Success Rate by Executor",
            series=[ChartSeries(
                name="Success Rate",
                data=data_points,
            )],
            options={"show_percentage": True},
        )

    def generate_cost_breakdown(self) -> ChartData:
        """Generate cost breakdown by executor.

        Returns:
            ChartData for bar chart.
        """
        by_executor: Dict[str, float] = defaultdict(float)
        currency = "CNY"

        for r in self.records:
            by_executor[r.executor_id] += r.total_cost
            if r.currency:
                currency = r.currency

        data_points = [
            ChartDataPoint(label=executor_id, value=cost)
            for executor_id, cost in sorted(by_executor.items(), key=lambda x: -x[1])
        ]

        return ChartData(
            chart_type="bar",
            title="Cost by Executor",
            x_axis_label="Executor",
            y_axis_label=f"Cost ({currency})",
            series=[ChartSeries(
                name="Total Cost",
                data=data_points,
                color=self.COLORS[3],
            )],
        )

    def generate_throughput_comparison(self) -> ChartData:
        """Generate throughput comparison chart.

        Returns:
            ChartData for bar chart.
        """
        by_executor: Dict[str, List[float]] = defaultdict(list)

        for r in self.records:
            if r.status == 200 and r.token_throughput > 0:
                by_executor[r.executor_id].append(r.token_throughput)

        data_points = []
        for executor_id, throughputs in sorted(by_executor.items()):
            avg = statistics.mean(throughputs) if throughputs else 0
            data_points.append(ChartDataPoint(
                label=executor_id,
                value=avg,
                extra={
                    "min": min(throughputs) if throughputs else 0,
                    "max": max(throughputs) if throughputs else 0,
                    "count": len(throughputs),
                },
            ))

        return ChartData(
            chart_type="bar",
            title="Average Throughput by Executor",
            x_axis_label="Executor",
            y_axis_label="Tokens/second",
            series=[ChartSeries(
                name="Throughput",
                data=data_points,
                color=self.COLORS[1],
            )],
        )

    def generate_time_series(
        self,
        metric: str = "latency",
        interval_minutes: int = 1,
    ) -> ChartData:
        """Generate time series chart data.

        Args:
            metric: Metric to plot (latency, throughput, cost, error_rate).
            interval_minutes: Aggregation interval.

        Returns:
            ChartData for line chart.
        """
        if not self.records:
            return ChartData(
                chart_type="line",
                title=f"{metric.title()} Over Time",
                x_axis_label="Time",
                y_axis_label=metric.title(),
                series=[],
            )

        # Sort by timestamp
        sorted_records = sorted(self.records, key=lambda r: r.created_at)
        min_ts = sorted_records[0].created_at

        # Group by time bucket
        interval_seconds = interval_minutes * 60
        buckets: Dict[int, List[RunRecord]] = defaultdict(list)

        for r in sorted_records:
            bucket = (r.created_at - min_ts) // interval_seconds
            buckets[bucket].append(r)

        # Calculate metric for each bucket
        data_points = []
        for bucket_idx in sorted(buckets.keys()):
            bucket_records = buckets[bucket_idx]
            ts = min_ts + bucket_idx * interval_seconds

            if metric == "latency":
                values = [r.first_resp_time for r in bucket_records if r.first_resp_time > 0]
                value = statistics.mean(values) if values else 0
            elif metric == "throughput":
                values = [r.token_throughput for r in bucket_records if r.token_throughput > 0]
                value = statistics.mean(values) if values else 0
            elif metric == "cost":
                value = sum(r.total_cost for r in bucket_records)
            elif metric == "error_rate":
                total = len(bucket_records)
                errors = sum(1 for r in bucket_records if r.status != 200)
                value = errors / total * 100 if total > 0 else 0
            else:
                value = 0

            data_points.append(ChartDataPoint(
                label=datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
                value=value,
                extra={"timestamp": ts},
            ))

        y_labels = {
            "latency": "Latency (ms)",
            "throughput": "Tokens/second",
            "cost": "Cost",
            "error_rate": "Error Rate (%)",
        }

        return ChartData(
            chart_type="line",
            title=f"{metric.replace('_', ' ').title()} Over Time",
            x_axis_label="Time",
            y_axis_label=y_labels.get(metric, metric),
            series=[ChartSeries(
                name=metric.title(),
                data=data_points,
                color=self.COLORS[0],
            )],
            options={"interval_minutes": interval_minutes},
        )

    def generate_model_comparison(self) -> ChartData:
        """Generate multi-metric model comparison radar chart.

        Returns:
            ChartData for radar chart.
        """
        # Group by executor
        by_executor: Dict[str, Dict[str, float]] = {}

        for r in self.records:
            if r.executor_id not in by_executor:
                by_executor[r.executor_id] = {
                    "requests": 0,
                    "success": 0,
                    "latency_sum": 0,
                    "latency_count": 0,
                    "throughput_sum": 0,
                    "throughput_count": 0,
                    "cost": 0,
                }

            stats = by_executor[r.executor_id]
            stats["requests"] += 1
            if r.status == 200:
                stats["success"] += 1
                if r.first_resp_time > 0:
                    stats["latency_sum"] += r.first_resp_time
                    stats["latency_count"] += 1
                if r.token_throughput > 0:
                    stats["throughput_sum"] += r.token_throughput
                    stats["throughput_count"] += 1
            stats["cost"] += r.total_cost

        # Normalize metrics (0-100 scale)
        series_list = []
        metrics = ["Success Rate", "Speed", "Cost Efficiency", "Reliability"]

        for i, (executor_id, stats) in enumerate(by_executor.items()):
            success_rate = stats["success"] / stats["requests"] * 100 if stats["requests"] > 0 else 0

            avg_latency = stats["latency_sum"] / stats["latency_count"] if stats["latency_count"] > 0 else 0
            # Invert latency (lower is better) and scale to 0-100
            speed_score = max(0, 100 - avg_latency / 10)

            cost = stats["cost"]
            # Invert cost (lower is better)
            cost_efficiency = max(0, 100 - cost * 10)

            reliability = success_rate  # Same as success rate for simplicity

            series_list.append(ChartSeries(
                name=executor_id,
                data=[
                    ChartDataPoint(label="Success Rate", value=success_rate),
                    ChartDataPoint(label="Speed", value=speed_score),
                    ChartDataPoint(label="Cost Efficiency", value=cost_efficiency),
                    ChartDataPoint(label="Reliability", value=reliability),
                ],
                color=self.COLORS[i % len(self.COLORS)],
            ))

        return ChartData(
            chart_type="radar",
            title="Model Comparison",
            series=series_list,
            options={"metrics": metrics},
        )

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate all chart data for a dashboard.

        Returns:
            Dictionary with all chart configurations.
        """
        return {
            "latency_distribution": self.generate_latency_distribution().model_dump(),
            "success_rate": self.generate_success_rate_by_executor().model_dump(),
            "cost_breakdown": self.generate_cost_breakdown().model_dump(),
            "throughput": self.generate_throughput_comparison().model_dump(),
            "latency_over_time": self.generate_time_series("latency").model_dump(),
            "error_rate_over_time": self.generate_time_series("error_rate").model_dump(),
            "model_comparison": self.generate_model_comparison().model_dump(),
        }
