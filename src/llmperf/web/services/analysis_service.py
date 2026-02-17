"""Analysis service for data processing and visualization."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from llmperf.config.runtime import load_runtime_config
from llmperf.records.storage import Storage
from llmperf.export.base import ExportConfig
from llmperf.export.registry import create_exporter

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for data analysis and export.

    This service provides:
    - Summary statistics
    - Time series analysis
    - Model comparison
    - Anomaly detection
    - Data export in multiple formats
    """

    def __init__(self):
        """Initialize analysis service."""
        runtime = load_runtime_config()
        self._db_path = str(runtime.db_path)
        self._storage = Storage(self._db_path)

    def get_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a run.

        Args:
            run_id: The run ID.

        Returns:
            Summary statistics dictionary, or None if no data.
        """
        records = list(self._storage.fetch_run_records(run_id))

        if not records:
            return None

        total = len(records)
        successful = [r for r in records if r.status == 200]
        failed = [r for r in records if r.status != 200]

        # Group by executor
        by_executor: Dict[str, List] = defaultdict(list)
        for r in records:
            by_executor[r.executor_id].append(r)

        # Calculate metrics
        first_resp_times = [r.first_resp_time for r in successful if r.first_resp_time > 0]
        char_per_sec = [r.char_per_second for r in successful if r.char_per_second > 0]
        token_throughput = [r.token_throughput for r in successful if r.token_throughput > 0]

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            k = int(len(sorted_data) * p)
            return sorted_data[min(k, len(sorted_data) - 1)]

        # Per-executor summary
        executor_summaries = {}
        for executor_id, exec_records in by_executor.items():
            exec_successful = [r for r in exec_records if r.status == 200]
            exec_first_resp = [r.first_resp_time for r in exec_successful if r.first_resp_time > 0]

            executor_summaries[executor_id] = {
                "total_requests": len(exec_records),
                "success_count": len(exec_successful),
                "error_count": len(exec_records) - len(exec_successful),
                "success_rate": len(exec_successful) / len(exec_records) * 100 if exec_records else 0,
                "total_cost": sum(r.total_cost for r in exec_records),
                "avg_first_resp_time": avg(exec_first_resp),
                "p95_first_resp_time": percentile(exec_first_resp, 0.95),
            }

        return {
            "run_id": run_id,
            "total_requests": total,
            "success_count": len(successful),
            "error_count": len(failed),
            "success_rate": len(successful) / total * 100 if total > 0 else 0,
            "total_cost": sum(r.total_cost for r in records),
            "currency": records[0].currency if records else "CNY",
            "avg_first_resp_time": avg(first_resp_times),
            "p50_first_resp_time": percentile(first_resp_times, 0.50),
            "p95_first_resp_time": percentile(first_resp_times, 0.95),
            "p99_first_resp_time": percentile(first_resp_times, 0.99),
            "avg_char_per_second": avg(char_per_sec),
            "avg_token_throughput": avg(token_throughput),
            "total_input_tokens": sum(r.qtokens for r in records),
            "total_output_tokens": sum(r.atokens for r in records),
            "total_cached_tokens": sum(r.ctokens for r in records),
            "by_executor": executor_summaries,
        }

    def get_timeseries(
        self,
        run_id: str,
        metric: str,
        interval: str = "1m",
    ) -> List[Dict[str, Any]]:
        """Get time series data for a metric.

        Args:
            run_id: The run ID.
            metric: Metric name (latency, throughput, cost, error_rate).
            interval: Aggregation interval (1m, 5m, 15m, 1h).

        Returns:
            List of time series points.
        """
        records = list(self._storage.fetch_run_records(run_id))

        if not records:
            return []

        # Parse interval
        interval_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
        }.get(interval, 1)

        # Group records by time bucket
        buckets: Dict[int, List] = defaultdict(list)

        for record in records:
            # Use created_at timestamp
            ts = record.created_at
            bucket = (ts // (interval_minutes * 60)) * (interval_minutes * 60)
            buckets[bucket].append(record)

        # Calculate metric for each bucket
        result = []
        for bucket_ts in sorted(buckets.keys()):
            bucket_records = buckets[bucket_ts]

            if metric == "latency":
                values = [r.first_resp_time for r in bucket_records if r.first_resp_time > 0]
                value = sum(values) / len(values) if values else 0
            elif metric == "throughput":
                values = [r.token_throughput for r in bucket_records if r.token_throughput > 0]
                value = sum(values) / len(values) if values else 0
            elif metric == "cost":
                value = sum(r.total_cost for r in bucket_records)
            elif metric == "error_rate":
                total = len(bucket_records)
                errors = sum(1 for r in bucket_records if r.status != 200)
                value = errors / total * 100 if total > 0 else 0
            else:
                value = 0

            result.append({
                "timestamp": datetime.fromtimestamp(bucket_ts).isoformat(),
                "value": value,
            })

        return result

    def compare_executors(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Compare performance across executors.

        Args:
            run_id: The run ID.

        Returns:
            Comparison data dictionary, or None if no data.
        """
        records = list(self._storage.fetch_run_records(run_id))

        if not records:
            return None

        # Group by (executor_id, provider, model)
        groups: Dict[tuple, List] = defaultdict(list)
        for r in records:
            key = (r.executor_id, r.provider, r.model)
            groups[key].append(r)

        items = []
        for (executor_id, provider, model), group_records in groups.items():
            successful = [r for r in group_records if r.status == 200]
            total = len(group_records)

            first_resp_times = [r.first_resp_time for r in successful if r.first_resp_time > 0]
            char_per_sec = [r.char_per_second for r in successful if r.char_per_second > 0]

            def avg(lst):
                return sum(lst) / len(lst) if lst else 0

            items.append({
                "executor_id": executor_id,
                "provider": provider,
                "model": model,
                "metrics": {
                    "total_requests": total,
                    "success_rate": len(successful) / total * 100 if total > 0 else 0,
                    "avg_first_resp_time": avg(first_resp_times),
                    "avg_char_per_second": avg(char_per_sec),
                    "total_cost": sum(r.total_cost for r in group_records),
                },
            })

        # Calculate ranking based on success rate and latency
        def score(item):
            m = item["metrics"]
            return m["success_rate"] - m["avg_first_resp_time"] / 1000

        ranking = sorted(items, key=score, reverse=True)
        ranking_ids = [item["executor_id"] for item in ranking]

        return {
            "items": items,
            "ranking": ranking_ids,
        }

    def detect_anomalies(
        self,
        run_id: str,
        sensitivity: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in run data.

        Args:
            run_id: The run ID.
            sensitivity: Z-score threshold for anomaly detection.

        Returns:
            List of detected anomalies.
        """
        records = list(self._storage.fetch_run_records(run_id))
        anomalies = []

        if len(records) < 10:
            return anomalies

        # Check latency anomalies
        latencies = [r.first_resp_time for r in records if r.first_resp_time > 0]
        if latencies:
            mean = statistics.mean(latencies)
            stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0

            if stdev > 0:
                for record in records:
                    if record.first_resp_time > 0:
                        z_score = abs(record.first_resp_time - mean) / stdev
                        if z_score > sensitivity:
                            anomalies.append({
                                "type": "high_latency",
                                "record_id": record.dataset_row_id,
                                "value": record.first_resp_time,
                                "expected": mean,
                                "z_score": z_score,
                                "severity": "high" if z_score > 3 else "medium",
                            })

        # Check error patterns
        error_statuses = defaultdict(int)
        for record in records:
            if record.status != 200:
                error_statuses[record.status] += 1

        for status, count in error_statuses.items():
            error_rate = count / len(records)
            if error_rate > 0.1:  # More than 10% error rate
                anomalies.append({
                    "type": "high_error_rate",
                    "status_code": status,
                    "count": count,
                    "rate": error_rate,
                    "severity": "high" if error_rate > 0.3 else "medium",
                })

        return anomalies

    def export_data(
        self,
        run_id: str,
        format_name: str,
        output_dir: str = ".",
    ) -> Any:
        """Export run data in specified format.

        Args:
            run_id: The run ID.
            format_name: Export format (csv, jsonl, json, html).
            output_dir: Output directory.

        Returns:
            ExportResult.
        """
        records = list(self._storage.fetch_run_records(run_id))

        if not records:
            from llmperf.export.base import ExportResult
            return ExportResult(
                success=False,
                format=format_name,
                message="No records to export",
            )

        config = ExportConfig(output_dir=output_dir)
        exporter = create_exporter(format_name, config)

        if not exporter:
            from llmperf.export.base import ExportResult
            return ExportResult(
                success=False,
                format=format_name,
                message=f"Unknown export format: {format_name}",
            )

        return exporter.export(
            records=records,
            run_id=run_id,
            task_name=f"run-{run_id[:8]}",
        )

    def get_history(
        self,
        limit: int = 20,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """Get historical runs.

        Args:
            limit: Maximum number of runs.
            days: Number of days to look back.

        Returns:
            List of historical run summaries.
        """
        # Get runs from database
        runs = self._storage.list_runs(limit=limit)

        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        filtered_runs = []

        for run in runs:
            # Run is a tuple or dict depending on storage implementation
            if isinstance(run, dict):
                created_at = run.get("created_at")
                if created_at:
                    if isinstance(created_at, int):
                        created_at = datetime.fromtimestamp(created_at)
                    if created_at >= cutoff:
                        filtered_runs.append(run)
            else:
                filtered_runs.append(run)

        return filtered_runs[:limit]
