"""Anomaly detection analysis for identifying unusual patterns."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from llmperf.analysis.base_analysis import BaseAnalysis
from llmperf.analysis.record_query import RecordQuery
from llmperf.analysis.analysis_registry import register_analysis
from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)


class AnomalyConfig(BaseModel):
    """Configuration for anomaly detection."""
    run_id: str = Field(..., min_length=1)
    """Run ID to analyze."""

    sensitivity: float = Field(default=2.0, ge=0.5, le=5.0)
    """Z-score threshold for anomaly detection."""

    min_sample_size: int = Field(default=10, ge=5)
    """Minimum sample size for statistical analysis."""

    check_metrics: List[str] = Field(
        default_factory=lambda: ["latency", "throughput", "error_rate", "cost"]
    )
    """Metrics to check for anomalies."""

    query: RecordQuery = Field(default_factory=RecordQuery)
    """Query configuration."""


class Anomaly(BaseModel):
    """Detected anomaly."""
    anomaly_type: str
    severity: str  # low, medium, high, critical
    executor_id: Optional[str] = None
    record_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    expected_value: Optional[float] = None
    deviation: Optional[float] = None
    description: str
    metadata: Dict[str, Any] = {}


class AnomalyReport(BaseModel):
    """Complete anomaly report."""
    run_id: str
    analysis_time: datetime
    total_records: int
    anomaly_count: int
    anomalies: List[Anomaly]
    summary: Dict[str, int]
    recommendations: List[str]


@register_analysis("anomaly")
class AnomalyDetectionAnalysis(BaseAnalysis[AnomalyConfig]):
    """Detect anomalies in run data.

    Uses statistical methods to identify:
    - Outlier latency values
    - Unusual error patterns
    - Cost anomalies
    - Performance degradation
    """

    type_name = "anomaly"
    Config = AnomalyConfig

    def run(self) -> Dict[str, Any]:
        """Run anomaly detection analysis.

        Returns:
            Dictionary with anomaly report.
        """
        storage = self.config.query.storage()
        records = list(storage.fetch_run_records(self.config.run_id))

        anomalies: List[Anomaly] = []

        if len(records) < self.config.min_sample_size:
            return {
                "run_id": self.config.run_id,
                "error": f"Insufficient data (need at least {self.config.min_sample_size} records)",
                "anomalies": [],
            }

        # Run various anomaly checks
        anomalies.extend(self._check_latency_anomalies(records))
        anomalies.extend(self._check_error_anomalies(records))
        anomalies.extend(self._check_cost_anomalies(records))
        anomalies.extend(self._check_throughput_anomalies(records))
        anomalies.extend(self._check_pattern_anomalies(records))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 4))

        # Generate summary
        summary = defaultdict(int)
        for anomaly in anomalies:
            summary[anomaly.anomaly_type] += 1

        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies)

        report = AnomalyReport(
            run_id=self.config.run_id,
            analysis_time=datetime.now(),
            total_records=len(records),
            anomaly_count=len(anomalies),
            anomalies=anomalies,
            summary=dict(summary),
            recommendations=recommendations,
        )

        return report.model_dump()

    def _check_latency_anomalies(self, records: List[RunRecord]) -> List[Anomaly]:
        """Check for latency anomalies using z-score method.

        Args:
            records: List of records.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        # Group by executor for more accurate baseline
        by_executor: Dict[str, List[RunRecord]] = defaultdict(list)
        for r in records:
            by_executor[r.executor_id].append(r)

        for executor_id, exec_records in by_executor.items():
            successful = [r for r in exec_records if r.status == 200 and r.first_resp_time > 0]

            if len(successful) < self.config.min_sample_size:
                continue

            latencies = [r.first_resp_time for r in successful]
            mean = statistics.mean(latencies)
            stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0

            if stdev == 0:
                continue

            for record in successful:
                z_score = abs(record.first_resp_time - mean) / stdev

                if z_score > self.config.sensitivity:
                    severity = self._get_severity(z_score)

                    anomalies.append(Anomaly(
                        anomaly_type="latency_outlier",
                        severity=severity,
                        executor_id=executor_id,
                        record_id=record.dataset_row_id,
                        timestamp=datetime.fromtimestamp(record.created_at),
                        metric="latency",
                        value=record.first_resp_time,
                        expected_value=mean,
                        deviation=z_score,
                        description=f"Latency {record.first_resp_time:.0f}ms is {z_score:.1f} standard deviations from mean ({mean:.0f}ms)",
                        metadata={
                            "z_score": z_score,
                            "mean_latency": mean,
                            "std_dev": stdev,
                        },
                    ))

        return anomalies

    def _check_error_anomalies(self, records: List[RunRecord]) -> List[Anomaly]:
        """Check for unusual error patterns.

        Args:
            records: List of records.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        total = len(records)
        errors = [r for r in records if r.status != 200]
        error_count = len(errors)
        error_rate = error_count / total if total > 0 else 0

        # Check overall error rate
        if error_rate > 0.5:
            anomalies.append(Anomaly(
                anomaly_type="high_error_rate",
                severity="critical",
                metric="error_rate",
                value=error_rate * 100,
                description=f"Very high error rate: {error_rate:.1%} ({error_count}/{total} requests)",
                metadata={"error_count": error_count, "total": total},
            ))
        elif error_rate > 0.3:
            anomalies.append(Anomaly(
                anomaly_type="high_error_rate",
                severity="high",
                metric="error_rate",
                value=error_rate * 100,
                description=f"High error rate: {error_rate:.1%} ({error_count}/{total} requests)",
                metadata={"error_count": error_count, "total": total},
            ))
        elif error_rate > 0.1:
            anomalies.append(Anomaly(
                anomaly_type="elevated_error_rate",
                severity="medium",
                metric="error_rate",
                value=error_rate * 100,
                description=f"Elevated error rate: {error_rate:.1%}",
                metadata={"error_count": error_count, "total": total},
            ))

        # Check for specific error codes
        error_codes: Dict[int, List[RunRecord]] = defaultdict(list)
        for r in errors:
            error_codes[r.status].append(r)

        for status, error_records in error_codes.items():
            code_rate = len(error_records) / total

            if code_rate > 0.05:  # More than 5% with same error code
                anomalies.append(Anomaly(
                    anomaly_type="repeated_error_code",
                    severity="high" if code_rate > 0.2 else "medium",
                    metric="error_rate",
                    value=code_rate * 100,
                    description=f"Repeated error code {status}: {len(error_records)} occurrences ({code_rate:.1%})",
                    metadata={"status_code": status, "count": len(error_records)},
                ))

        return anomalies

    def _check_cost_anomalies(self, records: List[RunRecord]) -> List[Anomaly]:
        """Check for cost anomalies.

        Args:
            records: List of records.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        costs = [r.total_cost for r in records]
        if not costs or all(c == 0 for c in costs):
            return anomalies

        mean_cost = statistics.mean(costs)
        stdev = statistics.stdev(costs) if len(costs) > 1 else 0

        if stdev == 0:
            return anomalies

        # Check for individual request cost outliers
        for record in records:
            if record.total_cost == 0:
                continue

            z_score = abs(record.total_cost - mean_cost) / stdev

            if z_score > self.config.sensitivity * 1.5:  # Higher threshold for cost
                anomalies.append(Anomaly(
                    anomaly_type="cost_outlier",
                    severity="medium",
                    executor_id=record.executor_id,
                    record_id=record.dataset_row_id,
                    metric="cost",
                    value=record.total_cost,
                    expected_value=mean_cost,
                    deviation=z_score,
                    description=f"Request cost {record.total_cost:.4f} is unusually high (expected ~{mean_cost:.4f})",
                    metadata={"z_score": z_score},
                ))

        return anomalies

    def _check_throughput_anomalies(self, records: List[RunRecord]) -> List[Anomaly]:
        """Check for throughput anomalies.

        Args:
            records: List of records.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        successful = [r for r in records if r.status == 200 and r.token_throughput > 0]

        if len(successful) < self.config.min_sample_size:
            return anomalies

        throughputs = [r.token_throughput for r in successful]
        mean = statistics.mean(throughputs)
        stdev = statistics.stdev(throughputs) if len(throughputs) > 1 else 0

        if stdev == 0:
            return anomalies

        # Check for unusually low throughput (performance degradation)
        for record in successful:
            if record.token_throughput < mean - self.config.sensitivity * stdev:
                deviation = (mean - record.token_throughput) / stdev

                anomalies.append(Anomaly(
                    anomaly_type="low_throughput",
                    severity="medium" if deviation < 3 else "high",
                    executor_id=record.executor_id,
                    record_id=record.dataset_row_id,
                    metric="throughput",
                    value=record.token_throughput,
                    expected_value=mean,
                    deviation=deviation,
                    description=f"Low throughput {record.token_throughput:.1f} tok/s (expected ~{mean:.1f})",
                    metadata={"deviation": deviation},
                ))

        return anomalies

    def _check_pattern_anomalies(self, records: List[RunRecord]) -> List[Anomaly]:
        """Check for unusual patterns in the data.

        Args:
            records: List of records.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        # Check for time-based patterns
        sorted_records = sorted(records, key=lambda r: r.created_at)

        # Detect burst errors
        window_size = 10
        for i in range(len(sorted_records) - window_size):
            window = sorted_records[i:i + window_size]
            error_count = sum(1 for r in window if r.status != 200)

            if error_count >= window_size * 0.8:  # 80% errors in window
                anomalies.append(Anomaly(
                    anomaly_type="error_burst",
                    severity="high",
                    timestamp=datetime.fromtimestamp(window[0].created_at),
                    description=f"Error burst detected: {error_count}/{window_size} consecutive errors",
                    metadata={"window_start": i, "error_count": error_count},
                ))

        return anomalies

    def _get_severity(self, z_score: float) -> str:
        """Determine anomaly severity based on z-score.

        Args:
            z_score: The z-score value.

        Returns:
            Severity string.
        """
        if z_score > 4:
            return "critical"
        elif z_score > 3:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, anomalies: List[Anomaly]) -> List[str]:
        """Generate recommendations based on detected anomalies.

        Args:
            anomalies: List of detected anomalies.

        Returns:
            List of recommendation strings.
        """
        recommendations = []
        seen_types: Set[str] = set()

        for anomaly in anomalies:
            if anomaly.anomaly_type in seen_types:
                continue
            seen_types.add(anomaly.anomaly_type)

            if anomaly.anomaly_type == "high_error_rate":
                recommendations.append(
                    "Investigate API errors - check API status, rate limits, and credentials"
                )
            elif anomaly.anomaly_type == "latency_outlier":
                recommendations.append(
                    "Review slow requests - consider reducing prompt length or enabling caching"
                )
            elif anomaly.anomaly_type == "low_throughput":
                recommendations.append(
                    "Check for network issues or API throttling affecting throughput"
                )
            elif anomaly.anomaly_type == "cost_outlier":
                recommendations.append(
                    "Review high-cost requests - consider optimizing prompts or response length limits"
                )
            elif anomaly.anomaly_type == "error_burst":
                recommendations.append(
                    "Investigate transient issues - implement retry logic with exponential backoff"
                )
            elif anomaly.anomaly_type == "repeated_error_code":
                recommendations.append(
                    f"Address recurring error code {anomaly.metadata.get('status_code')} - check API documentation"
                )

        return recommendations
