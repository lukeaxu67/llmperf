"""
Comprehensive monitoring report for 24-hour LLM performance analysis.

Generates a detailed HTML/Excel report containing:
1. Service Stability Report
2. Scheduling Behavior Report
3. Model Output Stability Report
4. Executive Summary
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery
from ..analysis_registry import register_analysis
from ..statistics import percentile, mean
from ..timeseries import get_hour_of_day, BEIJING_TZ

from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage


@register_analysis("monitoring_report")
class MonitoringReportAnalysis(BaseAnalysis["MonitoringReportAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        query: RecordQuery = Field(default_factory=RecordQuery)
        output_dir: str = Field(default=".")
        report_name: str = Field(default="monitoring_report")
        include_html: bool = Field(default=True)
        include_excel: bool = Field(default=True)
        segment_size: int = Field(default=6, description="Hours per time segment")

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

        # Group by provider/model
        grouped: Dict[tuple[str, str], List[RunRecord]] = defaultdict(list)
        for rec in records:
            grouped[(rec.provider, rec.model)].append(rec)

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d-%H%M%S")
        report_base_name = f"{self.config.report_name}-{timestamp}"

        results = {
            "report_name": report_base_name,
            "generated_at": datetime.now(BEIJING_TZ).isoformat(),
            "total_records": len(records),
            "by_provider_model": {},
        }

        for (provider, model), recs in grouped.items():
            key = f"{provider}.{model}"
            results["by_provider_model"][key] = self._generate_report(
                provider, model, recs, output_path, report_base_name
            )

        # Save JSON summary
        json_path = output_path / f"{report_base_name}.json"
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        return {"output_path": str(json_path), "reports": results}

    def _generate_report(
        self,
        provider: str,
        model: str,
        records: List[RunRecord],
        output_path: Path,
        base_name: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive report for a single provider/model."""

        # Run all analyses
        stability_data = self._compute_stability_metrics(records)
        health_data = self._compute_health_metrics(records)
        resource_data = self._compute_resource_metrics(records)
        temporal_data = self._compute_temporal_metrics(records)

        # Generate executive summary
        summary = self._generate_executive_summary(
            stability_data, health_data, resource_data, temporal_data
        )

        report_data = {
            "provider": provider,
            "model": model,
            "total_requests": len(records),
            "time_range": self._get_time_range(records),
            "executive_summary": summary,
            "stability": stability_data,
            "health": health_data,
            "resource_behavior": resource_data,
            "temporal_patterns": temporal_data,
        }

        # Generate HTML report
        if self.config.include_html:
            html_path = output_path / f"{base_name}-{provider}-{model}.html"
            html_content = self._generate_html_report(report_data)
            html_path.write_text(html_content, encoding="utf-8")
            report_data["html_path"] = str(html_path)

        return report_data

    def _compute_stability_metrics(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Compute stability metrics."""
        from ..statistics import std_dev
        successful = [r for r in records if r.status == 200 and r.is_valid()]

        ttft_values = [float(r.first_resp_time) for r in successful if r.first_resp_time > 0]
        throughput_values = [float(r.token_throughput) for r in successful if r.token_throughput > 0]
        session_times = [float(r.session_time) for r in successful if r.session_time > 0]

        return {
            "request_count": len(records),
            "success_count": len(successful),
            "success_rate": len(successful) / len(records) if records else 0,
            "ttft": {
                "mean": mean(ttft_values) if ttft_values else 0,
                "std": std_dev(ttft_values) if ttft_values else 0,
                "p50": percentile(ttft_values, 0.5) if ttft_values else 0,
                "p90": percentile(ttft_values, 0.9) if ttft_values else 0,
                "p95": percentile(ttft_values, 0.95) if ttft_values else 0,
                "p99": percentile(ttft_values, 0.99) if ttft_values else 0,
                "min": min(ttft_values) if ttft_values else 0,
                "max": max(ttft_values) if ttft_values else 0,
            },
            "throughput": {
                "mean": mean(throughput_values) if throughput_values else 0,
                "p95": percentile(throughput_values, 0.95) if throughput_values else 0,
            },
            "session_time": {
                "mean": mean(session_times) if session_times else 0,
                "p95": percentile(session_times, 0.95) if session_times else 0,
            },
        }

    def _compute_health_metrics(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Compute health metrics."""
        errors = [r for r in records if r.status != 200]
        timeouts = [r for r in errors if r.status == 0 or 'timeout' in (r.info or "").lower()]
        server_errors = [r for r in errors if 500 <= r.status < 600]

        output_lengths = [int(r.atokens) for r in records if r.atokens > 0]
        empty_responses = [r for r in records if r.status == 200 and r.atokens == 0]

        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(records) if records else 0,
            "timeout_count": len(timeouts),
            "server_error_count": len(server_errors),
            "empty_response_count": len(empty_responses),
            "output_length": {
                "mean": mean(output_lengths) if output_lengths else 0,
                "min": min(output_lengths) if output_lengths else 0,
                "max": max(output_lengths) if output_lengths else 0,
            },
        }

    def _compute_resource_metrics(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Compute resource behavior metrics."""
        successful = [r for r in records if r.status == 200 and r.is_valid()]

        ttft_values = [float(r.first_resp_time) for r in successful if r.first_resp_time > 0]
        throughput_values = [float(r.token_throughput) for r in successful if r.token_throughput > 0]

        # Simple correlation check
        if len(ttft_values) > 10 and len(throughput_values) > 10:
            # Check if high TTFT correlates with high/low throughput
            p75_ttft = percentile(ttft_values, 0.75)

            high_ttft_tp = [tp for ttft, tp in zip(ttft_values, throughput_values) if ttft > p75_ttft]
            low_ttft_tp = [tp for ttft, tp in zip(ttft_values, throughput_values) if ttft <= p75_ttft]

            high_ttft_mean = mean(high_ttft_tp) if high_ttft_tp else 0
            low_ttft_mean = mean(low_ttft_tp) if low_ttft_tp else 0
        else:
            high_ttft_mean = 0
            low_ttft_mean = 0

        return {
            "high_ttft_avg_throughput": high_ttft_mean,
            "low_ttft_avg_throughput": low_ttft_mean,
            "interpretation": self._interpret_resource_pattern(high_ttft_mean, low_ttft_mean),
        }

    def _interpret_resource_pattern(self, high_ttft_tp: float, low_ttft_tp: float) -> str:
        """Interpret resource pattern."""
        if high_ttft_tp > low_ttft_tp * 1.2:
            return "High TTFT with high throughput - likely batch processing"
        elif high_ttft_tp < low_ttft_tp * 0.8:
            return "High TTFT with low throughput - likely resource contention"
        else:
            return "TTFT and throughput are independent"

    def _compute_temporal_metrics(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Compute temporal patterns."""
        # Group by hour
        hour_stats: Dict[int, List[float]] = defaultdict(list)

        for r in records:
            if r.status == 200 and r.first_resp_time > 0:
                hour = get_hour_of_day(r.created_at)
                hour_stats[hour].append(float(r.first_resp_time))

        hourly_means = {}
        for hour in range(24):
            if hour_stats[hour]:
                hourly_means[hour] = mean(hour_stats[hour])

        # Find best and worst hours
        if hourly_means:
            worst_hour = max(hourly_means.items(), key=lambda x: x[1])
            best_hour = min(hourly_means.items(), key=lambda x: x[1])
        else:
            worst_hour = (0, 0)
            best_hour = (0, 0)

        return {
            "hourly_means": hourly_means,
            "worst_hour": {"hour": worst_hour[0], "mean_ttft": worst_hour[1]},
            "best_hour": {"hour": best_hour[0], "mean_ttft": best_hour[1]},
        }

    def _generate_executive_summary(
        self,
        stability: Dict[str, Any],
        health: Dict[str, Any],
        resource: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        insights = []

        # Stability insights
        ttft_cv = (
            (stability["ttft"]["std"] / stability["ttft"]["mean"])
            if stability["ttft"]["mean"] > 0 else 0
        )
        if ttft_cv > 0.5:
            insights.append("High volatility in response times (CV > 0.5)")
        elif ttft_cv < 0.2:
            insights.append("Very stable response times (CV < 0.2)")

        # Health insights
        if health["error_rate"] > 0.05:
            insights.append(f"Elevated error rate: {health['error_rate']*100:.1f}%")
        if health["timeout_count"] > 0:
            insights.append(f"{health['timeout_count']} timeout(s) detected")

        # Temporal insights
        worst_ratio = temporal["worst_hour"]["mean_ttft"] / temporal["best_hour"]["mean_ttft"] if temporal["best_hour"]["mean_ttft"] > 0 else 1
        if worst_ratio > 2:
            insights.append(f"Hour {temporal['worst_hour']['hour']} is {worst_ratio:.1f}x slower than hour {temporal['best_hour']['hour']}")

        # Resource insights
        insights.append(resource["interpretation"])

        return {
            "overall_health": "healthy" if health["error_rate"] < 0.01 and ttft_cv < 0.3 else "degraded",
            "key_insights": insights,
            "recommendations": self._generate_recommendations(
                stability, health, resource, temporal
            ),
        }

    def _generate_recommendations(
        self,
        stability: Dict[str, Any],
        health: Dict[str, Any],
        resource: Dict[str, Any],
        temporal: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        ttft_cv = (
            (stability["ttft"]["std"] / stability["ttft"]["mean"])
            if stability["ttft"]["mean"] > 0 else 0
        )

        if ttft_cv > 0.5:
            recommendations.append("Consider investigating causes of high latency variance")

        if health["error_rate"] > 0.01:
            recommendations.append("Review error logs and consider implementing retry logic")

        if health["timeout_count"] > 0:
            recommendations.append("Increase timeout values or investigate network issues")

        worst_ratio = temporal["worst_hour"]["mean_ttft"] / temporal["best_hour"]["mean_ttft"] if temporal["best_hour"]["mean_ttft"] > 0 else 1
        if worst_ratio > 2:
            recommendations.append(f"Consider avoiding requests during hour {temporal['worst_hour']['hour']} or implementing adaptive timing")

        return recommendations

    def _get_time_range(self, records: List[RunRecord]) -> Dict[str, str]:
        """Get the time range of records."""
        if not records:
            return {}

        timestamps = [r.created_at for r in records]
        start = datetime.fromtimestamp(min(timestamps) / 1000, tz=BEIJING_TZ)
        end = datetime.fromtimestamp(max(timestamps) / 1000, tz=BEIJING_TZ)

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "duration_hours": (max(timestamps) - min(timestamps)) / (1000 * 3600),
        }

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        provider = data["provider"]
        model = data["model"]
        summary = data["executive_summary"]
        stability = data["stability"]
        health = data["health"]
        resource = data["resource_behavior"]
        temporal = data["temporal_patterns"]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Monitoring Report - {provider}.{model}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .metric.warning {{ border-left-color: #ff9800; }}
        .metric.error {{ border-left-color: #f44336; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .insights {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .recommendations {{ background: #fff3e0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .section {{ margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Monitoring Report</h1>
        <p><strong>Provider:</strong> {provider} | <strong>Model:</strong> {model}</p>
        <p><strong>Total Requests:</strong> {data['total_requests']:,}</p>
        <p><strong>Time Range:</strong> {data['time_range'].get('start', 'N/A')} to {data['time_range'].get('end', 'N/A')}</p>
        <p><strong>Duration:</strong> {data['time_range'].get('duration_hours', 0):.1f} hours</p>

        <h2>Executive Summary</h2>
        <div class="metric {'warning' if summary['overall_health'] == 'degraded' else ''}">
            <div class="metric-label">Overall Health</div>
            <div class="metric-value">{summary['overall_health'].upper()}</div>
        </div>

        <h3>Key Insights</h3>
        <div class="insights">
            <ul>
                {"".join(f"<li>{insight}</li>" for insight in summary['key_insights'])}
            </ul>
        </div>

        <h3>Recommendations</h3>
        <div class="recommendations">
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in summary['recommendations'])}
            </ul>
        </div>

        <h2>Service Stability</h2>
        <div class="section">
            <div class="metric">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{stability['success_rate']*100:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg TTFT</div>
                <div class="metric-value">{stability['ttft']['mean']:.0f} ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">P95 TTFT</div>
                <div class="metric-value">{stability['ttft']['p95']:.0f} ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">P99 TTFT</div>
                <div class="metric-value">{stability['ttft']['p99']:.0f} ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Throughput</div>
                <div class="metric-value">{stability['throughput']['mean']:.1f} tok/s</div>
            </div>
        </div>

        <h2>Service Health</h2>
        <div class="section">
            <div class="metric {'warning' if health['error_rate'] > 0.01 else ''}">
                <div class="metric-label">Error Rate</div>
                <div class="metric-value">{health['error_rate']*100:.2f}%</div>
            </div>
            <div class="metric {'warning' if health['timeout_count'] > 0 else ''}">
                <div class="metric-label">Timeouts</div>
                <div class="metric-value">{health['timeout_count']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Server Errors</div>
                <div class="metric-value">{health['server_error_count']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Empty Responses</div>
                <div class="metric-value">{health['empty_response_count']}</div>
            </div>
        </div>

        <h2>Resource Behavior</h2>
        <div class="section">
            <p>{resource['interpretation']}</p>
        </div>

        <h2>Temporal Patterns</h2>
        <div class="section">
            <p><strong>Slowest Hour:</strong> {temporal['worst_hour']['hour']}:00 - {temporal['worst_hour']['mean_ttft']:.0f}ms avg</p>
            <p><strong>Fastest Hour:</strong> {temporal['best_hour']['hour']}:00 - {temporal['best_hour']['mean_ttft']:.0f}ms avg</p>
        </div>

        <p style="color: #999; font-size: 12px; margin-top: 50px;">
            Generated by llmperf
        </p>
    </div>
</body>
</html>"""
        return html
