"""
Service health analysis for LLM performance monitoring.

Analyzes error rates, error types, truncation patterns, output length distribution,
and temporal error patterns.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List
from datetime import datetime

from pydantic import BaseModel, Field

from ..base_analysis import BaseAnalysis
from ..record_query import RecordQuery
from ..analysis_registry import register_analysis
from ..statistics import percentile, mean
from ..timeseries import get_hour_of_day, group_by_time_segment

from llmperf.records.model import RunRecord
from llmperf.records.storage import Storage


@register_analysis("health")
class HealthAnalysis(BaseAnalysis["HealthAnalysis.Config"]):

    config: Config

    class Config(BaseModel):
        query: RecordQuery = Field(default_factory=RecordQuery)
        segment_size: int = Field(default=6, description="Hours per time segment")
        max_tokens_threshold: float = Field(
            default=0.95,
            description="Ratio to max_tokens considered as truncation"
        )

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

        # Add duration info
        timestamps = [r.created_at for r in records]
        duration_hours = (max(timestamps) - min(timestamps)) / (1000 * 3600) if timestamps else 0

        results = {
            "config": {
                "segment_size": self.config.segment_size,
                "max_tokens_threshold": self.config.max_tokens_threshold,
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

    def _analyze_group(
        self, provider: str, model: str, records: List[RunRecord]
    ) -> Dict[str, Any]:
        """Analyze health metrics for a group of records."""

        total_count = len(records)

        # Categorize errors
        error_categories = self._categorize_errors(records)

        # Analyze truncation
        truncation_stats = self._analyze_truncation(records)

        # Output length distribution
        output_lengths = [int(rec.atokens) for rec in records if rec.atokens > 0]

        # Temporal error patterns
        temporal_errors = self._analyze_temporal_errors(records)

        # Empty responses
        empty_responses = [
            rec for rec in records
            if rec.status == 200 and rec.atokens == 0 and not rec.content
        ]

        return {
            "total_requests": total_count,
            "successful_requests": sum(1 for r in records if r.status == 200),
            "overall_error_rate": 1 - (sum(1 for r in records if r.status == 200) / total_count if total_count else 0),

            # Error breakdown
            "error_breakdown": error_categories,

            # Truncation analysis
            "truncation": truncation_stats,

            # Output length
            "output_length": {
                "mean": mean(output_lengths) if output_lengths else 0,
                "min": min(output_lengths) if output_lengths else 0,
                "max": max(output_lengths) if output_lengths else 0,
                "p50": percentile(output_lengths, 0.5) if output_lengths else 0,
                "p95": percentile(output_lengths, 0.95) if output_lengths else 0,
                "empty_count": len(empty_responses),
            },

            # Temporal patterns
            "temporal_error_patterns": temporal_errors,

            # Insights
            "insights": self._generate_insights(
                error_categories, truncation_stats, temporal_errors, len(empty_responses)
            ),
        }

    def _categorize_errors(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Categorize errors by type."""
        categories = defaultdict(int)

        for rec in records:
            if rec.status == 200:
                continue

            status = rec.status
            info = rec.info or ""

            # Timeouts
            if status == 0 or 'timeout' in info.lower() or 'timed out' in info.lower():
                categories['timeout'] += 1
            # Server errors (5xx)
            elif 500 <= status < 600:
                categories['5xx'] += 1
            # Rate limiting
            elif status == 429 or 'rate limit' in info.lower():
                categories['rate_limited'] += 1
            # Client errors (4xx, excluding rate limit)
            elif 400 <= status < 500:
                categories['4xx_client'] += 1
            # Network errors
            elif 'connection' in info.lower() or 'network' in info.lower():
                categories['network'] += 1
            # Unknown/Other
            else:
                categories['other'] += 1

        # Convert to dict with ratios
        total_errors = sum(categories.values())
        result = {}
        for cat, count in categories.items():
            result[cat] = {
                "count": count,
                "ratio": count / total_errors if total_errors > 0 else 0,
            }

        return result

    def _analyze_truncation(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Analyze response truncation patterns."""
        # Get max_tokens from request params
        potentially_truncated = []
        for rec in records:
            if rec.status != 200 or rec.atokens == 0:
                continue

            max_tokens = rec.request_params.get('max_tokens') or rec.request_params.get('max_completion_tokens')
            if max_tokens and isinstance(max_tokens, int):
                if rec.atokens >= max_tokens * self.config.max_tokens_threshold:
                    potentially_truncated.append({
                        'output_tokens': rec.atokens,
                        'max_tokens': max_tokens,
                        'ratio': rec.atokens / max_tokens,
                    })

        # Check for sudden drop in output length
        output_lengths = [int(rec.atokens) for rec in records if rec.atokens > 0]
        if len(output_lengths) > 10:
            median_length = percentile(output_lengths, 0.5)
            sudden_drops = [
                l for l in output_lengths
                if l < median_length * 0.5
            ]
        else:
            sudden_drops = []

        return {
            "potentially_truncated_count": len(potentially_truncated),
            "truncation_ratio": len(potentially_truncated) / len(records) if records else 0,
            "sudden_short_response_count": len(sudden_drops),
            "short_response_ratio": len(sudden_drops) / len(output_lengths) if output_lengths else 0,
        }

    def _analyze_temporal_errors(self, records: List[RunRecord]) -> Dict[str, Any]:
        """Analyze error patterns over time segments."""
        # Group by time segment
        error_by_segment = defaultdict(lambda: {'total': 0, 'errors': 0})

        for rec in records:
            hour = get_hour_of_day(rec.created_at)
            segment_name = get_time_segment_name(hour, self.config.segment_size)

            error_by_segment[segment_name]['total'] += 1
            if rec.status != 200:
                error_by_segment[segment_name]['errors'] += 1

        # Calculate error rates by segment
        segment_errors = {}
        for segment, counts in error_by_segment.items():
            if counts['total'] > 0:
                segment_errors[segment] = {
                    'error_count': counts['errors'],
                    'total_count': counts['total'],
                    'error_rate': counts['errors'] / counts['total'],
                }

        # Find segments with elevated error rates
        if segment_errors:
            avg_error_rate = mean([s['error_rate'] for s in segment_errors.values()])
            elevated_segments = {
                seg: data for seg, data in segment_errors.items()
                if data['error_rate'] > avg_error_rate * 1.5
            }
        else:
            elevated_segments = {}
            avg_error_rate = 0

        return {
            "by_segment": segment_errors,
            "average_error_rate": avg_error_rate,
            "elevated_segments": elevated_segments,
        }

    def _generate_insights(
        self,
        error_categories: Dict[str, Any],
        truncation_stats: Dict[str, Any],
        temporal_errors: Dict[str, Any],
        empty_count: int,
    ) -> List[str]:
        """Generate health insights."""
        insights = []

        # Error type insights
        total_errors = sum(cat['count'] for cat in error_categories.values())
        if total_errors > 0:
            top_error = max(error_categories.items(), key=lambda x: x[1]['count'])
            insights.append(f"Most common error: {top_error[0]} ({top_error[1]['count']} occurrences)")

            if 'timeout' in error_categories and error_categories['timeout']['ratio'] > 0.3:
                insights.append("High timeout ratio, may indicate overload or network issues")
            if 'rate_limited' in error_categories:
                insights.append(f"Rate limiting detected ({error_categories['rate_limited']['count']} times)")

        # Truncation insights
        if truncation_stats['truncation_ratio'] > 0.1:
            insights.append(f"High truncation ratio ({truncation_stats['truncation_ratio']*100:.1f}%), outputs may be cut short")

        if truncation_stats['short_response_ratio'] > 0.05:
            insights.append(f"Sudden short responses detected ({truncation_stats['short_response_ratio']*100:.1f}%), possible dynamic throttling")

        # Empty response insight
        if empty_count > 0:
            insights.append(f"Found {empty_count} empty responses, check for policy rejections")

        # Temporal insight
        elevated = temporal_errors.get('elevated_segments', {})
        if elevated:
            for seg, data in elevated.items():
                insights.append(f"Elevated error rate in segment {seg} ({data['error_rate']*100:.1f}%)")

        return insights


def get_time_segment_name(hour: int, segment_size: int = 6) -> str:
    """Get time segment name for a given hour."""
    segment_idx = hour // segment_size
    segment_start = segment_idx * segment_size
    segment_end = segment_start + segment_size - 1
    return f"{segment_start:02d}-{segment_end:02d}"
