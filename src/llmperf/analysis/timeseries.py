"""
Time-series analysis utilities for LLM performance monitoring.

Provides functions for temporal segmentation, trend analysis,
periodicity detection, and time-of-day analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any
import collections

from .statistics import mean, variance, std_dev, percentile

# Beijing Timezone (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))


@dataclass
class TimeSegment:
    """A time segment with aggregated statistics."""
    name: str
    start_hour: int
    end_hour: int
    count: int
    mean_ttft: float
    mean_total_time: float
    mean_throughput: float
    error_rate: float
    mean_output_length: float


def get_hour_of_day(timestamp_ms: int) -> int:
    """Get hour of day (0-23) from millisecond timestamp using Beijing time (UTC+8)."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ).hour


def get_time_segment_name(hour: int, segment_size: int = 6) -> str:
    """
    Get time segment name for a given hour.

    Args:
        hour: Hour of day (0-23)
        segment_size: Size of each segment in hours (default: 6)
    """
    segment_idx = hour // segment_size
    segment_start = segment_idx * segment_size
    segment_end = segment_start + segment_size - 1
    return f"{segment_start:02d}-{segment_end:02d}"


def group_by_time_segment(
    timestamps: List[int],
    values_ttft: List[float],
    values_total: List[float],
    values_throughput: List[float],
    values_error: List[bool],
    values_output_len: List[int],
    segment_size: int = 6,
) -> Dict[str, TimeSegment]:
    """
    Group metrics by time segments.
    """
    # Ensure all lists have the same length
    min_len = min(len(timestamps), len(values_ttft), len(values_total),
                  len(values_throughput), len(values_error), len(values_output_len))

    timestamps = timestamps[:min_len]
    values_ttft = values_ttft[:min_len]
    values_total = values_total[:min_len]
    values_throughput = values_throughput[:min_len]
    values_error = values_error[:min_len]
    values_output_len = values_output_len[:min_len]

    segments: Dict[str, List[Dict]] = {}

    for i, ts in enumerate(timestamps):
        hour = get_hour_of_day(ts)
        segment_name = get_time_segment_name(hour, segment_size)

        if segment_name not in segments:
            segments[segment_name] = []

        segments[segment_name].append({
            'ttft': values_ttft[i] if i < len(values_ttft) else 0,
            'total': values_total[i] if i < len(values_total) else 0,
            'throughput': values_throughput[i] if i < len(values_throughput) else 0,
            'error': values_error[i] if i < len(values_error) else False,
            'output_len': values_output_len[i] if i < len(values_output_len) else 0,
        })

    result: Dict[str, TimeSegment] = {}
    for name, items in segments.items():
        segment_idx = int(name.split('-')[0]) // segment_size
        result[name] = TimeSegment(
            name=name,
            start_hour=segment_idx * segment_size,
            end_hour=segment_idx * segment_size + segment_size - 1,
            count=len(items),
            mean_ttft=mean([x['ttft'] for x in items if x['ttft'] > 0]),
            mean_total_time=mean([x['total'] for x in items if x['total'] > 0]),
            mean_throughput=mean([x['throughput'] for x in items if x['throughput'] > 0]),
            error_rate=sum(1 for x in items if x['error']) / len(items) if items else 0,
            mean_output_length=mean([x['output_len'] for x in items if x['output_len'] > 0]),
        )

    return result


def compare_day_night(
    timestamps: List[int],
    values: List[float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compare day vs night metrics."""
    # Ensure length consistency
    min_len = min(len(timestamps), len(values))
    timestamps = timestamps[:min_len]
    values = values[:min_len]

    day_values = []
    night_values = []

    for ts, val in zip(timestamps, values):
        hour = get_hour_of_day(ts)
        if 6 <= hour < 18:
            day_values.append(val)
        else:
            night_values.append(val)

    def _stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        return {
            'mean': mean(vals),
            'std': std_dev(vals),
            'min': min(vals),
            'max': max(vals),
            'count': len(vals),
        }

    return _stats(day_values), _stats(night_values)


def sliding_window_analysis(
    timestamps: List[int],
    values: List[float],
    window_minutes: int = 30,
    step_minutes: int = 5,
) -> List[Dict[str, Any]]:
    """Perform sliding window analysis on time-series data."""
    # Ensure length consistency
    min_len = min(len(timestamps), len(values))
    timestamps = timestamps[:min_len]
    values = values[:min_len]

    if not timestamps:
        return []

    # Convert to (timestamp, value) pairs and sort by timestamp
    pairs = sorted(zip(timestamps, values))
    timestamps_sorted = [p[0] for p in pairs]
    values_sorted = [p[1] for p in pairs]

    window_ms = window_minutes * 60 * 1000
    step_ms = step_minutes * 60 * 1000

    start_ts = timestamps_sorted[0]
    end_ts = timestamps_sorted[-1]

    results = []
    window_start = start_ts

    while window_start + window_ms <= end_ts:
        window_end = window_start + window_ms

        # Find values within window
        window_values = []
        for ts, val in zip(timestamps_sorted, values_sorted):
            if window_start <= ts < window_end:
                window_values.append(val)

        if window_values:
            results.append({
                'window_start': window_start,
                'window_end': window_end,
                'mean': mean(window_values),
                'std': std_dev(window_values),
                'min': min(window_values),
                'max': max(window_values),
                'count': len(window_values),
                'p50': percentile(window_values, 0.5),
                'p95': percentile(window_values, 0.95),
            })

        window_start += step_ms

    return results


def detect_periodicity(
    timestamps: List[int],
    values: List[float],
    period_minutes: int = 60,
) -> Dict[str, Any]:
    """Detect if there's a periodic pattern at the given period."""
    # Ensure length consistency
    min_len = min(len(timestamps), len(values))
    timestamps = timestamps[:min_len]
    values = values[:min_len]

    if len(timestamps) < 10:
        return {'period_minutes': period_minutes, 'pattern_strength': 0, 'note': 'Insufficient data'}

    period_ms = period_minutes * 60 * 1000

    # Group by position within period
    position_groups: Dict[int, List[float]] = collections.defaultdict(list)

    for ts, val in zip(timestamps, values):
        # Calculate position within period (0 to period_minutes-1)
        period_offset = (ts // 1000) % (period_minutes * 60)
        position_key = int(period_offset // 60)  # Group by minute within period

        position_groups[position_key].append(val)

    # Calculate mean for each position
    position_means = {k: mean(v) for k, v in position_groups.items() if v}

    if len(position_means) < 3:
        return {
            'period_minutes': period_minutes,
            'pattern_strength': 0,
            'note': 'Insufficient unique positions',
        }

    # Calculate variance of means (higher = stronger pattern)
    means_list = list(position_means.values())
    overall_mean = mean(means_list)
    variance_of_means = sum((m - overall_mean) ** 2 for m in means_list) / len(means_list)

    # Normalize by overall variance
    overall_var = variance(values)

    if overall_var == 0:
        pattern_strength = 0
    else:
        pattern_strength = variance_of_means / overall_var

    return {
        'period_minutes': period_minutes,
        'pattern_strength': pattern_strength,
        'positions_analyzed': len(position_means),
        'highest_mean': max(means_list) if means_list else 0,
        'lowest_mean': min(means_list) if means_list else 0,
        'interpretation': _interpret_pattern_strength(pattern_strength),
    }


def _interpret_pattern_strength(strength: float) -> str:
    """Interpret pattern strength value."""
    if strength < 0.1:
        return "No significant pattern"
    elif strength < 0.3:
        return "Weak pattern"
    elif strength < 0.5:
        return "Moderate pattern"
    else:
        return "Strong pattern"


def calculate_volatility_series(
    window_stats: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Calculate volatility metrics from sliding window statistics."""
    if len(window_stats) < 3:
        return []

    result = []
    prev_std = 0

    for i, stat in enumerate(window_stats):
        volatility_change = 0
        if i > 0:
            volatility_change = stat['std'] - prev_std

        result.append({
            'window_start': stat['window_start'],
            'std': stat['std'],
            'volatility_change': volatility_change,
            'is_spike': volatility_change > 0 and prev_std > 0 and (volatility_change / prev_std) > 0.5,
        })
        prev_std = stat['std']

    return result


def find_anomaly_windows(
    window_stats: List[Dict[str, Any]],
    std_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """Find windows where metrics deviate significantly from the overall baseline."""
    if len(window_stats) < 5:
        return []

    means = [w['mean'] for w in window_stats]
    overall_mean = mean(means)
    overall_std = std_dev(means)

    anomalies = []
    for stat in window_stats:
        z_score = (stat['mean'] - overall_mean) / overall_std if overall_std > 0 else 0

        if abs(z_score) > std_threshold:
            anomalies.append({
                **stat,
                'z_score': z_score,
                'anomaly_type': 'high' if z_score > 0 else 'low',
            })

    return anomalies
