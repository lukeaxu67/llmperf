"""
Statistical analysis utilities for LLM performance monitoring.

Provides functions for calculating percentiles, variance, outliers,
correlation, autocorrelation, and other statistical metrics.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


def percentile(values: List[float], p: float) -> float:
    """
    Calculate the p-th percentile of a list of values.

    Uses linear interpolation between closest ranks for better accuracy.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if not 0 <= p <= 1:
        raise ValueError(f"Percentile must be between 0 and 1, got {p}")

    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[f])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


def mean(values: List[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: List[float], sample: bool = True) -> float:
    """
    Calculate variance.
    Use sample=True for sample variance (n-1 denominator), False for population.
    """
    if len(values) < 2:
        return 0.0
    m = mean(values)
    squared_diffs = [(x - m) ** 2 for x in values]
    denominator = len(values) - 1 if sample else len(values)
    return sum(squared_diffs) / denominator


def std_dev(values: List[float], sample: bool = True) -> float:
    """Calculate standard deviation."""
    return math.sqrt(variance(values, sample))


def coefficient_of_variation(values: List[float]) -> float:
    """
    Calculate coefficient of variation (CV) = std_dev / mean.
    Higher CV indicates higher relative volatility.
    """
    m = mean(values)
    if m == 0:
        return 0.0
    return std_dev(values) / abs(m)


@dataclass
class PercentileStats:
    """Container for percentile statistics."""
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float
    mean: float
    std: float

    @classmethod
    def from_values(cls, values: List[float]) -> "PercentileStats":
        """Create PercentileStats from a list of values."""
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        return cls(
            p50=percentile(values, 0.50),
            p90=percentile(values, 0.90),
            p95=percentile(values, 0.95),
            p99=percentile(values, 0.99),
            min=min(values),
            max=max(values),
            mean=mean(values),
            std=std_dev(values),
        )


def outlier_count(values: List[float], threshold_multiplier: float = 2.0) -> Tuple[int, List[float]]:
    """
    Count outliers beyond threshold_multiplier * P95.

    Returns:
        (outlier_count, outlier_values)
    """
    if len(values) < 10:
        return 0, []
    p95 = percentile(values, 0.95)
    threshold = p95 * threshold_multiplier
    outliers = [v for v in values if v > threshold]
    return len(outliers), outliers


def outlier_ratio(values: List[float], threshold_multiplier: float = 2.0) -> float:
    """Calculate the ratio of outliers to total values."""
    count, _ = outlier_count(values, threshold_multiplier)
    return count / len(values) if values else 0.0


def correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two lists.

    Returns:
        Correlation coefficient between -1 and 1.
        Returns 0 if lists are too short or have zero variance.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


def autocorrelation(values: List[float], lag: int) -> float:
    """
    Calculate autocorrelation at a specific lag.

    Measures how correlated a value is with its value `lag` steps ago.
    Useful for detecting patterns like "slow requests cluster together".
    """
    if lag <= 0 or lag >= len(values):
        return 0.0

    n = len(values) - lag
    if n < 2:
        return 0.0

    # Split into two aligned series
    series1 = values[:n]
    series2 = values[lag:]

    return correlation(series1, series2)


@dataclass
class AutocorrelationStats:
    """Container for autocorrelation statistics at multiple lags."""
    lag1: float
    lag5: float
    lag10: float
    lag30: float

    @classmethod
    def from_values(cls, values: List[float]) -> "AutocorrelationStats":
        """Create AutocorrelationStats from a list of values."""
        return cls(
            lag1=autocorrelation(values, 1),
            lag5=autocorrelation(values, 5),
            lag10=autocorrelation(values, 10),
            lag30=autocorrelation(values, 30),
        )


def sliding_window_stats(
    values: List[float],
    window_size: int,
    step: int = 1,
) -> List[Tuple[int, float, float, float]]:
    """
    Calculate statistics over sliding windows.

    Args:
        values: List of time-ordered values
        window_size: Size of each window
        step: Step size between windows (default: 1 = overlapping)

    Returns:
        List of (window_index, mean, variance, std_dev) tuples
    """
    if window_size >= len(values) or window_size < 2:
        return []

    results = []
    for i in range(0, len(values) - window_size + 1, step):
        window = values[i:i + window_size]
        results.append((
            i,
            mean(window),
            variance(window),
            std_dev(window),
        ))
    return results


def change_point_detection(values: List[float], window_size: int = 10) -> List[int]:
    """
    Detect significant change points in a time series.

    Uses a simple approach: flag points where the mean differs significantly
    from the previous window.

    Args:
        values: List of time-ordered values
        window_size: Size of window to compare

    Returns:
        List of indices where significant changes were detected
    """
    if len(values) < window_size * 2:
        return []

    changes = []
    prev_mean = mean(values[:window_size])
    prev_std = std_dev(values[:window_size])

    for i in range(window_size, len(values) - window_size, window_size // 2):
        window = values[i:i + window_size]
        curr_mean = mean(window)
        curr_std = std_dev(window)

        # If means differ by more than 1 standard deviation, flag as change
        if abs(curr_mean - prev_mean) > max(prev_std, curr_std):
            changes.append(i)

        prev_mean = curr_mean
        prev_std = curr_std

    return changes


@dataclass
class DistributionSummary:
    """Summary statistics for a distribution."""
    count: int
    mean: float
    std: float
    cv: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    outlier_count: int
    outlier_ratio: float

    @classmethod
    def from_values(cls, values: List[float], outlier_threshold: float = 2.0) -> "DistributionSummary":
        """Create DistributionSummary from a list of values."""
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        pct = PercentileStats.from_values(values)
        outliers, _ = outlier_count(values, outlier_threshold)

        return cls(
            count=len(values),
            mean=pct.mean,
            std=pct.std,
            cv=coefficient_of_variation(values),
            min=pct.min,
            max=pct.max,
            p50=pct.p50,
            p90=pct.p90,
            p95=pct.p95,
            p99=pct.p99,
            outlier_count=outliers,
            outlier_ratio=outlier_ratio(values, outlier_threshold),
        )
