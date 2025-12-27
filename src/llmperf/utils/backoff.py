from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional, Protocol


class SupportsBackoff(Protocol):
    initial_ms: int
    factor: float
    max_ms: int
    jitter_ratio: float
    max_attempts: int


@dataclass
class BackoffConfig:
    """Lightweight container mirroring BackoffPolicyModel."""

    initial_ms: int = 500
    factor: float = 2.0
    max_ms: int = 8000
    jitter_ratio: float = 0.2
    max_attempts: int = 5


def should_retry_status(status: int) -> bool:
    return status == 429 or (500 <= status < 600)


def compute_backoff_delay_ms(attempt: int, cfg: SupportsBackoff) -> float:
    exp = cfg.initial_ms * (cfg.factor ** max(0, attempt - 1))
    exp = min(exp, cfg.max_ms)
    jitter = exp * cfg.jitter_ratio
    delay = exp + random.uniform(-jitter, jitter)
    return max(delay, 0.0)


def sleep_with_backoff(attempt: int, cfg: SupportsBackoff) -> float:
    delay_ms = compute_backoff_delay_ms(attempt, cfg)
    time.sleep(delay_ms / 1000.0)
    return delay_ms


async def async_sleep_with_backoff(attempt: int, cfg: SupportsBackoff) -> float:
    delay_ms = compute_backoff_delay_ms(attempt, cfg)
    import asyncio

    await asyncio.sleep(delay_ms / 1000.0)
    return delay_ms
