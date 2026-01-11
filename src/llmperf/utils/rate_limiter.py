from __future__ import annotations

import threading
import time


class RateLimiter:
    """
    Smooth rate limiter.

    - Default: unlimited (no-op)
    - qps or interval_seconds enables limiting
    - Thread-safe
    """

    def __init__(
        self,
        *,
        qps: float | None = None,
        interval_seconds: float | None = None,
    ):
        self._enabled = False
        self._interval: float | None = None
        self._next_ts: float = 0.0
        self._lock = threading.Lock()

        if qps is not None or interval_seconds is not None:
            if qps is not None and interval_seconds is not None:
                raise ValueError("Specify only one of qps or interval_seconds")

            if interval_seconds is not None:
                if interval_seconds <= 0:
                    raise ValueError("interval_seconds must be > 0")
                self._interval = float(interval_seconds)
            else:
                if qps is None or qps <= 0:
                    raise ValueError("qps must be > 0")
                self._interval = 1.0 / float(qps)

            self._enabled = True
            self._next_ts = time.monotonic()

    def acquire(self) -> None:
        if not self._enabled:
            return

        if self._interval is None:  # pragma: no cover - defensive
            return

        with self._lock:
            now = time.monotonic()
            if now < self._next_ts:
                time.sleep(self._next_ts - now)
            self._next_ts = max(self._next_ts + self._interval, time.monotonic())

