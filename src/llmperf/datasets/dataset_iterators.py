from __future__ import annotations

import threading
import time
from typing import Optional, Sequence
from dataclasses import dataclass

from .types import TestCase
from .mutation_chain import MutationChain
from .mutation_context import MutationContext


@dataclass
class ExecutionPlan:
    max_rounds: Optional[int] = None
    max_total_seconds: Optional[float] = None


class DatasetIterator:
    """Thread-safe iterator with explicit time/round strategy, no implicit looping."""

    def __init__(
        self,
        testcases: Sequence[TestCase],
        *,
        mutation_chain: MutationChain | None = None,
        max_rounds: Optional[int] = None,
        max_total_seconds: Optional[int | float] = None,
    ):
        if not testcases:
            raise ValueError("DatasetIterator requires at least one TestCase")

        self._cases = list(testcases)
        self._chain = mutation_chain or MutationChain()

        self._plan = ExecutionPlan(
            max_rounds=max_rounds,
            max_total_seconds=(float(max_total_seconds) if max_total_seconds else None),
        )

        self._index = 0
        self._round = 1
        self._consumed = 0
        self._start_ts: Optional[float] = None
        self._lock = threading.RLock()

    def _time_exceeded(self) -> bool:
        if self._plan.max_total_seconds is None or self._start_ts is None:
            return False
        return (time.monotonic() - self._start_ts) >= self._plan.max_total_seconds

    def _rounds_exceeded(self) -> bool:
        return self._plan.max_rounds is not None and self._round > self._plan.max_rounds

    def __iter__(self):
        return self

    # def commit(self, case: TestCase) -> None:
    #     """Persist updated case so future rounds see the changes."""
    #     with self._lock:
    #         for i, original in enumerate(self._cases):
    #             if original.id == case.id:
    #                 self._cases[i] = case.model_copy(deep=True)
    #                 break

    def __next__(self) -> TestCase:
        with self._lock:
            if self._start_ts is None:
                self._start_ts = time.monotonic()

            if self._time_exceeded() or self._rounds_exceeded():
                raise StopIteration

            if self._index >= len(self._cases):
                self._index = 0
                self._round += 1

                if self._time_exceeded() or self._rounds_exceeded():
                    raise StopIteration

            self._consumed += 1
            current_index = self._index
            self._index += 1

            ctx = MutationContext(
                round_index=self._round,
                case_index=current_index,
                consumed=self._consumed,
                total_rounds=self._plan.max_rounds,
                elapsed=time.monotonic() - self._start_ts,
                max_seconds=self._plan.max_total_seconds,
            )

            return self._chain.apply(self._cases[current_index], ctx)
