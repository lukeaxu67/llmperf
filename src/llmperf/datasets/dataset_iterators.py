from __future__ import annotations

import threading
from typing import Sequence

from .types import TestCase
from .mutation_chain import MutationChain
from .mutation_context import MutationContext


class DatasetIterator:
    """
    Thread-safe dataset iterator.

    Responsibilities:
    - Iterate over test cases and apply the mutation chain.
    - Provide `round_index` / `case_index` semantics (as metadata and mutation context).

    Non-responsibilities:
    - Any runtime/execution termination decisions (time limits, max rows, deadlines, QPS, etc.).
    """

    def __init__(
        self,
        testcases: Sequence[TestCase],
        *,
        mutation_chain: MutationChain | None = None,
    ):
        if not testcases:
            raise ValueError("DatasetIterator requires at least one TestCase")

        self._cases = list(testcases)
        self._chain = mutation_chain or MutationChain()

        self._index = 0
        self._round = 1
        self._consumed = 0
        self._lock = threading.RLock()

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
            if self._index >= len(self._cases):
                self._index = 0
                self._round += 1

            self._consumed += 1
            current_index = self._index
            self._index += 1

            ctx = MutationContext(
                round_index=self._round,
                case_index=current_index,
                consumed=self._consumed,
            )

            base = self._cases[current_index]
            mutated = self._chain.apply(base, ctx)

            meta = dict(mutated.metadata or {})
            meta.setdefault("base_id", base.id)
            meta.setdefault("pass_index", self._round)
            meta.setdefault("case_index", current_index)
            mutated.metadata = meta
            return mutated
