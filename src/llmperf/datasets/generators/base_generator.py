from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..types import TestCase


class BaseGenerator:
    """Shared generator contract that yields TestCase objects."""

    def __init__(self, dataset_name: str, parameters: Mapping[str, Any] | None = None):
        self.dataset_name = dataset_name
        self.parameters = dict(parameters or {})

    def generate(self) -> Iterable[TestCase]:
        raise NotImplementedError
