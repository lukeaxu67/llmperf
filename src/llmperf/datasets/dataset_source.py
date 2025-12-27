from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .types import TestCase


class DatasetSource(ABC):
    """Base abstraction for loading TestCase collections."""

    source_type: str

    def __init__(self, *, name: str, config: Dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def load(self) -> List[TestCase]:
        """Return the full dataset for this source."""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}(name={self.name!r}, config={self.config!r})"
