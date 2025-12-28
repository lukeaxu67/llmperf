from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

from llmperf.pricing.loader import load_pricing_entries
from llmperf.pricing.catalog import PriceCatalog


ConfigT = TypeVar("ConfigT", bound=BaseModel)


class BaseAnalysis(ABC, Generic[ConfigT]):

    type_name: str

    Config: Type[ConfigT]

    def __init__(self, config: ConfigT):
        self.config: ConfigT = config

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _price_catalog(self, *, path: Optional[str]) -> Optional[PriceCatalog]:
        if not path:
            return None
        return PriceCatalog(load_pricing_entries(Path(path)))
