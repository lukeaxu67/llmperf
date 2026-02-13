from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Type, TypeVar, List

from pydantic import BaseModel

from llmperf.pricing.loader import load_pricing_entries
from llmperf.pricing.catalog import PriceCatalog
from llmperf.records.model import RunRecord


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

    @staticmethod
    def _apply_duration_limit(
        records: List[RunRecord],
        max_duration_hours: Optional[float],
    ) -> List[RunRecord]:
        """
        Apply max duration limit to records, keeping the most recent N hours.

        Args:
            records: List of run records
            max_duration_hours: Maximum duration in hours (None = no limit)

        Returns:
            Filtered list of records
        """
        if not records or max_duration_hours is None:
            return records

        # Sort by timestamp
        sorted_records = sorted(records, key=lambda r: r.created_at)

        # Get the latest timestamp
        latest_ts = sorted_records[-1].created_at

        # Calculate cutoff timestamp
        max_duration_ms = int(max_duration_hours * 3600 * 1000)
        cutoff_ts = latest_ts - max_duration_ms

        # Filter records after cutoff (keep most recent N hours)
        filtered = [r for r in sorted_records if r.created_at >= cutoff_ts]

        return filtered
