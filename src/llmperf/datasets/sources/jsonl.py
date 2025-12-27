from __future__ import annotations

from pathlib import Path
from typing import List

from ..types import TestCase
from ..dataset_source import DatasetSource
from ..dataset_source_registry import register_source


def _load_from_path(path: Path, encoding: str, limit: int | None) -> List[TestCase]:
    items: List[TestCase] = []
    with path.open("r", encoding=encoding) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            test_case = TestCase.model_validate_json(stripped)
            items.append(test_case)
            if limit and len(items) >= limit:
                break
    return items


@register_source("jsonl")
class JsonlDatasetSource(DatasetSource):
    """DatasetSource backed by a JSONL file on disk."""

    def __init__(self, *, name: str, config: dict[str, object] | None = None):
        super().__init__(name=name, config=config)
        path_value = (config or {}).get("path")
        if not path_value:
            raise ValueError("JsonlDatasetSource requires a 'path' configuration value")
        self.path = Path(str(path_value)).expanduser().resolve()
        self.encoding = str((config or {}).get("encoding", "utf-8"))
        limit_value = (config or {}).get("limit")
        self.limit = int(limit_value) if limit_value is not None else None

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file {self.path} not found")

    def load(self) -> List[TestCase]:
        return _load_from_path(self.path, self.encoding, self.limit)
