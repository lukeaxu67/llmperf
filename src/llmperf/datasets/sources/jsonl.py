from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..types import Message, TestCase
from ..dataset_source import DatasetSource
from ..dataset_source_registry import register_source


def parse_jsonl_test_case(raw_line: str, index: int = 0) -> TestCase:
    data = json.loads(raw_line)
    row_id = f"row-{index + 1}"

    if isinstance(data, list):
        messages = [Message.model_validate(item) for item in data]
        return TestCase(id=row_id, messages=messages)

    if isinstance(data, dict):
        normalized = dict(data)
        if "messages" in normalized and not normalized.get("id"):
            normalized["id"] = row_id
        return TestCase.model_validate(normalized)

    raise ValueError("Each JSONL line must be a TestCase object or a message array")


def _load_from_path(path: Path, encoding: str, limit: int | None) -> List[TestCase]:
    items: List[TestCase] = []
    with path.open("r", encoding=encoding) as fh:
        for index, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            test_case = parse_jsonl_test_case(stripped, index=index)
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
