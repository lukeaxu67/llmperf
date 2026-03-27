"""CSV dataset source for LLMPerf."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

from ..types import TestCase, Message
from ..dataset_source import DatasetSource
from ..dataset_source_registry import register_source

logger = logging.getLogger(__name__)


def _parse_messages(value: str) -> List[Dict[str, str]]:
    """Parse messages from CSV value.

    Supports multiple formats:
    - JSON array: [{"role": "user", "content": "..."}]
    - Simple format: role:content|role:content

    Args:
        value: String value from CSV.

    Returns:
        List of message dictionaries.
    """
    import json

    value = value.strip()
    if not value:
        return []

    # Try JSON first
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # Try pipe-separated format
    messages = []
    for part in value.split("|"):
        if ":" in part:
            role, content = part.split(":", 1)
            messages.append({"role": role.strip(), "content": content.strip()})

    return messages


@register_source("csv")
class CSVDatasetSource(DatasetSource):
    """DatasetSource backed by a CSV file.

    Configuration:
        path: Path to CSV file (required)
        encoding: File encoding (default: utf-8)
        delimiter: CSV delimiter (default: ,)
        has_header: Whether CSV has header row (default: True)
        id_column: Column name/index for test case ID (default: id)
        messages_column: Column name/index for messages (default: messages)
        metadata_columns: List of columns to include as metadata
        messages_format: Format of messages field (json, pipe)
    """

    def __init__(self, *, name: str, config: dict[str, object] | None = None):
        super().__init__(name=name, config=config)

        cfg = config or {}
        path_value = cfg.get("path")
        if not path_value:
            raise ValueError("CSVDatasetSource requires a 'path' configuration value")

        self.path = Path(str(path_value)).expanduser().resolve()
        self.encoding = str(cfg.get("encoding", "utf-8"))
        self.delimiter = str(cfg.get("delimiter", ","))
        self.has_header = bool(cfg.get("has_header", True))
        self.id_column = cfg.get("id_column", "id")
        self.messages_column = cfg.get("messages_column", "messages")
        self.metadata_columns = cfg.get("metadata_columns", [])
        self.messages_format = cfg.get("messages_format", "json")
        self.limit = int(cfg["limit"]) if cfg.get("limit") else None

        if not self.path.exists():
            raise FileNotFoundError(f"CSV file {self.path} not found")

    def _get_column_value(
        self,
        row: Dict[str, str],
        column: Any,
        default: str = "",
    ) -> str:
        """Get value from row by column name or index.

        Args:
            row: Row dictionary.
            column: Column name or index.
            default: Default value if not found.

        Returns:
            Column value.
        """
        if isinstance(column, int):
            keys = list(row.keys())
            if 0 <= column < len(keys):
                return row.get(keys[column], default)
            return default
        return row.get(str(column), default)

    def _parse_row(self, row: Dict[str, str]) -> TestCase:
        """Parse a CSV row into a TestCase.

        Args:
            row: CSV row dictionary.

        Returns:
            TestCase instance.
        """
        # Get ID
        test_id = self._get_column_value(row, self.id_column, "")

        # Get messages
        messages_value = self._get_column_value(row, self.messages_column, "[]")
        if self.messages_format == "pipe":
            messages_raw = _parse_messages(messages_value)
        else:
            import json
            try:
                messages_raw = json.loads(messages_value)
            except json.JSONDecodeError:
                messages_raw = _parse_messages(messages_value)

        # Convert to Message objects
        messages = [
            Message(role=m.get("role", "user"), content=m.get("content", ""))
            for m in messages_raw
        ]

        # Get metadata
        metadata = {}
        if self.metadata_columns:
            for col in self.metadata_columns:
                if col in row:
                    metadata[col] = row[col]
        else:
            # Include all columns except id and messages as metadata
            for key, value in row.items():
                if key not in [str(self.id_column), str(self.messages_column)]:
                    metadata[key] = value

        return TestCase(
            id=test_id or None,
            messages=messages,
            metadata=metadata if metadata else None,
        )

    def load(self) -> List[TestCase]:
        """Load test cases from CSV file.

        Returns:
            List of TestCase instances.
        """
        items: List[TestCase] = []

        with self.path.open("r", encoding=self.encoding, newline="") as fh:
            if self.has_header:
                reader = csv.DictReader(fh, delimiter=self.delimiter)
            else:
                # Read without header, create dummy column names
                reader = csv.reader(fh, delimiter=self.delimiter)
                rows = list(reader)
                if not rows:
                    return []

                # Create column names
                num_cols = len(rows[0])
                fieldnames = [f"col_{i}" for i in range(num_cols)]

                # Convert to dict
                reader = [
                    dict(zip(fieldnames, row))
                    for row in rows
                ]

            for row in reader:
                try:
                    test_case = self._parse_row(row)
                    if test_case.messages:  # Only add if has messages
                        items.append(test_case)
                except Exception as e:
                    logger.warning("Failed to parse CSV row: %s", e)
                    continue

                if self.limit and len(items) >= self.limit:
                    break

        return items
