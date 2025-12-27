from __future__ import annotations

from typing import Any, Iterable, Mapping
from pydantic import BaseModel, Field
from pathlib import Path

from .base_generator import BaseGenerator
from .utils import build_test_case
from ..types import TestCase


class NovelSliceSummaryGenerator(BaseGenerator):
    """Generate test cases from a text file, slicing it into chunks."""

    class Parameters(BaseModel):
        path: str = Field(min_length=1)
        chunk_size: int = Field(default=800, gt=0)
        limit: int = Field(default=1000, gt=0)

    def __init__(self, dataset_name: str, parameters: Mapping[str, Any] | None = None):
        super().__init__(dataset_name, parameters)
        self.parameters: NovelSliceSummaryGenerator.Parameters = self.Parameters(
            **self.parameters
        )

    def generate(self) -> Iterable[TestCase]:
        text = ""
        if self.parameters.path:
            path = Path(str(self.parameters.path)).expanduser()
            if path.exists():
                text = path.read_text(encoding="utf-8")
            else:
                raise ValueError(f"Path does not exist: {path}")
        else:
            raise ValueError("NovelSliceGenerator requires a non-empty path parameter")
        chunks = [
            text[i : i + self.parameters.chunk_size]
            for i in range(0, len(text), self.parameters.chunk_size)
        ]
        for idx, chunk in enumerate(chunks):
            if idx >= self.parameters.limit:
                break
            content = chunk.replace("\n", " ")
            head = "[[START_OF_CHUNK]]"
            tail = "[[END_OF_CHUNK]]"
            user_text = f"Summarize this narrative thoughtfully: {head}{content}{tail}"
            yield build_test_case(
                f"{self.dataset_name}-novel-{idx}",
                user_text,
                metadata={"task": "summary"},
            )
