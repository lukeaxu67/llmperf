from __future__ import annotations

from typing import Any, Dict, List, Mapping, Type

from ..generators.base_generator import BaseGenerator
from ..generators.novel_slice_summary_generator import NovelSliceSummaryGenerator
from ..generators.simple_questions_generator import SimpleQuestionsGenerator
from ..dataset_source import DatasetSource
from ..dataset_source_registry import register_source
from ..types import TestCase

_GENERATOR_TYPES: Dict[str, Type[BaseGenerator]] = {
    "novel_slice_summary": NovelSliceSummaryGenerator,
    "simple_questions": SimpleQuestionsGenerator,
}


def build_generator(
    class_name: str, dataset_name: str, parameters: Mapping[str, Any] | None = None
) -> BaseGenerator:
    normalized = class_name.replace("Generator", "").replace("-", "_").lower()
    generator_cls = _GENERATOR_TYPES.get(normalized)
    if not generator_cls:
        raise ValueError(f"Unknown generator class '{class_name}'")
    return generator_cls(dataset_name=dataset_name, parameters=parameters)


@register_source(source_type="generator")
class GeneratorDatasetSource(DatasetSource):
    """DatasetSource backed by an in-process generator implementation."""

    def __init__(self, *, name: str, config: Dict[str, Any] | None = None):
        super().__init__(name=name, config=config)
        generator_config = dict((config or {}).get("generator") or {})
        class_name = generator_config.get("class_name")
        if not class_name:
            raise ValueError("GeneratorDatasetSource requires generator.class_name")
        parameters = generator_config.get("parameters") or {}
        self.generator = build_generator(
            class_name, dataset_name=name, parameters=parameters
        )

    def load(self) -> List[TestCase]:
        return list(self.generator.generate())
