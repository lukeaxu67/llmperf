from __future__ import annotations

from typing import Any, Dict, List, Mapping, Type
from .dataset_source import DatasetSource
from .types import TestCase


_SOURCE_REGISTRY: Dict[str, Type[DatasetSource]] = {}


def register_source(source_type: str):
    """Class decorator for registering DatasetSource implementations."""

    def _decorator(cls: Type[DatasetSource]) -> Type[DatasetSource]:
        if source_type in _SOURCE_REGISTRY:
            raise ValueError(f"Dataset source '{source_type}' already registered")
        cls.source_type = source_type
        _SOURCE_REGISTRY[source_type] = cls
        return cls

    return _decorator


def get_source_class(source_type: str) -> Type[DatasetSource]:
    try:
        return _SOURCE_REGISTRY[source_type]
    except KeyError as exc:  # pragma: no cover - defensive path
        available = ", ".join(sorted(_SOURCE_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown dataset source '{source_type}'. Available: {available}"
        ) from exc


def create_source(
    source_type: str, *, name: str, config: Dict[str, Any] | None = None
) -> DatasetSource:
    """Instantiate a DatasetSource by type."""

    source_cls = get_source_class(source_type)
    return source_cls(name=name, config=config or {})


def list_sources() -> Dict[str, Type[DatasetSource]]:
    """Expose registered sources for diagnostics/testing."""

    return dict(_SOURCE_REGISTRY)


def load_dataset(source_type: str, *, name: str, config: Mapping[str, Any] | None = None) -> List[TestCase]:
    """
    Load a dataset using the refactored DatasetSource stack.

    `name` is the logical dataset name (used by generators and for diagnostics).
    """
    source = create_source(source_type, name=name, config=dict(config or {}))
    return list(source.load())
