from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Tuple, Type, TypeVar


T = TypeVar("T")


@dataclass
class RegistryEntry(Generic[T]):
    key: Tuple[str, str]
    target: Type[T]
    metadata: Dict[str, Any]


class DoubleRegistry(Generic[T]):
    """Registry that indexes callables/classes via (type, impl) keys."""

    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str], RegistryEntry[T]] = {}

    def register(self, type_name: str, impl: str, target: Type[T], **metadata: Any) -> None:
        key = (type_name, impl)
        self._items[key] = RegistryEntry(key, target, metadata)

    def get(self, type_name: str, impl: str) -> Type[T]:
        key = (type_name, impl)
        if key in self._items:
            return self._items[key].target
        # fallback to "<type_name>, default" or "default, default"
        fallback_keys = [
            (type_name, "default"),
            ("default", impl),
            (type_name, ""),
            ("default", ""),
        ]
        for fb in fallback_keys:
            if fb in self._items:
                return self._items[fb].target
        raise KeyError(f"No registered target for key={type_name}.{impl}")

    def get_metadata(self, type_name: str, impl: str) -> Dict[str, Any]:
        key = (type_name, impl)
        entry = self._items.get(key)
        if entry:
            return entry.metadata
        for fb in [
            (type_name, "default"),
            ("default", impl),
            (type_name, ""),
            ("default", ""),
        ]:
            entry = self._items.get(fb)
            if entry:
                return entry.metadata
        raise KeyError(f"No metadata for key={type_name}.{impl}")

    def available_keys(self) -> Dict[str, Any]:
        return {f"{k[0]}.{k[1]}": entry.target for k, entry in self._items.items()}


def registration_decorator(registry: DoubleRegistry[T], type_name: str, impl: str, **metadata: Any) -> Callable[[Type[T]], Type[T]]:
    def decorator(cls: Type[T]) -> Type[T]:
        registry.register(type_name, impl, cls, **metadata)
        return cls

    return decorator
