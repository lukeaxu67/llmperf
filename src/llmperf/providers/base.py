from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..records.model import RunRecord
from ..utils.registry import DoubleRegistry, registration_decorator


@dataclass
class ProviderRequest:
    run_id: str
    executor_id: str
    dataset_row_id: str
    provider: str
    model: str
    messages: List[Dict[str, str]]
    options: Dict[str, Any]


def normalize_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in messages:
        role = item.get("role", "user")
        content = item.get("content", "")
        normalized.append({"role": role, "content": content})
    return normalized


class BaseProvider:
    """
    Provider abstraction responsible for talking to provider APIs and returning
    a unified RunRecord.
    """

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def invoke(self, request: ProviderRequest) -> RunRecord:
        raise NotImplementedError


provider_registry: DoubleRegistry[BaseProvider] = DoubleRegistry()


def register_provider(type_name: str, impl: str = "default", **metadata):
    return registration_decorator(provider_registry, type_name, impl, **metadata)


def create_provider(type_name: str, impl: str, provider_name: Optional[str] = None) -> BaseProvider:
    provider_cls = provider_registry.get(type_name, impl)
    return provider_cls(provider_name or type_name)
