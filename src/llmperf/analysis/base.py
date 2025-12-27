from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, Field

from ..config.runtime import load_runtime_config
from ..pricing.catalog import PriceCatalog
from ..pricing.loader import load_pricing_entries
from ..records.storage import Storage


class Query(BaseModel):
    db_path: Optional[str] = None
    run_ids: list[str] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None

    def storage(self) -> Storage:
        runtime = load_runtime_config()
        return Storage(self.db_path or str(runtime.db_path))


ConfigT = TypeVar("ConfigT", bound=BaseModel)


class BaseAnalysis(ABC, Generic[ConfigT]):
    type_name: str
    Config: Type[ConfigT]

    def __init__(self, config: ConfigT):
        self.config = config

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _price_catalog(self, *, path: Optional[str]) -> Optional[PriceCatalog]:
        if not path:
            return None
        return PriceCatalog(load_pricing_entries(Path(path)))


_REGISTRY: Dict[str, Type[BaseAnalysis]] = {}


def register_analysis(type_name: str):
    def _decorator(cls: Type[BaseAnalysis]) -> Type[BaseAnalysis]:
        if type_name in _REGISTRY:
            raise ValueError(f"analysis type already registered: {type_name}")
        cls.type_name = type_name
        _REGISTRY[type_name] = cls
        return cls

    return _decorator


def create_analysis(type_name: str, config_data: dict) -> BaseAnalysis:
    cls = _REGISTRY.get(type_name)
    if not cls:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"unknown analysis type '{type_name}', available: {available}")
    cfg_model = getattr(cls, "Config", None)
    if cfg_model is None:
        raise ValueError(f"analysis '{type_name}' is missing Config model")
    config = cfg_model.model_validate(config_data)
    return cls(config)  # type: ignore[call-arg]


def load_analysis_config(path: str | Path) -> dict:
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Analysis config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Analysis config must be a YAML mapping")
    return data

