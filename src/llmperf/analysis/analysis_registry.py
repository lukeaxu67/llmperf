from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

import yaml

from .base_analysis import BaseAnalysis

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
