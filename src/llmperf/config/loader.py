from __future__ import annotations

import pathlib

import yaml

from .models import RunConfig


def load_config(path: str) -> RunConfig:
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config {path} not found")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    return RunConfig.model_validate(data)
