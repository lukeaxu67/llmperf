from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _default_db_path() -> Path:
    home = Path(os.getenv("USERPROFILE") or os.getenv("HOME") or ".").expanduser()
    return (home / "llmperf" / "db" / "perf.sqlite").resolve()


class RuntimeConfig(BaseModel):
    db_path: Path = Field(default_factory=_default_db_path)

    @field_validator("db_path", mode="before")
    @classmethod
    def _expand_path(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()


def load_runtime_config(overrides: Optional[dict] = None) -> RuntimeConfig:
    """
    Load runtime configuration.

    This project does not use environment-variable overrides; callers should
    pass explicit overrides when needed.
    """
    return RuntimeConfig.model_validate(overrides or {})

