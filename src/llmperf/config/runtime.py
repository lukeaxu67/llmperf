from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _default_db_path() -> Path:
    env_path = os.getenv("LLMPerf_DB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    home = Path(os.getenv("USERPROFILE") or os.getenv("HOME") or ".").expanduser()
    return (home / "llmperf" / "db" / "perf.sqlite").resolve()


def _default_log_dir() -> Path:
    env_dir = os.getenv("LLMPerf_LOG_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    home = Path(os.getenv("USERPROFILE") or os.getenv("HOME") or ".").expanduser()
    return (home / "llmperf" / "logs").resolve()


def _default_log_level() -> str:
    return os.getenv("LLMPerf_LOG_LEVEL", "INFO").upper()


def _default_datasets_dir() -> Path:
    env_dir = os.getenv("LLMPerf_DATASETS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    home = Path(os.getenv("USERPROFILE") or os.getenv("HOME") or ".").expanduser()
    return (home / "llmperf" / "datasets").resolve()


class RuntimeConfig(BaseModel):
    """Runtime configuration for LLMPerf.

    Configuration can be set via environment variables:
    - LLMPerf_DB_PATH: Database file path
    - LLMPerf_LOG_DIR: Log directory
    - LLMPerf_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR)
    """

    db_path: Path = Field(default_factory=_default_db_path)
    """Path to the SQLite database file."""

    log_dir: Path = Field(default_factory=_default_log_dir)
    """Directory for log files."""

    datasets_dir: Path = Field(default_factory=_default_datasets_dir)
    """Directory for runtime uploaded datasets."""

    log_level: str = Field(default_factory=_default_log_level)
    """Logging level."""

    cache_enabled: bool = Field(default=True)
    """Whether caching is enabled."""

    cache_ttl_seconds: int = Field(default=3600)
    """Cache time-to-live in seconds."""

    @field_validator("db_path", "log_dir", "datasets_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = str(value).upper()
        if level not in valid_levels:
            raise ValueError(f"Invalid log level: {value}. Must be one of {valid_levels}")
        return level

    def setup_logging(self) -> None:
        """Configure logging based on this configuration."""
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.log_dir / "llmperf.log",
                    encoding="utf-8",
                ),
            ],
        )


def load_runtime_config(overrides: Optional[dict] = None) -> RuntimeConfig:
    """Load runtime configuration.

    Args:
        overrides: Optional dictionary of configuration overrides.

    Returns:
        RuntimeConfig instance.
    """
    config_data = {}

    # Load from environment variables
    if os.getenv("LLMPerf_DB_PATH"):
        config_data["db_path"] = os.getenv("LLMPerf_DB_PATH")
    if os.getenv("LLMPerf_LOG_DIR"):
        config_data["log_dir"] = os.getenv("LLMPerf_LOG_DIR")
    if os.getenv("LLMPerf_DATASETS_DIR"):
        config_data["datasets_dir"] = os.getenv("LLMPerf_DATASETS_DIR")
    if os.getenv("LLMPerf_LOG_LEVEL"):
        config_data["log_level"] = os.getenv("LLMPerf_LOG_LEVEL")

    # Apply overrides
    if overrides:
        config_data.update(overrides)

    return RuntimeConfig.model_validate(config_data)
