"""Helpers for normalizing Web-submitted run configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmperf.config.models import RunConfig

from .dataset_service import get_dataset_service


def _looks_like_legacy_web_dataset_path(path_value: Any) -> bool:
    if not path_value:
        return True
    normalized = str(path_value).replace("\\", "/").lower()
    return normalized.startswith("resource/") or normalized.startswith("./resource/")


def _path_exists(path_value: Any) -> bool:
    if not path_value:
        return False
    try:
        return Path(str(path_value)).expanduser().exists()
    except OSError:
        return False


def normalize_run_config(config: RunConfig) -> RunConfig:
    """Resolve Web-managed dataset names to their actual on-disk files."""

    dataset_source = config.dataset.source
    dataset_name = dataset_source.name
    if not dataset_name:
        return config

    source_config = dict(dataset_source.config or {})
    source_path = source_config.get("path")
    should_replace_path = (
        not source_path
        or _looks_like_legacy_web_dataset_path(source_path)
        or not _path_exists(source_path)
    )
    if not should_replace_path:
        return config

    dataset_service = get_dataset_service()
    metadata = dataset_service.get_dataset(dataset_name)
    if not metadata:
        return config

    dataset_source.type = metadata.file_type.value
    source_config["path"] = str(dataset_service._get_data_path(metadata.name, metadata.file_type))
    if metadata.encoding and "encoding" not in source_config:
        source_config["encoding"] = metadata.encoding
    dataset_source.config = source_config
    return config


def load_run_config_content(config_content: str) -> tuple[RunConfig, str]:
    """Parse, normalize, and re-serialize YAML config content."""

    config_dict = yaml.safe_load(config_content) or {}
    config = normalize_run_config(RunConfig.model_validate(config_dict))
    normalized_content = yaml.safe_dump(
        config.model_dump(exclude_none=True),
        allow_unicode=True,
        sort_keys=False,
    )
    return config, normalized_content
