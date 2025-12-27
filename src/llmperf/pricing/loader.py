from __future__ import annotations

from pathlib import Path
from typing import Any, List

import yaml

from ..config.models import PricingEntry


def load_pricing_entries(path: str | Path) -> List[PricingEntry]:
    pricing_path = Path(path).expanduser()
    if not pricing_path.exists():
        raise FileNotFoundError(f"Price catalog file not found: {pricing_path}")
    data = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("pricing") or []
    else:
        items = []
    if not isinstance(items, list):
        raise ValueError("Price catalog must be a list or {pricing: [...]} mapping")
    return [PricingEntry.model_validate(item) for item in items]


def read_text(path: str | Path) -> str:
    return Path(path).expanduser().read_text(encoding="utf-8")

