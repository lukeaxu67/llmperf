from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


class SecretNotFoundError(RuntimeError):
    """Raised when a requested secret cannot be resolved."""


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"


def get_secret(key: str, *, raise_on_missing: bool = True) -> Optional[str]:
    """Resolve a secret from environment variables or a local .env file."""
    value = os.getenv(key)
    if value:
        return value

    dotenv_values = _load_dotenv()
    if key in dotenv_values:
        return dotenv_values[key]

    if raise_on_missing:
        raise SecretNotFoundError(f"Secret '{key}' not found in env or {DOTENV_PATH}")
    return None


def resolve_api_key(ref: str) -> str:
    """Resolve a provider api_key_ref value."""
    if ref.startswith("literal:"):
        return ref.split("literal:", 1)[1]
    return get_secret(ref, raise_on_missing=True) or ""


@lru_cache(maxsize=1)
def _load_dotenv() -> Dict[str, str]:
    if not DOTENV_PATH.exists():
        return {}
    lines = DOTENV_PATH.read_text().splitlines()
    values: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")
    return values
