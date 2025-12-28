import json
import re
from typing import Tuple
from llmperf.records.model import RunRecord


def _params_key(params: dict) -> str:

    return json.dumps(params or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[int(k)])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


def _safe_filename(name: str) -> str:
    value = (name or "run").strip()
    value = re.sub(r"[<>:\"/\\\\|?*]+", "_", value)
    value = re.sub(r"\\s+", " ", value).strip()
    return value or "run"


def _group_key(record: RunRecord) -> Tuple[str, str, str]:
    return (record.provider, record.model, _params_key(record.request_params))
