from __future__ import annotations

import re
from collections.abc import Iterable

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def get_word_num(text: str, debug: bool = False) -> int:
    """
    Heuristic token-ish counter used for throughput metrics.

    Counts:
    - CJK characters (each counts as 1)
    - ASCII word chunks (letters/digits counts as 1 per chunk)
    - punctuation/symbols (each counts as 1)
    """
    value = (text or "").strip()
    if not value:
        return 0

    cjk_count = len(_CJK_RE.findall(value))
    ascii_words = _ASCII_WORD_RE.findall(value)
    ascii_count = len(ascii_words)
    punct_count = len(_PUNCT_RE.findall(value))

    total = cjk_count + ascii_count + punct_count
    if debug:
        details = {
            "cjk": cjk_count,
            "ascii_words": ascii_count,
            "punct": punct_count,
            "total": total,
        }
        print(details)
    return total


def word_count(text: str, punctuation_as_breaker: bool = False) -> dict[str, object]:
    """
    A minimal word counter for debugging / optional tooling.
    """
    value = (text or "").strip()
    if not value:
        return {"words": [], "count": 0}

    if punctuation_as_breaker:
        value = _PUNCT_RE.sub(" ", value)

    words = [w for w in re.split(r"\s+", value) if w]
    return {"words": words, "count": len(words)}


def iter_chunks(text: str, chunk_size: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    value = text or ""
    for i in range(0, len(value), chunk_size):
        yield value[i : i + chunk_size]

