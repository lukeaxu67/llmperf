from dataclasses import dataclass
from typing import Optional


@dataclass
class MutationContext:
    round_index: int
    case_index: int
    consumed: int
    total_rounds: Optional[int]
    elapsed: float
    max_seconds: Optional[float]
