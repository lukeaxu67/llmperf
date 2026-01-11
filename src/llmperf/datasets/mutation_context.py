from dataclasses import dataclass


@dataclass
class MutationContext:
    round_index: int
    case_index: int
    consumed: int
