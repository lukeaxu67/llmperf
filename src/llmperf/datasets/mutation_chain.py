from typing import Callable, Dict, Iterable
from .types import TestCase
from .mutation_context import MutationContext

ChainStep = Callable[[TestCase, MutationContext], TestCase]
STEP_REGISTRY: Dict[str, ChainStep] = {}


def register_step(name: str, func: ChainStep) -> None:
    STEP_REGISTRY[name] = func


def identity(case: TestCase, ctx: MutationContext) -> TestCase:
    return case.model_copy(deep=True)


register_step("identity", identity)


class MutationChain:
    """Applies a sequence of TestCase mutations with context."""

    def __init__(self, steps: Iterable[str] | None = None):
        self.steps = list(steps or ["identity"])

    def apply(self, case: TestCase, ctx: MutationContext) -> TestCase:
        current = case.model_copy(deep=True)
        for step in self.steps:
            func = STEP_REGISTRY.get(step, identity)
            current = func(current, ctx)
        return current
