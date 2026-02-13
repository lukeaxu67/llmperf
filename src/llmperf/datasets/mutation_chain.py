import random
import string
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


def rdmprefix(case: TestCase, ctx: MutationContext) -> TestCase:
    """在问题前面添加 IGNORE-THIS 前缀来干扰模型"""
    mutated = case.model_copy(deep=True)
    # 生成随机的8位前缀
    random_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    # 在最后一条消息前添加前缀
    last_message = mutated.messages[-1]
    prefix = f"IGNORE-THIS:{random_prefix}. Pay attention to my question. My question is: "
    last_message.content = prefix + last_message.content
    return mutated


register_step("rdmprefix", rdmprefix)


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
