from __future__ import annotations

import random
import re
import string
from typing import Callable, Dict, Iterable

from .mutation_context import MutationContext
from .types import TestCase

ChainStep = Callable[[TestCase, MutationContext], TestCase]
STEP_REGISTRY: Dict[str, ChainStep] = {}


def register_step(name: str, func: ChainStep) -> None:
    STEP_REGISTRY[name] = func


def identity(case: TestCase, ctx: MutationContext) -> TestCase:
    return case.model_copy(deep=True)


register_step("identity", identity)


def rdmprefix(case: TestCase, ctx: MutationContext) -> TestCase:
    """Add a randomized benign prefix before the final message content."""
    mutated = case.model_copy(deep=True)
    random_prefix = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    first_message = mutated.messages[0]
    prefix = f"IGNORE-THIS:{random_prefix}. Pay attention to my question. My question is: "
    first_message.content = prefix + first_message.content
    return mutated


register_step("rdmprefix", rdmprefix)


def _collapse_duplicate_spaces(text: str) -> str:
    normalized_lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()]
    collapsed = "\n".join(normalized_lines).strip()
    return collapsed or text.strip()


def rmdupspaces(case: TestCase, ctx: MutationContext) -> TestCase:
    """Collapse repeated spaces and tabs while preserving line breaks."""
    mutated = case.model_copy(deep=True)
    for message in mutated.messages:
        message.content = _collapse_duplicate_spaces(message.content)
    return mutated


register_step("rmdupspaces", rmdupspaces)


def _strip_comments(text: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    without_html_comments = re.sub(r"<!--.*?-->", "", without_block_comments, flags=re.DOTALL)
    cleaned_lines = []
    for line in without_html_comments.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("--"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text.strip()


def rmcomments(case: TestCase, ctx: MutationContext) -> TestCase:
    """Remove common standalone comment lines and block comments from messages."""
    mutated = case.model_copy(deep=True)
    for message in mutated.messages:
        message.content = _strip_comments(message.content)
    return mutated


register_step("rmcomments", rmcomments)


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
