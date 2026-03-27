from __future__ import annotations

from llmperf.datasets.mutation_chain import MutationChain
from llmperf.datasets.mutation_context import MutationContext
from llmperf.datasets.types import TestCase


def _case(content: str) -> TestCase:
    return TestCase.model_validate(
        {
            "id": "1",
            "messages": [{"role": "user", "content": content}],
        }
    )


def test_rdmprefix_adds_random_prefix():
    chain = MutationChain(["rdmprefix"])
    mutated = chain.apply(_case("please summarize"), MutationContext(round_index=1, case_index=0, consumed=1))
    assert mutated.messages[-1].content.startswith("IGNORE-THIS:")
    assert mutated.messages[-1].content.endswith("please summarize")


def test_rmdupspaces_collapses_duplicate_spaces():
    chain = MutationChain(["rmdupspaces"])
    mutated = chain.apply(_case("hello   world\t\tagain"), MutationContext(round_index=1, case_index=0, consumed=1))
    assert mutated.messages[-1].content == "hello world again"


def test_rmcomments_removes_comment_lines_and_blocks():
    chain = MutationChain(["rmcomments"])
    content = "# note\nkeep this\n// another note\n/* block */\n<!-- html -->\nfinal line"
    mutated = chain.apply(_case(content), MutationContext(round_index=1, case_index=0, consumed=1))
    assert mutated.messages[-1].content == "keep this\n\n\nfinal line"
