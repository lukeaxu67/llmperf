from typing import Dict, Any
from ..types import Message, TestCase


def build_test_case(
    case_id: str,
    user_prompt: str,
    assistant_text: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> TestCase:
    messages = [Message(role="user", content=user_prompt)]
    if assistant_text:
        messages.append(Message(role="assistant", content=assistant_text))
    return TestCase(id=case_id, messages=messages, metadata=metadata or {})
