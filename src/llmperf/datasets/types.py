from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1, max_length=1000000)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("content")
    @classmethod
    def _normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Message content cannot be empty.")
        return normalized


class TestCase(BaseModel):
    id: str = Field(min_length=1)
    messages: List[Message] = Field(min_length=1)
    metadata: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("TestCase id cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def _ensure_trailing_assistant(self) -> "TestCase":
        if not self.messages:
            raise ValueError("TestCase requires at least one message.")
        if self.messages[-1].role != "user":
            raise ValueError("The final message must have role 'user'.")
        return self
