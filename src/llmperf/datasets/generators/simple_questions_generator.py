"""
Simple questions generator for continuous monitoring.

Generates a variety of simple questions for monitoring LLM API performance.
"""
from __future__ import annotations

import random
from typing import Iterable, Mapping, Any
from datetime import datetime, timezone

from .base_generator import BaseGenerator
from ..types import TestCase, Message


class SimpleQuestionsGenerator(BaseGenerator):
    """
    Generator that produces simple questions for monitoring.

    Rotates through different question types to test various aspects:
    - Factual questions
    - Creative questions
    - Analytical questions
    - Code questions
    """

    # Question templates
    QUESTIONS = [
        # Analytical questions
        "What are the pros and cons of remote work?",
        "Compare Python and JavaScript.",
        "Explain the concept of recursion.",
        "What makes a good user interface?",
        "How do you balance work and life?",

        # Code questions
        "Write a function to reverse a string in Python.",
        "Explain what Big O notation means.",
        "What's the difference between SQL and NoSQL?",
        "How does a hash table work?",
        "Explain the concept of a REST API.",

        # Factual questions
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When was the first moon landing?",
        "What is the largest ocean on Earth?",

        # Creative questions
        "Write a haiku about programming.",
        "Tell me a short joke.",
        "Describe the color blue to someone who has never seen it.",
        "Write a catchy slogan for a coffee shop.",
        "Invent a word and define it.",
    ]

    def __init__(
        self,
        dataset_name: str,
        parameters: Mapping[str, Any] | None = None,
    ):
        super().__init__(dataset_name, parameters)
        self.count = int(parameters.get("count", 10) if parameters else 10)
        self.seed = parameters.get("seed") if parameters else None

        if self.seed is not None:
            random.seed(self.seed)

    def generate(self) -> Iterable[TestCase]:
        """Generate test cases with simple questions."""
        timestamp = datetime.now(timezone.utc).isoformat()

        for i in range(self.count):
            # 循环使用问题列表
            question = self.QUESTIONS[i % len(self.QUESTIONS)]

            yield TestCase(
                id=f"{self.dataset_name}-{i}",
                messages=[
                    Message(role="user", content=question)
                ],
                metadata={
                    "question_type": self._classify_question(question),
                    "generated_at": timestamp,
                },
            )

    def _classify_question(self, question: str) -> str:
        """Classify the question type."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["what", "who", "when", "where", "how"]):
            if any(word in question_lower for word in ["write", "code", "function"]):
                return "code"
            elif any(word in question_lower for word in ["explain", "compare", "pros", "cons"]):
                return "analytical"
            else:
                return "factual"
        elif any(word in question_lower for word in ["write", "tell", "describe", "invent"]):
            return "creative"
        else:
            return "general"
