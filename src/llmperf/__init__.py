"""
LLMPerf package entry. Importing this package registers built-in
datasets, message builders, providers, and executors.
"""

import importlib


def _safe_import(module: str) -> None:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        # Provider SDKs are optional at import time; errors surface when the provider/executor is used.
        return


_safe_import("llmperf.datasets.sources.jsonl")
_safe_import("llmperf.datasets.sources.generator")
_safe_import("llmperf.providers.openai_chat")
_safe_import("llmperf.providers.response_api")
_safe_import("llmperf.providers.mock")
_safe_import("llmperf.executors.openai_chat")
_safe_import("llmperf.executors.response_api")
