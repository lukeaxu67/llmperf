"""
LLMPerf package entry. Importing this package registers built-in
datasets, message builders, providers, executors, notifications, and exporters.
"""

import importlib


def _safe_import(module: str) -> None:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        # Provider SDKs and optional dependencies are optional at import time;
        # errors surface when the provider/executor is used.
        return


# Dataset sources
_safe_import("llmperf.datasets.sources.jsonl")
_safe_import("llmperf.datasets.sources.generator")
_safe_import("llmperf.datasets.sources.csv")
_safe_import("llmperf.datasets.sources.http")

# Providers
_safe_import("llmperf.providers.openai_chat")
_safe_import("llmperf.providers.response_api")
_safe_import("llmperf.providers.mock")

# Executors
_safe_import("llmperf.executors.openai_chat")
_safe_import("llmperf.executors.response_api")

# Notifications
_safe_import("llmperf.notifications.channels.email")
_safe_import("llmperf.notifications.channels.webhook")
_safe_import("llmperf.notifications.channels.dingtalk")
_safe_import("llmperf.notifications.channels.wechat")
_safe_import("llmperf.notifications.channels.log")

# Exporters
_safe_import("llmperf.export.csv")
_safe_import("llmperf.export.jsonl")
_safe_import("llmperf.export.html")

# Analysis v2
_safe_import("llmperf.analysis.v2.timeseries")
_safe_import("llmperf.analysis.v2.comparison")
_safe_import("llmperf.analysis.v2.anomaly")
