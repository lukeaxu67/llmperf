from __future__ import annotations

import pytest

from llmperf.web import main as web_main
from llmperf.web.services import dataset_service, pricing_service


@pytest.fixture(autouse=True)
def isolate_runtime_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMPerf_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("LLMPerf_DATASETS_DIR", str(tmp_path / "datasets"))

    web_main._task_service = None
    web_main._analysis_service = None
    dataset_service._dataset_service = None
    pricing_service._pricing_service = None

    yield

    web_main._task_service = None
    web_main._analysis_service = None
    dataset_service._dataset_service = None
    pricing_service._pricing_service = None
