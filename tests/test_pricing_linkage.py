from __future__ import annotations

from fastapi.testclient import TestClient

from llmperf.web.main import create_app
from llmperf.web.services import pricing_service


def test_pricing_matches_executor_provider_case_insensitively(tmp_path, monkeypatch):
    db_path = tmp_path / "pricing.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    monkeypatch.setattr(pricing_service, "_pricing_service", None)

    with TestClient(create_app()) as client:
        add_resp = client.post(
            "/api/pricing",
            json={
                "provider": "OpenAI",
                "model": "gpt-4o-mini",
                "input_price": 1.23,
                "output_price": 4.56,
            },
        )
        assert add_resp.status_code == 200
        body = add_resp.json()
        assert body["provider"] == "openai"

        list_resp = client.get("/api/pricing", params={"provider": "openai", "model": "gpt-4o-mini"})
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] == 1

        current_resp = client.get(
            "/api/pricing/current",
            params={"provider": "openai", "model": "gpt-4o-mini"},
        )
        assert current_resp.status_code == 200
        current = current_resp.json()
        assert current["found"] is True
        assert current["input_price"] == 1.23
        assert current["output_price"] == 4.56


def test_pricing_matches_model_with_trimmed_whitespace(tmp_path, monkeypatch):
    db_path = tmp_path / "pricing-trim.sqlite"
    monkeypatch.setenv("LLMPerf_DB_PATH", str(db_path))
    monkeypatch.setattr(pricing_service, "_pricing_service", None)

    with TestClient(create_app()) as client:
        add_resp = client.post(
            "/api/pricing",
            json={
                "provider": "openai ",
                "model": " gpt-4o ",
                "input_price": 2.0,
                "output_price": 3.0,
            },
        )
        assert add_resp.status_code == 200

        current_resp = client.get(
            "/api/pricing/current",
            params={"provider": "openai", "model": "gpt-4o"},
        )
        assert current_resp.status_code == 200
        assert current_resp.json()["found"] is True
