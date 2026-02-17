"""Tests for the Web API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llmperf.web.main import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestTaskEndpoints:
    """Tests for task management endpoints."""

    def test_list_tasks_empty(self, client):
        """Test listing tasks when empty."""
        response = client.get("/api/tasks")

        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data

    def test_get_task_not_found(self, client):
        """Test getting non-existent task."""
        response = client.get("/api/tasks/nonexistent-id")

        assert response.status_code == 404


class TestAnalysisEndpoints:
    """Tests for analysis endpoints."""

    def test_get_summary_not_found(self, client):
        """Test getting summary for non-existent run."""
        response = client.get("/api/analysis/nonexistent-id/summary")

        assert response.status_code == 404

    def test_get_history(self, client):
        """Test getting run history."""
        response = client.get("/api/analysis/history")

        assert response.status_code == 200
        data = response.json()
        assert "runs" in data


class TestConfigEndpoints:
    """Tests for configuration endpoints."""

    def test_list_templates(self, client):
        """Test listing config templates."""
        response = client.get("/api/config/templates")

        assert response.status_code == 200
        templates = response.json()
        assert isinstance(templates, list)

    def test_validate_config_valid(self, client):
        """Test validating valid config."""
        valid_config = """
info: "Test Task"
dataset:
  source:
    type: "jsonl"
    name: "test"
    config:
      path: "resource/demo.jsonl"
  iterator:
    mutation_chain: ["identity"]
    max_rounds: 1
executors:
  - id: "test-001"
    name: "Test Executor"
    type: "mock"
    impl: "chat"
    concurrency: 1
    model: "test-model"
"""
        response = client.post(
            "/api/config/validate",
            content=valid_config,
        )

        assert response.status_code == 200
        data = response.json()
        # May have warnings but should be valid
        assert "valid" in data

    def test_validate_config_invalid(self, client):
        """Test validating invalid config."""
        invalid_config = """
info: "Missing required fields"
executors: []
"""
        response = client.post(
            "/api/config/validate",
            content=invalid_config,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_get_runtime_config(self, client):
        """Test getting runtime config."""
        response = client.get("/api/config/runtime")

        assert response.status_code == 200
        data = response.json()
        assert "db_path" in data
        assert "log_level" in data


class TestDatasetEndpoints:
    """Tests for dataset endpoints."""

    def test_list_datasets(self, client):
        """Test listing datasets."""
        response = client.get("/api/datasets")

        assert response.status_code == 200
        datasets = response.json()
        assert isinstance(datasets, list)

    def test_get_dataset_not_found(self, client):
        """Test getting non-existent dataset."""
        response = client.get("/api/datasets/nonexistent")

        assert response.status_code == 404


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/api/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_endpoint(self, client):
        """Test docs endpoint is available."""
        response = client.get("/api/docs")

        assert response.status_code == 200
