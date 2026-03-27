"""Tests for the export system."""

from __future__ import annotations

import json
from pathlib import Path


from llmperf.export import (
    ExportConfig,
    ExportResult,
    CSVExporter,
    JSONLExporter,
    HTMLReportExporter,
    exporter_registry,
    create_exporter,
)
from llmperf.records.model import RunRecord


def create_test_records(count: int = 10) -> list[RunRecord]:
    """Create test records for testing."""
    records = []
    for i in range(count):
        record = RunRecord(
            run_id="test-run-001",
            executor_id=f"executor-{i % 3}",
            dataset_row_id=f"row-{i}",
            provider="test-provider",
            model="test-model",
            status=200 if i % 5 != 0 else 500,
            qtokens=100 + i * 10,
            atokens=50 + i * 5,
            ctokens=10,
            prompt_cost=0.001 * (i + 1),
            completion_cost=0.002 * (i + 1),
            total_cost=0.003 * (i + 1),
            currency="USD",
            action_times=[1000, 1100 + i * 10, 2000 + i * 20],
        )
        records.append(record)
    return records


class TestCSVExporter:
    """Tests for CSV exporter."""

    def test_export_basic(self, tmp_path):
        """Test basic CSV export."""
        config = ExportConfig(output_dir=str(tmp_path))
        exporter = CSVExporter(config)
        records = create_test_records(10)

        result = exporter.export(
            records=records,
            run_id="test-run-001",
            task_name="Test Export",
        )

        assert result.success
        assert result.format == "csv"
        assert result.records_exported == 10
        assert Path(result.output_path).exists()

        # Verify file content
        with open(result.output_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.strip().split("\n")
            # Header + 10 records
            assert len(lines) == 11

    def test_export_empty_records(self, tmp_path):
        """Test export with empty records."""
        config = ExportConfig(output_dir=str(tmp_path))
        exporter = CSVExporter(config)

        result = exporter.export(
            records=[],
            run_id="test-run-empty",
            task_name="Empty Export",
        )

        assert not result.success
        assert "No records" in result.message


class TestJSONLExporter:
    """Tests for JSONL exporter."""

    def test_export_basic(self, tmp_path):
        """Test basic JSONL export."""
        config = ExportConfig(output_dir=str(tmp_path))
        exporter = JSONLExporter(config)
        records = create_test_records(5)

        result = exporter.export(
            records=records,
            run_id="test-run-001",
            task_name="Test JSONL",
        )

        assert result.success
        assert result.format == "jsonl"
        assert result.records_exported == 5
        assert Path(result.output_path).exists()

        # Verify file content
        with open(result.output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 5

            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert "run_id" in data
                assert "tokens" in data
                assert "performance" in data

    def test_export_with_compression(self, tmp_path):
        """Test JSONL export with gzip compression."""
        from llmperf.export.jsonl import JSONLExportConfig

        config = JSONLExportConfig(output_dir=str(tmp_path), compress=True)
        exporter = JSONLExporter(config)
        records = create_test_records(3)

        result = exporter.export(
            records=records,
            run_id="test-run-compressed",
            task_name="Compressed Export",
        )

        assert result.success
        assert result.output_path.endswith(".gz")
        assert result.metadata.get("compressed") is True


class TestHTMLReportExporter:
    """Tests for HTML report exporter."""

    def test_export_basic(self, tmp_path):
        """Test basic HTML report export."""
        from llmperf.export.html import HTMLReportConfig

        config = HTMLReportConfig(
            output_dir=str(tmp_path),
            title="Test Report",
        )
        exporter = HTMLReportExporter(config)
        records = create_test_records(20)

        result = exporter.export(
            records=records,
            run_id="test-run-001",
            task_name="Test Report",
        )

        assert result.success
        assert result.format == "html"
        assert Path(result.output_path).exists()

        # Verify HTML content
        with open(result.output_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "Test Report" in content
            assert "Summary" in content

    def test_export_with_dark_theme(self, tmp_path):
        """Test HTML export with dark theme."""
        from llmperf.export.html import HTMLReportConfig

        config = HTMLReportConfig(
            output_dir=str(tmp_path),
            theme="dark",
        )
        exporter = HTMLReportExporter(config)
        records = create_test_records(5)

        result = exporter.export(
            records=records,
            run_id="test-run-dark",
            task_name="Dark Theme Report",
        )

        assert result.success

        with open(result.output_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Dark theme uses darker colors
            assert "#1a1a2e" in content


class TestExporterRegistry:
    """Tests for exporter registry."""

    def test_list_formats(self):
        """Test listing available formats."""
        formats = exporter_registry.list_formats()

        assert "csv" in formats
        assert "jsonl" in formats
        assert "json" in formats
        assert "html" in formats

    def test_create_exporter(self):
        """Test creating exporters from registry."""
        config = ExportConfig()

        csv_exporter = create_exporter("csv", config)
        assert isinstance(csv_exporter, CSVExporter)

        jsonl_exporter = create_exporter("jsonl", config)
        assert isinstance(jsonl_exporter, JSONLExporter)

    def test_create_unknown_format(self):
        """Test creating unknown format returns None."""
        config = ExportConfig()
        exporter = create_exporter("unknown_format", config)
        assert exporter is None


class TestExportResult:
    """Tests for ExportResult."""

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ExportResult(
            success=True,
            output_path="/tmp/test.csv",
            format="csv",
            records_exported=100,
            message="Success",
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["output_path"] == "/tmp/test.csv"
        assert d["format"] == "csv"
        assert d["records_exported"] == 100
        assert "timestamp" in d
