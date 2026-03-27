"""JSONL exporter for LLMPerf records."""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any, Dict, List, Optional

from .base import Exporter, ExportConfig, ExportResult
from .registry import register_exporter
from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)


class JSONLExportConfig(ExportConfig):
    """Configuration for JSONL export."""
    encoding: str = "utf-8"
    """File encoding."""

    indent: Optional[int] = None
    """JSON indentation (None for compact)."""

    ensure_ascii: bool = False
    """Whether to escape non-ASCII characters."""

    include_content: bool = True
    """Whether to include reasoning and content text."""

    include_raw_usage: bool = True
    """Whether to include raw usage data."""


@register_exporter("jsonl")
class JSONLExporter(Exporter):
    """JSONL (JSON Lines) format exporter.

    Exports records to newline-delimited JSON format.
    Each line is a valid JSON object representing one record.

    Configuration:
        output_dir: Output directory (default: ".")
        encoding: File encoding (default: "utf-8")
        include_content: Include reasoning/content text (default: True)
        include_raw_usage: Include raw usage data (default: True)
        compress: Gzip compress output (default: False)
    """

    format_name = "jsonl"
    def __init__(self, config: ExportConfig):
        """Initialize JSONL exporter.

        Args:
            config: Export configuration.
        """
        # Convert to JSONL-specific config if needed
        if not isinstance(config, JSONLExportConfig):
            config = JSONLExportConfig(**config.model_dump())
        super().__init__(config)
        self._file_extension_override = None

    @property
    def jsonl_config(self) -> JSONLExportConfig:
        """Get JSONL-specific configuration."""
        return self.config  # type: ignore

    @property
    def file_extension(self) -> str:  # type: ignore
        """Get file extension considering compression."""
        if self.jsonl_config.compress:
            return ".jsonl.gz"
        return ".jsonl"

    def _prepare_record_for_jsonl(
        self,
        record: RunRecord,
    ) -> Dict[str, Any]:
        """Prepare a record for JSONL output.

        Args:
            record: The record to prepare.

        Returns:
            Dictionary ready for JSON serialization.
        """
        result = {
            "run_id": record.run_id,
            "executor_id": record.executor_id,
            "dataset_row_id": record.dataset_row_id,
            "provider": record.provider,
            "model": record.model,
            "status": record.status,
            "info": record.info,
            "tokens": {
                "prompt": record.qtokens,
                "completion": record.atokens,
                "cached": record.ctokens,
            },
            "cost": {
                "prompt": record.prompt_cost,
                "completion": record.completion_cost,
                "cache": record.cache_cost,
                "total": record.total_cost,
                "currency": record.currency,
            },
            "timing": {
                "first_response_ms": record.first_resp_time,
                "total_ms": record.last_resp_time,
            },
            "performance": {
                "chars_per_second": record.char_per_second,
                "tokens_per_second": record.token_per_second,
                "throughput": record.token_throughput,
            },
            "content_stats": {
                "total_chars": record.char_count,
                "reasoning_chars": record.reasoning_char_count,
                "content_chars": record.content_char_count,
            },
            "cache": {
                "hit": record.cache_hit,
                "ratio": record.cache_ratio,
            },
            "created_at": record.created_at,
        }

        # Optionally include raw content
        if self.jsonl_config.include_content:
            result["content"] = {
                "reasoning": record.reasoning,
                "text": record.content,
            }

        # Optionally include raw usage
        if self.jsonl_config.include_raw_usage:
            result["raw_usage"] = record.usage

        # Include extra fields
        if record.extra:
            result["extra"] = record.extra

        # Include request params if available
        if record.request_params:
            result["request_params"] = record.request_params

        return result

    def export(
        self,
        records: List[RunRecord],
        run_id: str,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Export records to JSONL.

        Args:
            records: List of records to export.
            run_id: Run identifier.
            task_name: Task name for filename.
            metadata: Optional additional metadata.

        Returns:
            ExportResult indicating success or failure.
        """
        if not records:
            return ExportResult(
                success=False,
                format=self.format_name,
                message="No records to export",
            )

        output_path = self._build_output_path(task_name)

        try:
            json_kwargs = {
                "ensure_ascii": self.jsonl_config.ensure_ascii,
            }
            if self.jsonl_config.indent is not None:
                json_kwargs["indent"] = self.jsonl_config.indent

            if self.jsonl_config.compress:
                with gzip.open(output_path, "wt", encoding=self.jsonl_config.encoding) as f:
                    for record in records:
                        record_dict = self._prepare_record_for_jsonl(record)
                        f.write(json.dumps(record_dict, **json_kwargs) + "\n")
            else:
                with output_path.open("w", encoding=self.jsonl_config.encoding) as f:
                    for record in records:
                        record_dict = self._prepare_record_for_jsonl(record)
                        f.write(json.dumps(record_dict, **json_kwargs) + "\n")

            # Calculate file size
            file_size = output_path.stat().st_size

            logger.info(
                "Exported %d records to %s (%.2f KB)",
                len(records),
                output_path,
                file_size / 1024,
            )

            return ExportResult(
                success=True,
                output_path=str(output_path),
                format=self.format_name,
                records_exported=len(records),
                message=f"Successfully exported {len(records)} records",
                metadata={
                    "encoding": self.jsonl_config.encoding,
                    "compressed": self.jsonl_config.compress,
                    "file_size_bytes": file_size,
                },
            )

        except Exception as e:
            logger.error("Failed to export JSONL: %s", e)
            return ExportResult(
                success=False,
                format=self.format_name,
                message=f"Export failed: {e}",
            )


@register_exporter("json")
class JSONExporter(JSONLExporter):
    """JSON array format exporter.

    Exports records to a single JSON array instead of JSON Lines.
    """

    format_name = "json"
    file_extension = ".json"

    def export(
        self,
        records: List[RunRecord],
        run_id: str,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Export records to JSON array.

        Args:
            records: List of records to export.
            run_id: Run identifier.
            task_name: Task name for filename.
            metadata: Optional additional metadata.

        Returns:
            ExportResult indicating success or failure.
        """
        if not records:
            return ExportResult(
                success=False,
                format=self.format_name,
                message="No records to export",
            )

        output_path = self._build_output_path(task_name)

        try:
            json_kwargs = {
                "ensure_ascii": self.jsonl_config.ensure_ascii,
                "indent": self.jsonl_config.indent or 2,
            }

            # Prepare all records
            records_data = [
                self._prepare_record_for_jsonl(record)
                for record in records
            ]

            # Build complete export data
            export_data = {
                "run_id": run_id,
                "task_name": task_name,
                "exported_at": __import__("datetime").datetime.now().isoformat(),
                "record_count": len(records),
                "records": records_data,
            }

            if metadata:
                export_data["metadata"] = metadata

            with output_path.open("w", encoding=self.jsonl_config.encoding) as f:
                json.dump(export_data, f, **json_kwargs)

            file_size = output_path.stat().st_size

            logger.info(
                "Exported %d records to %s (%.2f KB)",
                len(records),
                output_path,
                file_size / 1024,
            )

            return ExportResult(
                success=True,
                output_path=str(output_path),
                format=self.format_name,
                records_exported=len(records),
                message=f"Successfully exported {len(records)} records",
                metadata={
                    "encoding": self.jsonl_config.encoding,
                    "file_size_bytes": file_size,
                },
            )

        except Exception as e:
            logger.error("Failed to export JSON: %s", e)
            return ExportResult(
                success=False,
                format=self.format_name,
                message=f"Export failed: {e}",
            )
