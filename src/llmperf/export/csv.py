"""CSV exporter for LLMPerf records."""

from __future__ import annotations

import csv
import logging
from typing import Any, Dict, List, Optional

from .base import Exporter, ExportConfig, ExportResult
from .registry import register_exporter
from llmperf.records.model import RunRecord, field_aliases

logger = logging.getLogger(__name__)


class CSVExportConfig(ExportConfig):
    """Configuration for CSV export."""
    delimiter: str = ","
    """CSV delimiter character."""

    quotechar: str = '"'
    """CSV quote character."""

    include_headers: bool = True
    """Whether to include header row."""

    encoding: str = "utf-8"
    """File encoding."""

    flatten_nested: bool = True
    """Whether to flatten nested fields to JSON strings."""

    columns: Optional[List[str]] = None
    """Specific columns to include. If None, includes all."""


@register_exporter("csv")
class CSVExporter(Exporter):
    """CSV format exporter.

    Exports records to comma-separated values format.

    Configuration:
        output_dir: Output directory (default: ".")
        delimiter: CSV delimiter (default: ",")
        quotechar: Quote character (default: '"')
        include_headers: Include header row (default: True)
        encoding: File encoding (default: "utf-8")
        columns: Specific columns to export (default: all)
    """

    format_name = "csv"
    file_extension = ".csv"

    def __init__(self, config: ExportConfig):
        """Initialize CSV exporter.

        Args:
            config: Export configuration.
        """
        # Convert to CSV-specific config if needed
        if not isinstance(config, CSVExportConfig):
            config = CSVExportConfig(**config.model_dump())
        super().__init__(config)
        self._field_aliases = field_aliases()

    @property
    def csv_config(self) -> CSVExportConfig:
        """Get CSV-specific configuration."""
        return self.config  # type: ignore

    def _get_column_headers(self) -> List[str]:
        """Get column headers with aliases applied.

        Returns:
            List of column header names.
        """
        # Define standard column order
        standard_columns = [
            "run_id",
            "executor_id",
            "dataset_row_id",
            "provider",
            "model",
            "status",
            "qtokens",
            "atokens",
            "ctokens",
            "prompt_cost",
            "completion_cost",
            "cache_cost",
            "total_cost",
            "currency",
            "first_resp_time",
            "last_resp_time",
            "char_per_second",
            "token_per_second",
            "token_throughput",
            "char_count",
            "reasoning_char_count",
            "content_char_count",
            "cache_hit",
            "cache_ratio",
            "created_at",
        ]

        # Apply aliases
        headers = []
        for col in standard_columns:
            alias = self._field_aliases.get(col, col)
            headers.append(alias)

        return headers

    def _flatten_value(self, value: Any) -> str:
        """Flatten a value for CSV output.

        Args:
            value: The value to flatten.

        Returns:
            String representation of the value.
        """
        if value is None:
            return ""
        elif isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, (list, dict)):
            import json
            return json.dumps(value, ensure_ascii=False)
        else:
            return str(value)

    def export(
        self,
        records: List[RunRecord],
        run_id: str,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Export records to CSV.

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
            headers = self._get_column_headers()

            with output_path.open(
                "w",
                newline="",
                encoding=self.csv_config.encoding,
            ) as csvfile:
                writer = csv.writer(
                    csvfile,
                    delimiter=self.csv_config.delimiter,
                    quotechar=self.csv_config.quotechar,
                    quoting=csv.QUOTE_MINIMAL,
                )

                if self.csv_config.include_headers:
                    writer.writerow(headers)

                for record in records:
                    row_dict = self._prepare_record_dict(
                        record,
                        include_extra=self.csv_config.flatten_nested,
                    )
                    row = [self._flatten_value(row_dict.get(col, "")) for col in [
                        "run_id",
                        "executor_id",
                        "dataset_row_id",
                        "provider",
                        "model",
                        "status",
                        "qtokens",
                        "atokens",
                        "ctokens",
                        "prompt_cost",
                        "completion_cost",
                        "cache_cost",
                        "total_cost",
                        "currency",
                        "first_resp_time",
                        "last_resp_time",
                        "char_per_second",
                        "token_per_second",
                        "token_throughput",
                        "char_count",
                        "reasoning_char_count",
                        "content_char_count",
                        "cache_hit",
                        "cache_ratio",
                        "created_at",
                    ]]
                    writer.writerow(row)

            logger.info(
                "Exported %d records to %s",
                len(records),
                output_path,
            )

            return ExportResult(
                success=True,
                output_path=str(output_path),
                format=self.format_name,
                records_exported=len(records),
                message=f"Successfully exported {len(records)} records",
                metadata={
                    "encoding": self.csv_config.encoding,
                    "delimiter": self.csv_config.delimiter,
                },
            )

        except Exception as e:
            logger.error("Failed to export CSV: %s", e)
            return ExportResult(
                success=False,
                format=self.format_name,
                message=f"Export failed: {e}",
            )
