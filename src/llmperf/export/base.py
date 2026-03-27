"""Base classes for export functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from llmperf.records.model import RunRecord


class ExportConfig(BaseModel):
    """Base configuration for exporters."""
    output_dir: str = "."
    """Directory to write output files."""

    filename_prefix: str = "export"
    """Prefix for output filename."""

    include_metadata: bool = True
    """Whether to include run metadata."""

    compress: bool = False
    """Whether to compress output (if supported)."""


@dataclass
class ExportResult:
    """Result of an export operation.

    Attributes:
        success: Whether the export was successful.
        output_path: Path to the output file(s).
        format: Export format name.
        records_exported: Number of records exported.
        message: Optional success/error message.
        metadata: Additional metadata about the export.
        timestamp: When the export was performed.
    """
    success: bool
    output_path: Optional[str] = None
    format: str = "unknown"
    records_exported: int = 0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "format": self.format,
            "records_exported": self.records_exported,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class Exporter(ABC):
    """Abstract base class for exporters.

    All exporters must implement this interface.
    """

    # Format identifier (e.g., "csv", "json", "html")
    format_name: str = "base"
    # File extension (e.g., ".csv", ".json")
    file_extension: str = ".txt"

    def __init__(self, config: ExportConfig):
        """Initialize the exporter with configuration.

        Args:
            config: Export configuration.
        """
        self.config = config

    @abstractmethod
    def export(
        self,
        records: List[RunRecord],
        run_id: str,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Export records to the configured format.

        Args:
            records: List of records to export.
            run_id: Run identifier.
            task_name: Task name for filename.
            metadata: Optional additional metadata.

        Returns:
            ExportResult indicating success or failure.
        """
        raise NotImplementedError

    def _build_output_path(
        self,
        task_name: str,
        suffix: str = "",
    ) -> Path:
        """Build the output file path.

        Args:
            task_name: Task name for filename.
            suffix: Optional suffix to add.

        Returns:
            Path object for the output file.
        """
        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize task name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)
        safe_name = safe_name[:50]  # Limit length

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        filename_parts = [self.config.filename_prefix, safe_name, timestamp]
        if suffix:
            filename_parts.append(suffix)

        filename = "-".join(filename_parts) + self.file_extension

        return output_dir / filename

    def _prepare_record_dict(
        self,
        record: RunRecord,
        include_extra: bool = True,
    ) -> Dict[str, Any]:
        """Prepare a record for export.

        Args:
            record: The record to prepare.
            include_extra: Whether to include extra fields.

        Returns:
            Dictionary representation of the record.
        """
        result = {
            "run_id": record.run_id,
            "executor_id": record.executor_id,
            "dataset_row_id": record.dataset_row_id,
            "provider": record.provider,
            "model": record.model,
            "status": record.status,
            "info": record.info,
            "qtokens": record.qtokens,
            "atokens": record.atokens,
            "ctokens": record.ctokens,
            "prompt_cost": record.prompt_cost,
            "completion_cost": record.completion_cost,
            "cache_cost": record.cache_cost,
            "total_cost": record.total_cost,
            "currency": record.currency,
            "first_resp_time": record.first_resp_time,
            "last_resp_time": record.last_resp_time,
            "char_per_second": record.char_per_second,
            "token_per_second": record.token_per_second,
            "token_throughput": record.token_throughput,
            "char_count": record.char_count,
            "reasoning_char_count": record.reasoning_char_count,
            "content_char_count": record.content_char_count,
            "cache_hit": record.cache_hit,
            "cache_ratio": record.cache_ratio,
            "created_at": record.created_at,
        }

        if include_extra and record.extra:
            result["extra"] = record.extra

        return result
