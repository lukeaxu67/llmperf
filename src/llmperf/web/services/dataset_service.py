"""Dataset management service."""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatasetType(str, Enum):
    """Supported dataset file types."""
    JSONL = "jsonl"
    CSV = "csv"


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    name: str
    description: str = ""
    file_path: str = ""
    file_type: DatasetType = DatasetType.JSONL
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    created_at: int = 0
    updated_at: int = 0
    file_size: int = 0
    encoding: str = "utf-8"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "file_type": self.file_type.value,
            "row_count": self.row_count,
            "columns": self.columns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "file_size": self.file_size,
            "encoding": self.encoding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create from dictionary."""
        file_type = DatasetType(data.get("file_type", "jsonl"))
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            file_path=data.get("file_path", ""),
            file_type=file_type,
            row_count=data.get("row_count", 0),
            columns=data.get("columns", []),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
            file_size=data.get("file_size", 0),
            encoding=data.get("encoding", "utf-8"),
        )


class DatasetService:
    """Service for managing datasets.

    This service handles:
    - Dataset scanning and listing
    - Dataset upload with metadata generation
    - Dataset deletion
    - Dataset preview
    """

    def __init__(self, datasets_dir: Optional[str | Path] = None):
        """Initialize dataset service.

        Args:
            datasets_dir: Directory to store datasets. Defaults to data/datasets/
        """
        if datasets_dir is None:
            # Default to project root / data / datasets
            project_root = Path(__file__).parent.parent.parent.parent.parent
            datasets_dir = project_root / "data" / "datasets"

        self._datasets_dir = Path(datasets_dir)
        self._datasets_dir.mkdir(parents=True, exist_ok=True)

        # Cache for metadata
        self._metadata_cache: Dict[str, DatasetMetadata] = {}
        self._cache_loaded = False

        logger.info(f"Dataset service initialized with directory: {self._datasets_dir}")

    def _get_meta_path(self, name: str) -> Path:
        """Get metadata file path for a dataset.

        Args:
            name: Dataset name.

        Returns:
            Path to metadata file.
        """
        return self._datasets_dir / f"{name}.meta.json"

    def _get_data_path(self, name: str, file_type: DatasetType) -> Path:
        """Get data file path for a dataset.

        Args:
            name: Dataset name.
            file_type: File type.

        Returns:
            Path to data file.
        """
        return self._datasets_dir / f"{name}.{file_type.value}"

    def _detect_file_type(self, filename: str) -> Optional[DatasetType]:
        """Detect file type from filename.

        Args:
            filename: Filename.

        Returns:
            DatasetType or None.
        """
        if filename.endswith(".jsonl"):
            return DatasetType.JSONL
        elif filename.endswith(".csv"):
            return DatasetType.CSV
        return None

    def _load_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Load metadata from file.

        Args:
            name: Dataset name.

        Returns:
            DatasetMetadata or None.
        """
        meta_path = self._get_meta_path(name)
        if not meta_path.exists():
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return DatasetMetadata.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load metadata for %s: %s", name, e)
            return None

    def _save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save metadata to file.

        Args:
            metadata: Metadata to save.
        """
        meta_path = self._get_meta_path(metadata.name)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    def _count_jsonl_rows(self, path: Path) -> int:
        """Count rows in JSONL file.

        Args:
            path: Path to JSONL file.

        Returns:
            Number of rows.
        """
        count = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            logger.warning("Failed to count JSONL rows: %s", e)
        return count

    def _extract_jsonl_columns(self, path: Path) -> List[str]:
        """Extract columns from JSONL file.

        Args:
            path: Path to JSONL file.

        Returns:
            List of column names.
        """
        columns = set()
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                columns.update(data.keys())
                                if len(columns) > 100:  # Limit for performance
                                    break
                        except json.JSONDecodeError:
                            pass
                    if len(columns) > 0:
                        break  # Got columns from first valid record
        except Exception as e:
            logger.warning("Failed to extract JSONL columns: %s", e)
        return sorted(columns)

    def _count_csv_rows(self, path: Path, encoding: str = "utf-8") -> int:
        """Count rows in CSV file.

        Args:
            path: Path to CSV file.
            encoding: File encoding.

        Returns:
            Number of data rows (excluding header).
        """
        count = 0
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if any(row):  # Only count non-empty rows
                        count += 1
        except Exception as e:
            logger.warning("Failed to count CSV rows: %s", e)
        return count

    def _extract_csv_columns(self, path: Path, encoding: str = "utf-8") -> List[str]:
        """Extract columns from CSV file.

        Args:
            path: Path to CSV file.
            encoding: File encoding.

        Returns:
            List of column names.
        """
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    return [col.strip() for col in header if col.strip()]
        except Exception as e:
            logger.warning("Failed to extract CSV columns: %s", e)
        return []

    def scan(self) -> List[DatasetMetadata]:
        """Scan datasets directory and return all datasets.

        Returns:
            List of DatasetMetadata.
        """
        datasets = []

        # First, load from metadata files
        for meta_path in self._datasets_dir.glob("*.meta.json"):
            name = meta_path.stem.replace(".meta", "")
            metadata = self._load_metadata(name)
            if metadata:
                # Verify data file exists
                data_path = self._get_data_path(metadata.name, metadata.file_type)
                if data_path.exists():
                    datasets.append(metadata)
                    self._metadata_cache[name] = metadata
                else:
                    logger.warning("Data file missing for dataset: %s", name)

        self._cache_loaded = True
        logger.info("Scanned %d datasets", len(datasets))
        return datasets

    def list_datasets(self) -> List[DatasetMetadata]:
        """List all datasets.

        Returns:
            List of DatasetMetadata.
        """
        if not self._cache_loaded:
            self.scan()
        return list(self._metadata_cache.values())

    def get_dataset(self, name: str) -> Optional[DatasetMetadata]:
        """Get dataset by name.

        Args:
            name: Dataset name.

        Returns:
            DatasetMetadata or None.
        """
        if not self._cache_loaded:
            self.scan()

        if name in self._metadata_cache:
            return self._metadata_cache[name]

        # Try loading from file
        metadata = self._load_metadata(name)
        if metadata:
            self._metadata_cache[name] = metadata
        return metadata

    def upload_dataset(
        self,
        filename: str,
        content: bytes,
        description: str = "",
        encoding: str = "utf-8",
    ) -> DatasetMetadata:
        """Upload a dataset file.

        Args:
            filename: Original filename.
            content: File content.
            description: Optional description.
            encoding: File encoding.

        Returns:
            DatasetMetadata.

        Raises:
            ValueError: If file format is not supported or content is invalid.
        """
        # Detect file type
        file_type = self._detect_file_type(filename)
        if not file_type:
            raise ValueError(f"Unsupported file type: {filename}. Supported: .jsonl, .csv")

        # Extract base name without extension
        base_name = Path(filename).stem
        name = base_name

        # Handle duplicate names
        counter = 1
        while self._get_data_path(name, file_type).exists():
            name = f"{base_name}_{counter}"
            counter += 1

        # Get file paths
        data_path = self._get_data_path(name, file_type)
        meta_path = self._get_meta_path(name)

        # Save data file
        data_path.write_bytes(content)
        file_size = len(content)

        # Analyze content
        now = int(time.time())
        row_count = 0
        columns = []

        try:
            text = content.decode(encoding)

            if file_type == DatasetType.JSONL:
                # Validate JSONL
                row_count = self._count_jsonl_rows(data_path)
                columns = self._extract_jsonl_columns(data_path)

                # Validate first few lines
                for i, line in enumerate(text.split("\n")[:5]):
                    if line.strip():
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as e:
                            data_path.unlink(missing_ok=True)
                            raise ValueError(f"Invalid JSONL at line {i + 1}: {e}")

            elif file_type == DatasetType.CSV:
                # Validate CSV
                row_count = self._count_csv_rows(data_path, encoding)
                columns = self._extract_csv_columns(data_path, encoding)

                if row_count == 0:
                    data_path.unlink(missing_ok=True)
                    raise ValueError("CSV file is empty or has no data rows")

        except UnicodeDecodeError as e:
            data_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to decode file with encoding {encoding}: {e}")
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            data_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to process file: {e}")

        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            description=description,
            file_path=data_path.name,
            file_type=file_type,
            row_count=row_count,
            columns=columns,
            created_at=now,
            updated_at=now,
            file_size=file_size,
            encoding=encoding,
        )

        # Save metadata
        self._save_metadata(metadata)
        self._metadata_cache[name] = metadata

        logger.info("Uploaded dataset: %s (%d rows)", name, row_count)
        return metadata

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset.

        Args:
            name: Dataset name.

        Returns:
            True if deleted, False if not found.
        """
        metadata = self.get_dataset(name)
        if not metadata:
            return False

        try:
            # Delete data file
            data_path = self._get_data_path(name, metadata.file_type)
            if data_path.exists():
                data_path.unlink()

            # Delete metadata file
            meta_path = self._get_meta_path(name)
            if meta_path.exists():
                meta_path.unlink()

            # Remove from cache
            self._metadata_cache.pop(name, None)

            logger.info("Deleted dataset: %s", name)
            return True

        except Exception as e:
            logger.error("Failed to delete dataset %s: %s", name, e)
            return False

    def preview_dataset(
        self,
        name: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Preview dataset content.

        Args:
            name: Dataset name.
            limit: Number of rows to return.

        Returns:
            Dictionary with preview data.

        Raises:
            ValueError: If dataset not found.
        """
        metadata = self.get_dataset(name)
        if not metadata:
            raise ValueError(f"Dataset not found: {name}")

        data_path = self._get_data_path(name, metadata.file_type)
        if not data_path.exists():
            raise ValueError(f"Data file not found: {data_path}")

        records = []

        try:
            if metadata.file_type == DatasetType.JSONL:
                with data_path.open("r", encoding=metadata.encoding) as f:
                    for i, line in enumerate(f):
                        if i >= limit:
                            break
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                records.append({"error": "Invalid JSON", "raw": line})

            elif metadata.file_type == DatasetType.CSV:
                with data_path.open("r", encoding=metadata.encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= limit:
                            break
                        records.append(dict(row))

        except Exception as e:
            logger.error("Failed to preview dataset %s: %s", name, e)
            raise ValueError(f"Failed to preview dataset: {e}")

        return {
            "name": name,
            "total_rows": metadata.row_count,
            "preview_rows": len(records),
            "columns": metadata.columns,
            "records": records,
        }

    def update_metadata(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> Optional[DatasetMetadata]:
        """Update dataset metadata.

        Args:
            name: Dataset name.
            description: New description.

        Returns:
            Updated DatasetMetadata or None if not found.
        """
        metadata = self.get_dataset(name)
        if not metadata:
            return None

        if description is not None:
            metadata.description = description

        metadata.updated_at = int(time.time())

        self._save_metadata(metadata)
        self._metadata_cache[name] = metadata

        return metadata


# Global service instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Get or create dataset service instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service


def set_dataset_service(service: DatasetService) -> None:
    """Set dataset service instance (for testing)."""
    global _dataset_service
    _dataset_service = service
