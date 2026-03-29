"""Dataset management service."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmperf.config.runtime import load_runtime_config
from llmperf.datasets.sources.jsonl import parse_jsonl_test_case
from llmperf.datasets.validator import validate_jsonl_content

logger = logging.getLogger(__name__)


class DatasetType(str, Enum):
    """Supported dataset file types."""
    JSONL = "jsonl"
    CSV = "csv"


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""
    name: str
    id: str = ""
    description: str = ""
    file_path: str = ""
    file_type: DatasetType = DatasetType.JSONL
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    created_at: int = 0
    updated_at: int = 0
    file_size: int = 0
    encoding: str = "utf-8"
    source: str = "runtime"
    read_only: bool = False

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
            "source": self.source,
            "read_only": self.read_only,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], dataset_id: str = "") -> "DatasetMetadata":
        """Create from dictionary."""
        file_type = DatasetType(data.get("file_type", "jsonl"))
        return cls(
            id=dataset_id,
            name=data.get("name") or dataset_id,
            description=data.get("description", ""),
            file_path=data.get("file_path", ""),
            file_type=file_type,
            row_count=data.get("row_count", 0),
            columns=data.get("columns", []),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
            file_size=data.get("file_size", 0),
            encoding=data.get("encoding", "utf-8"),
            source=data.get("source", "runtime"),
            read_only=data.get("read_only", False),
        )


class DatasetService:
    """Service for managing datasets.

    This service handles:
    - Dataset scanning and listing
    - Dataset upload with metadata generation
    - Dataset deletion
    - Dataset preview
    """

    def __init__(
        self,
        datasets_dir: Optional[str | Path] = None,
        preset_datasets_dir: Optional[str | Path] = None,
    ):
        """Initialize dataset service.

        Args:
            datasets_dir: Directory to store datasets. Defaults to data/datasets/
        """
        project_root = Path(__file__).parent.parent.parent.parent.parent
        self._project_root = project_root

        if datasets_dir is None:
            runtime = load_runtime_config()
            self._datasets_dir = runtime.datasets_dir
            self._preset_datasets_dir = Path(preset_datasets_dir).resolve() if preset_datasets_dir else (project_root / "data" / "datasets")
        else:
            self._datasets_dir = Path(datasets_dir).resolve()
            self._preset_datasets_dir = Path(preset_datasets_dir).resolve() if preset_datasets_dir else self._datasets_dir

        self._datasets_dir.mkdir(parents=True, exist_ok=True)
        if self._preset_datasets_dir != self._datasets_dir:
            self._preset_datasets_dir.mkdir(parents=True, exist_ok=True)

        self._scan_dirs: List[Path] = [self._datasets_dir]
        if self._preset_datasets_dir not in self._scan_dirs:
            self._scan_dirs.append(self._preset_datasets_dir)

        # Cache for metadata
        self._metadata_cache: Dict[str, DatasetMetadata] = {}
        self._metadata_roots: Dict[str, Path] = {}
        self._cache_loaded = False

        logger.info(
            "Dataset service initialized with runtime dir=%s preset dir=%s",
            self._datasets_dir,
            self._preset_datasets_dir,
        )

    def _get_meta_path(self, dataset_id: str, base_dir: Optional[Path] = None) -> Path:
        """Get metadata file path for a dataset.

        Args:
            dataset_id: Stable dataset identifier.

        Returns:
            Path to metadata file.
        """
        target_dir = base_dir or self._datasets_dir
        return target_dir / f"{dataset_id}.meta.json"

    def _get_data_path(self, dataset_id: str, file_type: DatasetType, base_dir: Optional[Path] = None) -> Path:
        """Get data file path for a dataset.

        Args:
            dataset_id: Stable dataset identifier.
            file_type: File type.

        Returns:
            Path to data file.
        """
        target_dir = base_dir or self._datasets_dir
        return target_dir / f"{dataset_id}.{file_type.value}"

    def _resolve_data_path(self, metadata: DatasetMetadata) -> Path:
        """Resolve the actual dataset file from metadata."""
        base_dir = self._metadata_roots.get(metadata.id or metadata.name, self._datasets_dir)
        if metadata.file_path:
            file_path = Path(metadata.file_path)
            if file_path.is_absolute():
                return file_path

            candidates = [
                base_dir / file_path,
                self._project_root / file_path,
                file_path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        return self._get_data_path(metadata.id, metadata.file_type, base_dir=base_dir)

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

    def _load_metadata(self, dataset_id: str, meta_dir: Optional[Path] = None) -> Optional[DatasetMetadata]:
        """Load metadata from file.

        Args:
            dataset_id: Stable dataset identifier.

        Returns:
            DatasetMetadata or None.
        """
        meta_root = meta_dir or self._datasets_dir
        meta_path = self._get_meta_path(dataset_id, base_dir=meta_root)
        if not meta_path.exists():
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            metadata = DatasetMetadata.from_dict(data, dataset_id=dataset_id)
            if not metadata.file_path:
                metadata.file_path = self._get_data_path(dataset_id, metadata.file_type, base_dir=meta_root).name
            metadata.source = "runtime" if meta_root == self._datasets_dir else "builtin"
            metadata.read_only = meta_root != self._datasets_dir
            self._metadata_roots[dataset_id] = meta_root
            return metadata
        except Exception as e:
            logger.warning("Failed to load metadata for %s: %s", dataset_id, e)
            return None

    def _save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save metadata to file.

        Args:
            metadata: Metadata to save.
        """
        meta_root = self._metadata_roots.get(metadata.id or metadata.name, self._datasets_dir)
        meta_path = self._get_meta_path(metadata.id or metadata.name, base_dir=meta_root)
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

    @staticmethod
    def _normalize_preview_jsonl_record(line: str, index: int) -> Dict[str, Any]:
        test_case = parse_jsonl_test_case(line, index=index)
        return test_case.model_dump()

    def _analyze_dataset_content(
        self,
        *,
        filename: str,
        content: bytes,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        file_type = self._detect_file_type(filename)
        if not file_type:
            raise ValueError(f"Unsupported file type: {filename}. Supported: .jsonl, .csv")

        row_count = 0
        columns: List[str] = []
        preview_records: List[Dict[str, Any]] = []

        try:
            text = content.decode(encoding)
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode file with encoding {encoding}: {e}") from e

        if file_type == DatasetType.JSONL:
            validation = validate_jsonl_content(text)
            if not validation.valid:
                first_error = validation.errors[0].message if validation.errors else "Invalid JSONL content"
                raise ValueError(first_error)
            row_count = int(validation.statistics.get("total_records", 0))
            columns = ["id", "messages"]
            for index, line in enumerate(text.splitlines()):
                stripped = line.strip()
                if not stripped:
                    continue
                preview_records.append(self._normalize_preview_jsonl_record(stripped, index))
                if len(preview_records) >= 5:
                    break
        elif file_type == DatasetType.CSV:
            temp_path = self._datasets_dir / f".tmp-validate-{int(time.time() * 1000)}.csv"
            try:
                temp_path.write_bytes(content)
                row_count = self._count_csv_rows(temp_path, encoding)
                columns = self._extract_csv_columns(temp_path, encoding)
                with temp_path.open("r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    for index, row in enumerate(reader):
                        preview_records.append(dict(row))
                        if index >= 4:
                            break
            finally:
                temp_path.unlink(missing_ok=True)
            if row_count == 0:
                raise ValueError("CSV file is empty or has no data rows")

        return {
            "file_type": file_type,
            "row_count": row_count,
            "columns": columns,
            "preview_records": preview_records,
            "encoding": encoding,
        }

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
        self._metadata_cache = {}
        self._metadata_roots = {}

        for root_dir in self._scan_dirs:
            if not root_dir.exists():
                continue
            for meta_path in root_dir.glob("*.meta.json"):
                dataset_id = meta_path.stem.replace(".meta", "")
                if dataset_id in self._metadata_cache:
                    continue
                metadata = self._load_metadata(dataset_id, meta_dir=root_dir)
                if metadata:
                    data_path = self._resolve_data_path(metadata)
                    if data_path.exists():
                        datasets.append(metadata)
                        self._metadata_cache[dataset_id] = metadata
                        self._metadata_roots[dataset_id] = root_dir
                    else:
                        logger.warning("Data file missing for dataset: %s (%s)", dataset_id, data_path)

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

    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get dataset by name.

        Args:
            dataset_id: Stable dataset identifier.

        Returns:
            DatasetMetadata or None.
        """
        if not self._cache_loaded:
            self.scan()

        if dataset_id in self._metadata_cache:
            return self._metadata_cache[dataset_id]

        # Try loading from file
        for root_dir in self._scan_dirs:
            metadata = self._load_metadata(dataset_id, meta_dir=root_dir)
            if metadata:
                self._metadata_cache[dataset_id] = metadata
                self._metadata_roots[dataset_id] = root_dir
                return metadata
        return None

    def upload_dataset(
        self,
        filename: str,
        content: bytes,
        name: str = "",
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
        analysis = self._analyze_dataset_content(
            filename=filename,
            content=content,
            encoding=encoding,
        )
        file_type: DatasetType = analysis["file_type"]

        # Extract base name without extension
        base_name = Path(filename).stem
        dataset_id = base_name

        # Handle duplicate names
        counter = 1
        while any(
            self._get_data_path(dataset_id, file_type, base_dir=scan_dir).exists()
            or self._get_meta_path(dataset_id, base_dir=scan_dir).exists()
            for scan_dir in self._scan_dirs
        ):
            dataset_id = f"{base_name}_{counter}"
            counter += 1

        # Get file paths
        data_path = self._get_data_path(dataset_id, file_type, base_dir=self._datasets_dir)
        data_path.write_bytes(content)
        file_size = len(content)
        now = int(time.time())

        # Create metadata
        metadata = DatasetMetadata(
            id=dataset_id,
            name=name.strip() or dataset_id,
            description=description,
            file_path=data_path.name,
            file_type=file_type,
            row_count=analysis["row_count"],
            columns=analysis["columns"],
            created_at=now,
            updated_at=now,
            file_size=file_size,
            encoding=encoding,
            source="runtime",
            read_only=False,
        )

        # Save metadata
        self._metadata_roots[dataset_id] = self._datasets_dir
        self._save_metadata(metadata)
        self._metadata_cache[dataset_id] = metadata

        logger.info("Uploaded dataset: %s (%d rows)", dataset_id, metadata.row_count)
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
        if metadata.read_only:
            raise ValueError("Built-in datasets are read-only and cannot be deleted")

        try:
            # Delete data file
            data_path = self._resolve_data_path(metadata)
            if data_path.exists():
                data_path.unlink()

            # Delete metadata file
            meta_root = self._metadata_roots.get(name, self._datasets_dir)
            meta_path = self._get_meta_path(name, base_dir=meta_root)
            if meta_path.exists():
                meta_path.unlink()

            # Remove from cache
            self._metadata_cache.pop(name, None)
            self._metadata_roots.pop(name, None)

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

        data_path = self._resolve_data_path(metadata)
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
                                records.append(self._normalize_preview_jsonl_record(line, i))
                            except Exception:
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
            "id": metadata.id,
            "name": metadata.name,
            "file_path": metadata.file_path,
            "total_rows": metadata.row_count,
            "preview_rows": len(records),
            "columns": metadata.columns,
            "records": records,
        }

    def update_metadata(
        self,
        name: str,
        display_name: Optional[str] = None,
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
        if metadata.read_only:
            raise ValueError("Built-in datasets are read-only and cannot be modified")

        if display_name is not None and display_name.strip():
            metadata.name = display_name.strip()

        if description is not None:
            metadata.description = description

        metadata.updated_at = int(time.time())

        self._save_metadata(metadata)
        self._metadata_cache[name] = metadata

        return metadata

    def validate_upload(
        self,
        *,
        filename: str,
        content: bytes,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        analysis = self._analyze_dataset_content(
            filename=filename,
            content=content,
            encoding=encoding,
        )
        return {
            "valid": True,
            "file_type": analysis["file_type"].value,
            "row_count": analysis["row_count"],
            "columns": analysis["columns"],
            "preview_records": analysis["preview_records"],
            "encoding": analysis["encoding"],
        }


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
