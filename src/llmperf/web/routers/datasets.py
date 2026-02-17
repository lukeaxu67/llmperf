"""Dataset management API router."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Default resource directory
RESOURCE_DIR = Path(__file__).parent.parent.parent.parent.parent / "resource"


class DatasetInfo(BaseModel):
    """Information about a dataset."""
    name: str
    path: str
    size: int
    record_count: int
    format: str
    encoding: str = "utf-8"


class DatasetPreview(BaseModel):
    """Preview of dataset contents."""
    name: str
    records: List[Dict[str, Any]]
    total_records: int
    preview_count: int


class ValidationResult(BaseModel):
    """Result of dataset validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    record_count: int = 0


class DatasetUploadResponse(BaseModel):
    """Response for dataset upload."""
    name: str
    path: str
    size: int
    record_count: int
    message: str


@router.get(
    "",
    response_model=List[DatasetInfo],
    summary="List datasets",
    description="Get a list of available datasets.",
)
async def list_datasets():
    """List available datasets."""
    datasets = []

    if RESOURCE_DIR.exists():
        for path in RESOURCE_DIR.glob("*.jsonl"):
            try:
                # Count records
                record_count = 0
                with path.open("r", encoding="utf-8") as f:
                    for _ in f:
                        record_count += 1

                datasets.append(DatasetInfo(
                    name=path.stem,
                    path=str(path.relative_to(RESOURCE_DIR.parent)),
                    size=path.stat().st_size,
                    record_count=record_count,
                    format="jsonl",
                    encoding="utf-8",
                ))
            except Exception as e:
                logger.warning("Failed to read dataset %s: %s", path, e)

    return datasets


@router.get(
    "/{name}",
    response_model=DatasetPreview,
    summary="Get dataset preview",
    description="Get a preview of dataset contents.",
)
async def get_dataset(
    name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of records to preview"),
):
    """Get dataset preview."""
    # Security: prevent path traversal
    safe_name = name.replace("..", "").replace("/", "").replace("\\", "")
    dataset_path = RESOURCE_DIR / f"{safe_name}.jsonl"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    records = []
    total_count = 0

    try:
        with dataset_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                total_count += 1

                if len(records) < limit:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        records.append({"error": str(e), "line": i + 1})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}")

    return DatasetPreview(
        name=name,
        records=records,
        total_records=total_count,
        preview_count=len(records),
    )


@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="Validate dataset",
    description="Validate a dataset file format and content.",
)
async def validate_dataset(file: UploadFile = File(...)):
    """Validate a dataset file."""
    errors = []
    warnings = []
    record_count = 0

    if not file.filename:
        return ValidationResult(
            valid=False,
            errors=["No filename provided"],
        )

    if not file.filename.endswith(".jsonl"):
        errors.append("File must be a JSONL file (.jsonl)")
        return ValidationResult(valid=False, errors=errors)

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return ValidationResult(
            valid=False,
            errors=["File must be UTF-8 encoded"],
        )

    lines = text.strip().split("\n")

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
            record_count += 1

            # Validate record structure
            if not isinstance(record, dict):
                errors.append(f"Line {i}: Record must be a JSON object")
                continue

            # Check for required fields
            if "messages" not in record:
                errors.append(f"Line {i}: Missing required field 'messages'")
            elif not isinstance(record["messages"], list):
                errors.append(f"Line {i}: 'messages' must be a list")
            else:
                for j, msg in enumerate(record["messages"]):
                    if not isinstance(msg, dict):
                        errors.append(f"Line {i}: messages[{j}] must be an object")
                    elif "role" not in msg:
                        errors.append(f"Line {i}: messages[{j}] missing 'role'")
                    elif "content" not in msg:
                        errors.append(f"Line {i}: messages[{j}] missing 'content'")

            # Check for optional but recommended fields
            if "id" not in record:
                warnings.append(f"Line {i}: Missing optional field 'id'")

            # Check message roles
            valid_roles = {"system", "user", "assistant", "tool"}
            if isinstance(record.get("messages"), list):
                for j, msg in enumerate(record.get("messages", [])):
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        if role and role not in valid_roles:
                            warnings.append(
                                f"Line {i}: messages[{j}] has unusual role '{role}'"
                            )

        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: Invalid JSON - {e}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        record_count=record_count,
    )


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    summary="Upload dataset",
    description="Upload a new dataset file.",
)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.endswith(".jsonl"):
        raise HTTPException(
            status_code=400,
            detail="File must be a JSONL file (.jsonl)",
        )

    # Ensure resource directory exists
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

    # Save file
    dest_path = RESOURCE_DIR / file.filename
    content = await file.read()

    # Validate content
    try:
        text = content.decode("utf-8")
        record_count = 0
        for line in text.strip().split("\n"):
            if line.strip():
                json.loads(line)
                record_count += 1
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset file: {e}")

    # Write file
    dest_path.write_bytes(content)

    return DatasetUploadResponse(
        name=file.filename,
        path=str(dest_path),
        size=len(content),
        record_count=record_count,
        message=f"Dataset uploaded successfully with {record_count} records",
    )


@router.delete(
    "/{name}",
    summary="Delete dataset",
    description="Delete a dataset file.",
)
async def delete_dataset(name: str):
    """Delete a dataset file."""
    # Security: prevent path traversal
    safe_name = name.replace("..", "").replace("/", "").replace("\\", "")
    dataset_path = RESOURCE_DIR / f"{safe_name}.jsonl"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        dataset_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}")

    return {"message": "Dataset deleted", "name": name}


@router.get(
    "/generators/list",
    summary="List data generators",
    description="Get a list of available data generators.",
)
async def list_generators():
    """List available data generators."""
    from llmperf.datasets.dataset_source_registry import list_sources

    generators = []
    for source_type, source_info in list_sources().items():
        generators.append({
            "type": source_type,
            "description": source_info.get("description", ""),
        })

    return {"generators": generators}
