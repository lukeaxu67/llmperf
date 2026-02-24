"""Dataset management API router."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body
from pydantic import BaseModel, Field

from ..services.dataset_service import (
    DatasetService,
    DatasetMetadata,
    DatasetType,
    get_dataset_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class DatasetListResponse(BaseModel):
    """Response for dataset list."""
    datasets: List[Dict[str, Any]]
    total: int


class DatasetDetailResponse(BaseModel):
    """Response for dataset details."""
    name: str
    description: str
    file_path: str
    file_type: str
    row_count: int
    columns: List[str]
    created_at: int
    updated_at: int
    file_size: int
    encoding: str


class DatasetPreviewResponse(BaseModel):
    """Response for dataset preview."""
    name: str
    total_rows: int
    preview_rows: int
    columns: List[str]
    records: List[Dict[str, Any]]


class DatasetUploadResponse(BaseModel):
    """Response for dataset upload."""
    name: str
    description: str
    file_type: str
    row_count: int
    columns: List[str]
    file_size: int
    message: str


class DatasetUpdateRequest(BaseModel):
    """Request to update dataset metadata."""
    description: str = Field("", description="Dataset description")


def get_service() -> DatasetService:
    """Get dataset service instance."""
    return get_dataset_service()


def _metadata_to_dict(metadata: DatasetMetadata) -> Dict[str, Any]:
    """Convert metadata to dictionary."""
    return {
        "name": metadata.name,
        "description": metadata.description,
        "file_path": metadata.file_path,
        "file_type": metadata.file_type.value,
        "row_count": metadata.row_count,
        "columns": metadata.columns,
        "created_at": metadata.created_at,
        "updated_at": metadata.updated_at,
        "file_size": metadata.file_size,
        "encoding": metadata.encoding,
    }


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List all datasets",
    description="Get a list of all available datasets with their metadata.",
)
async def list_datasets():
    """List all datasets."""
    service = get_service()
    datasets = service.list_datasets()

    return DatasetListResponse(
        datasets=[_metadata_to_dict(d) for d in datasets],
        total=len(datasets),
    )


@router.get(
    "/{name}",
    response_model=DatasetDetailResponse,
    summary="Get dataset details",
    description="Get detailed information about a specific dataset.",
)
async def get_dataset(name: str):
    """Get dataset details."""
    service = get_service()
    metadata = service.get_dataset(name)

    if not metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetDetailResponse(**_metadata_to_dict(metadata))


@router.get(
    "/{name}/preview",
    response_model=DatasetPreviewResponse,
    summary="Preview dataset content",
    description="Preview the first few rows of a dataset.",
)
async def preview_dataset(
    name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of rows to preview"),
):
    """Preview dataset content."""
    service = get_service()

    try:
        preview = service.preview_dataset(name, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return DatasetPreviewResponse(**preview)


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    summary="Upload a dataset",
    description="Upload a new dataset file (JSONL or CSV). Metadata will be auto-generated.",
)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file (.jsonl or .csv)"),
    description: str = Body("", description="Optional dataset description"),
):
    """Upload a dataset file."""
    service = get_service()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Read file content
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    # Upload dataset
    try:
        metadata = service.upload_dataset(
            filename=file.filename,
            content=content,
            description=description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to upload dataset")
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {e}")

    return DatasetUploadResponse(
        name=metadata.name,
        description=metadata.description,
        file_type=metadata.file_type.value,
        row_count=metadata.row_count,
        columns=metadata.columns,
        file_size=metadata.file_size,
        message=f"Dataset uploaded successfully with {metadata.row_count} rows",
    )


@router.delete(
    "/{name}",
    summary="Delete a dataset",
    description="Delete a dataset and its metadata file.",
)
async def delete_dataset(name: str):
    """Delete a dataset."""
    service = get_service()
    success = service.delete_dataset(name)

    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {"message": "Dataset deleted", "name": name}


@router.post(
    "/scan",
    response_model=DatasetListResponse,
    summary="Scan for datasets",
    description="Manually trigger a scan of the datasets directory.",
)
async def scan_datasets():
    """Scan for datasets."""
    service = get_service()
    datasets = service.scan()

    return DatasetListResponse(
        datasets=[_metadata_to_dict(d) for d in datasets],
        total=len(datasets),
    )


@router.patch(
    "/{name}",
    response_model=DatasetDetailResponse,
    summary="Update dataset metadata",
    description="Update dataset metadata (e.g., description).",
)
async def update_dataset(
    name: str,
    request: DatasetUpdateRequest,
):
    """Update dataset metadata."""
    service = get_service()
    metadata = service.update_metadata(name, description=request.description)

    if not metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetDetailResponse(**_metadata_to_dict(metadata))


@router.get(
    "/formats/list",
    summary="List supported formats",
    description="Get a list of supported dataset file formats.",
)
async def list_formats():
    """List supported dataset formats."""
    return {
        "formats": [
            {
                "type": "jsonl",
                "extension": ".jsonl",
                "description": "JSON Lines - one JSON object per line",
                "columns": "auto-detected from keys",
            },
            {
                "type": "csv",
                "extension": ".csv",
                "description": "Comma Separated Values",
                "columns": "from header row",
            },
        ]
    }


# Legacy endpoints for compatibility
from pathlib import Path
RESOURCE_DIR = Path(__file__).parent.parent.parent.parent.parent / "resource"


class LegacyDatasetInfo(BaseModel):
    """Information about a dataset (legacy format)."""
    name: str
    path: str
    size: int
    record_count: int
    format: str
    encoding: str = "utf-8"


@router.get(
    "/legacy",
    response_model=List[LegacyDatasetInfo],
    summary="List datasets (legacy)",
    description="Legacy endpoint for backward compatibility.",
    include_in_schema=False,
)
async def list_datasets_legacy():
    """List available datasets (legacy endpoint)."""
    import json

    datasets = []

    if RESOURCE_DIR.exists():
        for path in RESOURCE_DIR.glob("*.jsonl"):
            try:
                record_count = 0
                with path.open("r", encoding="utf-8") as f:
                    for _ in f:
                        record_count += 1

                datasets.append(LegacyDatasetInfo(
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
