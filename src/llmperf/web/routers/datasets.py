"""Dataset management API router."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from ..services.dataset_service import DatasetMetadata, DatasetService, get_dataset_service

logger = logging.getLogger(__name__)

router = APIRouter()


class DatasetListResponse(BaseModel):
    datasets: List[Dict[str, Any]]
    total: int


class DatasetDetailResponse(BaseModel):
    id: str
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
    source: str = "runtime"
    read_only: bool = False


class DatasetPreviewResponse(BaseModel):
    id: str
    name: str
    file_path: str
    total_rows: int
    preview_rows: int
    columns: List[str]
    records: List[Dict[str, Any]]


class DatasetUploadResponse(BaseModel):
    id: str
    name: str
    description: str
    file_type: str
    row_count: int
    columns: List[str]
    file_size: int
    encoding: str
    source: str = "runtime"
    read_only: bool = False
    message: str


class DatasetValidateResponse(BaseModel):
    valid: bool
    file_type: str
    row_count: int
    columns: List[str]
    preview_records: List[Dict[str, Any]]
    encoding: str


class DatasetUpdateRequest(BaseModel):
    name: str = Field("", description="Display name")
    description: str = Field("", description="Dataset description")


def get_service() -> DatasetService:
    return get_dataset_service()


def _metadata_to_dict(metadata: DatasetMetadata) -> Dict[str, Any]:
    return {
        "id": metadata.id,
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
        "source": metadata.source,
        "read_only": metadata.read_only,
    }


@router.get("", response_model=DatasetListResponse, summary="List all datasets")
async def list_datasets():
    service = get_service()
    datasets = service.list_datasets()
    return DatasetListResponse(datasets=[_metadata_to_dict(item) for item in datasets], total=len(datasets))


@router.post("/validate", response_model=DatasetValidateResponse, summary="Validate dataset upload")
async def validate_dataset_upload(
    file: UploadFile = File(..., description="Dataset file (.jsonl or .csv)"),
    encoding: str = Form("utf-8", description="File encoding"),
):
    service = get_service()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        result = service.validate_upload(filename=file.filename, content=content, encoding=encoding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DatasetValidateResponse(**result)


@router.post("/upload", response_model=DatasetUploadResponse, summary="Upload a dataset")
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file (.jsonl or .csv)"),
    name: str = Form("", description="Dataset display name"),
    description: str = Form("", description="Optional dataset description"),
    encoding: str = Form("utf-8", description="File encoding"),
):
    service = get_service()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        metadata = service.upload_dataset(
            filename=file.filename,
            content=content,
            name=name,
            description=description,
            encoding=encoding,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to upload dataset")
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {exc}") from exc

    return DatasetUploadResponse(
        id=metadata.id,
        name=metadata.name,
        description=metadata.description,
        file_type=metadata.file_type.value,
        row_count=metadata.row_count,
        columns=metadata.columns,
        file_size=metadata.file_size,
        encoding=metadata.encoding,
        source=metadata.source,
        read_only=metadata.read_only,
        message=f"Dataset uploaded successfully with {metadata.row_count} rows",
    )


@router.post("/scan", response_model=DatasetListResponse, summary="Scan for datasets")
async def scan_datasets():
    service = get_service()
    datasets = service.scan()
    return DatasetListResponse(datasets=[_metadata_to_dict(item) for item in datasets], total=len(datasets))


@router.get("/formats/list", summary="List supported formats")
async def list_formats():
    return {
        "formats": [
            {
                "type": "jsonl",
                "extension": ".jsonl",
                "description": "JSON Lines - one object or message array per line",
                "columns": ["id", "messages"],
            },
            {
                "type": "csv",
                "extension": ".csv",
                "description": "Comma separated values",
                "columns": "from header row",
            },
        ]
    }


RESOURCE_DIR = Path(__file__).parent.parent.parent.parent.parent / "resource"


class LegacyDatasetInfo(BaseModel):
    name: str
    path: str
    size: int
    record_count: int
    format: str
    encoding: str = "utf-8"


@router.get("/legacy", response_model=List[LegacyDatasetInfo], include_in_schema=False)
async def list_datasets_legacy():
    datasets: List[LegacyDatasetInfo] = []

    if RESOURCE_DIR.exists():
        for path in RESOURCE_DIR.glob("*.jsonl"):
            try:
                record_count = 0
                with path.open("r", encoding="utf-8") as handle:
                    for _ in handle:
                        record_count += 1
                datasets.append(
                    LegacyDatasetInfo(
                        name=path.stem,
                        path=str(path.relative_to(RESOURCE_DIR.parent)),
                        size=path.stat().st_size,
                        record_count=record_count,
                        format="jsonl",
                        encoding="utf-8",
                    )
                )
            except Exception as exc:
                logger.warning("Failed to read dataset %s: %s", path, exc)

    return datasets


@router.get("/{name}", response_model=DatasetDetailResponse, summary="Get dataset details")
async def get_dataset(name: str):
    service = get_service()
    metadata = service.get_dataset(name)
    if not metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetDetailResponse(**_metadata_to_dict(metadata))


@router.get("/{name}/preview", response_model=DatasetPreviewResponse, summary="Preview dataset content")
async def preview_dataset(
    name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of rows to preview"),
):
    service = get_service()
    try:
        preview = service.preview_dataset(name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DatasetPreviewResponse(**preview)


@router.delete("/{name}", summary="Delete a dataset")
async def delete_dataset(name: str):
    service = get_service()
    try:
        success = service.delete_dataset(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {"message": "Dataset deleted", "name": name}


@router.patch("/{name}", response_model=DatasetDetailResponse, summary="Update dataset metadata")
async def update_dataset(name: str, request: DatasetUpdateRequest):
    service = get_service()
    try:
        metadata = service.update_metadata(
            name,
            display_name=request.name,
            description=request.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not metadata:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetDetailResponse(**_metadata_to_dict(metadata))
