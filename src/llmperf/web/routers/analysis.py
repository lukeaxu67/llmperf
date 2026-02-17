"""Analysis API router."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class SummaryResponse(BaseModel):
    """Response for summary analysis."""
    run_id: str
    summary: Dict[str, Any]


class TimeSeriesPoint(BaseModel):
    """A single point in time series."""
    timestamp: str
    value: float


class TimeSeriesResponse(BaseModel):
    """Response for time series analysis."""
    run_id: str
    metric: str
    data: List[TimeSeriesPoint]


class ComparisonItem(BaseModel):
    """Comparison data for a single executor/model."""
    executor_id: str
    provider: str
    model: str
    metrics: Dict[str, float]


class ComparisonResponse(BaseModel):
    """Response for model comparison."""
    run_id: str
    comparison: List[ComparisonItem]
    ranking: List[str]


class ExportRequest(BaseModel):
    """Request to export data."""
    format: str = Field("jsonl", description="Export format: csv, jsonl, json, html")
    output_dir: str = Field(".", description="Output directory")


class ExportResponse(BaseModel):
    """Response for export operation."""
    success: bool
    output_path: Optional[str] = None
    format: str
    records_exported: int = 0
    message: str = ""


def get_service() -> AnalysisService:
    """Get analysis service instance."""
    from ..main import get_analysis_service
    return get_analysis_service()


@router.get(
    "/{run_id}/summary",
    response_model=SummaryResponse,
    summary="Get summary statistics",
    description="Get aggregated summary statistics for a run.",
)
async def get_summary(run_id: str):
    """Get summary statistics for a run."""
    service = get_service()
    summary = service.get_summary(run_id)

    if not summary:
        raise HTTPException(status_code=404, detail="Run not found or no data")

    return SummaryResponse(run_id=run_id, summary=summary)


@router.get(
    "/{run_id}/timeseries",
    response_model=TimeSeriesResponse,
    summary="Get time series data",
    description="Get time series data for a specific metric.",
)
async def get_timeseries(
    run_id: str,
    metric: str = Query(
        ...,
        description="Metric name: latency, throughput, cost, error_rate",
    ),
    interval: str = Query(
        "1m",
        description="Aggregation interval: 1m, 5m, 15m, 1h",
    ),
):
    """Get time series data for a run."""
    service = get_service()
    data = service.get_timeseries(run_id, metric, interval)

    if not data:
        raise HTTPException(status_code=404, detail="Run not found or no data")

    points = [TimeSeriesPoint(timestamp=p["timestamp"], value=p["value"]) for p in data]

    return TimeSeriesResponse(
        run_id=run_id,
        metric=metric,
        data=points,
    )


@router.get(
    "/{run_id}/compare",
    response_model=ComparisonResponse,
    summary="Compare executors/models",
    description="Compare performance across executors and models.",
)
async def compare_executors(run_id: str):
    """Compare performance across executors."""
    service = get_service()
    comparison = service.compare_executors(run_id)

    if not comparison:
        raise HTTPException(status_code=404, detail="Run not found or no data")

    items = [
        ComparisonItem(
            executor_id=item["executor_id"],
            provider=item["provider"],
            model=item["model"],
            metrics=item["metrics"],
        )
        for item in comparison["items"]
    ]

    return ComparisonResponse(
        run_id=run_id,
        comparison=items,
        ranking=comparison.get("ranking", []),
    )


@router.get(
    "/{run_id}/anomalies",
    summary="Detect anomalies",
    description="Detect anomalies in the run data.",
)
async def detect_anomalies(
    run_id: str,
    sensitivity: float = Query(2.0, description="Z-score threshold for anomalies"),
):
    """Detect anomalies in run data."""
    service = get_service()
    anomalies = service.detect_anomalies(run_id, sensitivity)

    return {
        "run_id": run_id,
        "anomalies": anomalies,
        "sensitivity": sensitivity,
    }


@router.post(
    "/{run_id}/export",
    response_model=ExportResponse,
    summary="Export run data",
    description="Export run data in specified format.",
)
async def export_data(run_id: str, request: ExportRequest):
    """Export run data."""
    service = get_service()

    # Validate format
    valid_formats = ["csv", "jsonl", "json", "html"]
    if request.format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Supported: {valid_formats}",
        )

    result = service.export_data(
        run_id=run_id,
        format_name=request.format,
        output_dir=request.output_dir,
    )

    return ExportResponse(
        success=result.success,
        output_path=result.output_path,
        format=result.format,
        records_exported=result.records_exported,
        message=result.message,
    )


@router.get(
    "/{run_id}/report",
    summary="Generate HTML report",
    description="Generate a self-contained HTML report.",
)
async def generate_report(run_id: str):
    """Generate HTML report."""
    service = get_service()
    result = service.export_data(
        run_id=run_id,
        format_name="html",
        output_dir=".",
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)

    return {
        "run_id": run_id,
        "report_path": result.output_path,
        "message": result.message,
    }


@router.get(
    "/history",
    summary="Get historical runs",
    description="Get list of historical runs with optional filtering.",
)
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    days: int = Query(7, ge=1, le=365),
):
    """Get historical runs."""
    service = get_service()
    runs = service.get_history(limit=limit, days=days)

    return {
        "runs": runs,
        "limit": limit,
        "days": days,
    }
