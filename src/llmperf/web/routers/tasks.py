"""Task management API router."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.task_service import TaskService, TaskStatus, TaskInfo

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class TaskCreateRequest(BaseModel):
    """Request to create a new task."""
    config_path: Optional[str] = Field(None, description="Path to YAML config file")
    config_content: Optional[str] = Field(None, description="YAML config content")
    pricing_path: Optional[str] = Field(None, description="Path to pricing file")
    run_id: Optional[str] = Field(None, description="Optional run ID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "config_path": "template/0.1.0.yaml",
            }
        }
    }


class TaskResponse(BaseModel):
    """Response for task operations."""
    run_id: str
    status: TaskStatus
    message: str = ""
    created_at: Optional[datetime] = None


class TaskListResponse(BaseModel):
    """Response for task list."""
    tasks: List[TaskInfo]
    total: int


class TaskProgressResponse(BaseModel):
    """Response for task progress."""
    run_id: str
    status: TaskStatus
    progress_percent: float
    completed: int
    total: int
    elapsed_seconds: float
    eta_seconds: Optional[float] = None
    success_count: int = 0
    error_count: int = 0
    current_cost: float = 0.0
    currency: str = "CNY"


class TaskStatsResponse(BaseModel):
    """Response for task statistics."""
    run_id: str
    total_requests: int
    success_count: int
    error_count: int
    success_rate: float
    total_cost: float
    currency: str
    avg_first_resp_time: float
    avg_char_per_second: float
    avg_token_throughput: float


def get_service() -> TaskService:
    """Get task service instance."""
    from ..main import get_task_service
    return get_task_service()


@router.post(
    "",
    response_model=TaskResponse,
    summary="Create a new task",
    description="Create and start a new benchmarking task from a YAML configuration.",
)
async def create_task(
    request: TaskCreateRequest,
    background_tasks: BackgroundTasks,
):
    """Create a new benchmarking task."""
    service = get_service()

    # Validate request
    if not request.config_path and not request.config_content:
        raise HTTPException(
            status_code=400,
            detail="Either config_path or config_content must be provided",
        )

    try:
        task_info = service.create_task(
            config_path=request.config_path,
            config_content=request.config_content,
            pricing_path=request.pricing_path,
            run_id=request.run_id,
        )

        # Start task in background
        background_tasks.add_task(
            service.run_task,
            task_info.run_id,
        )

        return TaskResponse(
            run_id=task_info.run_id,
            status=task_info.status,
            message="Task created and started",
            created_at=task_info.created_at,
        )

    except Exception as e:
        logger.error("Failed to create task: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "",
    response_model=TaskListResponse,
    summary="List tasks",
    description="Get a list of all tasks, optionally filtered by status.",
)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of tasks"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List all tasks."""
    service = get_service()
    tasks_list = service.list_tasks(status=status, limit=limit, offset=offset)
    total = service.count_tasks(status=status)

    return TaskListResponse(
        tasks=tasks_list,
        total=total,
    )


@router.get(
    "/{run_id}",
    summary="Get task details",
    description="Get detailed information about a specific task.",
)
async def get_task(run_id: str):
    """Get task details."""
    service = get_service()
    task_info = service.get_task(run_id)

    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_info


@router.get(
    "/{run_id}/progress",
    response_model=TaskProgressResponse,
    summary="Get task progress",
    description="Get real-time progress information for a running task.",
)
async def get_task_progress(run_id: str):
    """Get task progress."""
    service = get_service()
    progress = service.get_progress(run_id)

    if not progress:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskProgressResponse(
        run_id=progress.run_id,
        status=progress.status,
        progress_percent=progress.progress_percent,
        completed=progress.completed,
        total=progress.total,
        elapsed_seconds=progress.elapsed_seconds,
        eta_seconds=progress.eta_seconds,
        success_count=progress.success_count,
        error_count=progress.error_count,
        current_cost=progress.current_cost,
        currency=progress.currency,
    )


@router.get(
    "/{run_id}/stats",
    response_model=TaskStatsResponse,
    summary="Get task statistics",
    description="Get statistical summary for a completed task.",
)
async def get_task_stats(run_id: str):
    """Get task statistics."""
    service = get_service()
    stats = service.get_stats(run_id)

    if not stats:
        raise HTTPException(status_code=404, detail="Task not found or no data")

    return TaskStatsResponse(
        run_id=run_id,
        total_requests=stats.get("total_requests", 0),
        success_count=stats.get("success_count", 0),
        error_count=stats.get("error_count", 0),
        success_rate=stats.get("success_rate", 0.0),
        total_cost=stats.get("total_cost", 0.0),
        currency=stats.get("currency", "CNY"),
        avg_first_resp_time=stats.get("avg_first_resp_time", 0.0),
        avg_char_per_second=stats.get("avg_char_per_second", 0.0),
        avg_token_throughput=stats.get("avg_token_throughput", 0.0),
    )


@router.post(
    "/{run_id}/cancel",
    response_model=TaskResponse,
    summary="Cancel a task",
    description="Cancel a running task.",
)
async def cancel_task(run_id: str):
    """Cancel a running task."""
    service = get_service()
    success = service.cancel_task(run_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be cancelled (not found or already completed)",
        )

    return TaskResponse(
        run_id=run_id,
        status=TaskStatus.CANCELLED,
        message="Task cancelled",
    )


@router.post(
    "/{run_id}/retry",
    response_model=TaskResponse,
    summary="Retry a failed task",
    description="Retry a failed task from the beginning.",
)
async def retry_task(
    run_id: str,
    background_tasks: BackgroundTasks,
):
    """Retry a failed task."""
    service = get_service()
    task_info = service.retry_task(run_id)

    if not task_info:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be retried (not found or not failed)",
        )

    # Start task in background
    background_tasks.add_task(
        service.run_task,
        task_info.run_id,
    )

    return TaskResponse(
        run_id=task_info.run_id,
        status=task_info.status,
        message="Task retry started",
        created_at=task_info.created_at,
    )


@router.delete(
    "/{run_id}",
    summary="Delete a task",
    description="Delete a task and its data.",
)
async def delete_task(run_id: str):
    """Delete a task."""
    service = get_service()
    success = service.delete_task(run_id)

    if not success:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"message": "Task deleted", "run_id": run_id}
