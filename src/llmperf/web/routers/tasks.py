"""Task management API router."""

from __future__ import annotations

import io
import json
import csv
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..services.task_service import TaskService, TaskStatus, TaskInfo
from ..services.pricing_service import PricingService, PriceInfo
from llmperf.config.runtime import load_runtime_config
from llmperf.records.storage import Storage
from llmperf.records.model import RunRecord

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class TaskCreateRequest(BaseModel):
    """Request to create a new task."""
    config_path: Optional[str] = Field(None, description="Path to YAML config file")
    config_content: Optional[str] = Field(None, description="YAML config content")
    run_id: Optional[str] = Field(None, description="Optional run ID")
    auto_start: bool = Field(True, description="Auto start task after creation")


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
    p50_first_resp_time: float
    p95_first_resp_time: float
    p99_first_resp_time: float
    avg_char_per_second: float
    avg_token_throughput: float


class QuickReportResponse(BaseModel):
    """Quick report response."""
    run_id: str
    task_name: str
    completed_at: Optional[datetime]
    duration_seconds: float
    score: int
    grade: str
    dimension_scores: Dict[str, int]
    metrics: Dict[str, Any]
    executor_summary: List[Dict[str, Any]]
    cost_analysis: Optional[Dict[str, Any]] = None
    alerts: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]


class TestRunRequest(BaseModel):
    """Request for a test run."""
    config_content: str = Field(..., description="YAML config content")


class TestRunResponse(BaseModel):
    """Response for a test run."""
    success: bool
    duration_ms: float = 0
    first_token_ms: float = 0
    tokens_per_second: float = 0
    response: str = ""
    error: str = ""


def get_service() -> TaskService:
    """Get task service instance."""
    from ..main import get_task_service
    return get_task_service()


@router.post(
    "",
    response_model=TaskResponse,
    summary="Create a new task",
)
async def create_task(
    request: TaskCreateRequest = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
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
            run_id=request.run_id,
        )

        # Start task in background if auto_start
        if request.auto_start:
            background_tasks.add_task(
                service.run_task,
                task_info.run_id,
            )

        return TaskResponse(
            run_id=task_info.run_id,
            status=task_info.status,
            message="Task created" + (" and started" if request.auto_start else ""),
            created_at=task_info.created_at,
        )

    except Exception as e:
        logger.error("Failed to create task: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "",
    response_model=TaskListResponse,
    summary="List tasks",
)
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 50,
    offset: int = 0,
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
        p50_first_resp_time=stats.get("p50_first_resp_time", 0.0),
        p95_first_resp_time=stats.get("p95_first_resp_time", 0.0),
        p99_first_resp_time=stats.get("p99_first_resp_time", 0.0),
        avg_char_per_second=stats.get("avg_char_per_second", 0.0),
        avg_token_throughput=stats.get("avg_token_throughput", 0.0),
    )


@router.get(
    "/{run_id}/report",
    response_model=QuickReportResponse,
    summary="Get quick report",
)
async def get_quick_report(run_id: str):
    """Get quick report for a completed task.

    This endpoint always recalculates the report from the latest database data.
    No caching is applied to ensure fresh results on each request.
    """
    service = get_service()
    # Always recalculate report from database - no caching
    report = service.get_quick_report(run_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    from fastapi import Response
    response = QuickReportResponse(**report)
    # Add headers to prevent client-side caching
    return Response(
        content=response.model_dump_json(),
        media_type="application/json",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


@router.post(
    "/{run_id}/cancel",
    response_model=TaskResponse,
    summary="Cancel a task",
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
    "/{run_id}/pause",
    response_model=TaskResponse,
    summary="Pause a running task",
)
async def pause_task(run_id: str):
    """Pause a running task.

    The pause is graceful - in-flight requests will complete before
    the task enters paused state.
    """
    service = get_service()
    success = service.pause_task(run_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be paused (not found or not running)",
        )

    return TaskResponse(
        run_id=run_id,
        status=TaskStatus.PAUSED,
        message="Task pause requested",
    )


@router.post(
    "/{run_id}/resume",
    response_model=TaskResponse,
    summary="Resume a paused task",
)
async def resume_task(run_id: str):
    """Resume a paused task."""
    service = get_service()
    success = service.resume_task(run_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be resumed (not found or not paused)",
        )

    return TaskResponse(
        run_id=run_id,
        status=TaskStatus.RUNNING,
        message="Task resumed",
    )


@router.post(
    "/{run_id}/stop",
    response_model=TaskResponse,
    summary="Stop a task (not recoverable)",
)
async def stop_task(run_id: str):
    """Stop a running or paused task.

    This is equivalent to cancel and is not recoverable.
    """
    service = get_service()
    success = service.stop_task(run_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be stopped (not found or already completed)",
        )

    return TaskResponse(
        run_id=run_id,
        status=TaskStatus.CANCELLED,
        message="Task stopped",
    )


@router.post(
    "/{run_id}/retry",
    response_model=TaskResponse,
    summary="Retry a failed task",
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
)
async def delete_task(run_id: str):
    """Delete a task."""
    service = get_service()
    success = service.delete_task(run_id)

    if not success:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"message": "Task deleted", "run_id": run_id}


@router.post(
    "/test-run",
    response_model=TestRunResponse,
    summary="Run a test with first record only",
)
async def test_run(request: TestRunRequest = Body(...)):
    """Run a test with the first record only, without saving to database.

    This endpoint is useful for validating executor configurations before
    running a full benchmark task.
    """
    import time
    import yaml
    from llmperf.config.models import RunConfig
    from llmperf.executors.base import create_executor
    from llmperf.providers.base import create_provider
    from llmperf.datasets.dataset_source_registry import create_dataset_source

    try:
        # Parse config
        config_dict = yaml.safe_load(request.config_content)
        config = RunConfig.model_validate(config_dict)

        if not config.executors:
            return TestRunResponse(
                success=False,
                error="No executors configured",
            )

        # Get first executor
        executor_config = config.executors[0]

        # Get first record from dataset
        dataset_source = create_dataset_source(config.dataset.source)
        records = list(dataset_source.load())
        if not records:
            return TestRunResponse(
                success=False,
                error="Dataset is empty",
            )

        first_record = records[0]

        # Create provider and run single request
        provider = create_provider(
            executor_config.type,
            executor_config.impl or "default",
            executor_config.type,
        )

        from llmperf.providers.base import ProviderRequest

        test_request = ProviderRequest(
            run_id="test-run",
            executor_id=executor_config.id,
            dataset_row_id=first_record.id or "test-0",
            provider=executor_config.type,
            model=executor_config.model or "unknown",
            messages=[msg.model_dump() for msg in first_record.messages],
            options=executor_config.param or {},
        )

        # Override API settings if provided
        if executor_config.api_url:
            test_request.options["api_url"] = executor_config.api_url
        if executor_config.api_key:
            test_request.options["api_key"] = executor_config.api_key

        start_time = time.time()
        record = provider.invoke(test_request)
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000

        # Calculate metrics
        first_token_ms = record.first_resp_time or 0
        tokens_per_second = 0
        if record.duration and record.duration > 0 and record.atokens:
            tokens_per_second = (record.atokens / record.duration) * 1000

        return TestRunResponse(
            success=record.status == 200,
            duration_ms=duration_ms,
            first_token_ms=first_token_ms,
            tokens_per_second=tokens_per_second,
            response=record.response or "",
            error="" if record.status == 200 else f"Request failed with status {record.status}",
        )

    except Exception as e:
        logger.exception("Test run failed: %s", e)
        return TestRunResponse(
            success=False,
            error=str(e),
        )


# ==================== Export & Pricing APIs ====================

class CurrentPriceResponse(BaseModel):
    """Response for current price query."""
    provider: str
    model: str
    input_price: float  # CNY per million tokens
    output_price: float  # CNY per million tokens
    cache_read_price: float = 0.0
    cache_write_price: float = 0.0
    currency: str = "CNY"


def get_storage() -> Storage:
    """Get storage instance."""
    runtime = load_runtime_config()
    return Storage(str(runtime.db_path))


def get_pricing_service() -> PricingService:
    """Get pricing service instance."""
    runtime = load_runtime_config()
    return PricingService(str(runtime.db_path))


@router.get(
    "/pricing/current",
    response_model=CurrentPriceResponse,
    summary="Get current price for provider/model",
)
async def get_current_price(
    provider: str,
    model: str,
):
    """Get the current effective price for a provider/model combination.

    This returns the latest pricing from the pricing_history table.
    """
    pricing_service = get_pricing_service()
    price_info = pricing_service.get_current_price(provider, model)

    return CurrentPriceResponse(
        provider=price_info.provider,
        model=price_info.model,
        input_price=price_info.input_price,
        output_price=price_info.output_price,
        cache_read_price=price_info.cache_read_price,
        cache_write_price=price_info.cache_write_price,
        currency=price_info.currency,
    )


@router.get(
    "/pricing/batch",
    response_model=List[CurrentPriceResponse],
    summary="Get current prices for multiple provider/models",
)
async def get_current_prices_batch(
    providers: Optional[str] = None,  # Comma-separated list
    models: Optional[str] = None,  # Comma-separated list
):
    """Get current prices for multiple provider/model combinations.

    If no filters provided, returns all available prices.
    """
    storage = get_storage()
    provider_list = providers.split(",") if providers else None
    model_list = models.split(",") if models else None

    # Get all provider/model combinations with latest pricing
    all_prices = storage.get_providers_models()

    # Filter if needed
    if provider_list:
        all_prices = [p for p in all_prices if p["provider"] in provider_list]
    if model_list:
        all_prices = [p for p in all_prices if p["model"] in model_list]

    return [
        CurrentPriceResponse(
            provider=p["provider"],
            model=p["model"],
            input_price=p["input_price"],
            output_price=p["output_price"],
            currency="CNY",
        )
        for p in all_prices
    ]


@router.get(
    "/{run_id}/export",
    summary="Export task results",
)
async def export_task_results(
    run_id: str,
    format: str = "jsonl",  # jsonl, csv
):
    """Export task execution results in various formats.

    Supported formats:
    - jsonl: JSON Lines format (default), one record per line
    - csv: Comma-separated values format

    The export includes:
    - Input/output content
    - Token usage
    - Timing metrics (TTFT, total time)
    - Cost information
    - Price snapshots
    """
    storage = get_storage()
    records = list(storage.fetch_run_records(run_id))

    if not records:
        raise HTTPException(status_code=404, detail="No records found for this run")

    if format.lower() == "csv":
        # Generate CSV
        output = io.StringIO()
        fieldnames = [
            "dataset_row_id",
            "provider",
            "model",
            "status",
            "input",
            "output",
            "reasoning",
            "qtokens",
            "atokens",
            "ctokens",
            "ttft_ms",
            "total_time_ms",
            "prompt_cost",
            "completion_cost",
            "cache_cost",
            "total_cost",
            "currency",
            "input_price_snapshot",
            "output_price_snapshot",
            "cache_price_snapshot",
            "created_at",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            # Extract input messages text
            input_text = ""
            try:
                request_params = record.request_params or {}
                messages = request_params.get("messages", [])
                if messages:
                    input_text = "\n".join(
                        f"{m.get('role', '')}: {m.get('content', '')}"
                        for m in messages
                    )
            except Exception:
                pass

            # Extract output content
            output_text = "".join(record.content) if record.content else ""

            # Extract reasoning
            reasoning_text = "".join(record.reasoning) if record.reasoning else ""

            writer.writerow({
                "dataset_row_id": record.dataset_row_id,
                "provider": record.provider,
                "model": record.model,
                "status": record.status,
                "input": input_text,
                "output": output_text,
                "reasoning": reasoning_text,
                "qtokens": record.qtokens,
                "atokens": record.atokens,
                "ctokens": record.ctokens,
                "ttft_ms": record.first_resp_time,
                "total_time_ms": record.last_resp_time,
                "prompt_cost": round(record.prompt_cost, 6),
                "completion_cost": round(record.completion_cost, 6),
                "cache_cost": round(record.cache_cost, 6),
                "total_cost": round(record.total_cost, 6),
                "currency": record.currency,
                "input_price_snapshot": getattr(record, "input_price_snapshot", 0.0),
                "output_price_snapshot": getattr(record, "output_price_snapshot", 0.0),
                "cache_price_snapshot": getattr(record, "cache_price_snapshot", 0.0),
                "created_at": record.created_at,
            })

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={run_id}_results.csv",
            },
        )

    else:  # jsonl (default)
        def generate_jsonl():
            """Generate JSONL output incrementally."""
            for record in records:
                # Extract input messages text
                input_text = ""
                input_messages = []
                try:
                    request_params = record.request_params or {}
                    messages = request_params.get("messages", [])
                    input_messages = messages
                    if messages:
                        input_text = "\n".join(
                            f"{m.get('role', '')}: {m.get('content', '')}"
                            for m in messages
                        )
                except Exception:
                    pass

                # Extract output content
                output_text = "".join(record.content) if record.content else ""

                # Extract reasoning
                reasoning_text = "".join(record.reasoning) if record.reasoning else ""

                export_record = {
                    "dataset_row_id": record.dataset_row_id,
                    "input": input_text,
                    "input_messages": input_messages,
                    "output": output_text,
                    "reasoning": reasoning_text,
                    "qtokens": record.qtokens,
                    "atokens": record.atokens,
                    "ctokens": record.ctokens,
                    "ttft_ms": record.first_resp_time,
                    "total_time_ms": record.last_resp_time,
                    "cost": record.total_cost,
                    "currency": record.currency,
                    "model": record.model,
                    "provider": record.provider,
                    "status": record.status,
                    "created_at": record.created_at,
                    # Price snapshots
                    "input_price_snapshot": getattr(record, "input_price_snapshot", 0.0),
                    "output_price_snapshot": getattr(record, "output_price_snapshot", 0.0),
                    "cache_price_snapshot": getattr(record, "cache_price_snapshot", 0.0),
                }
                yield json.dumps(export_record, ensure_ascii=False) + "\n"

        return StreamingResponse(
            generate_jsonl(),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f"attachment; filename={run_id}_results.jsonl",
            },
        )


@router.post(
    "/export/batch",
    summary="Export multiple tasks results",
)
async def export_batch_results(
    run_ids: List[str] = Body(..., description="List of run IDs to export"),
    format: str = "jsonl",
):
    """Export results from multiple tasks merged together.

    This endpoint combines results from multiple runs into a single export file.
    Each record includes a "run_id" field to identify the source task.
    """
    storage = get_storage()
    all_records = []

    for run_id in run_ids:
        records = list(storage.fetch_run_records(run_id))
        all_records.extend(records)

    if not all_records:
        raise HTTPException(status_code=404, detail="No records found for the specified runs")

    if format.lower() == "csv":
        # Generate CSV
        output = io.StringIO()
        fieldnames = [
            "run_id",
            "dataset_row_id",
            "provider",
            "model",
            "status",
            "input",
            "output",
            "reasoning",
            "qtokens",
            "atokens",
            "ctokens",
            "ttft_ms",
            "total_time_ms",
            "total_cost",
            "currency",
            "created_at",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for record in all_records:
            # Extract input messages text
            input_text = ""
            try:
                request_params = record.request_params or {}
                messages = request_params.get("messages", [])
                if messages:
                    input_text = "\n".join(
                        f"{m.get('role', '')}: {m.get('content', '')}"
                        for m in messages
                    )
            except Exception:
                pass

            output_text = "".join(record.content) if record.content else ""
            reasoning_text = "".join(record.reasoning) if record.reasoning else ""

            writer.writerow({
                "run_id": record.run_id,
                "dataset_row_id": record.dataset_row_id,
                "provider": record.provider,
                "model": record.model,
                "status": record.status,
                "input": input_text,
                "output": output_text,
                "reasoning": reasoning_text,
                "qtokens": record.qtokens,
                "atokens": record.atokens,
                "ctokens": record.ctokens,
                "ttft_ms": record.first_resp_time,
                "total_time_ms": record.last_resp_time,
                "total_cost": round(record.total_cost, 6),
                "currency": record.currency,
                "created_at": record.created_at,
            })

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=batch_results.csv",
            },
        )

    else:  # jsonl (default)
        def generate_jsonl():
            """Generate JSONL output incrementally."""
            for record in all_records:
                input_text = ""
                input_messages = []
                try:
                    request_params = record.request_params or {}
                    messages = request_params.get("messages", [])
                    input_messages = messages
                    if messages:
                        input_text = "\n".join(
                            f"{m.get('role', '')}: {m.get('content', '')}"
                            for m in messages
                        )
                except Exception:
                    pass

                output_text = "".join(record.content) if record.content else ""
                reasoning_text = "".join(record.reasoning) if record.reasoning else ""

                export_record = {
                    "run_id": record.run_id,
                    "dataset_row_id": record.dataset_row_id,
                    "input": input_text,
                    "input_messages": input_messages,
                    "output": output_text,
                    "reasoning": reasoning_text,
                    "qtokens": record.qtokens,
                    "atokens": record.atokens,
                    "ctokens": record.ctokens,
                    "ttft_ms": record.first_resp_time,
                    "total_time_ms": record.last_resp_time,
                    "cost": record.total_cost,
                    "currency": record.currency,
                    "model": record.model,
                    "provider": record.provider,
                    "status": record.status,
                    "created_at": record.created_at,
                }
                yield json.dumps(export_record, ensure_ascii=False) + "\n"

        return StreamingResponse(
            generate_jsonl(),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": "attachment; filename=batch_results.jsonl",
            },
        )
