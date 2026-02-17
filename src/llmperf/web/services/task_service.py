"""Task management service."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmperf.config.loader import load_config
from llmperf.config.models import RunConfig
from llmperf.config.runtime import load_runtime_config
from llmperf.records.storage import Storage
from llmperf.runner import RunManager

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Progress information for a running task."""
    run_id: str
    status: TaskStatus
    progress_percent: float = 0.0
    completed: int = 0
    total: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    success_count: int = 0
    error_count: int = 0
    current_cost: float = 0.0
    currency: str = "CNY"
    started_at: Optional[datetime] = None


@dataclass
class TaskInfo:
    """Information about a task."""
    run_id: str
    status: TaskStatus
    config_path: Optional[str] = None
    task_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class TaskService:
    """Service for managing benchmarking tasks.

    This service handles:
    - Task creation and execution
    - Progress tracking
    - Task cancellation
    - Integration with WebSocket for real-time updates
    """

    def __init__(self):
        """Initialize task service."""
        self._tasks: Dict[str, TaskInfo] = {}
        self._progress: Dict[str, TaskProgress] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._config_contents: Dict[str, str] = {}  # Store config content for tasks
        self._lock = threading.Lock()

        runtime = load_runtime_config()
        self._db_path = str(runtime.db_path)
        self._storage = Storage(self._db_path)

    def create_task(
        self,
        config_path: Optional[str] = None,
        config_content: Optional[str] = None,
        pricing_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> TaskInfo:
        """Create a new task.

        Args:
            config_path: Path to YAML config file.
            config_content: YAML config content (used if config_path is None).
            pricing_path: Path to pricing file.
            run_id: Optional run ID.

        Returns:
            TaskInfo for the created task.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Generate run ID if not provided
        if not run_id:
            run_id = uuid.uuid4().hex

        # Load configuration
        if config_path:
            config = load_config(config_path)
        elif config_content:
            import yaml
            config_dict = yaml.safe_load(config_content)
            config = RunConfig.model_validate(config_dict)
        else:
            raise ValueError("Either config_path or config_content must be provided")

        # Create task info
        task_info = TaskInfo(
            run_id=run_id,
            status=TaskStatus.PENDING,
            config_path=config_path,
            task_name=config.info or "Untitled Task",
        )

        # Create progress tracker
        progress = TaskProgress(
            run_id=run_id,
            status=TaskStatus.PENDING,
        )

        with self._lock:
            self._tasks[run_id] = task_info
            self._progress[run_id] = progress
            if config_content:
                self._config_contents[run_id] = config_content

        return task_info

    def run_task(self, run_id: str) -> None:
        """Run a task in the background.

        Args:
            run_id: The run ID to execute.
        """
        task_info = self._tasks.get(run_id)
        if not task_info:
            logger.error("Task not found: %s", run_id)
            return

        # Update status
        task_info.status = TaskStatus.RUNNING
        task_info.started_at = datetime.now()

        progress = self._progress.get(run_id)
        if progress:
            progress.status = TaskStatus.RUNNING
            progress.started_at = task_info.started_at

        # Create cancel event
        cancel_event = threading.Event()
        self._cancel_events[run_id] = cancel_event

        try:
            # Load config
            temp_config_path = None
            if task_info.config_path:
                config = load_config(task_info.config_path)
                config_path = task_info.config_path
            elif self._config_contents.get(run_id):
                # Load from config content - create temp file
                import yaml
                import tempfile
                config_dict = yaml.safe_load(self._config_contents[run_id])
                config = RunConfig.model_validate(config_dict)

                # Create temporary file for config
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.yaml',
                    delete=False,
                    encoding='utf-8'
                )
                temp_file.write(self._config_contents[run_id])
                temp_file.close()
                temp_config_path = temp_file.name
                config_path = temp_config_path
            else:
                raise ValueError("No configuration available")

            # Create run manager
            manager = RunManager(
                config,
                config_path=config_path,
                pricing_path=None,
                run_id=run_id,
            )

            # Start progress monitoring in background
            monitor_thread = threading.Thread(
                target=self._monitor_progress,
                args=(run_id, cancel_event),
                daemon=True,
            )
            monitor_thread.start()

            # Run the task
            manager.run()

            # Update completion status
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.now()

            if progress:
                progress.status = TaskStatus.COMPLETED
                progress.progress_percent = 100.0

        except Exception as e:
            logger.exception("Task failed: %s", e)
            task_info.status = TaskStatus.FAILED
            task_info.error_message = str(e)
            task_info.completed_at = datetime.now()

            if progress:
                progress.status = TaskStatus.FAILED

        finally:
            # Cleanup cancel event
            self._cancel_events.pop(run_id, None)

            # Cleanup temp config file
            if temp_config_path:
                try:
                    import os
                    os.unlink(temp_config_path)
                except Exception:
                    pass

            # Broadcast final status via WebSocket
            asyncio.run(self._broadcast_status(run_id))

    def _monitor_progress(
        self,
        run_id: str,
        cancel_event: threading.Event,
    ) -> None:
        """Monitor task progress and update progress tracker.

        Args:
            run_id: The run ID to monitor.
            cancel_event: Event to check for cancellation.
        """
        progress = self._progress.get(run_id)
        if not progress or not progress.started_at:
            return

        while not cancel_event.is_set():
            try:
                # Fetch current stats from database
                records = list(self._storage.fetch_run_records(run_id))

                if records:
                    total = len(records)
                    success_count = sum(1 for r in records if r.status == 200)
                    error_count = total - success_count
                    total_cost = sum(r.total_cost for r in records)

                    progress.completed = total
                    progress.success_count = success_count
                    progress.error_count = error_count
                    progress.current_cost = total_cost
                    progress.currency = records[0].currency if records else "CNY"

                    # Calculate elapsed time
                    elapsed = (datetime.now() - progress.started_at).total_seconds()
                    progress.elapsed_seconds = elapsed

                    # Broadcast progress update
                    asyncio.run(self._broadcast_progress(run_id))

                time.sleep(2)  # Poll every 2 seconds

            except Exception as e:
                logger.warning("Error monitoring progress: %s", e)
                time.sleep(5)

    async def _broadcast_progress(self, run_id: str) -> None:
        """Broadcast progress update via WebSocket.

        Args:
            run_id: The run ID.
        """
        try:
            from ..routers.websocket import get_manager

            progress = self._progress.get(run_id)
            if not progress:
                return

            manager = get_manager()
            await manager.broadcast_progress(
                run_id=run_id,
                progress_percent=progress.progress_percent,
                completed=progress.completed,
                total=progress.total,
                elapsed_seconds=progress.elapsed_seconds,
                eta_seconds=progress.eta_seconds,
                success_count=progress.success_count,
                error_count=progress.error_count,
                current_cost=progress.current_cost,
                currency=progress.currency,
                status=progress.status.value,
            )
        except Exception as e:
            logger.warning("Failed to broadcast progress: %s", e)

    async def _broadcast_status(self, run_id: str) -> None:
        """Broadcast status update via WebSocket.

        Args:
            run_id: The run ID.
        """
        try:
            from ..routers.websocket import get_manager

            task_info = self._tasks.get(run_id)
            if not task_info:
                return

            manager = get_manager()
            await manager.broadcast_event(
                run_id=run_id,
                event_type="status_change",
                data={
                    "status": task_info.status.value,
                    "error_message": task_info.error_message,
                },
            )
        except Exception as e:
            logger.warning("Failed to broadcast status: %s", e)

    def get_task(self, run_id: str) -> Optional[TaskInfo]:
        """Get task information.

        Args:
            run_id: The run ID.

        Returns:
            TaskInfo if found, None otherwise.
        """
        return self._tasks.get(run_id)

    def get_progress(self, run_id: str) -> Optional[TaskProgress]:
        """Get task progress.

        Args:
            run_id: The run ID.

        Returns:
            TaskProgress if found, None otherwise.
        """
        return self._progress.get(run_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TaskInfo]:
        """List tasks from both memory and database.

        Args:
            status: Filter by status.
            limit: Maximum number of tasks.
            offset: Offset for pagination.

        Returns:
            List of TaskInfo.
        """
        tasks = list(self._tasks.values())

        # Also load tasks from database
        try:
            db_runs = self._storage.list_runs(limit=limit * 2)  # Get more for filtering
            for run in db_runs:
                run_id = run.get("run_id")
                if run_id and run_id not in self._tasks:
                    # Convert database run to TaskInfo
                    task = TaskInfo(
                        run_id=run_id,
                        status=TaskStatus.COMPLETED,  # Assume completed if in DB
                        config_path=run.get("config_path"),
                        task_name=run.get("info", ""),
                        created_at=datetime.fromtimestamp(run.get("created_at", 0)) if run.get("created_at") else datetime.now(),
                    )
                    tasks.append(task)
        except Exception as e:
            logger.warning("Failed to load tasks from database: %s", e)

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[offset:offset + limit]

    def count_tasks(self, status: Optional[TaskStatus] = None) -> int:
        """Count tasks from both memory and database.

        Args:
            status: Filter by status.

        Returns:
            Number of tasks.
        """
        # Count from database
        try:
            db_runs = self._storage.list_runs(limit=1000)
            db_count = len(db_runs)
        except Exception:
            db_count = 0

        if status:
            memory_count = sum(1 for t in self._tasks.values() if t.status == status)
            # For database tasks, we assume completed status
            if status == TaskStatus.COMPLETED:
                return memory_count + db_count
            return memory_count

        return len(self._tasks) + db_count

    def cancel_task(self, run_id: str) -> bool:
        """Cancel a running task.

        Args:
            run_id: The run ID.

        Returns:
            True if cancelled, False if not cancellable.
        """
        task_info = self._tasks.get(run_id)
        if not task_info:
            return False

        if task_info.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return False

        # Set cancel event
        cancel_event = self._cancel_events.get(run_id)
        if cancel_event:
            cancel_event.set()

        # Update status
        task_info.status = TaskStatus.CANCELLED
        task_info.completed_at = datetime.now()

        progress = self._progress.get(run_id)
        if progress:
            progress.status = TaskStatus.CANCELLED

        return True

    def retry_task(self, run_id: str) -> Optional[TaskInfo]:
        """Retry a failed task.

        Args:
            run_id: The run ID.

        Returns:
            New TaskInfo if retry started, None otherwise.
        """
        task_info = self._tasks.get(run_id)
        if not task_info or task_info.status != TaskStatus.FAILED:
            return None

        # Create new task with same config
        new_run_id = uuid.uuid4().hex
        return self.create_task(
            config_path=task_info.config_path,
            run_id=new_run_id,
        )

    def delete_task(self, run_id: str) -> bool:
        """Delete a task.

        Args:
            run_id: The run ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if run_id in self._tasks:
                del self._tasks[run_id]
            if run_id in self._progress:
                del self._progress[run_id]
            if run_id in self._config_contents:
                del self._config_contents[run_id]
            return True

    def get_stats(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a completed task.

        Args:
            run_id: The run ID.

        Returns:
            Statistics dictionary, or None if no data.
        """
        records = list(self._storage.fetch_run_records(run_id))

        if not records:
            return None

        total = len(records)
        successful = [r for r in records if r.status == 200]
        failed = [r for r in records if r.status != 200]

        first_resp_times = [r.first_resp_time for r in successful if r.first_resp_time > 0]
        char_per_sec = [r.char_per_second for r in successful if r.char_per_second > 0]
        token_throughput = [r.token_throughput for r in successful if r.token_throughput > 0]

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "total_requests": total,
            "success_count": len(successful),
            "error_count": len(failed),
            "success_rate": len(successful) / total * 100 if total > 0 else 0,
            "total_cost": sum(r.total_cost for r in records),
            "currency": records[0].currency if records else "CNY",
            "avg_first_resp_time": avg(first_resp_times),
            "avg_char_per_second": avg(char_per_sec),
            "avg_token_throughput": avg(token_throughput),
            "total_input_tokens": sum(r.qtokens for r in records),
            "total_output_tokens": sum(r.atokens for r in records),
        }
