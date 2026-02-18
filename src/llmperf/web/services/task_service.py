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

            # Update completion status and calculate total cost
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.now()

            if progress:
                progress.status = TaskStatus.COMPLETED
                progress.progress_percent = 100.0

            # Calculate and save total cost to database
            try:
                records = list(self._storage.fetch_run_records(run_id))
                total_cost = sum(r.total_cost for r in records)
                currency = records[0].currency if records else "CNY"
                self._storage.update_run_cost(run_id, total_cost, currency)
                logger.info("Task %s completed with total cost: %.4f %s", run_id, total_cost, currency)
            except Exception as e:
                logger.warning("Failed to update run cost: %s", e)

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
        # First check memory
        task_info = self._tasks.get(run_id)
        if task_info:
            return task_info

        # If not in memory, try to load from database
        try:
            runs = self._storage.list_runs(limit=1000)
            for run in runs:
                if run.get("run_id") == run_id:
                    return TaskInfo(
                        run_id=run_id,
                        status=TaskStatus.COMPLETED,
                        config_path=run.get("config_path"),
                        task_name=run.get("info", ""),
                        created_at=datetime.fromtimestamp(run.get("created_at", 0)) if run.get("created_at") else datetime.now(),
                        completed_at=datetime.fromtimestamp(run.get("completed_at", 0)) if run.get("completed_at") else None,
                    )
        except Exception as e:
            logger.warning("Failed to load task from database: %s", e)

        return None

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

        def percentile(lst, p):
            if not lst:
                return 0
            sorted_lst = sorted(lst)
            k = (len(sorted_lst) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_lst) else f
            return sorted_lst[f] + (k - f) * (sorted_lst[c] - sorted_lst[f])

        return {
            "total_requests": total,
            "success_count": len(successful),
            "error_count": len(failed),
            "success_rate": len(successful) / total * 100 if total > 0 else 0,
            "total_cost": sum(r.total_cost for r in records),
            "currency": records[0].currency if records else "CNY",
            "avg_first_resp_time": avg(first_resp_times),
            "p50_first_resp_time": percentile(first_resp_times, 50),
            "p95_first_resp_time": percentile(first_resp_times, 95),
            "p99_first_resp_time": percentile(first_resp_times, 99),
            "avg_char_per_second": avg(char_per_sec),
            "avg_token_throughput": avg(token_throughput),
            "total_input_tokens": sum(r.qtokens for r in records),
            "total_output_tokens": sum(r.atokens for r in records),
        }

    def get_quick_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Generate a quick report for a completed task.

        Args:
            run_id: The run ID.

        Returns:
            Quick report dictionary, or None if no data.
        """
        task_info = self.get_task(run_id)  # Use get_task to check both memory and database
        stats = self.get_stats(run_id)

        if not stats:
            return None

        # Calculate dimension scores
        latency_score = self._calculate_latency_score(stats.get("avg_first_resp_time", 0))
        throughput_score = self._calculate_throughput_score(stats.get("avg_token_throughput", 0))
        success_score = self._calculate_success_score(stats.get("success_rate", 0))
        cost_score = self._calculate_cost_score(stats.get("total_cost", 0), stats.get("total_requests", 1))

        # Overall score (weighted average)
        overall_score = round(
            latency_score * 0.25 +
            throughput_score * 0.25 +
            success_score * 0.25 +
            cost_score * 0.25
        )

        # Determine grade
        grade = self._score_to_grade(overall_score)

        # Generate alerts
        alerts = self._generate_alerts(stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(stats, latency_score, throughput_score, success_score, cost_score)

        # Get executor summary
        executor_summary = self._get_executor_summary(run_id)

        # Calculate duration
        duration_seconds = 0
        if task_info and task_info.started_at and task_info.completed_at:
            duration_seconds = (task_info.completed_at - task_info.started_at).total_seconds()

        return {
            "run_id": run_id,
            "task_name": task_info.task_name if task_info else "Unknown Task",
            "completed_at": task_info.completed_at if task_info else None,
            "duration_seconds": duration_seconds,
            "score": overall_score,
            "grade": grade,
            "dimension_scores": {
                "latency": latency_score,
                "throughput": throughput_score,
                "success_rate": success_score,
                "cost": cost_score,
            },
            "metrics": {
                "total_requests": stats.get("total_requests", 0),
                "success_rate": stats.get("success_rate", 0),
                "avg_ttft": stats.get("avg_first_resp_time", 0),
                "p95_ttft": stats.get("p95_first_resp_time", 0),
                "avg_tps": stats.get("avg_token_throughput", 0),
                "total_cost": stats.get("total_cost", 0),
                "currency": stats.get("currency", "CNY"),
                "total_input_tokens": stats.get("total_input_tokens", 0),
                "total_output_tokens": stats.get("total_output_tokens", 0),
            },
            "executor_summary": executor_summary,
            "alerts": alerts,
            "recommendations": recommendations,
        }

    def _calculate_latency_score(self, avg_ttft_ms: float) -> int:
        """Calculate latency score based on average TTFT."""
        if avg_ttft_ms < 200:
            return 100
        elif avg_ttft_ms < 500:
            return 90
        elif avg_ttft_ms < 1000:
            return 75
        elif avg_ttft_ms < 2000:
            return 60
        elif avg_ttft_ms < 5000:
            return 40
        else:
            return 20

    def _calculate_throughput_score(self, avg_tps: float) -> int:
        """Calculate throughput score based on average TPS."""
        if avg_tps > 100:
            return 100
        elif avg_tps > 50:
            return 90
        elif avg_tps > 20:
            return 75
        elif avg_tps > 10:
            return 60
        elif avg_tps > 5:
            return 40
        else:
            return 20

    def _calculate_success_score(self, success_rate: float) -> int:
        """Calculate success rate score."""
        if success_rate >= 99.9:
            return 100
        elif success_rate >= 99:
            return 95
        elif success_rate >= 95:
            return 80
        elif success_rate >= 90:
            return 60
        elif success_rate >= 80:
            return 40
        else:
            return 20

    def _calculate_cost_score(self, total_cost: float, total_requests: int) -> int:
        """Calculate cost efficiency score."""
        if total_requests == 0:
            return 75  # Default baseline

        cost_per_request = total_cost / total_requests

        # Cost per request thresholds (adjust based on typical LLM API costs)
        if cost_per_request < 0.001:  # Very cheap
            return 100
        elif cost_per_request < 0.005:
            return 90
        elif cost_per_request < 0.01:
            return 75
        elif cost_per_request < 0.05:
            return 60
        elif cost_per_request < 0.1:
            return 40
        else:
            return 20

    def _score_to_grade(self, score: int) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "S"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    def _generate_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on statistics."""
        alerts = []

        # Check success rate
        success_rate = stats.get("success_rate", 0)
        if success_rate < 90:
            alerts.append({
                "type": "success_rate",
                "severity": "error" if success_rate < 80 else "warning",
                "message": f"成功率较低 ({success_rate:.1f}%)，存在较多失败请求",
                "suggestion": "检查API配置、网络连接或降低并发数",
            })

        # Check latency
        avg_ttft = stats.get("avg_first_resp_time", 0)
        p95_ttft = stats.get("p95_first_resp_time", 0)
        if avg_ttft > 2000:
            alerts.append({
                "type": "latency",
                "severity": "warning",
                "message": f"平均延迟较高 ({avg_ttft:.0f}ms)，影响用户体验",
                "suggestion": "考虑优化prompt长度或使用更快的模型",
            })
        if p95_ttft > 5000:
            alerts.append({
                "type": "latency_p95",
                "severity": "warning",
                "message": f"P95延迟过高 ({p95_ttft:.0f}ms)，部分请求响应缓慢",
                "suggestion": "检查是否有冷启动或网络波动问题",
            })

        # Check error distribution
        error_count = stats.get("error_count", 0)
        if error_count > 0:
            alerts.append({
                "type": "errors",
                "severity": "info",
                "message": f"发现 {error_count} 个错误请求",
                "suggestion": "查看详细日志了解错误原因",
            })

        # Check throughput
        avg_tps = stats.get("avg_token_throughput", 0)
        if avg_tps < 10:
            alerts.append({
                "type": "throughput",
                "severity": "info",
                "message": f"吞吐量较低 ({avg_tps:.1f} TPS)",
                "suggestion": "考虑增加并发数或使用流式输出",
            })

        return alerts

    def _generate_recommendations(
        self,
        stats: Dict[str, Any],
        latency_score: int,
        throughput_score: int,
        success_score: int,
        cost_score: int,
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []

        # Latency recommendations
        if latency_score < 75:
            recommendations.append({
                "category": "performance",
                "title": "优化响应延迟",
                "description": "当前延迟偏高，建议优化prompt长度、使用prompt缓存或选择响应更快的模型",
                "impact": "high",
            })

        # Throughput recommendations
        if throughput_score < 60:
            recommendations.append({
                "category": "performance",
                "title": "提升吞吐量",
                "description": "当前吞吐量较低，可考虑增加并发请求数或启用流式输出",
                "impact": "medium",
            })

        # Success rate recommendations
        if success_score < 80:
            recommendations.append({
                "category": "reliability",
                "title": "提高成功率",
                "description": "当前错误率较高，建议检查API配置、添加重试机制或降低请求频率",
                "impact": "high",
            })

        # Cost recommendations
        if cost_score < 75:
            recommendations.append({
                "category": "cost",
                "title": "优化成本",
                "description": "可考虑启用prompt缓存、优化prompt长度或选择性价比更高的模型",
                "impact": "medium",
            })

        # Always add cache recommendation if not extremely low cost
        total_cost = stats.get("total_cost", 0)
        total_requests = stats.get("total_requests", 1)
        if total_requests > 0 and total_cost / total_requests > 0.001:
            recommendations.append({
                "category": "cost",
                "title": "启用Prompt缓存",
                "description": "对于重复的prompt，启用缓存可节省20-50%的API成本",
                "impact": "low",
            })

        return recommendations

    def _get_executor_summary(self, run_id: str) -> List[Dict[str, Any]]:
        """Get summary statistics by executor."""
        records = list(self._storage.fetch_run_records(run_id))

        # Group by executor
        executors: Dict[str, List[Any]] = {}
        for r in records:
            exec_id = r.executor_id or "default"
            if exec_id not in executors:
                executors[exec_id] = []
            executors[exec_id].append(r)

        summary = []
        for exec_id, exec_records in executors.items():
            total = len(exec_records)
            successful = [r for r in exec_records if r.status == 200]
            ttfts = [r.first_resp_time for r in successful if r.first_resp_time > 0]
            tps_list = [r.token_throughput for r in successful if r.token_throughput > 0]

            summary.append({
                "id": exec_id,
                "requests": total,
                "success_rate": len(successful) / total * 100 if total > 0 else 0,
                "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else 0,
                "avg_tps": sum(tps_list) / len(tps_list) if tps_list else 0,
                "cost": sum(r.total_cost for r in exec_records),
            })

        return summary
