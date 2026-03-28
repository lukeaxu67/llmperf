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
from typing import Any, Dict, List, Optional

from llmperf.config.loader import load_config
from llmperf.config.models import ExecutorConfig, PricingEntry, RunConfig
from llmperf.config.runtime import load_runtime_config
from llmperf.records.storage import Storage
from llmperf.runner import RunManager
from llmperf.executors.process_manager import TaskCancelledError
import yaml
from .run_config_service import load_run_config_content, normalize_run_config

logger = logging.getLogger(__name__)


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile
    floor_index = int(rank)
    ceil_index = min(floor_index + 1, len(sorted_values) - 1)
    return sorted_values[floor_index] + (rank - floor_index) * (
        sorted_values[ceil_index] - sorted_values[floor_index]
    )


class TaskStatus(str, Enum):
    """Status of a task."""
    SCHEDULED = "scheduled"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
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
    paused_duration_seconds: float = 0.0  # Total time spent paused
    current_rate: float = 0.0
    concurrency: int = 0
    paused_at: Optional[datetime] = None
    dataset_total_per_executor: int = 0
    executors: List[Dict[str, Any]] = field(default_factory=list)
    topology: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskInfo:
    """Information about a task."""
    run_id: str
    status: TaskStatus
    config_path: Optional[str] = None
    task_name: str = ""
    task_type: str = "benchmark"
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
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
        self._pause_events: Dict[str, threading.Event] = {}  # Events for pause signaling
        self._resume_events: Dict[str, threading.Event] = {}  # Events for resume signaling
        self._scheduled_timers: Dict[str, threading.Timer] = {}
        self._config_contents: Dict[str, str] = {}  # Store config content for tasks
        self._lock = threading.Lock()

        runtime = load_runtime_config()
        self._db_path = str(runtime.db_path)
        self._storage = Storage(self._db_path)
        self._restore_recoverable_tasks()

    def _restore_recoverable_tasks(self) -> None:
        try:
            recoverable_runs = self._storage.list_runs(limit=1000)
        except Exception as exc:
            logger.warning("Failed to restore tasks after restart: %s", exc)
            return

        now = datetime.now()
        for run in recoverable_runs:
            task_info = self._build_task_info_from_snapshot(run)
            if not task_info.run_id:
                continue

            if task_info.status not in (TaskStatus.SCHEDULED, TaskStatus.RUNNING):
                continue

            self._tasks[task_info.run_id] = task_info
            self._progress.setdefault(
                task_info.run_id,
                TaskProgress(
                    run_id=task_info.run_id,
                    status=task_info.status,
                    started_at=task_info.started_at,
                ),
            )
            config_content = run.get("config_content") or ""
            if config_content:
                self._config_contents[task_info.run_id] = config_content

            if task_info.status == TaskStatus.SCHEDULED:
                if task_info.scheduled_at and task_info.scheduled_at > now:
                    self.schedule_task(task_info.run_id, task_info.scheduled_at)
                else:
                    threading.Thread(target=self.run_task, args=(task_info.run_id,), daemon=True).start()
                continue

            logger.info("Restoring interrupted running task %s after service restart", task_info.run_id)
            threading.Thread(target=self.run_task, args=(task_info.run_id,), daemon=True).start()

    def _load_run_snapshot(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._storage.get_run(run_id)

    @staticmethod
    def _dispatch_async(coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return
        loop.create_task(coro)

    @staticmethod
    def _task_status_from_value(value: Any) -> TaskStatus:
        try:
            return TaskStatus(str(value))
        except Exception:
            return TaskStatus.PENDING

    @staticmethod
    def _normalize_datetime(value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is not None:
            return value.astimezone().replace(tzinfo=None)
        return value

    @staticmethod
    def _executor_pricing_provider(executor: ExecutorConfig) -> str:
        provider_name = executor.param.get("provider_name") if isinstance(executor.param, dict) else None
        return str(provider_name or executor.type or "").strip().lower()

    def _inject_pricing_snapshot(self, config: RunConfig) -> RunConfig:
        entries_by_key: Dict[tuple[str, str], PricingEntry] = {
            (str(entry.provider).strip().lower(), str(entry.model).strip().lower()): entry
            for entry in config.pricing
        }
        pricing_timestamp = int(time.time())
        for executor in config.executors:
            provider = self._executor_pricing_provider(executor)
            model = str(executor.model or "").strip()
            if not provider or not model:
                continue
            key = (provider, model.lower())
            if key in entries_by_key:
                continue
            price_record = self._storage.get_pricing_at_time(
                provider=provider,
                model=model,
                timestamp=pricing_timestamp,
            )
            if not price_record:
                continue
            cache_input_discount = (
                max(price_record.cache_read_price, 0.0) / price_record.input_price
                if price_record.input_price > 0
                else 0.0
            )
            entries_by_key[key] = PricingEntry(
                provider=provider,
                model=model,
                unit="per_1m",
                input_price=price_record.input_price,
                output_price=price_record.output_price,
                cache_input_discount=min(max(cache_input_discount, 0.0), 1.0),
                cache_output_discount=0.0,
                currency="CNY",
            )

        config.pricing = list(entries_by_key.values())
        return config

    def _normalize_and_snapshot_config_content(self, config: RunConfig) -> str:
        self._inject_pricing_snapshot(config)
        return yaml.safe_dump(
            config.model_dump(exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        )

    def _build_task_info_from_snapshot(self, run: Dict[str, Any]) -> TaskInfo:
        return TaskInfo(
            run_id=str(run.get("run_id") or ""),
            status=self._task_status_from_value(run.get("status")),
            config_path=run.get("config_path"),
            task_name=run.get("info", ""),
            task_type=run.get("task_type", "benchmark"),
            created_at=datetime.fromtimestamp(run.get("created_at", 0)) if run.get("created_at") else datetime.now(),
            scheduled_at=datetime.fromtimestamp(run.get("scheduled_at", 0)) if run.get("scheduled_at") else None,
            started_at=datetime.fromtimestamp(run.get("started_at", 0)) if run.get("started_at") else None,
            completed_at=datetime.fromtimestamp(run.get("completed_at", 0)) if run.get("completed_at") else None,
            error_message=run.get("error_message") or None,
        )

    def _load_run_config(self, run_id: str) -> Optional[RunConfig]:
        task_info = self._tasks.get(run_id)
        try:
            if run_id in self._config_contents:
                config, _ = load_run_config_content(self._config_contents[run_id])
                return config
            if task_info and task_info.config_path:
                return normalize_run_config(load_config(task_info.config_path))
            run_snapshot = self._load_run_snapshot(run_id)
            if not run_snapshot:
                return None
            config_content = run_snapshot.get("config_content") or ""
            if config_content:
                config, _ = load_run_config_content(config_content)
                return config
            config_path = run_snapshot.get("config_path")
            if config_path:
                return normalize_run_config(load_config(str(config_path)))
        except Exception as exc:
            logger.warning("Failed to load config for run %s: %s", run_id, exc)
        return None

    def _estimate_dataset_total(self, config: Optional[RunConfig]) -> int:
        if not config or not config.dataset:
            return 0
        try:
            from llmperf.datasets.dataset_source_registry import create_source

            dataset_source = create_source(
                config.dataset.source.type,
                name=config.dataset.source.name or "dataset",
                config=dict(config.dataset.source.config or {}),
            )
            dataset_total = 0
            if hasattr(dataset_source, "__len__"):
                try:
                    dataset_total = len(dataset_source)
                except TypeError:
                    dataset_total = 0
            if dataset_total == 0:
                dataset_total = len(list(dataset_source.load()))
            max_rounds = config.dataset.iterator.max_rounds if config.dataset.iterator else None
            if max_rounds:
                dataset_total *= max_rounds
            return max(dataset_total, 0)
        except Exception as exc:
            logger.warning("Failed to estimate dataset total for run: %s", exc)
            return 0

    def _executor_levels(self, executors: List[ExecutorConfig]) -> Dict[str, int]:
        executor_map = {executor.id: executor for executor in executors}
        levels: Dict[str, int] = {}

        def visit(executor_id: str, path: set[str]) -> int:
            if executor_id in levels:
                return levels[executor_id]
            if executor_id in path:
                return 0
            path.add(executor_id)
            executor = executor_map.get(executor_id)
            if not executor or not executor.after:
                level = 0
            else:
                level = 1 + max(visit(dep, path) for dep in executor.after if dep in executor_map)
            path.remove(executor_id)
            levels[executor_id] = level
            return level

        for executor in executors:
            visit(executor.id, set())
        return levels

    def _topological_order(self, executors: List[ExecutorConfig]) -> List[str]:
        pending = {executor.id: executor for executor in executors}
        completed: set[str] = set()
        ordered: List[str] = []
        while pending:
            ready_ids = [
                executor_id
                for executor_id, executor in pending.items()
                if all(dep in completed or dep not in pending for dep in executor.after)
            ]
            if not ready_ids:
                ordered.extend(sorted(pending))
                break
            for executor_id in ready_ids:
                ordered.append(executor_id)
                pending.pop(executor_id, None)
                completed.add(executor_id)
        return ordered

    def _build_topology(self, config: Optional[RunConfig], executors: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not config:
            return {"nodes": [], "edges": [], "layers": []}

        id_to_executor = {item["id"]: item for item in executors}
        levels = self._executor_levels(config.executors)
        layers_map: Dict[int, List[str]] = {}
        edges: List[Dict[str, str]] = []
        downstream_count: Dict[str, int] = {executor.id: 0 for executor in config.executors}

        for executor in config.executors:
            level = levels.get(executor.id, 0)
            layers_map.setdefault(level, []).append(executor.id)
            if executor.after:
                for dep in executor.after:
                    edges.append({"source": dep, "target": executor.id})
                    if dep in downstream_count:
                        downstream_count[dep] += 1
            else:
                edges.append({"source": "__start__", "target": executor.id})

        for executor in config.executors:
            if downstream_count.get(executor.id, 0) == 0:
                edges.append({"source": executor.id, "target": "__end__"})

        end_status = self._resolve_end_node_status(executors)

        node_items = [
            {
                "id": "__start__",
                "name": "Start",
                "kind": "boundary",
                "status": "completed",
                "level": -1,
            }
        ]
        node_items.extend(
            {
                **id_to_executor.get(executor.id, {"id": executor.id, "name": executor.name}),
                "kind": "executor",
                "level": levels.get(executor.id, 0),
            }
            for executor in config.executors
        )
        node_items.append(
            {
                "id": "__end__",
                "name": "End",
                "kind": "boundary",
                "status": end_status,
                "level": max(levels.values(), default=0) + 1,
            }
        )

        layers = [
            {
                "level": level,
                "node_ids": sorted(node_ids),
            }
            for level, node_ids in sorted(layers_map.items())
        ]

        return {
            "nodes": node_items,
            "edges": edges,
            "layers": layers,
        }

    @staticmethod
    def _resolve_end_node_status(executors: List[Dict[str, Any]]) -> str:
        if not executors:
            return TaskStatus.PENDING.value

        statuses = {str(item.get("status") or TaskStatus.PENDING.value) for item in executors}
        if statuses == {TaskStatus.COMPLETED.value}:
            return TaskStatus.COMPLETED.value

        for status in (
            TaskStatus.RUNNING.value,
            TaskStatus.PAUSED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
            "blocked",
            TaskStatus.PENDING.value,
        ):
            if status in statuses:
                return status

        return TaskStatus.COMPLETED.value if TaskStatus.COMPLETED.value in statuses else TaskStatus.PENDING.value

    def _build_executor_progress(
        self,
        config: Optional[RunConfig],
        records: List[Any],
        task_status: TaskStatus,
        dataset_total: int,
    ) -> List[Dict[str, Any]]:
        if not config:
            return []

        per_executor: Dict[str, List[Any]] = {}
        for record in records:
            per_executor.setdefault(record.executor_id or "default", []).append(record)

        completed_executor_ids = {
            executor_id
            for executor_id, executor_records in per_executor.items()
            if dataset_total > 0 and len(executor_records) >= dataset_total
        }
        ordered_ids = self._topological_order(config.executors)
        max_workers = (
            config.multiprocess.max_workers
            if config.multiprocess and config.multiprocess.max_workers
            else max(1, len(config.executors))
        )

        ready_batch: List[str] = []
        pending_ids = {executor.id for executor in config.executors if executor.id not in completed_executor_ids}
        if pending_ids:
            for executor_id in ordered_ids:
                executor = next((item for item in config.executors if item.id == executor_id), None)
                if not executor or executor.id not in pending_ids:
                    continue
                if all(dep in completed_executor_ids for dep in executor.after):
                    ready_batch.append(executor.id)
                if len(ready_batch) >= max_workers:
                    break

        executor_items: List[Dict[str, Any]] = []
        for order, executor in enumerate(config.executors):
            executor_records = per_executor.get(executor.id, [])
            success_records = [record for record in executor_records if record.status == 200]
            ttfts = [float(record.first_resp_time) for record in success_records if record.first_resp_time > 0]
            total_times = [float(record.last_resp_time) for record in success_records if record.last_resp_time > 0]
            input_tokens = [float(record.qtokens) for record in success_records if record.qtokens >= 0]
            output_tokens = [float(record.atokens) for record in success_records if record.atokens >= 0]
            tps = [float(record.token_per_second) for record in success_records if record.token_per_second > 0]
            tps_with_ttft = [
                float(record.token_per_second_with_calltime)
                for record in success_records
                if record.token_per_second_with_calltime > 0
            ]

            completed = len(executor_records)
            total = dataset_total
            success_count = len(success_records)
            error_count = completed - success_count
            progress_percent = min(100.0, (completed / total) * 100) if total > 0 else 0.0

            if total > 0 and completed >= total:
                status = TaskStatus.FAILED.value if error_count > 0 else TaskStatus.COMPLETED.value
            elif completed > 0:
                status = TaskStatus.PAUSED.value if task_status == TaskStatus.PAUSED else TaskStatus.RUNNING.value
            elif task_status in (TaskStatus.RUNNING, TaskStatus.PAUSED) and executor.id in ready_batch:
                status = TaskStatus.PAUSED.value if task_status == TaskStatus.PAUSED else TaskStatus.RUNNING.value
            elif executor.after and not all(dep in completed_executor_ids for dep in executor.after):
                status = "blocked"
            elif task_status == TaskStatus.CANCELLED:
                status = TaskStatus.CANCELLED.value
            elif task_status == TaskStatus.FAILED:
                status = TaskStatus.FAILED.value
            else:
                status = TaskStatus.PENDING.value

            score = round(
                min(100.0, (success_count / completed * 100) if completed else 0.0) * 0.45
                + min(100.0, max(0.0, 100.0 - (_avg(ttfts) / 50.0))) * 0.25
                + min(100.0, _avg(tps_with_ttft) * 4.0) * 0.30
            ) if completed else 0

            if completed == 0:
                conclusion = "尚无样本结果"
            elif error_count > 0 and success_count == 0:
                conclusion = "当前样本全部失败，需要先排查接口或参数配置"
            elif error_count > 0:
                conclusion = "已有可用结果，但存在失败样本，结论需结合错误分布复核"
            elif progress_percent < 100:
                conclusion = "基于当前已完成样本给出阶段性结论，任务完成后可继续复核"
            elif _avg(ttfts) > 3000:
                conclusion = "结果稳定但首响偏慢，更适合离线或低交互场景"
            elif _avg(tps_with_ttft) < 10:
                conclusion = "结果可用，但生成速率偏低，吞吐能力一般"
            else:
                conclusion = "结果稳定，当前样本下延迟和生成效率表现正常"

            executor_items.append(
                {
                    "id": executor.id,
                    "name": executor.name,
                    "provider": executor.type,
                    "model": executor.model,
                    "after": list(executor.after),
                    "order": order,
                    "status": status,
                    "completed": completed,
                    "total": total,
                    "progress_percent": progress_percent,
                    "success_count": success_count,
                    "error_count": error_count,
                    "success_rate": (success_count / completed * 100.0) if completed else 0.0,
                    "avg_input_tokens": _avg(input_tokens),
                    "avg_output_tokens": _avg(output_tokens),
                    "avg_ttft": _avg(ttfts),
                    "p95_ttft": _percentile(ttfts, 0.95),
                    "avg_total_time": _avg(total_times),
                    "avg_token_per_second": _avg(tps),
                    "avg_token_per_second_with_calltime": _avg(tps_with_ttft),
                    "cost": sum(record.total_cost for record in executor_records),
                    "avg_cost_per_request": (sum(record.total_cost for record in executor_records) / completed) if completed else 0.0,
                    "score": score,
                    "conclusion": conclusion,
                }
            )

        return executor_items

    def _update_progress_snapshot(
        self,
        run_id: str,
        *,
        task_status: Optional[TaskStatus] = None,
        progress: Optional[TaskProgress] = None,
    ) -> Optional[TaskProgress]:
        current_progress = progress or self._progress.get(run_id)
        if not current_progress:
            return None

        config = self._load_run_config(run_id)
        if config:
            dataset_total = current_progress.dataset_total_per_executor or self._estimate_dataset_total(config)
            current_progress.dataset_total_per_executor = dataset_total
            current_progress.total = dataset_total * len(config.executors)
            current_progress.concurrency = sum(max(1, executor.concurrency) for executor in config.executors)
        else:
            dataset_total = current_progress.dataset_total_per_executor

        records = list(self._storage.fetch_run_records(run_id))
        completed = len(records)
        success_count = sum(1 for record in records if record.status == 200)
        error_count = completed - success_count
        current_progress.completed = completed
        current_progress.success_count = success_count
        current_progress.error_count = error_count
        current_progress.current_cost = sum(record.total_cost for record in records)
        if records:
            current_progress.currency = records[0].currency
        if current_progress.total > 0:
            current_progress.progress_percent = min(100.0, (completed / current_progress.total) * 100.0)

        effective_status = task_status or current_progress.status
        current_progress.executors = self._build_executor_progress(config, records, effective_status, dataset_total)
        current_progress.topology = self._build_topology(config, current_progress.executors)

        if current_progress.started_at and effective_status in (TaskStatus.RUNNING, TaskStatus.PAUSED):
            elapsed = (datetime.now() - current_progress.started_at).total_seconds() - current_progress.paused_duration_seconds
            current_progress.elapsed_seconds = max(elapsed, 0.0)
            if current_progress.completed > 0 and current_progress.total > current_progress.completed:
                remaining = current_progress.total - current_progress.completed
                rate = current_progress.completed / max(current_progress.elapsed_seconds, 1.0)
                current_progress.eta_seconds = (remaining / rate) if rate > 0 else None
            else:
                current_progress.eta_seconds = 0.0 if current_progress.total and current_progress.completed >= current_progress.total else None

        return current_progress

    def create_task(
        self,
        config_path: Optional[str] = None,
        config_content: Optional[str] = None,
        run_id: Optional[str] = None,
        task_type: str = "benchmark",
        scheduled_at: Optional[datetime] = None,
    ) -> TaskInfo:
        """Create a new task.

        Args:
            config_path: Path to YAML config file.
            config_content: YAML config content (used if config_path is None).
            run_id: Optional run ID.

        Returns:
            TaskInfo for the created task.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Generate run ID if not provided
        if not run_id:
            run_id = uuid.uuid4().hex
        scheduled_at = self._normalize_datetime(scheduled_at)

        # Load configuration
        if config_path:
            config = normalize_run_config(load_config(config_path))
            normalized_config_content = self._normalize_and_snapshot_config_content(config)
        elif config_content:
            config, _ = load_run_config_content(config_content)
            normalized_config_content = self._normalize_and_snapshot_config_content(config)
        else:
            raise ValueError("Either config_path or config_content must be provided")

        initial_status = (
            TaskStatus.SCHEDULED
            if scheduled_at and scheduled_at > datetime.now()
            else TaskStatus.PENDING
        )

        # Create task info
        task_info = TaskInfo(
            run_id=run_id,
            status=initial_status,
            config_path=config_path,
            task_name=config.info or "Untitled Task",
            task_type=task_type,
            scheduled_at=scheduled_at,
        )

        # Create progress tracker
        progress = TaskProgress(
            run_id=run_id,
            status=initial_status,
        )

        with self._lock:
            self._tasks[run_id] = task_info
            self._progress[run_id] = progress
            if normalized_config_content:
                self._config_contents[run_id] = normalized_config_content

        self._storage.register_run(
            run_id,
            config,
            config_path=config_path or "",
            config_content=normalized_config_content or "",
            task_type=task_type,
            status=initial_status.value,
            scheduled_at=int(scheduled_at.timestamp()) if scheduled_at else 0,
        )

        return task_info

    def schedule_task(self, run_id: str, scheduled_at: datetime) -> bool:
        scheduled_at = self._normalize_datetime(scheduled_at)
        if scheduled_at is None:
            return False
        task_info = self._tasks.get(run_id)
        if not task_info:
            snapshot = self._load_run_snapshot(run_id)
            if not snapshot:
                return False
            task_info = self._build_task_info_from_snapshot(snapshot)
            self._tasks[run_id] = task_info

        delay_seconds = max((scheduled_at - datetime.now()).total_seconds(), 0.0)
        existing_timer = self._scheduled_timers.pop(run_id, None)
        if existing_timer:
            existing_timer.cancel()

        task_info.status = TaskStatus.SCHEDULED
        task_info.scheduled_at = scheduled_at

        progress = self._progress.get(run_id)
        if progress:
            progress.status = TaskStatus.SCHEDULED

        self._storage.update_run_status(
            run_id,
            status=TaskStatus.SCHEDULED.value,
            scheduled_at=int(scheduled_at.timestamp()),
            error_message="",
        )

        timer = threading.Timer(delay_seconds, self.run_task, args=(run_id,))
        timer.daemon = True
        self._scheduled_timers[run_id] = timer
        timer.start()
        return True

    def get_task_config_content(self, run_id: str) -> Optional[str]:
        if run_id in self._config_contents:
            return self._config_contents[run_id]
        snapshot = self._load_run_snapshot(run_id)
        if snapshot:
            content = snapshot.get("config_content") or ""
            return content or None
        return None

    def rerun_task(
        self,
        source_run_id: str,
        *,
        scheduled_at: Optional[datetime] = None,
    ) -> Optional[TaskInfo]:
        config_content = self.get_task_config_content(source_run_id)
        if not config_content:
            return None
        source_snapshot = self._load_run_snapshot(source_run_id) or {}
        return self.create_task(
            config_content=config_content,
            task_type=str(source_snapshot.get("task_type") or "benchmark"),
            scheduled_at=scheduled_at,
        )

    def start_task(self, run_id: str) -> bool:
        task_info = self._tasks.get(run_id)
        if not task_info:
            snapshot = self._load_run_snapshot(run_id)
            if not snapshot:
                return False
            task_info = self._build_task_info_from_snapshot(snapshot)
            self._tasks[run_id] = task_info
            self._progress.setdefault(run_id, TaskProgress(run_id=run_id, status=task_info.status))

        if task_info.status not in (TaskStatus.SCHEDULED, TaskStatus.PENDING):
            return False

        scheduled_timer = self._scheduled_timers.pop(run_id, None)
        if scheduled_timer:
            scheduled_timer.cancel()

        task_info.status = TaskStatus.PENDING
        task_info.scheduled_at = None

        progress = self._progress.get(run_id)
        if progress:
            progress.status = TaskStatus.PENDING

        self._storage.update_run_status(
            run_id,
            status=TaskStatus.PENDING.value,
            scheduled_at=0,
            error_message="",
        )
        return True

    def recover_task(self, run_id: str) -> bool:
        task_info = self._tasks.get(run_id)
        if not task_info:
            snapshot = self._load_run_snapshot(run_id)
            if not snapshot:
                return False
            task_info = self._build_task_info_from_snapshot(snapshot)
            self._tasks[run_id] = task_info
            self._progress.setdefault(run_id, TaskProgress(run_id=run_id, status=task_info.status))

        if task_info.status not in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False

        if not self.get_task_config_content(run_id) and not task_info.config_path:
            return False

        scheduled_timer = self._scheduled_timers.pop(run_id, None)
        if scheduled_timer:
            scheduled_timer.cancel()

        task_info.status = TaskStatus.PENDING
        task_info.completed_at = None
        task_info.error_message = None
        task_info.scheduled_at = None

        progress = self._progress.get(run_id)
        if progress:
            progress.status = TaskStatus.PENDING
            progress.paused_at = None

        self._storage.update_run_status(
            run_id,
            status=TaskStatus.PENDING.value,
            completed_at=0,
            scheduled_at=0,
            error_message="",
        )
        return True

    def run_task(self, run_id: str) -> None:
        """Run a task in the background.

        Args:
            run_id: The run ID to execute.
        """
        task_info = self._tasks.get(run_id)
        if not task_info:
            snapshot = self._load_run_snapshot(run_id)
            if not snapshot:
                logger.error("Task not found: %s", run_id)
                return
            task_info = self._build_task_info_from_snapshot(snapshot)
            self._tasks[run_id] = task_info
            self._progress.setdefault(run_id, TaskProgress(run_id=run_id, status=task_info.status))

        existing_timer = self._scheduled_timers.pop(run_id, None)
        if existing_timer:
            existing_timer.cancel()

        if task_info.status == TaskStatus.CANCELLED:
            logger.info("Task %s has been cancelled before execution start", run_id)
            return

        task_info.status = TaskStatus.RUNNING
        task_info.started_at = datetime.now()
        task_info.completed_at = None
        task_info.error_message = None
        task_info.scheduled_at = task_info.scheduled_at

        progress = self._progress.get(run_id)
        if progress:
            progress.status = TaskStatus.RUNNING
            progress.started_at = task_info.started_at
            progress.paused_at = None
            progress.paused_duration_seconds = 0.0

        self._storage.update_run_status(
            run_id,
            status=TaskStatus.RUNNING.value,
            started_at=int(task_info.started_at.timestamp()),
            completed_at=0,
            scheduled_at=0,
            error_message="",
        )

        config = self._load_run_config(run_id)
        dataset_total = self._estimate_dataset_total(config)
        if progress and config:
            progress.dataset_total_per_executor = dataset_total
            progress.total = dataset_total * len(config.executors)
            progress.concurrency = sum(max(1, executor.concurrency) for executor in config.executors)
            progress.executors = self._build_executor_progress(config, [], TaskStatus.RUNNING, dataset_total)
            progress.topology = self._build_topology(config, progress.executors)

        cancel_event = self._cancel_events.get(run_id)
        if cancel_event is None:
            cancel_event = threading.Event()
            self._cancel_events[run_id] = cancel_event
        pause_event = self._pause_events.get(run_id)
        if pause_event is None:
            pause_event = threading.Event()
            self._pause_events[run_id] = pause_event
        self._resume_events[run_id] = threading.Event()

        temp_config_path = None
        try:
            if self._config_contents.get(run_id):
                import tempfile

                config, _ = load_run_config_content(self._config_contents[run_id])
                config = self._inject_pricing_snapshot(config)
                normalized_content = self._normalize_and_snapshot_config_content(config)
                self._config_contents[run_id] = normalized_content
                temp_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".yaml",
                    delete=False,
                    encoding="utf-8",
                )
                temp_file.write(normalized_content)
                temp_file.close()
                temp_config_path = temp_file.name
                config_path = temp_config_path
                self._storage.register_run(
                    run_id,
                    config,
                    config_path="",
                    config_content=normalized_content,
                    task_type=task_info.task_type,
                    status=TaskStatus.RUNNING.value,
                    scheduled_at=0,
                )
            elif task_info.config_path:
                config = normalize_run_config(load_config(task_info.config_path))
                config = self._inject_pricing_snapshot(config)
                config_path = task_info.config_path
            else:
                raise ValueError("No configuration available")

            manager = RunManager(
                config,
                config_path=config_path,
                pricing_path=None,
                run_id=run_id,
                register_run=False,
                cancel_event=cancel_event,
            )

            monitor_thread = threading.Thread(
                target=self._monitor_progress,
                args=(run_id, cancel_event),
                daemon=True,
            )
            monitor_thread.start()

            manager.run()

            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.now()
            self._storage.update_run_status(
                run_id,
                status=TaskStatus.COMPLETED.value,
                completed_at=int(task_info.completed_at.timestamp()),
                error_message="",
            )

            if progress:
                progress.status = TaskStatus.COMPLETED
                if progress.started_at and task_info.completed_at:
                    progress.elapsed_seconds = (task_info.completed_at - progress.started_at).total_seconds()
                self._update_progress_snapshot(run_id, task_status=TaskStatus.COMPLETED, progress=progress)

            try:
                records = list(self._storage.fetch_run_records(run_id))
                total_cost = sum(r.total_cost for r in records)
                currency = records[0].currency if records else "CNY"
                self._storage.update_run_cost(run_id, total_cost, currency)
                logger.info("Task %s completed with total cost: %.4f %s", run_id, total_cost, currency)
            except Exception as e:
                logger.warning("Failed to update run cost: %s", e)

        except TaskCancelledError:
            logger.info("Task %s cancelled during execution", run_id)
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = datetime.now()
            task_info.error_message = "Task cancelled"
            self._storage.update_run_status(
                run_id,
                status=TaskStatus.CANCELLED.value,
                completed_at=int(task_info.completed_at.timestamp()),
                error_message=task_info.error_message,
            )
            if progress:
                progress.status = TaskStatus.CANCELLED
                self._update_progress_snapshot(run_id, task_status=TaskStatus.CANCELLED, progress=progress)
        except Exception as e:
            logger.exception("Task failed: %s", e)
            task_info.status = TaskStatus.FAILED
            task_info.error_message = str(e)
            task_info.completed_at = datetime.now()
            self._storage.update_run_status(
                run_id,
                status=TaskStatus.FAILED.value,
                completed_at=int(task_info.completed_at.timestamp()),
                error_message=task_info.error_message,
            )

            if progress:
                progress.status = TaskStatus.FAILED
                self._update_progress_snapshot(run_id, task_status=TaskStatus.FAILED, progress=progress)

        finally:
            self._cancel_events.pop(run_id, None)
            self._pause_events.pop(run_id, None)
            self._resume_events.pop(run_id, None)

            if temp_config_path:
                try:
                    import os

                    os.unlink(temp_config_path)
                except Exception:
                    pass

            self._dispatch_async(self._broadcast_status(run_id))

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

        pause_event = self._pause_events.get(run_id)
        last_completed_count = 0
        last_update_time = datetime.now()

        while not cancel_event.is_set():
            try:
                # Check for pause request
                if pause_event and pause_event.is_set() and progress.status == TaskStatus.RUNNING:
                    # Transition to paused state
                    task_info = self._tasks.get(run_id)
                    if task_info:
                        task_info.status = TaskStatus.PAUSED
                    progress.status = TaskStatus.PAUSED
                    progress.paused_at = datetime.now()

                    # Broadcast status change
                    self._dispatch_async(self._broadcast_status(run_id))
                    self._dispatch_async(self._broadcast_progress(run_id))

                    # Wait for resume or cancel
                    while pause_event.is_set() and not cancel_event.is_set():
                        time.sleep(0.5)

                    # Check if we were cancelled while paused
                    if cancel_event.is_set():
                        break

                    # Resume - update status
                    if task_info:
                        task_info.status = TaskStatus.RUNNING
                    progress.status = TaskStatus.RUNNING

                    # Calculate paused duration
                    if progress.paused_at:
                        paused_duration = (datetime.now() - progress.paused_at).total_seconds()
                        progress.paused_duration_seconds += paused_duration
                        progress.paused_at = None

                    # Broadcast resume
                    self._dispatch_async(self._broadcast_status(run_id))

                # Check if task has completed/failed/cancelled - exit loop if so
                task_info = self._tasks.get(run_id)
                if task_info and task_info.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    logger.info("Task %s finished with status %s, stopping progress monitor", run_id, task_info.status.value)
                    break

                # Only fetch stats and update if not paused
                if progress.status != TaskStatus.PAUSED:
                    self._update_progress_snapshot(run_id, task_status=progress.status, progress=progress)

                    # Calculate current rate (requests per second)
                    now = datetime.now()
                    time_delta = (now - last_update_time).total_seconds()
                    if time_delta > 0:
                        completed_delta = progress.completed - last_completed_count
                        progress.current_rate = completed_delta / time_delta
                    last_update_time = now
                    last_completed_count = progress.completed

                    # Broadcast progress update
                    self._dispatch_async(self._broadcast_progress(run_id))

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
                current_rate=progress.current_rate,
                concurrency=progress.concurrency,
                paused_at=progress.paused_at,
                executors=progress.executors,
                topology=progress.topology,
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
            run = self._storage.get_run(run_id)
            if run:
                return self._build_task_info_from_snapshot(run)
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
        progress = self._progress.get(run_id)
        if progress:
            return self._update_progress_snapshot(run_id, task_status=progress.status, progress=progress)

        task_info = self.get_task(run_id)
        if not task_info:
            return None

        synthetic_progress = TaskProgress(
            run_id=run_id,
            status=task_info.status,
            started_at=task_info.started_at or task_info.created_at,
        )
        if task_info.completed_at and synthetic_progress.started_at:
            synthetic_progress.elapsed_seconds = max(
                (task_info.completed_at - synthetic_progress.started_at).total_seconds(),
                0.0,
            )
        return self._update_progress_snapshot(run_id, task_status=task_info.status, progress=synthetic_progress)

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
        try:
            db_runs = self._storage.list_runs(
                limit=limit,
                offset=offset,
                status=status.value if status else None,
            )
        except Exception as e:
            logger.warning("Failed to load tasks from database: %s", e)
            db_runs = []

        tasks: List[TaskInfo] = []
        for run in db_runs:
            run_id = run.get("run_id")
            memory_task = self._tasks.get(run_id or "")
            tasks.append(memory_task or self._build_task_info_from_snapshot(run))
        return tasks

    def count_tasks(self, status: Optional[TaskStatus] = None) -> int:
        """Count tasks from both memory and database.

        Args:
            status: Filter by status.

        Returns:
            Number of tasks.
        """
        try:
            return self._storage.count_runs(status.value if status else None)
        except Exception:
            return 0

    def cancel_task(self, run_id: str) -> bool:
        """Cancel a running task.

        Args:
            run_id: The run ID.

        Returns:
            True if cancelled, False if not cancellable.
        """
        task_info = self._tasks.get(run_id)
        if not task_info:
            snapshot = self._load_run_snapshot(run_id)
            if not snapshot:
                return False
            task_info = self._build_task_info_from_snapshot(snapshot)
            self._tasks[run_id] = task_info

        if task_info.status not in (TaskStatus.SCHEDULED, TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED):
            return False

        scheduled_timer = self._scheduled_timers.pop(run_id, None)
        if scheduled_timer:
            scheduled_timer.cancel()

        # Clear pause event if paused to allow cancellation
        if task_info.status == TaskStatus.PAUSED:
            pause_event = self._pause_events.get(run_id)
            if pause_event:
                pause_event.clear()

        # Set cancel event
        cancel_event = self._cancel_events.get(run_id)
        if cancel_event:
            cancel_event.set()

        # Update status
        task_info.status = TaskStatus.CANCELLED
        task_info.completed_at = datetime.now()
        task_info.error_message = "Task cancelled"

        progress = self._progress.get(run_id)
        if progress:
            progress.status = TaskStatus.CANCELLED
            self._update_progress_snapshot(run_id, task_status=TaskStatus.CANCELLED, progress=progress)
        self._storage.update_run_status(
            run_id,
            status=TaskStatus.CANCELLED.value,
            completed_at=int(task_info.completed_at.timestamp()),
            scheduled_at=0,
            error_message=task_info.error_message,
        )

        # Broadcast status change
        self._dispatch_async(self._broadcast_status(run_id))

        return True

    def pause_task(self, run_id: str) -> bool:
        """Pause a running task.

        The pause is graceful - the task will complete any in-flight requests
        before pausing. Progress monitoring will detect the pause and stop
        polling until resume is called.

        Args:
            run_id: The run ID.

        Returns:
            True if paused, False if not pausable.
        """
        task_info = self._tasks.get(run_id)
        if not task_info:
            return False

        if task_info.status != TaskStatus.RUNNING:
            return False

        # Set pause event
        pause_event = self._pause_events.get(run_id)
        if pause_event:
            pause_event.set()

        # Note: Actual status change happens in _monitor_progress
        # to ensure graceful pause
        logger.info("Pause requested for task %s", run_id)
        self._storage.update_run_status(run_id, status=TaskStatus.PAUSED.value)

        # Broadcast that pause was requested
        self._dispatch_async(self._broadcast_status(run_id))

        return True

    def resume_task(self, run_id: str) -> bool:
        """Resume a paused task.

        Args:
            run_id: The run ID.

        Returns:
            True if resumed, False if not resumable.
        """
        task_info = self._tasks.get(run_id)
        if not task_info:
            return False

        if task_info.status != TaskStatus.PAUSED:
            return False

        # Clear pause event to allow resumption
        pause_event = self._pause_events.get(run_id)
        if pause_event:
            pause_event.clear()

        # Set resume event
        resume_event = self._resume_events.get(run_id)
        if resume_event:
            resume_event.set()

        # Note: Actual status change happens in _monitor_progress
        logger.info("Resume requested for task %s", run_id)
        self._storage.update_run_status(run_id, status=TaskStatus.RUNNING.value)

        # Broadcast that resume was requested
        self._dispatch_async(self._broadcast_status(run_id))

        return True

    def stop_task(self, run_id: str) -> bool:
        """Stop a task (alias for cancel, not recoverable).

        Args:
            run_id: The run ID.

        Returns:
            True if stopped, False if not stoppable.
        """
        # Stop is the same as cancel - not recoverable
        return self.cancel_task(run_id)

    def retry_task(self, run_id: str) -> Optional[TaskInfo]:
        """Retry a failed task.

        Args:
            run_id: The run ID.

        Returns:
            New TaskInfo if retry started, None otherwise.
        """
        task_info = self.get_task(run_id)
        if not task_info or task_info.status != TaskStatus.FAILED:
            return None

        return self.rerun_task(run_id)

    def delete_task(self, run_id: str) -> bool:
        """Delete a task.

        Args:
            run_id: The run ID.

        Returns:
            True if deleted, False if not found.
        """
        found = False
        with self._lock:
            if run_id in self._tasks:
                found = True
                del self._tasks[run_id]
            if run_id in self._progress:
                del self._progress[run_id]
            if run_id in self._config_contents:
                del self._config_contents[run_id]
            # Also cleanup control events if exist
            if run_id in self._cancel_events:
                del self._cancel_events[run_id]
            if run_id in self._pause_events:
                del self._pause_events[run_id]
            if run_id in self._resume_events:
                del self._resume_events[run_id]
            scheduled_timer = self._scheduled_timers.pop(run_id, None)
            if scheduled_timer:
                scheduled_timer.cancel()
        found = self._storage.delete_run(run_id) or found
        return found

    def get_stats(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a task using current persisted records.

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

        first_resp_times = [float(r.first_resp_time) for r in successful if r.first_resp_time > 0]
        last_resp_times = [float(r.last_resp_time) for r in successful if r.last_resp_time > 0]
        char_per_sec = [float(r.char_per_second) for r in successful if r.char_per_second > 0]
        token_throughput = [float(r.token_throughput) for r in successful if r.token_throughput > 0]
        token_per_second = [float(r.token_per_second) for r in successful if r.token_per_second > 0]
        token_per_second_with_calltime = [
            float(r.token_per_second_with_calltime)
            for r in successful
            if r.token_per_second_with_calltime > 0
        ]
        input_tokens = [float(r.qtokens) for r in successful if r.qtokens >= 0]
        output_tokens = [float(r.atokens) for r in successful if r.atokens >= 0]

        return {
            "total_requests": total,
            "success_count": len(successful),
            "error_count": len(failed),
            "success_rate": len(successful) / total * 100 if total > 0 else 0,
            "total_cost": sum(r.total_cost for r in records),
            "currency": records[0].currency if records else "CNY",
            "avg_first_resp_time": _avg(first_resp_times),
            "p50_first_resp_time": _percentile(first_resp_times, 0.50),
            "p95_first_resp_time": _percentile(first_resp_times, 0.95),
            "p99_first_resp_time": _percentile(first_resp_times, 0.99),
            "avg_last_resp_time": _avg(last_resp_times),
            "p95_last_resp_time": _percentile(last_resp_times, 0.95),
            "avg_char_per_second": _avg(char_per_sec),
            "avg_token_throughput": _avg(token_throughput),
            "avg_token_per_second": _avg(token_per_second),
            "avg_token_per_second_with_calltime": _avg(token_per_second_with_calltime),
            "avg_input_tokens": _avg(input_tokens),
            "avg_output_tokens": _avg(output_tokens),
            "total_input_tokens": sum(r.qtokens for r in records),
            "total_output_tokens": sum(r.atokens for r in records),
        }

    def get_quick_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Generate a fresh report snapshot from current run records.

        Args:
            run_id: The run ID.

        Returns:
            Quick report dictionary, or None if no data.
        """
        task_info = self.get_task(run_id)  # Use get_task to check both memory and database
        stats = self.get_stats(run_id)

        if not stats:
            return None

        progress = self.get_progress(run_id)
        task_status = progress.status if progress else (task_info.status if task_info else TaskStatus.COMPLETED)
        executor_summary = progress.executors if progress else self._get_executor_summary(run_id)

        # Calculate dimension scores
        latency_score = self._calculate_latency_score(stats.get("avg_first_resp_time", 0))
        throughput_score = self._calculate_throughput_score(stats.get("avg_token_per_second_with_calltime", 0))
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

        # Calculate cost analysis
        cost_analysis = self._calculate_cost_analysis(run_id, executor_summary, stats)

        # Calculate duration
        duration_seconds = 0
        if task_info and task_info.started_at and task_info.completed_at:
            duration_seconds = (task_info.completed_at - task_info.started_at).total_seconds()

        return {
            "run_id": run_id,
            "task_name": task_info.task_name if task_info else "Unknown Task",
            "status": task_status.value if isinstance(task_status, TaskStatus) else str(task_status),
            "is_partial": task_status != TaskStatus.COMPLETED,
            "completed_at": task_info.completed_at if task_info else None,
            "duration_seconds": duration_seconds,
            "generated_at": datetime.now(),
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
                "success_count": stats.get("success_count", 0),
                "error_count": stats.get("error_count", 0),
                "success_rate": stats.get("success_rate", 0),
                "avg_ttft": stats.get("avg_first_resp_time", 0),
                "p50_ttft": stats.get("p50_first_resp_time", 0),
                "p95_ttft": stats.get("p95_first_resp_time", 0),
                "avg_total_time": stats.get("avg_last_resp_time", 0),
                "p95_total_time": stats.get("p95_last_resp_time", 0),
                "avg_tps": stats.get("avg_token_per_second", 0),
                "avg_tps_with_ttft": stats.get("avg_token_per_second_with_calltime", 0),
                "avg_input_tokens": stats.get("avg_input_tokens", 0),
                "avg_output_tokens": stats.get("avg_output_tokens", 0),
                "total_cost": stats.get("total_cost", 0),
                "currency": stats.get("currency", "CNY"),
                "total_input_tokens": stats.get("total_input_tokens", 0),
                "total_output_tokens": stats.get("total_output_tokens", 0),
            },
            "executor_summary": executor_summary,
            "cost_analysis": cost_analysis,
            "topology": progress.topology if progress else {"nodes": [], "edges": [], "layers": []},
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

    def _calculate_cost_analysis(
        self,
        run_id: str,
        executor_summary: List[Dict[str, Any]],
        stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate cost analysis by executor."""
        records = list(self._storage.fetch_run_records(run_id))

        if not records:
            return {
                "total_cost": 0,
                "currency": "CNY",
                "by_executor": [],
            }

        total_cost = stats.get("total_cost", 0)
        currency = stats.get("currency", "CNY")

        # Group by executor for cost breakdown
        executor_costs: Dict[str, Dict[str, Any]] = {}
        for r in records:
            exec_id = r.executor_id or "default"
            if exec_id not in executor_costs:
                executor_costs[exec_id] = {
                    "cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "request_count": 0,
                }
            executor_costs[exec_id]["cost"] += r.total_cost
            executor_costs[exec_id]["input_tokens"] += r.qtokens
            executor_costs[exec_id]["output_tokens"] += r.atokens
            executor_costs[exec_id]["request_count"] += 1

        by_executor = []
        for exec_id, cost_data in executor_costs.items():
            avg_cost = cost_data["cost"] / cost_data["request_count"] if cost_data["request_count"] > 0 else 0
            by_executor.append({
                "executor": exec_id,
                "cost": round(cost_data["cost"], 6),
                "input_tokens": cost_data["input_tokens"],
                "output_tokens": cost_data["output_tokens"],
                "request_count": cost_data["request_count"],
                "avg_cost_per_request": round(avg_cost, 6),
            })

        # Sort by cost descending
        by_executor.sort(key=lambda x: x["cost"], reverse=True)

        return {
            "total_cost": round(total_cost, 6),
            "currency": currency,
            "by_executor": by_executor,
        }

    def _get_executor_summary(self, run_id: str) -> List[Dict[str, Any]]:
        """Get summary statistics by executor."""
        progress = self.get_progress(run_id)
        if progress and progress.executors:
            return progress.executors
        return []
