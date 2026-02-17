"""Base classes for notification channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(Enum):
    """Status of notification delivery."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


@dataclass
class NotificationMessage:
    """Represents a notification message to be sent.

    Attributes:
        title: The title/subject of the notification.
        content: The main content/body of the notification.
        event_type: The type of event that triggered this notification.
        priority: Priority level of the notification.
        metadata: Additional metadata about the event.
        timestamp: When the notification was created.
        run_id: Optional run ID associated with the notification.
        executor_id: Optional executor ID associated with the notification.
    """
    title: str
    content: str
    event_type: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    executor_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "title": self.title,
            "content": self.content,
            "event_type": self.event_type,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "executor_id": self.executor_id,
        }

    @classmethod
    def create_task_started(
        cls,
        run_id: str,
        task_name: str,
        executor_count: int,
        dataset_size: int,
    ) -> "NotificationMessage":
        """Create a notification for task started event."""
        return cls(
            title=f"Task Started: {task_name}",
            content=f"Run ID: {run_id}\n"
                    f"Executors: {executor_count}\n"
                    f"Dataset Size: {dataset_size}",
            event_type="task_started",
            priority=NotificationPriority.NORMAL,
            run_id=run_id,
            metadata={
                "task_name": task_name,
                "executor_count": executor_count,
                "dataset_size": dataset_size,
            },
        )

    @classmethod
    def create_task_progress(
        cls,
        run_id: str,
        task_name: str,
        progress_percent: float,
        completed: int,
        total: int,
        elapsed_seconds: float,
        eta_seconds: Optional[float] = None,
    ) -> "NotificationMessage":
        """Create a notification for task progress event."""
        content = (
            f"Run ID: {run_id}\n"
            f"Progress: {progress_percent:.1f}%\n"
            f"Completed: {completed}/{total}\n"
            f"Elapsed: {elapsed_seconds:.1f}s"
        )
        if eta_seconds is not None:
            content += f"\nETA: {eta_seconds:.1f}s"

        return cls(
            title=f"Task Progress: {task_name} ({progress_percent:.0f}%)",
            content=content,
            event_type="task_progress",
            priority=NotificationPriority.LOW,
            run_id=run_id,
            metadata={
                "task_name": task_name,
                "progress_percent": progress_percent,
                "completed": completed,
                "total": total,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
            },
        )

    @classmethod
    def create_task_completed(
        cls,
        run_id: str,
        task_name: str,
        total_requests: int,
        success_count: int,
        error_count: int,
        total_cost: float,
        currency: str,
        elapsed_seconds: float,
    ) -> "NotificationMessage":
        """Create a notification for task completed event."""
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        return cls(
            title=f"Task Completed: {task_name}",
            content=(
                f"Run ID: {run_id}\n"
                f"Total Requests: {total_requests}\n"
                f"Success Rate: {success_rate:.1f}%\n"
                f"Errors: {error_count}\n"
                f"Total Cost: {total_cost:.4f} {currency}\n"
                f"Duration: {elapsed_seconds:.1f}s"
            ),
            event_type="task_completed",
            priority=NotificationPriority.NORMAL,
            run_id=run_id,
            metadata={
                "task_name": task_name,
                "total_requests": total_requests,
                "success_count": success_count,
                "error_count": error_count,
                "total_cost": total_cost,
                "currency": currency,
                "elapsed_seconds": elapsed_seconds,
            },
        )

    @classmethod
    def create_task_failed(
        cls,
        run_id: str,
        task_name: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> "NotificationMessage":
        """Create a notification for task failed event."""
        return cls(
            title=f"Task Failed: {task_name}",
            content=(
                f"Run ID: {run_id}\n"
                f"Error: {error_message}"
            ),
            event_type="task_failed",
            priority=NotificationPriority.HIGH,
            run_id=run_id,
            metadata={
                "task_name": task_name,
                "error_message": error_message,
                "error_details": error_details or {},
            },
        )

    @classmethod
    def create_error_threshold(
        cls,
        run_id: str,
        task_name: str,
        current_error_rate: float,
        threshold: float,
        recent_errors: List[str],
    ) -> "NotificationMessage":
        """Create a notification for error threshold exceeded event."""
        return cls(
            title=f"Error Threshold Exceeded: {task_name}",
            content=(
                f"Run ID: {run_id}\n"
                f"Current Error Rate: {current_error_rate:.1%}\n"
                f"Threshold: {threshold:.1%}\n"
                f"Recent Errors:\n" + "\n".join(f"  - {e}" for e in recent_errors[:5])
            ),
            event_type="error_threshold",
            priority=NotificationPriority.CRITICAL,
            run_id=run_id,
            metadata={
                "task_name": task_name,
                "current_error_rate": current_error_rate,
                "threshold": threshold,
                "recent_errors": recent_errors,
            },
        )

    @classmethod
    def create_cost_alert(
        cls,
        run_id: str,
        task_name: str,
        current_cost: float,
        threshold: float,
        currency: str,
    ) -> "NotificationMessage":
        """Create a notification for cost alert event."""
        return cls(
            title=f"Cost Alert: {task_name}",
            content=(
                f"Run ID: {run_id}\n"
                f"Current Cost: {current_cost:.4f} {currency}\n"
                f"Threshold: {threshold:.4f} {currency}"
            ),
            event_type="cost_alert",
            priority=NotificationPriority.HIGH,
            run_id=run_id,
            metadata={
                "task_name": task_name,
                "current_cost": current_cost,
                "threshold": threshold,
                "currency": currency,
            },
        )


@dataclass
class NotificationResult:
    """Result of a notification delivery attempt.

    Attributes:
        success: Whether the notification was sent successfully.
        channel_type: The type of channel that was used.
        message: Optional success/error message.
        timestamp: When the delivery was attempted.
        response: Optional response data from the channel.
    """
    success: bool
    channel_type: str
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "success": self.success,
            "channel_type": self.channel_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response": self.response,
        }


class NotificationChannel(ABC):
    """Abstract base class for notification channels.

    All notification channels must implement this interface.
    """

    # Channel type identifier (e.g., "email", "webhook", "dingtalk")
    channel_type: str = "base"

    def __init__(self, config: Dict[str, Any]):
        """Initialize the channel with configuration.

        Args:
            config: Channel-specific configuration dictionary.
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate channel configuration. Override in subclasses.

        Raises:
            ValueError: If configuration is invalid.
        """
        pass

    @abstractmethod
    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Send a notification message.

        Args:
            message: The notification message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check if the channel is healthy and ready to send.

        Returns:
            True if the channel is healthy, False otherwise.
        """
        return True

    def format_message(self, message: NotificationMessage) -> str:
        """Format a message for this channel. Override for custom formatting.

        Args:
            message: The notification message to format.

        Returns:
            Formatted message string.
        """
        return f"{message.title}\n\n{message.content}"
