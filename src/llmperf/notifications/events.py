"""Notification event definitions and configuration."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class NotificationEventType(str, Enum):
    """Types of events that can trigger notifications."""
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    ERROR_THRESHOLD = "error_threshold"
    COST_ALERT = "cost_alert"
    EXECUTOR_ERROR = "executor_error"
    RATE_LIMIT_HIT = "rate_limit_hit"


# Default progress percentages at which to send notifications
DEFAULT_PROGRESS_POINTS = [25, 50, 75, 100]


class ProgressConfig(BaseModel):
    """Configuration for progress notifications."""
    enabled: bool = True
    """Whether progress notifications are enabled."""

    points: List[int] = Field(default_factory=lambda: DEFAULT_PROGRESS_POINTS.copy())
    """Progress percentages at which to send notifications (1-100)."""

    min_interval_seconds: float = 60.0
    """Minimum interval between progress notifications in seconds."""


class ErrorThresholdConfig(BaseModel):
    """Configuration for error threshold notifications."""
    enabled: bool = True
    """Whether error threshold notifications are enabled."""

    threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    """Error rate threshold (0.0-1.0) that triggers notification."""

    min_sample_size: int = Field(default=10, ge=1)
    """Minimum number of requests before checking threshold."""

    cooldown_seconds: float = 300.0
    """Minimum time between error threshold notifications."""


class CostAlertConfig(BaseModel):
    """Configuration for cost alert notifications."""
    enabled: bool = True
    """Whether cost alert notifications are enabled."""

    threshold: float = Field(default=100.0, ge=0.0)
    """Cost threshold that triggers notification."""

    currency: str = "CNY"
    """Currency for the threshold."""

    cooldown_seconds: float = 600.0
    """Minimum time between cost alerts."""


class EventConfig(BaseModel):
    """Configuration for a single notification event."""
    enabled: bool = True
    """Whether this event type is enabled."""

    channels: List[str] = Field(default_factory=list)
    """List of channel types to notify for this event."""

    cooldown_seconds: float = 0.0
    """Minimum time between notifications for this event type."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Additional event-specific configuration."""


class NotificationEventsConfig(BaseModel):
    """Configuration for all notification events."""
    task_started: EventConfig = Field(default_factory=EventConfig)
    task_progress: EventConfig = Field(
        default_factory=lambda: EventConfig(
            enabled=True,
            metadata={"progress_config": ProgressConfig().model_dump()},
        )
    )
    task_completed: EventConfig = Field(default_factory=EventConfig)
    task_failed: EventConfig = Field(default_factory=EventConfig)
    error_threshold: EventConfig = Field(
        default_factory=lambda: EventConfig(
            enabled=True,
            metadata={"threshold_config": ErrorThresholdConfig().model_dump()},
        )
    )
    cost_alert: EventConfig = Field(
        default_factory=lambda: EventConfig(
            enabled=True,
            metadata={"alert_config": CostAlertConfig().model_dump()},
        )
    )

    def get_event_config(self, event_type: str) -> Optional[EventConfig]:
        """Get configuration for a specific event type.

        Args:
            event_type: The event type to get configuration for.

        Returns:
            EventConfig if found, None otherwise.
        """
        event_map = {
            NotificationEventType.TASK_STARTED.value: self.task_started,
            NotificationEventType.TASK_PROGRESS.value: self.task_progress,
            NotificationEventType.TASK_COMPLETED.value: self.task_completed,
            NotificationEventType.TASK_FAILED.value: self.task_failed,
            NotificationEventType.ERROR_THRESHOLD.value: self.error_threshold,
            NotificationEventType.COST_ALERT.value: self.cost_alert,
        }
        return event_map.get(event_type)

    def is_event_enabled(self, event_type: str) -> bool:
        """Check if an event type is enabled.

        Args:
            event_type: The event type to check.

        Returns:
            True if the event is enabled, False otherwise.
        """
        config = self.get_event_config(event_type)
        return config.enabled if config else False

    def get_channels_for_event(self, event_type: str) -> List[str]:
        """Get the list of channels configured for an event type.

        Args:
            event_type: The event type to get channels for.

        Returns:
            List of channel type identifiers.
        """
        config = self.get_event_config(event_type)
        return config.channels if config else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationEventsConfig":
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with event configurations.

        Returns:
            NotificationEventsConfig instance.
        """
        return cls.model_validate(data)
