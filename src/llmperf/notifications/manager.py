"""Notification manager for coordinating notification delivery."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .base import NotificationChannel, NotificationMessage, NotificationResult
from .events import (
    NotificationEventsConfig,
    NotificationEventType,
    ProgressConfig,
    ErrorThresholdConfig,
    CostAlertConfig,
)
from .registry import channel_registry

logger = logging.getLogger(__name__)


class ChannelConfig(BaseModel):
    """Configuration for a single notification channel."""
    type: str
    """Channel type identifier (e.g., 'email', 'webhook')."""

    enabled: bool = True
    """Whether this channel is enabled."""

    config: Dict[str, Any] = Field(default_factory=dict)
    """Channel-specific configuration."""


class NotificationConfig(BaseModel):
    """Complete notification system configuration."""
    enabled: bool = True
    """Whether the notification system is enabled."""

    channels: List[ChannelConfig] = Field(default_factory=list)
    """List of notification channel configurations."""

    events: NotificationEventsConfig = Field(default_factory=NotificationEventsConfig)
    """Event-specific configuration."""

    default_channels: List[str] = Field(default_factory=list)
    """Default channels to use when event has no specific channels configured."""

    async_delivery: bool = True
    """Whether to deliver notifications asynchronously."""

    retry_count: int = Field(default=3, ge=0)
    """Number of retry attempts for failed deliveries."""

    retry_delay_seconds: float = Field(default=5.0, ge=0)
    """Delay between retry attempts."""


@dataclass
class CooldownState:
    """Tracks cooldown state for events."""
    last_notification_time: Dict[str, float] = field(default_factory=dict)
    """Map of event key to last notification timestamp."""

    notified_progress_points: Dict[str, Set[int]] = field(default_factory=dict)
    """Map of run_id to set of notified progress points."""

    last_cost_check: Dict[str, float] = field(default_factory=dict)
    """Map of run_id to last cost at notification."""

    def is_in_cooldown(self, key: str, cooldown_seconds: float) -> bool:
        """Check if a key is in cooldown.

        Args:
            key: Unique identifier for the event.
            cooldown_seconds: Cooldown duration in seconds.

        Returns:
            True if in cooldown, False otherwise.
        """
        if cooldown_seconds <= 0:
            return False
        last_time = self.last_notification_time.get(key, 0)
        return (time.time() - last_time) < cooldown_seconds

    def record_notification(self, key: str) -> None:
        """Record that a notification was sent for a key.

        Args:
            key: Unique identifier for the event.
        """
        self.last_notification_time[key] = time.time()

    def was_progress_notified(self, run_id: str, point: int) -> bool:
        """Check if a progress point was already notified.

        Args:
            run_id: The run ID.
            point: Progress percentage point.

        Returns:
            True if already notified, False otherwise.
        """
        points = self.notified_progress_points.get(run_id, set())
        return point in points

    def record_progress_notification(self, run_id: str, point: int) -> None:
        """Record that a progress point was notified.

        Args:
            run_id: The run ID.
            point: Progress percentage point.
        """
        if run_id not in self.notified_progress_points:
            self.notified_progress_points[run_id] = set()
        self.notified_progress_points[run_id].add(point)

    def cleanup_run(self, run_id: str) -> None:
        """Clean up state for a completed run.

        Args:
            run_id: The run ID to clean up.
        """
        self.notified_progress_points.pop(run_id, None)
        self.last_cost_check.pop(run_id, None)
        # Clean up event keys for this run
        keys_to_remove = [k for k in self.last_notification_time if run_id in k]
        for key in keys_to_remove:
            self.last_notification_time.pop(key, None)


class NotificationManager:
    """Manages notification channels and event dispatching.

    This class coordinates:
    - Channel initialization and health checking
    - Event-based notification dispatching
    - Cooldown management to prevent notification spam
    - Asynchronous delivery with retry logic
    """

    def __init__(self, config: NotificationConfig):
        """Initialize the notification manager.

        Args:
            config: Notification system configuration.
        """
        self.config = config
        self._channels: Dict[str, NotificationChannel] = {}
        self._cooldown_state = CooldownState()
        self._delivery_lock = asyncio.Lock()
        self._initialize_channels()

    def _initialize_channels(self) -> None:
        """Initialize all configured channels."""
        for channel_cfg in self.config.channels:
            if not channel_cfg.enabled:
                logger.debug("Skipping disabled channel: %s", channel_cfg.type)
                continue

            try:
                channel = channel_registry.create(
                    channel_cfg.type,
                    channel_cfg.config,
                )
                if channel:
                    self._channels[channel_cfg.type] = channel
                    logger.info("Initialized notification channel: %s", channel_cfg.type)
            except Exception as e:
                logger.error(
                    "Failed to initialize channel '%s': %s",
                    channel_cfg.type,
                    e,
                )

    def get_channel(self, channel_type: str) -> Optional[NotificationChannel]:
        """Get a channel by type.

        Args:
            channel_type: The channel type identifier.

        Returns:
            The channel instance, or None if not available.
        """
        return self._channels.get(channel_type)

    def list_available_channels(self) -> List[str]:
        """Get list of available channel types.

        Returns:
            List of channel type identifiers.
        """
        return list(self._channels.keys())

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all channels.

        Returns:
            Dictionary mapping channel type to health status.
        """
        results = {}
        for channel_type, channel in self._channels.items():
            try:
                results[channel_type] = await channel.health_check()
            except Exception as e:
                logger.warning(
                    "Health check failed for channel '%s': %s",
                    channel_type,
                    e,
                )
                results[channel_type] = False
        return results

    def _should_send_notification(
        self,
        event_type: str,
        run_id: Optional[str] = None,
    ) -> bool:
        """Check if a notification should be sent for an event.

        Args:
            event_type: The event type.
            run_id: Optional run ID for cooldown tracking.

        Returns:
            True if notification should be sent, False otherwise.
        """
        # Check if system is enabled
        if not self.config.enabled:
            return False

        # Check if event is enabled
        event_config = self.config.events.get_event_config(event_type)
        if not event_config or not event_config.enabled:
            return False

        # Check cooldown
        if event_config.cooldown_seconds > 0:
            cooldown_key = f"{event_type}:{run_id}" if run_id else event_type
            if self._cooldown_state.is_in_cooldown(
                cooldown_key,
                event_config.cooldown_seconds,
            ):
                return False

        return True

    def _get_target_channels(self, event_type: str) -> List[NotificationChannel]:
        """Get the list of channels to notify for an event.

        Args:
            event_type: The event type.

        Returns:
            List of channel instances to notify.
        """
        event_config = self.config.events.get_event_config(event_type)
        channel_types = []

        if event_config and event_config.channels:
            channel_types = event_config.channels
        elif self.config.default_channels:
            channel_types = self.config.default_channels

        channels = []
        for channel_type in channel_types:
            channel = self._channels.get(channel_type)
            if channel:
                channels.append(channel)
            else:
                logger.warning(
                    "Channel '%s' not available for event '%s'",
                    channel_type,
                    event_type,
                )

        return channels

    async def notify(self, message: NotificationMessage) -> Dict[str, NotificationResult]:
        """Send a notification to all configured channels.

        Args:
            message: The notification message to send.

        Returns:
            Dictionary mapping channel type to delivery result.
        """
        if not self._should_send_notification(message.event_type, message.run_id):
            logger.debug(
                "Notification suppressed for event '%s' (run_id=%s)",
                message.event_type,
                message.run_id,
            )
            return {}

        channels = self._get_target_channels(message.event_type)
        if not channels:
            logger.debug("No channels configured for event '%s'", message.event_type)
            return {}

        results: Dict[str, NotificationResult] = {}

        async def deliver_to_channel(channel: NotificationChannel) -> tuple[str, NotificationResult]:
            """Deliver notification to a single channel with retry."""
            channel_type = channel.channel_type
            last_error = None

            for attempt in range(self.config.retry_count + 1):
                try:
                    result = await channel.send(message)
                    if result.success:
                        return channel_type, result
                    last_error = result.message
                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        "Notification delivery attempt %d failed for channel '%s': %s",
                        attempt + 1,
                        channel_type,
                        e,
                    )

                if attempt < self.config.retry_count:
                    await asyncio.sleep(self.config.retry_delay_seconds)

            # All retries exhausted
            return channel_type, NotificationResult(
                success=False,
                channel_type=channel_type,
                message=f"Failed after {self.config.retry_count + 1} attempts: {last_error}",
            )

        if self.config.async_delivery:
            # Deliver to all channels in parallel
            tasks = [deliver_to_channel(ch) for ch in channels]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    results[channels[i].channel_type] = NotificationResult(
                        success=False,
                        channel_type=channels[i].channel_type,
                        message=str(result),
                    )
                else:
                    channel_type, notification_result = result
                    results[channel_type] = notification_result
        else:
            # Deliver sequentially
            for channel in channels:
                channel_type, result = await deliver_to_channel(channel)
                results[channel_type] = result

        # Record notification time for cooldown
        if any(r.success for r in results.values()):
            cooldown_key = f"{message.event_type}:{message.run_id}" if message.run_id else message.event_type
            self._cooldown_state.record_notification(cooldown_key)

        return results

    async def notify_task_started(
        self,
        run_id: str,
        task_name: str,
        executor_count: int,
        dataset_size: int,
    ) -> Dict[str, NotificationResult]:
        """Send task started notification.

        Args:
            run_id: The run ID.
            task_name: Name of the task.
            executor_count: Number of executors.
            dataset_size: Size of the dataset.

        Returns:
            Delivery results per channel.
        """
        message = NotificationMessage.create_task_started(
            run_id=run_id,
            task_name=task_name,
            executor_count=executor_count,
            dataset_size=dataset_size,
        )
        return await self.notify(message)

    async def notify_task_progress(
        self,
        run_id: str,
        task_name: str,
        progress_percent: float,
        completed: int,
        total: int,
        elapsed_seconds: float,
        eta_seconds: Optional[float] = None,
        force: bool = False,
    ) -> Dict[str, NotificationResult]:
        """Send task progress notification.

        Only sends notification if progress crosses a configured threshold
        and cooldown period has passed.

        Args:
            run_id: The run ID.
            task_name: Name of the task.
            progress_percent: Current progress percentage (0-100).
            completed: Number of completed items.
            total: Total number of items.
            elapsed_seconds: Elapsed time in seconds.
            eta_seconds: Estimated time remaining in seconds.
            force: Force notification regardless of thresholds.

        Returns:
            Delivery results per channel.
        """
        # Get progress configuration
        event_config = self.config.events.get_event_config(
            NotificationEventType.TASK_PROGRESS.value
        )
        progress_cfg_data = (event_config.metadata or {}).get("progress_config", {})
        progress_cfg = ProgressConfig.model_validate(progress_cfg_data)

        if not force:
            # Check if we should send notification at this progress point
            should_notify = False
            for point in progress_cfg.points:
                if progress_percent >= point and not self._cooldown_state.was_progress_notified(run_id, point):
                    should_notify = True
                    self._cooldown_state.record_progress_notification(run_id, point)
                    break

            # Check minimum interval
            if should_notify:
                cooldown_key = f"progress:{run_id}"
                if self._cooldown_state.is_in_cooldown(cooldown_key, progress_cfg.min_interval_seconds):
                    should_notify = False

            if not should_notify:
                return {}

        message = NotificationMessage.create_task_progress(
            run_id=run_id,
            task_name=task_name,
            progress_percent=progress_percent,
            completed=completed,
            total=total,
            elapsed_seconds=elapsed_seconds,
            eta_seconds=eta_seconds,
        )
        return await self.notify(message)

    async def notify_task_completed(
        self,
        run_id: str,
        task_name: str,
        total_requests: int,
        success_count: int,
        error_count: int,
        total_cost: float,
        currency: str,
        elapsed_seconds: float,
    ) -> Dict[str, NotificationResult]:
        """Send task completed notification.

        Args:
            run_id: The run ID.
            task_name: Name of the task.
            total_requests: Total number of requests.
            success_count: Number of successful requests.
            error_count: Number of failed requests.
            total_cost: Total cost incurred.
            currency: Currency code.
            elapsed_seconds: Total elapsed time in seconds.

        Returns:
            Delivery results per channel.
        """
        # Clean up run state
        self._cooldown_state.cleanup_run(run_id)

        message = NotificationMessage.create_task_completed(
            run_id=run_id,
            task_name=task_name,
            total_requests=total_requests,
            success_count=success_count,
            error_count=error_count,
            total_cost=total_cost,
            currency=currency,
            elapsed_seconds=elapsed_seconds,
        )
        return await self.notify(message)

    async def notify_task_failed(
        self,
        run_id: str,
        task_name: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, NotificationResult]:
        """Send task failed notification.

        Args:
            run_id: The run ID.
            task_name: Name of the task.
            error_message: Error message.
            error_details: Additional error details.

        Returns:
            Delivery results per channel.
        """
        # Clean up run state
        self._cooldown_state.cleanup_run(run_id)

        message = NotificationMessage.create_task_failed(
            run_id=run_id,
            task_name=task_name,
            error_message=error_message,
            error_details=error_details,
        )
        return await self.notify(message)

    async def check_and_notify_error_threshold(
        self,
        run_id: str,
        task_name: str,
        total_requests: int,
        error_count: int,
        recent_errors: List[str],
    ) -> Optional[Dict[str, NotificationResult]]:
        """Check error rate and send notification if threshold exceeded.

        Args:
            run_id: The run ID.
            task_name: Name of the task.
            total_requests: Total number of requests so far.
            error_count: Number of errors so far.
            recent_errors: List of recent error messages.

        Returns:
            Delivery results if notification sent, None otherwise.
        """
        # Get threshold configuration
        event_config = self.config.events.get_event_config(
            NotificationEventType.ERROR_THRESHOLD.value
        )
        threshold_cfg_data = (event_config.metadata or {}).get("threshold_config", {})
        threshold_cfg = ErrorThresholdConfig.model_validate(threshold_cfg_data)

        if not threshold_cfg.enabled:
            return None

        # Check minimum sample size
        if total_requests < threshold_cfg.min_sample_size:
            return None

        error_rate = error_count / total_requests

        # Check threshold
        if error_rate < threshold_cfg.threshold:
            return None

        # Check cooldown
        cooldown_key = f"error_threshold:{run_id}"
        if self._cooldown_state.is_in_cooldown(cooldown_key, threshold_cfg.cooldown_seconds):
            return None

        message = NotificationMessage.create_error_threshold(
            run_id=run_id,
            task_name=task_name,
            current_error_rate=error_rate,
            threshold=threshold_cfg.threshold,
            recent_errors=recent_errors,
        )
        result = await self.notify(message)
        self._cooldown_state.record_notification(cooldown_key)
        return result

    async def check_and_notify_cost_alert(
        self,
        run_id: str,
        task_name: str,
        current_cost: float,
        currency: str = "CNY",
    ) -> Optional[Dict[str, NotificationResult]]:
        """Check cost and send notification if threshold exceeded.

        Args:
            run_id: The run ID.
            task_name: Name of the task.
            current_cost: Current total cost.
            currency: Currency code.

        Returns:
            Delivery results if notification sent, None otherwise.
        """
        # Get alert configuration
        event_config = self.config.events.get_event_config(
            NotificationEventType.COST_ALERT.value
        )
        alert_cfg_data = (event_config.metadata or {}).get("alert_config", {})
        alert_cfg = CostAlertConfig.model_validate(alert_cfg_data)

        if not alert_cfg.enabled:
            return None

        # Check threshold
        if current_cost < alert_cfg.threshold:
            return None

        # Check if we already notified for this cost level
        last_notified_cost = self._cooldown_state.last_cost_check.get(run_id, 0)
        if current_cost <= last_notified_cost + alert_cfg.threshold:
            return None

        # Check cooldown
        cooldown_key = f"cost_alert:{run_id}"
        if self._cooldown_state.is_in_cooldown(cooldown_key, alert_cfg.cooldown_seconds):
            return None

        message = NotificationMessage.create_cost_alert(
            run_id=run_id,
            task_name=task_name,
            current_cost=current_cost,
            threshold=alert_cfg.threshold,
            currency=currency,
        )
        result = await self.notify(message)
        self._cooldown_state.last_cost_check[run_id] = current_cost
        self._cooldown_state.record_notification(cooldown_key)
        return result

    def cleanup_run(self, run_id: str) -> None:
        """Clean up state for a completed run.

        Args:
            run_id: The run ID to clean up.
        """
        self._cooldown_state.cleanup_run(run_id)
