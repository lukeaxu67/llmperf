"""Log notification channel for testing and debugging."""

from __future__ import annotations

import json
import logging

from ..base import NotificationChannel, NotificationMessage, NotificationResult
from ..registry import register_channel

logger = logging.getLogger(__name__)


@register_channel("log")
class LogChannel(NotificationChannel):
    """Log notification channel.

    Simply logs notifications to the Python logging system.
    Useful for testing and debugging without external dependencies.

    Configuration:
        level: Log level (default: INFO)
        include_metadata: Whether to include full metadata (default: True)
        prefix: Prefix for log messages (default: "[Notification]")
    """

    channel_type = "log"

    @property
    def log_level(self) -> int:
        """Get the log level."""
        level_name = self.config.get("level", "INFO").upper()
        return getattr(logging, level_name, logging.INFO)

    @property
    def include_metadata(self) -> bool:
        """Get whether to include metadata."""
        return bool(self.config.get("include_metadata", True))

    @property
    def prefix(self) -> str:
        """Get the log prefix."""
        return self.config.get("prefix", "[Notification]")

    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Log the notification.

        Args:
            message: The notification message to log.

        Returns:
            NotificationResult indicating success (always succeeds).
        """
        log_parts = [
            self.prefix,
            f"Event: {message.event_type}",
            f"Title: {message.title}",
            f"Priority: {message.priority.value}",
        ]

        if message.run_id:
            log_parts.append(f"Run ID: {message.run_id}")

        log_parts.append(f"Content:\n{message.content}")

        if self.include_metadata and message.metadata:
            log_parts.append(f"Metadata: {json.dumps(message.metadata, ensure_ascii=False, indent=2)}")

        log_message = "\n".join(log_parts)
        logger.log(self.log_level, log_message)

        return NotificationResult(
            success=True,
            channel_type=self.channel_type,
            message="Notification logged successfully",
            response={"logged": True},
        )

    async def health_check(self) -> bool:
        """Check if channel is healthy (always returns True).

        Returns:
            Always True.
        """
        return True
