"""Notification system for LLMPerf.

This module provides a flexible notification system that supports multiple
channels (email, webhook, DingTalk, WeChat, etc.) and configurable events.
"""

from .base import (
    NotificationChannel,
    NotificationMessage,
    NotificationResult,
    NotificationPriority,
    NotificationStatus,
)
from .events import NotificationEventType, EventConfig, NotificationEventsConfig
from .manager import NotificationManager, NotificationConfig
from .registry import register_channel, create_channel, channel_registry

__all__ = [
    # Base classes
    "NotificationChannel",
    "NotificationMessage",
    "NotificationResult",
    "NotificationPriority",
    "NotificationStatus",
    # Events
    "NotificationEventType",
    "EventConfig",
    "NotificationEventsConfig",
    # Manager
    "NotificationManager",
    "NotificationConfig",
    # Registry
    "register_channel",
    "create_channel",
    "channel_registry",
]
