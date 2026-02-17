"""Notification channel registry for dynamic registration and creation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Type

from .base import NotificationChannel

logger = logging.getLogger(__name__)


class ChannelRegistry:
    """Registry for notification channel types.

    Supports dynamic registration of channel implementations and
    creation of channel instances from configuration.
    """

    def __init__(self):
        self._channels: Dict[str, Type[NotificationChannel]] = {}

    def register(
        self,
        channel_type: str,
    ) -> Callable[[Type[NotificationChannel]], Type[NotificationChannel]]:
        """Decorator to register a notification channel type.

        Args:
            channel_type: Unique identifier for this channel type.

        Returns:
            Decorator function.

        Example:
            @channel_registry.register("email")
            class EmailChannel(NotificationChannel):
                ...
        """
        def decorator(cls: Type[NotificationChannel]) -> Type[NotificationChannel]:
            if not issubclass(cls, NotificationChannel):
                raise TypeError(
                    f"Registered class must be a subclass of NotificationChannel, "
                    f"got {cls}"
                )
            self._channels[channel_type] = cls
            logger.debug("Registered notification channel: %s", channel_type)
            return cls
        return decorator

    def get(self, channel_type: str) -> Optional[Type[NotificationChannel]]:
        """Get a registered channel class by type.

        Args:
            channel_type: The channel type identifier.

        Returns:
            The channel class, or None if not found.
        """
        return self._channels.get(channel_type)

    def create(
        self,
        channel_type: str,
        config: Dict[str, Any],
    ) -> Optional[NotificationChannel]:
        """Create a channel instance from configuration.

        Args:
            channel_type: The channel type identifier.
            config: Channel-specific configuration.

        Returns:
            Channel instance, or None if type not found.

        Raises:
            ValueError: If channel configuration is invalid.
        """
        channel_cls = self.get(channel_type)
        if channel_cls is None:
            logger.warning("Unknown notification channel type: %s", channel_type)
            return None

        try:
            return channel_cls(config)
        except Exception as e:
            logger.error(
                "Failed to create notification channel '%s': %s",
                channel_type,
                e,
            )
            raise ValueError(f"Invalid configuration for channel '{channel_type}': {e}") from e

    def list_channels(self) -> Dict[str, Type[NotificationChannel]]:
        """Get all registered channel types.

        Returns:
            Dictionary mapping channel type to channel class.
        """
        return dict(self._channels)

    def has_channel(self, channel_type: str) -> bool:
        """Check if a channel type is registered.

        Args:
            channel_type: The channel type identifier.

        Returns:
            True if the channel is registered, False otherwise.
        """
        return channel_type in self._channels


# Global registry instance
channel_registry = ChannelRegistry()


# Convenience functions
def register_channel(channel_type: str) -> Callable[[Type[NotificationChannel]], Type[NotificationChannel]]:
    """Decorator to register a notification channel type.

    Args:
        channel_type: Unique identifier for this channel type.

    Returns:
        Decorator function.
    """
    return channel_registry.register(channel_type)


def create_channel(
    channel_type: str,
    config: Dict[str, Any],
) -> Optional[NotificationChannel]:
    """Create a channel instance from configuration.

    Args:
        channel_type: The channel type identifier.
        config: Channel-specific configuration.

    Returns:
        Channel instance, or None if type not found.
    """
    return channel_registry.create(channel_type, config)
