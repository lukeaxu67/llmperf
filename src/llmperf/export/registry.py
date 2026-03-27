"""Exporter registry for dynamic registration and creation."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Type

from .base import Exporter, ExportConfig

logger = logging.getLogger(__name__)


class ExporterRegistry:
    """Registry for exporter types.

    Supports dynamic registration of exporter implementations and
    creation of exporter instances from configuration.
    """

    def __init__(self):
        self._exporters: Dict[str, Type[Exporter]] = {}

    def register(
        self,
        format_name: str,
    ) -> Callable[[Type[Exporter]], Type[Exporter]]:
        """Decorator to register an exporter type.

        Args:
            format_name: Unique identifier for this format.

        Returns:
            Decorator function.

        Example:
            @exporter_registry.register("csv")
            class CSVExporter(Exporter):
                ...
        """
        def decorator(cls: Type[Exporter]) -> Type[Exporter]:
            if not issubclass(cls, Exporter):
                raise TypeError(
                    f"Registered class must be a subclass of Exporter, "
                    f"got {cls}"
                )
            self._exporters[format_name] = cls
            logger.debug("Registered exporter: %s", format_name)
            return cls
        return decorator

    def get(self, format_name: str) -> Optional[Type[Exporter]]:
        """Get a registered exporter class by format name.

        Args:
            format_name: The format identifier.

        Returns:
            The exporter class, or None if not found.
        """
        return self._exporters.get(format_name)

    def create(
        self,
        format_name: str,
        config: ExportConfig,
    ) -> Optional[Exporter]:
        """Create an exporter instance from configuration.

        Args:
            format_name: The format identifier.
            config: Export configuration.

        Returns:
            Exporter instance, or None if format not found.

        Raises:
            ValueError: If configuration is invalid.
        """
        exporter_cls = self.get(format_name)
        if exporter_cls is None:
            logger.warning("Unknown exporter format: %s", format_name)
            return None

        try:
            return exporter_cls(config)
        except Exception as e:
            logger.error(
                "Failed to create exporter '%s': %s",
                format_name,
                e,
            )
            raise ValueError(f"Invalid configuration for exporter '{format_name}': {e}") from e

    def list_formats(self) -> Dict[str, Type[Exporter]]:
        """Get all registered exporter formats.

        Returns:
            Dictionary mapping format name to exporter class.
        """
        return dict(self._exporters)

    def has_format(self, format_name: str) -> bool:
        """Check if an exporter format is registered.

        Args:
            format_name: The format identifier.

        Returns:
            True if the format is registered, False otherwise.
        """
        return format_name in self._exporters


# Global registry instance
exporter_registry = ExporterRegistry()


# Convenience functions
def register_exporter(format_name: str) -> Callable[[Type[Exporter]], Type[Exporter]]:
    """Decorator to register an exporter format.

    Args:
        format_name: Unique identifier for this format.

    Returns:
        Decorator function.
    """
    return exporter_registry.register(format_name)


def create_exporter(
    format_name: str,
    config: ExportConfig,
) -> Optional[Exporter]:
    """Create an exporter instance from configuration.

    Args:
        format_name: The format identifier.
        config: Export configuration.

    Returns:
        Exporter instance, or None if format not found.
    """
    return exporter_registry.create(format_name, config)
