"""Tests for the notification system."""

from __future__ import annotations

import pytest
from datetime import datetime

from llmperf.notifications import (
    NotificationChannel,
    NotificationMessage,
    NotificationResult,
    NotificationManager,
    NotificationConfig,
    NotificationPriority,
    register_channel,
    channel_registry,
)
from llmperf.notifications.channels.log import LogChannel


class TestNotificationMessage:
    """Tests for NotificationMessage."""

    def test_create_basic_message(self):
        """Test creating a basic notification message."""
        msg = NotificationMessage(
            title="Test Title",
            content="Test content",
            event_type="test_event",
        )

        assert msg.title == "Test Title"
        assert msg.content == "Test content"
        assert msg.event_type == "test_event"
        assert msg.priority == NotificationPriority.NORMAL

    def test_create_task_started(self):
        """Test creating task started message."""
        msg = NotificationMessage.create_task_started(
            run_id="test-123",
            task_name="Test Task",
            executor_count=3,
            dataset_size=100,
        )

        assert "Test Task" in msg.title
        assert msg.run_id == "test-123"
        assert msg.event_type == "task_started"
        assert msg.metadata["executor_count"] == 3
        assert msg.metadata["dataset_size"] == 100

    def test_create_task_completed(self):
        """Test creating task completed message."""
        msg = NotificationMessage.create_task_completed(
            run_id="test-456",
            task_name="Completed Task",
            total_requests=100,
            success_count=95,
            error_count=5,
            total_cost=1.23,
            currency="USD",
            elapsed_seconds=60.0,
        )

        assert "Completed" in msg.title
        assert msg.run_id == "test-456"
        assert msg.event_type == "task_completed"
        assert msg.metadata["success_count"] == 95

    def test_create_task_failed(self):
        """Test creating task failed message."""
        msg = NotificationMessage.create_task_failed(
            run_id="test-789",
            task_name="Failed Task",
            error_message="Connection timeout",
            error_details={"code": "ETIMEDOUT"},
        )

        assert "Failed" in msg.title
        assert msg.priority == NotificationPriority.HIGH
        assert msg.metadata["error_message"] == "Connection timeout"

    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = NotificationMessage(
            title="Test",
            content="Content",
            event_type="test",
            priority=NotificationPriority.HIGH,
            run_id="run-123",
        )

        d = msg.to_dict()

        assert d["title"] == "Test"
        assert d["priority"] == "high"
        assert d["run_id"] == "run-123"


class TestLogChannel:
    """Tests for LogChannel."""

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending a message via log channel."""
        channel = LogChannel({"level": "INFO"})

        msg = NotificationMessage(
            title="Test",
            content="Test content",
            event_type="test",
        )

        result = await channel.send(msg)

        assert result.success
        assert result.channel_type == "log"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check always returns True."""
        channel = LogChannel({})
        assert await channel.health_check()


class TestChannelRegistry:
    """Tests for channel registry."""

    def test_register_channel(self):
        """Test registering a custom channel."""
        @register_channel("test_custom")
        class CustomChannel(NotificationChannel):
            channel_type = "test_custom"

            async def send(self, message):
                return NotificationResult(
                    success=True,
                    channel_type=self.channel_type,
                )

        assert "test_custom" in channel_registry.list_channels()

    def test_create_channel(self):
        """Test creating a channel from registry."""
        channel = channel_registry.create("log", {})
        assert channel is not None
        assert channel.channel_type == "log"


class TestNotificationManager:
    """Tests for NotificationManager."""

    def test_create_manager(self):
        """Test creating a notification manager."""
        config = NotificationConfig(
            enabled=True,
            channels=[
                {"type": "log", "enabled": True, "config": {}},
            ],
        )

        manager = NotificationManager(config)
        assert "log" in manager.list_available_channels()

    @pytest.mark.asyncio
    async def test_notify_task_started(self):
        """Test sending task started notification."""
        config = NotificationConfig(
            enabled=True,
            channels=[
                {"type": "log", "enabled": True, "config": {}},
            ],
            default_channels=["log"],
        )

        manager = NotificationManager(config)
        results = await manager.notify_task_started(
            run_id="test-001",
            task_name="Test Task",
            executor_count=2,
            dataset_size=50,
        )

        # Should have results from log channel
        assert "log" in results

    @pytest.mark.asyncio
    async def test_disabled_system(self):
        """Test that disabled system doesn't send notifications."""
        config = NotificationConfig(
            enabled=False,
            channels=[
                {"type": "log", "enabled": True, "config": {}},
            ],
        )

        manager = NotificationManager(config)

        msg = NotificationMessage(
            title="Test",
            content="Content",
            event_type="test",
        )

        results = await manager.notify(msg)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check for all channels."""
        config = NotificationConfig(
            channels=[
                {"type": "log", "enabled": True, "config": {}},
            ],
        )

        manager = NotificationManager(config)
        health = await manager.health_check()

        assert "log" in health
        assert health["log"] is True
