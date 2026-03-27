"""Enterprise WeChat robot notification channel."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from ..base import NotificationChannel, NotificationMessage, NotificationResult, NotificationPriority
from ..registry import register_channel

logger = logging.getLogger(__name__)


@register_channel("wechat")
class WeChatChannel(NotificationChannel):
    """Enterprise WeChat robot notification channel.

    Sends notifications via WeChat Work group robot webhook.

    Configuration:
        webhook: WeChat webhook URL (required)
        mentioned_list: List of user IDs to mention (@all for everyone)
        mentioned_mobile_list: List of phone numbers to mention
    """

    channel_type = "wechat"

    def _validate_config(self) -> None:
        """Validate WeChat configuration."""
        if not self.config.get("webhook"):
            raise ValueError("WeChat channel requires 'webhook' configuration")

    @property
    def webhook(self) -> str:
        """Get the webhook URL."""
        return self.config["webhook"]

    @property
    def mentioned_list(self) -> List[str]:
        """Get user IDs to mention."""
        return self.config.get("mentioned_list", [])

    @property
    def mentioned_mobile_list(self) -> List[str]:
        """Get phone numbers to mention."""
        return self.config.get("mentioned_mobile_list", [])

    def _build_markdown_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build markdown format message.

        Args:
            message: The notification message.

        Returns:
            WeChat message payload.
        """
        priority_color = {
            NotificationPriority.CRITICAL: "<font color=\"warning\">",
            NotificationPriority.HIGH: "<font color=\"comment\">",
            NotificationPriority.NORMAL: "",
            NotificationPriority.LOW: "",
        }
        color_start = priority_color.get(message.priority, "")
        color_end = "</font>" if color_start else ""

        content = f"## {color_start}{message.title}{color_end}\n"
        content += f"> **Event:** {message.event_type}\n"
        content += f"> **Time:** {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "\n---\n\n"
        content += message.content

        return {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
                "mentioned_list": self.mentioned_list,
                "mentioned_mobile_list": self.mentioned_mobile_list,
            },
        }

    def _build_text_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build text format message.

        Args:
            message: The notification message.

        Returns:
            WeChat message payload.
        """
        content = f"【{message.title}】\n\n{message.content}"

        return {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": self.mentioned_list,
                "mentioned_mobile_list": self.mentioned_mobile_list,
            },
        }

    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Send notification via WeChat.

        Args:
            message: The notification message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        payload = self._build_markdown_message(message)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    self.webhook,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                result = response.json()

                if result.get("errcode") == 0:
                    return NotificationResult(
                        success=True,
                        channel_type=self.channel_type,
                        message="WeChat notification sent successfully",
                        response=result,
                    )
                else:
                    return NotificationResult(
                        success=False,
                        channel_type=self.channel_type,
                        message=f"WeChat error: {result.get('errmsg', 'Unknown error')}",
                        response=result,
                    )

        except httpx.TimeoutException:
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message="WeChat request timed out",
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message=f"WeChat request failed: {e}",
            )

    async def health_check(self) -> bool:
        """Check if WeChat webhook is valid.

        Returns:
            True if valid, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Send empty message to test webhook validity
                response = await client.post(
                    self.webhook,
                    json={"msgtype": "text", "text": {"content": ""}},
                )
                result = response.json()
                # errcode 40008 means empty content, which means webhook is valid
                return result.get("errcode") in [0, 40008]
        except Exception:
            return False
