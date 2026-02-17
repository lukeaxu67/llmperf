"""DingTalk robot notification channel."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

from ..base import NotificationChannel, NotificationMessage, NotificationResult, NotificationPriority
from ..registry import register_channel

logger = logging.getLogger(__name__)


@register_channel("dingtalk")
class DingTalkChannel(NotificationChannel):
    """DingTalk robot notification channel.

    Sends notifications via DingTalk group robot webhook.

    Configuration:
        webhook: DingTalk webhook URL (required)
        secret: Optional secret for signing (recommended)
        at_mobiles: List of phone numbers to @mention
        at_user_ids: List of user IDs to @mention
        is_at_all: Whether to @all members (default: False)
    """

    channel_type = "dingtalk"

    def _validate_config(self) -> None:
        """Validate DingTalk configuration."""
        if not self.config.get("webhook"):
            raise ValueError("DingTalk channel requires 'webhook' configuration")

    @property
    def webhook(self) -> str:
        """Get the webhook URL."""
        return self.config["webhook"]

    @property
    def secret(self) -> Optional[str]:
        """Get the signing secret."""
        return self.config.get("secret")

    @property
    def at_mobiles(self) -> List[str]:
        """Get phone numbers to @mention."""
        return self.config.get("at_mobiles", [])

    @property
    def at_user_ids(self) -> List[str]:
        """Get user IDs to @mention."""
        return self.config.get("at_user_ids", [])

    @property
    def is_at_all(self) -> bool:
        """Get whether to @all members."""
        return self.config.get("is_at_all", False)

    def _sign_webhook(self) -> str:
        """Add signature to webhook URL.

        Returns:
            Signed webhook URL.
        """
        if not self.secret:
            return self.webhook

        timestamp = str(int(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

        separator = "&" if "?" in self.webhook else "?"
        return f"{self.webhook}{separator}timestamp={timestamp}&sign={sign}"

    def _build_markdown_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build markdown format message.

        Args:
            message: The notification message.

        Returns:
            DingTalk message payload.
        """
        # Build markdown content
        priority_emoji = {
            NotificationPriority.CRITICAL: "🔴 ",
            NotificationPriority.HIGH: "🟠 ",
            NotificationPriority.NORMAL: "🔵 ",
            NotificationPriority.LOW: "⚪ ",
        }
        emoji = priority_emoji.get(message.priority, "")

        markdown_content = f"## {emoji}{message.title}\n\n"
        markdown_content += f"> **Event:** {message.event_type}\n\n"
        markdown_content += f"> **Time:** {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown_content += "---\n\n"
        markdown_content += message.content.replace("\n", "\n\n")

        # Add @mentions
        at_text = ""
        if self.at_mobiles:
            at_text = " ".join(f"@{mobile}" for mobile in self.at_mobiles)
        if self.is_at_all:
            at_text = "@all " + at_text

        if at_text:
            markdown_content += f"\n\n{at_text}"

        return {
            "msgtype": "markdown",
            "markdown": {
                "title": message.title,
                "text": markdown_content,
            },
            "at": {
                "atMobiles": self.at_mobiles,
                "atUserIds": self.at_user_ids,
                "isAtAll": self.is_at_all,
            },
        }

    def _build_text_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build text format message.

        Args:
            message: The notification message.

        Returns:
            DingTalk message payload.
        """
        content = f"【{message.title}】\n\n{message.content}"

        # Add @mentions
        if self.at_mobiles:
            content += "\n\n" + " ".join(f"@{mobile}" for mobile in self.at_mobiles)

        return {
            "msgtype": "text",
            "text": {
                "content": content,
            },
            "at": {
                "atMobiles": self.at_mobiles,
                "atUserIds": self.at_user_ids,
                "isAtAll": self.is_at_all,
            },
        }

    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Send notification via DingTalk.

        Args:
            message: The notification message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        url = self._sign_webhook()
        payload = self._build_markdown_message(message)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                result = response.json()

                if result.get("errcode") == 0:
                    return NotificationResult(
                        success=True,
                        channel_type=self.channel_type,
                        message="DingTalk notification sent successfully",
                        response=result,
                    )
                else:
                    return NotificationResult(
                        success=False,
                        channel_type=self.channel_type,
                        message=f"DingTalk error: {result.get('errmsg', 'Unknown error')}",
                        response=result,
                    )

        except httpx.TimeoutException:
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message="DingTalk request timed out",
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message=f"DingTalk request failed: {e}",
            )

    async def health_check(self) -> bool:
        """Check if DingTalk webhook is valid.

        Returns:
            True if valid, False otherwise.
        """
        # Send a test message without content to verify webhook
        # DingTalk will return an error but we can check the response
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                url = self._sign_webhook()
                response = await client.post(
                    url,
                    json={"msgtype": "text", "text": {"content": ""}},
                )
                # Even with empty content, a valid webhook returns errcode
                result = response.json()
                # errcode 40035 means empty content, which means webhook is valid
                return result.get("errcode") in [0, 40035]
        except Exception:
            return False
