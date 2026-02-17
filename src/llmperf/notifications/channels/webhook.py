"""HTTP Webhook notification channel."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional

import httpx

from ..base import NotificationChannel, NotificationMessage, NotificationResult
from ..registry import register_channel

logger = logging.getLogger(__name__)


@register_channel("webhook")
class WebhookChannel(NotificationChannel):
    """HTTP Webhook notification channel.

    Sends notifications via HTTP POST to a configured URL.

    Configuration:
        url: Webhook URL (required)
        method: HTTP method (default: POST)
        headers: Additional headers to include
        timeout_seconds: Request timeout (default: 30)
        secret: Optional secret for HMAC signature
        secret_header: Header name for signature (default: X-Signature)
    """

    channel_type = "webhook"

    def _validate_config(self) -> None:
        """Validate webhook configuration."""
        if not self.config.get("url"):
            raise ValueError("Webhook channel requires 'url' configuration")

    @property
    def url(self) -> str:
        """Get the webhook URL."""
        return self.config["url"]

    @property
    def method(self) -> str:
        """Get the HTTP method."""
        return self.config.get("method", "POST").upper()

    @property
    def headers(self) -> Dict[str, str]:
        """Get additional headers."""
        return self.config.get("headers", {})

    @property
    def timeout_seconds(self) -> float:
        """Get request timeout."""
        return float(self.config.get("timeout_seconds", 30))

    @property
    def secret(self) -> Optional[str]:
        """Get the signing secret."""
        return self.config.get("secret")

    @property
    def secret_header(self) -> str:
        """Get the signature header name."""
        return self.config.get("secret_header", "X-Signature")

    def _compute_signature(self, payload: bytes) -> str:
        """Compute HMAC signature for the payload.

        Args:
            payload: Raw payload bytes.

        Returns:
            Hexadecimal signature string.
        """
        if not self.secret:
            return ""
        return hmac.new(
            self.secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

    def _build_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Build the webhook payload.

        Override this method to customize the payload format.

        Args:
            message: The notification message.

        Returns:
            Payload dictionary.
        """
        return {
            "event": message.event_type,
            "title": message.title,
            "content": message.content,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "run_id": message.run_id,
            "executor_id": message.executor_id,
            "metadata": message.metadata,
        }

    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Send notification via webhook.

        Args:
            message: The notification message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        payload = self._build_payload(message)
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            **self.headers,
        }

        # Add signature if secret is configured
        if self.secret:
            signature = self._compute_signature(payload_bytes)
            headers[self.secret_header] = signature

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.request(
                    method=self.method,
                    url=self.url,
                    content=payload_bytes,
                    headers=headers,
                )

                if response.status_code >= 200 and response.status_code < 300:
                    return NotificationResult(
                        success=True,
                        channel_type=self.channel_type,
                        message=f"Webhook delivered successfully (status={response.status_code})",
                        response={
                            "status_code": response.status_code,
                            "body": response.text[:500] if response.text else "",
                        },
                    )
                else:
                    return NotificationResult(
                        success=False,
                        channel_type=self.channel_type,
                        message=f"Webhook returned status {response.status_code}",
                        response={
                            "status_code": response.status_code,
                            "body": response.text[:500] if response.text else "",
                        },
                    )

        except httpx.TimeoutException:
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message=f"Webhook request timed out after {self.timeout_seconds}s",
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message=f"Webhook request failed: {e}",
            )

    async def health_check(self) -> bool:
        """Check if the webhook endpoint is reachable.

        Returns:
            True if reachable, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Use HEAD request for health check
                response = await client.head(self.url)
                # Consider any response as healthy (even 404 means endpoint exists)
                return True
        except Exception:
            return False
