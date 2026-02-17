"""SMTP Email notification channel."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from ..base import NotificationChannel, NotificationMessage, NotificationResult, NotificationPriority
from ..registry import register_channel

logger = logging.getLogger(__name__)


@register_channel("email")
class EmailChannel(NotificationChannel):
    """SMTP Email notification channel.

    Sends notifications via email using SMTP.

    Configuration:
        smtp_host: SMTP server hostname (required)
        smtp_port: SMTP server port (default: 587)
        smtp_user: SMTP username (required)
        smtp_pass: SMTP password (required)
        use_tls: Whether to use TLS (default: True)
        from_address: Sender email address (default: smtp_user)
        to_addresses: List of recipient email addresses (required)
        subject_prefix: Prefix for email subjects (default: "[LLMPerf]")
    """

    channel_type = "email"

    def _validate_config(self) -> None:
        """Validate email configuration."""
        required_fields = ["smtp_host", "smtp_user", "smtp_pass", "to_addresses"]
        for field in required_fields:
            if not self.config.get(field):
                raise ValueError(f"Email channel requires '{field}' configuration")

        if not isinstance(self.config.get("to_addresses"), list):
            raise ValueError("Email channel 'to_addresses' must be a list")

    @property
    def smtp_host(self) -> str:
        """Get SMTP host."""
        return self.config["smtp_host"]

    @property
    def smtp_port(self) -> int:
        """Get SMTP port."""
        return int(self.config.get("smtp_port", 587))

    @property
    def smtp_user(self) -> str:
        """Get SMTP username."""
        return self.config["smtp_user"]

    @property
    def smtp_pass(self) -> str:
        """Get SMTP password."""
        return self.config["smtp_pass"]

    @property
    def use_tls(self) -> bool:
        """Get whether TLS is enabled."""
        return bool(self.config.get("use_tls", True))

    @property
    def from_address(self) -> str:
        """Get sender email address."""
        return self.config.get("from_address", self.smtp_user)

    @property
    def to_addresses(self) -> List[str]:
        """Get recipient email addresses."""
        return self.config["to_addresses"]

    @property
    def subject_prefix(self) -> str:
        """Get subject prefix."""
        return self.config.get("subject_prefix", "[LLMPerf]")

    def _build_subject(self, message: NotificationMessage) -> str:
        """Build email subject line.

        Args:
            message: The notification message.

        Returns:
            Subject line string.
        """
        priority_prefix = {
            NotificationPriority.CRITICAL: "[CRITICAL] ",
            NotificationPriority.HIGH: "[HIGH] ",
            NotificationPriority.NORMAL: "",
            NotificationPriority.LOW: "",
        }.get(message.priority, "")

        return f"{self.subject_prefix} {priority_prefix}{message.title}".strip()

    def _build_body(self, message: NotificationMessage) -> str:
        """Build email body.

        Args:
            message: The notification message.

        Returns:
            Email body string.
        """
        lines = [
            f"Event: {message.event_type}",
            f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            message.content,
        ]

        if message.run_id:
            lines.append(f"\nRun ID: {message.run_id}")

        if message.metadata:
            lines.append("\n--- Additional Information ---")
            for key, value in message.metadata.items():
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def _build_html_body(self, message: NotificationMessage) -> str:
        """Build HTML email body.

        Args:
            message: The notification message.

        Returns:
            HTML body string.
        """
        priority_colors = {
            NotificationPriority.CRITICAL: "#dc3545",
            NotificationPriority.HIGH: "#fd7e14",
            NotificationPriority.NORMAL: "#0d6efd",
            NotificationPriority.LOW: "#6c757d",
        }

        color = priority_colors.get(message.priority, "#0d6efd")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 10px; }}
                .metadata {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em; }}
                .footer {{ margin-top: 20px; font-size: 0.8em; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2 style="margin: 0;">{message.title}</h2>
                </div>
                <div class="content">
                    <p><strong>Event:</strong> {message.event_type}</p>
                    <p><strong>Time:</strong> {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr>
                    <pre style="white-space: pre-wrap;">{message.content}</pre>
                </div>
        """

        if message.metadata:
            html += """
                <div class="metadata">
                    <strong>Additional Information:</strong>
                    <ul>
            """
            for key, value in message.metadata.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += """
                    </ul>
                </div>
            """

        html += """
                <div class="footer">
                    <p>This is an automated notification from LLMPerf.</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Send notification via email.

        Args:
            message: The notification message to send.

        Returns:
            NotificationResult indicating success or failure.
        """
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)
            msg["Subject"] = self._build_subject(message)

            # Add body parts
            text_body = self._build_body(message)
            html_body = self._build_html_body(message)

            msg.attach(MIMEText(text_body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            # Send email
            if self.use_tls:
                smtp = smtplib.SMTP(self.smtp_host, self.smtp_port)
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
            else:
                smtp = smtplib.SMTP(self.smtp_host, self.smtp_port)

            try:
                smtp.login(self.smtp_user, self.smtp_pass)
                smtp.sendmail(
                    self.from_address,
                    self.to_addresses,
                    msg.as_string(),
                )
            finally:
                smtp.quit()

            logger.info(
                "Email sent successfully to %s",
                ", ".join(self.to_addresses),
            )

            return NotificationResult(
                success=True,
                channel_type=self.channel_type,
                message=f"Email sent to {len(self.to_addresses)} recipient(s)",
                response={
                    "recipients": self.to_addresses,
                },
            )

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {e}"
            logger.error(error_msg)
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message=error_msg,
            )
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error: {e}"
            logger.error(error_msg)
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message=error_msg,
            )
        except Exception as e:
            error_msg = f"Failed to send email: {e}"
            logger.error(error_msg)
            return NotificationResult(
                success=False,
                channel_type=self.channel_type,
                message=error_msg,
            )

    async def health_check(self) -> bool:
        """Check if SMTP server is reachable.

        Returns:
            True if reachable, False otherwise.
        """
        try:
            if self.use_tls:
                smtp = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
            else:
                smtp = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)

            smtp.quit()
            return True
        except Exception as e:
            logger.warning("SMTP health check failed: %s", e)
            return False
