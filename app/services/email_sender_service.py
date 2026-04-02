"""
SMTP Email Sender Service.
Sends emails via Gmail SMTP using credentials from settings.
Uses asyncio.to_thread so the async event loop is never blocked.
"""
import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class EmailSenderService:
    """Async wrapper around Python's built-in smtplib."""

    def _send_sync(
        self,
        to_email: str,
        subject: str,
        body: str,
        cc: Optional[List[str]],
    ) -> None:
        """Blocking SMTP call — always runs inside asyncio.to_thread."""
        msg = MIMEMultipart("alternative")
        msg["From"] = settings.smtp_email
        msg["To"] = to_email
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = ", ".join(cc)

        msg.attach(MIMEText(body, "plain"))

        recipients = [to_email] + (cc or [])

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.smtp_email, settings.smtp_password)
            server.sendmail(settings.smtp_email, recipients, msg.as_string())

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
    ) -> Dict:
        """
        Send an email asynchronously via SMTP.

        Returns:
            {"success": True,  "to": to_email, "subject": subject}
            {"success": False, "error": "<reason>"}
        """
        if not settings.smtp_email or not settings.smtp_password:
            return {
                "success": False,
                "error": (
                    "SMTP credentials not configured. "
                    "Add SMTP_EMAIL and SMTP_PASSWORD to your .env file."
                ),
            }

        try:
            await asyncio.to_thread(
                self._send_sync, to_email, subject, body, cc
            )
            logger.info("Email sent to=%s subject=%s", to_email, subject)
            return {"success": True, "to": to_email, "subject": subject}

        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP auth failed for sender=%s", settings.smtp_email)
            return {
                "success": False,
                "error": (
                    "Gmail authentication failed. "
                    "Make sure SMTP_PASSWORD is an App Password "
                    "(not your normal Gmail password)."
                ),
            }
        except smtplib.SMTPRecipientsRefused:
            logger.error("Recipient refused: %s", to_email)
            return {
                "success": False,
                "error": f"The address '{to_email}' was rejected by the server.",
            }
        except Exception as exc:
            logger.error("Failed to send email to %s: %s", to_email, exc)
            return {"success": False, "error": str(exc)}


email_sender_service = EmailSenderService()
