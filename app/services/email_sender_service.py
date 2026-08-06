"""
SMTP Email Sender Service.
Sends emails via Gmail SMTP using credentials from settings.
Uses asyncio.to_thread so the async event loop is never blocked.
"""
import asyncio
import hashlib
import logging
import re
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr
from typing import Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# Deliberately conservative: rejects display-name forms, header-injection
# attempts, and obvious malformations before anything reaches the SMTP server.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# Window in which an identical message to the same recipient is treated as a
# duplicate. The agent's reflect loop can re-run a specialist after a failure,
# and a send that succeeded before the failure would otherwise be repeated —
# delivering the same email to a real person twice.
_DEDUPE_WINDOW_SECONDS = 300.0


def _validate_address(address: str, field: str) -> Optional[str]:
    """Return an error string if `address` is not a safe single email address."""
    if not address:
        return f"{field} is required."
    if len(address) > 254:
        return f"{field} is too long to be a valid email address."
    # Header injection guard: CR/LF in a header value can forge extra headers.
    if any(ch in address for ch in "\r\n"):
        return f"{field} contains invalid characters."
    _, parsed = parseaddr(address)
    if parsed != address or not _EMAIL_RE.match(address):
        return f"'{address}' is not a valid email address."
    return None


class EmailSenderService:
    """Async wrapper around Python's built-in smtplib."""

    def __init__(self) -> None:
        # (fingerprint -> monotonic timestamp) of recently delivered messages.
        self._recent_sends: Dict[str, float] = {}

    def _fingerprint(self, to_email: str, subject: str, body: str) -> str:
        digest = hashlib.sha256(
            f"{to_email}\x00{subject}\x00{body}".encode("utf-8", errors="replace")
        ).hexdigest()
        return digest

    def _check_duplicate(self, fingerprint: str) -> Tuple[bool, float]:
        """Return (is_duplicate, age_seconds) and prune expired entries."""
        now = time.monotonic()
        for key, sent_at in list(self._recent_sends.items()):
            if now - sent_at > _DEDUPE_WINDOW_SECONDS:
                del self._recent_sends[key]

        sent_at = self._recent_sends.get(fingerprint)
        if sent_at is not None:
            return True, now - sent_at
        return False, 0.0

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

        # Validate every address before contacting the SMTP server. The
        # recipient originates from LLM output, so it is untrusted input.
        to_email = (to_email or "").strip()
        error = _validate_address(to_email, "Recipient address")
        if error:
            logger.warning("Rejected email send: %s", error)
            return {"success": False, "error": error}

        cc = [addr.strip() for addr in (cc or []) if addr and addr.strip()]
        for cc_address in cc:
            error = _validate_address(cc_address, "CC address")
            if error:
                logger.warning("Rejected email send: %s", error)
                return {"success": False, "error": error}

        subject = (subject or "").replace("\r", " ").replace("\n", " ").strip()
        if not subject:
            return {"success": False, "error": "Subject is required."}
        if not (body or "").strip():
            return {"success": False, "error": "Body is required."}

        # Suppress an identical resend within the dedupe window (agent retry).
        fingerprint = self._fingerprint(to_email, subject, body)
        is_duplicate, age = self._check_duplicate(fingerprint)
        if is_duplicate:
            logger.warning(
                "Suppressed duplicate email to=%s subject=%s (identical message sent %.0fs ago)",
                to_email, subject, age,
            )
            return {
                "success": True,
                "to": to_email,
                "subject": subject,
                "duplicate_suppressed": True,
                "message": "This exact email was already sent moments ago; not sending again.",
            }

        try:
            await asyncio.to_thread(
                self._send_sync, to_email, subject, body, cc
            )
            self._recent_sends[fingerprint] = time.monotonic()
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
            logger.error("Failed to send email to %s: %s", to_email, exc, exc_info=True)
            return {
                "success": False,
                "error": "The email could not be sent. Please try again.",
            }


email_sender_service = EmailSenderService()
