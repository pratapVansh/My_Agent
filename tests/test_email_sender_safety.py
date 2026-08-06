"""
Email send safety tests (audit findings C4 and H4).

send_email is reachable from LLM tool output, so the recipient address is
untrusted input and a retry must never re-deliver a message that already sent.
"""
import pytest

from app.services.email_sender_service import EmailSenderService, _validate_address


@pytest.mark.parametrize("address", [
    "",
    "not-an-email",
    "missing@tld",
    "two@@at.com",
    "spaced address@example.com",
    "victim@example.com\r\nBcc: attacker@evil.com",   # header injection
    "victim@example.com\nSubject: forged",
    "Display Name <someone@example.com>",             # not a bare address
    "a" * 250 + "@example.com",                       # over length
])
def test_rejects_invalid_recipients(address):
    assert _validate_address(address, "Recipient address") is not None


@pytest.mark.parametrize("address", [
    "user@example.com",
    "first.last+tag@sub.example.co.uk",
    "digits123@example.io",
])
def test_accepts_valid_recipients(address):
    assert _validate_address(address, "Recipient address") is None


@pytest.mark.asyncio
async def test_send_rejects_invalid_recipient_without_smtp_call(monkeypatch):
    service = EmailSenderService()
    called = False

    def _fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(service, "_send_sync", _fail_if_called)
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_email", "me@example.com")
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_password", "app-password")

    result = await service.send_email(
        to_email="not-an-email", subject="Hi", body="Body"
    )

    assert result["success"] is False
    assert not called, "SMTP must not be contacted for an invalid recipient"


@pytest.mark.asyncio
async def test_identical_resend_is_suppressed(monkeypatch):
    """A reflect-loop retry must not deliver the same email twice."""
    service = EmailSenderService()
    send_count = 0

    def _count(*args, **kwargs):
        nonlocal send_count
        send_count += 1

    monkeypatch.setattr(service, "_send_sync", _count)
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_email", "me@example.com")
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_password", "app-password")

    payload = dict(to_email="hr@example.com", subject="Application", body="Please consider me.")

    first = await service.send_email(**payload)
    second = await service.send_email(**payload)

    assert first["success"] is True
    assert second["success"] is True
    assert second.get("duplicate_suppressed") is True
    assert send_count == 1, "the duplicate must not reach SMTP"


@pytest.mark.asyncio
async def test_different_content_still_sends(monkeypatch):
    service = EmailSenderService()
    send_count = 0

    def _count(*args, **kwargs):
        nonlocal send_count
        send_count += 1

    monkeypatch.setattr(service, "_send_sync", _count)
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_email", "me@example.com")
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_password", "app-password")

    await service.send_email(to_email="hr@example.com", subject="A", body="First")
    await service.send_email(to_email="hr@example.com", subject="B", body="Second")

    assert send_count == 2
