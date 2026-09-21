"""
Email send safety tests (audit findings C4 and H4).

send_email is reachable from LLM tool output, so the recipient address is
untrusted input and a retry must never re-deliver a message that already sent.
"""
import pytest

from app.services.email_sender_service import (
    _RATE_LIMIT_MAX_SENDS,
    EmailSenderService,
    _validate_address,
)


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


# ── Configuration (checked at first use, not at startup) ─────────────────────

@pytest.mark.parametrize("email, password", [
    ("", "app-password"),
    ("me@example.com", ""),
    ("your_email@gmail.com", "app-password"),        # unedited .env.example
    ("me@example.com", "your_app_password_here"),    # unedited .env.example
])
@pytest.mark.asyncio
async def test_unconfigured_smtp_refuses_before_connecting(monkeypatch, email, password):
    """
    A placeholder is worse than a blank, because pydantic accepts it and the
    failure surfaces at Gmail as an auth error indistinguishable from a wrong
    password. Both are caught here, and the message names the app password.
    """
    service = EmailSenderService()
    called = False

    def _fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(service, "_send_sync", _fail_if_called)
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_email", email)
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_password", password)

    result = await service.send_email(
        to_email="hr@example.com", subject="Hi", body="Body", user_id="owner"
    )

    assert result["success"] is False
    assert "SMTP_EMAIL" in result["error"] and "SMTP_PASSWORD" in result["error"]
    assert "App Password" in result["error"], "Gmail needs an app password — say so"
    assert not called, "SMTP must not be contacted without credentials"


# ── Per-user send cap (audit item C4) ────────────────────────────────────────

def _configured(monkeypatch, service):
    """A service with working credentials and a counting SMTP stub."""
    sent = []
    monkeypatch.setattr(service, "_send_sync", lambda *a, **kw: sent.append(a))
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_email", "me@example.com")
    monkeypatch.setattr("app.services.email_sender_service.settings.smtp_password", "app-password")
    return sent


@pytest.mark.asyncio
async def test_a_user_cannot_exceed_the_hourly_send_cap(monkeypatch):
    """
    Confirmation gates every send, so this is the second line rather than the
    first. It bounds what a wrong approval — or a user clicking yes at a
    prompt loop — can actually deliver.
    """
    service = EmailSenderService()
    sent = _configured(monkeypatch, service)

    for i in range(_RATE_LIMIT_MAX_SENDS):
        result = await service.send_email(
            to_email="hr@example.com", subject=f"Application {i}",
            body=f"Body {i}", user_id="owner",
        )
        assert result["success"] is True, f"send {i} should have been allowed"

    blocked = await service.send_email(
        to_email="hr@example.com", subject="One too many",
        body="Body", user_id="owner",
    )

    assert blocked["success"] is False
    assert "Send limit reached" in blocked["error"]
    assert len(sent) == _RATE_LIMIT_MAX_SENDS, "the capped send must not reach SMTP"


@pytest.mark.asyncio
async def test_the_cap_is_per_user_not_global(monkeypatch):
    service = EmailSenderService()
    sent = _configured(monkeypatch, service)

    for i in range(_RATE_LIMIT_MAX_SENDS):
        await service.send_email(
            to_email="hr@example.com", subject=f"A{i}", body=f"B{i}", user_id="owner",
        )

    # Distinct content: the dedupe fingerprint is content-only and not
    # scoped by user, so reusing a subject here would be suppressed as a
    # duplicate and prove nothing about the cap.
    other = await service.send_email(
        to_email="hr@example.com", subject="Different", body="Different",
        user_id="someone-else",
    )

    assert other["success"] is True
    assert len(sent) == _RATE_LIMIT_MAX_SENDS + 1


@pytest.mark.asyncio
async def test_a_suppressed_duplicate_does_not_spend_the_budget(monkeypatch):
    """
    A retry that delivered nothing must not consume a slot — otherwise a
    reflect loop could exhaust an hour's allowance without a single message
    leaving the building.
    """
    service = EmailSenderService()
    sent = _configured(monkeypatch, service)
    payload = dict(to_email="hr@example.com", subject="Application", body="Please consider me.")

    await service.send_email(user_id="owner", **payload)
    for _ in range(_RATE_LIMIT_MAX_SENDS * 2):
        await service.send_email(user_id="owner", **payload)

    # One real send, the rest suppressed — so the budget is almost untouched.
    assert len(sent) == 1
    assert service._check_rate_limit("owner") is None


# ── The guard itself ─────────────────────────────────────────────────────────

def test_the_suite_cannot_open_a_real_smtp_connection():
    """
    The offline guarantee, asserted rather than assumed.

    Every other email test stubs the send. Each stub is one a test has to
    remember, and one test forgot — it passed a failing callable in a spec the
    gateway does not use, so the real `send_email` ran and, once credentials
    were configured, delivered live mail. This asserts the backstop underneath
    all of those stubs is actually installed.
    """
    import smtplib

    from tests.conftest import SMTPContactedInTests

    with pytest.raises(SMTPContactedInTests):
        smtplib.SMTP("smtp.gmail.com", 587)

    with pytest.raises(SMTPContactedInTests):
        smtplib.SMTP_SSL("smtp.gmail.com", 465)


@pytest.mark.asyncio
async def test_a_fully_configured_send_still_cannot_reach_a_mail_server(monkeypatch):
    """
    The end-to-end shape of the bug that motivated the guard: a service with
    working credentials and *no* stub on the send path.

    It must not deliver. The guard is a `BaseException` precisely so that
    `send_email`'s closing `except Exception` cannot turn this into a quiet
    `{"success": False}` — a contained failure nobody notices is how the
    original defect survived a full test run.
    """
    from tests.conftest import SMTPContactedInTests

    service = EmailSenderService()
    monkeypatch.setattr(
        "app.services.email_sender_service.settings.smtp_email", "me@example.com")
    monkeypatch.setattr(
        "app.services.email_sender_service.settings.smtp_password", "app-password")

    with pytest.raises(SMTPContactedInTests):
        await service.send_email(
            to_email="someone@example.com", subject="s", body="b", user_id="owner",
        )
