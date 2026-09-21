"""
Sensitive-value detection tests (audit finding M5).

Profile facts are injected verbatim into every agent's system prompt, so a
credential stored here reaches an LLM.
"""
import pytest

from app.memory.short_term_memory import short_term_memory

is_sensitive = short_term_memory._is_sensitive


@pytest.mark.parametrize("key,value", [
    ("password", "hunter2"),
    ("user-password", "hunter2"),
    ("My Password", "hunter2"),
    ("api_key", "abc"),
    ("openai_secret", "x"),
    ("bank_account", "123"),
    ("aadhaar", "1234"),
    ("session_key", "k"),
])
def test_blocks_sensitive_keys_regardless_of_separator(key, value):
    assert is_sensitive(key, value) is True


@pytest.mark.parametrize("value", [
    "sk-abcdefghijklmnopqrstuvwxyz",                       # prefixed API key
    "ghp_1234567890abcdefghijklmnopqrstuvwx",              # opaque token
    "4111111111111111",                                    # card-like PAN
    "123-45-6789",                                         # SSN
    "-----BEGIN RSA PRIVATE KEY-----",                     # PEM
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc",            # JWT
])
def test_blocks_credential_shaped_values_under_innocuous_key(value):
    # The original heuristic could be bypassed by choosing a harmless key name.
    assert is_sensitive("info", value) is True


@pytest.mark.parametrize("key,value", [
    ("preferred_tone", "concise"),
    ("name", "Vansh Pratap Singh"),
    ("role", "Backend engineer focused on distributed systems"),
    ("timezone", "Asia/Kolkata"),
    ("portfolio", "https://example.com/portfolio-page-with-a-long-path"),
    ("contact_email", "someone@example.com"),
    ("goal", "Find a machine learning internship this summer"),
])
def test_allows_ordinary_profile_facts(key, value):
    assert is_sensitive(key, value) is False
