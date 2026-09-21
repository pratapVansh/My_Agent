"""
Groq parameter-building tests (audit finding H3).

temperature=0.0 is falsy, so `temperature or default` silently discarded every
request for deterministic output — including the voice intent router's.
"""
from app.services.groq_service import GroqService

MESSAGES = [{"role": "user", "content": "hello"}]


def test_zero_temperature_is_preserved():
    service = GroqService()
    service.temperature = 0.7

    params = service._build_params(MESSAGES, temperature=0.0, max_tokens=None, model=None)

    assert params["temperature"] == 0.0, "explicit 0.0 must not fall back to the default"


def test_omitted_temperature_uses_default():
    service = GroqService()
    service.temperature = 0.7

    params = service._build_params(MESSAGES, temperature=None, max_tokens=None, model=None)

    assert params["temperature"] == 0.7


def test_zero_max_tokens_is_preserved():
    service = GroqService()
    service.max_tokens = 2048

    params = service._build_params(MESSAGES, temperature=None, max_tokens=0, model=None)

    assert params["max_tokens"] == 0


def test_model_override_and_passthrough_kwargs():
    service = GroqService()

    params = service._build_params(
        MESSAGES,
        temperature=0.2,
        max_tokens=100,
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
    )

    assert params["model"] == "llama3-8b-8192"
    assert params["response_format"] == {"type": "json_object"}
    assert params["messages"] is MESSAGES
