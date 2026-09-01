"""Unit tests for LLM client helpers (SDK-compat + retry classification)."""

from __future__ import annotations

from skill_factory.llm.client import _create_supports, _is_retryable_status


class _Status:
    def __init__(self, code):
        self.status_code = code


def test_is_retryable_status():
    assert _is_retryable_status(_Status(429)) is True
    assert _is_retryable_status(_Status(500)) is True
    assert _is_retryable_status(_Status(503)) is True
    assert _is_retryable_status(_Status(400)) is False  # low balance / bad request
    assert _is_retryable_status(_Status(401)) is False
    assert _is_retryable_status(_Status(404)) is False
    assert _is_retryable_status(object()) is True  # unknown -> assume retryable


def test_create_supports_detects_param():
    class Msgs:
        def create(self, model, max_tokens, temperature=0):
            ...

    class Client:
        messages = Msgs()

    assert _create_supports(Client(), "temperature") is True
    assert _create_supports(Client(), "top_k") is False


def test_create_supports_without_temperature():
    class Msgs:
        # Mirrors anthropic>=1.x: no temperature, no **kwargs.
        def create(self, model, max_tokens, system=None):
            ...

    class Client:
        messages = Msgs()

    assert _create_supports(Client(), "temperature") is False


def test_create_supports_with_var_kwargs():
    class Msgs:
        def create(self, model, **kwargs):
            ...

    class Client:
        messages = Msgs()

    assert _create_supports(Client(), "temperature") is True


def test_create_supports_uninspectable_defaults_true():
    class Client:
        messages = object()  # no create attribute

    assert _create_supports(Client(), "temperature") is True
