"""Thin, backend-agnostic LLM client.

The whole factory talks to models through :class:`LLMClient`. This keeps three
concerns swappable and testable:

    * the *target* model (runs the skill under optimization),
    * the *judge* model (scores rollouts),
    * the *optimizer* model (proposes skill edits).

Tests inject :class:`FakeLLMClient`; production uses :class:`AnthropicClient`.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every backend implements."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return the model's completion for ``prompt`` given an optional system prompt."""
        ...


class FakeLLMClient:
    """Deterministic client for tests.

    Pass either a list of canned responses (returned in order, last one repeats)
    or a callable ``responder(prompt, system) -> str``. Every call is recorded on
    :attr:`calls` for assertions.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        responder: Callable[[str, str | None], str] | None = None,
    ):
        if responses is None and responder is None:
            raise ValueError("FakeLLMClient needs either responses or a responder")
        self._responses = list(responses or [])
        self._responder = responder
        self._index = 0
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "system": system, "temperature": temperature}
        )
        if self._responder is not None:
            return self._responder(prompt, system)
        if not self._responses:
            return ""
        idx = min(self._index, len(self._responses) - 1)
        self._index += 1
        return self._responses[idx]


class AnthropicClient:
    """Claude backend (default). Requires ``pip install skill-factory[anthropic]``.

    The ``anthropic`` package is imported lazily so the core library — and the
    entire test suite — has no hard dependency on it.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package. "
                "Install with: pip install 'skill-factory[anthropic]'"
            ) from exc
        from anthropic import Anthropic

        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        import anthropic

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, object] = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system
                response = self._client.messages.create(**kwargs)
                return _first_text_block(response)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_base_delay * (2**attempt))
        raise RuntimeError(
            f"Anthropic request failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc


def _first_text_block(response: object) -> str:
    """Extract concatenated text from an Anthropic Messages response."""
    content = getattr(response, "content", None) or []
    parts = [getattr(block, "text", "") for block in content if getattr(block, "type", "") == "text"]
    return "".join(parts).strip()
