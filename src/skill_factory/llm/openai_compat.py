"""OpenAI-compatible LLM client.

Covers any provider that speaks the OpenAI Chat Completions API: OpenAI, Azure
OpenAI, OpenRouter, Together, Groq, vLLM, LM Studio, and Ollama (``/v1``). The
only differences are ``base_url`` and ``api_key``.
"""

from __future__ import annotations

import time


class OpenAICompatibleClient:
    """Talk to any OpenAI-compatible Chat Completions endpoint.

    Requires ``pip install skill-factory[openai]``. The ``openai`` package is
    imported lazily so the core has no hard dependency on it.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        extra_body: dict | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OpenAICompatibleClient requires the 'openai' package. "
                "Install with: pip install 'skill-factory[openai]'"
            ) from exc

        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._extra_body = extra_body or {}
        client_kwargs: dict[str, object] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        # Some local servers (Ollama/LM Studio) accept any key; supply a dummy if none.
        if base_url and not api_key:
            client_kwargs.setdefault("api_key", "not-needed")
        self._client = OpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        from openai import APIConnectionError, APIStatusError

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=self._extra_body or None,
                )
                return (response.choices[0].message.content or "").strip()
            except (APIStatusError, APIConnectionError) as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_base_delay * (2**attempt))
        raise RuntimeError(
            f"OpenAI-compatible request failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc
