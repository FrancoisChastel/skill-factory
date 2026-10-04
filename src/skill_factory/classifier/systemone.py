"""System One clients: send a state and choice questions, get P(true) per question.

System One is TypeSafe's decision protocol, served by jev on TypeSafe, OpenRouter,
Vercel AI Gateway and Cloudflare, and by local decision models on Ollama:

    POST {"model": ..., "state": "<text>", "questions": {"<id>": {"type": "choice", ...}}}
    ->   {"answers": {"<id>": {"type": "choice", "probabilities": {"true": 0.93, "false": 0.07}}},
          "model": "jev-1.13.0", "usage": {"input_tokens": 1234}}

The state is billed once per request, so asking 40 questions costs about what one
costs. The client is stdlib-only. A key is sent only to the host it belongs to and
never appears in an error message.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, runtime_checkable

from skill_factory.classifier.questions import Question


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    model: str
    key_env: str
    needs_key: bool = True
    headers: Mapping[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, Provider] = {
    "typesafe": Provider("typesafe", "https://api.typesafe.ai/v1/systemone", "jev-latest", "TYPESAFE_API_KEY"),
    "openrouter": Provider(
        "openrouter",
        "https://openrouter.ai/api/alpha/decisions",
        "typesafe/jev-1.13",
        "OPENROUTER_API_KEY",
        headers={"X-Title": "skill-factory"},
    ),
    "vercel": Provider(
        "vercel", "https://ai-gateway.vercel.sh/typesafe/v1/systemone", "typesafe-ai/jev", "AI_GATEWAY_API_KEY"
    ),
    "cloudflare": Provider(
        "cloudflare",
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
        "typesafe/jev",
        "CLOUDFLARE_API_TOKEN",
    ),
    "ollama": Provider("ollama", "http://localhost:11434/v1/systemone", "nimble", "", needs_key=False),
    "custom": Provider("custom", "", "jev-latest", ""),
}

#: A key for skill-factory only; wins over the provider's own variable.
OVERRIDE_KEY_ENV = "SKILL_FACTORY_SYSTEMONE_KEY"
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,99}$")


class SystemOneError(RuntimeError):
    """A request failed or the answer was malformed. Never carries the key."""


@dataclass(frozen=True)
class Answers:
    """P(true) per question id, the model that answered, and the input tokens it billed."""

    probabilities: Mapping[str, float]
    model: str
    input_tokens: int | None = None


@runtime_checkable
class SystemOneClient(Protocol):
    model: str

    def ask(self, state: str, questions: Mapping[str, Question]) -> Answers:
        ...


class HttpSystemOne:
    """A System One endpoint over HTTPS (or HTTP on localhost)."""

    def __init__(
        self,
        provider: str = "typesafe",
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        account_id: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        retry_delay: float = 1.0,
        opener: Callable[..., object] | None = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown System One provider {provider!r}; one of {', '.join(PROVIDERS)}")
        spec = PROVIDERS[provider]
        self.provider = spec
        self.model = model or spec.model
        url = base_url or spec.url
        if not url:
            raise ValueError("provider 'custom' needs base_url (the full System One endpoint)")
        if "{account_id}" in url:
            account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
            if not account_id:
                raise ValueError("provider 'cloudflare' needs account_id or CLOUDFLARE_ACCOUNT_ID")
            url = url.replace("{account_id}", account_id)
        self.url = url.replace("{model}", self.model)
        self._key = api_key or os.getenv(OVERRIDE_KEY_ENV) or (os.getenv(spec.key_env) if spec.key_env else None)
        if spec.needs_key and provider != "custom" and not self._key:
            raise ValueError(
                f"no key for {provider}: set {spec.key_env or OVERRIDE_KEY_ENV} (or systemone.api_key)"
            )
        if not self.url.startswith("https://") and not _is_local(self.url):
            raise ValueError(f"refusing to send a state over plain HTTP to a remote host: {self.url}")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._open = opener or urllib.request.urlopen

    def ask(self, state: str, questions: Mapping[str, Question]) -> Answers:
        body = json.dumps(
            {"model": self.model, "state": state, "questions": {k: q.to_wire() for k, q in questions.items()}},
            ensure_ascii=False,
        ).encode("utf-8")
        raw = self._post(body)
        return parse_answers(raw, list(questions), fallback_model=self.model)

    def _post(self, body: bytes) -> object:
        headers = {"content-type": "application/json", "accept": "application/json", **self.provider.headers}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"
        last = "no attempt"
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with self._open(request, timeout=self.timeout) as response:  # type: ignore[attr-defined]
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {self._redact(_error_text(exc))}"
                if exc.code != 429 and exc.code < 500:
                    break  # 4xx other than rate limiting will not get better
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = self._redact(str(getattr(exc, "reason", exc)))
            except json.JSONDecodeError as exc:
                raise SystemOneError(f"invalid answer: the response is not JSON ({exc})") from None
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (2**attempt))
        raise SystemOneError(f"{self.provider.name} request failed: {last}")

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "[key]") if self._key else text


class FakeSystemOne:
    """Deterministic System One for tests and offline demos.

    ``responder(state, question_id, question) -> P(true)``; every request is
    recorded on :attr:`requests` as ``(state, [question ids])``.
    """

    def __init__(
        self,
        responder: Callable[[str, str, Question], float],
        *,
        model: str = "fake-jev",
        served: str | None = None,
        fail_when: Callable[[str], bool] | None = None,
    ):
        self.model = model
        self._served = served or model
        self._responder = responder
        self._fail_when = fail_when
        self.requests: list[tuple[str, list[str]]] = []

    def ask(self, state: str, questions: Mapping[str, Question]) -> Answers:
        self.requests.append((state, list(questions)))
        if self._fail_when is not None and self._fail_when(state):
            raise SystemOneError("fake failure")
        probabilities = {qid: float(self._responder(state, qid, q)) for qid, q in questions.items()}
        return Answers(probabilities, self._served, input_tokens=max(1, len(state) // 4))


def parse_answers(raw: object, ids: list[str], *, fallback_model: str) -> Answers:
    """P(true) for every id. A missing or malformed answer fails the whole request: no partial opinions."""
    inner = _unwrap(raw)
    answers = inner.get("answers") if isinstance(inner, dict) else None
    if not isinstance(answers, dict):
        raise SystemOneError("invalid answer: the response has no answers object")
    probabilities = {}
    for qid in ids:
        answer = answers.get(qid)
        if not isinstance(answer, dict):
            raise SystemOneError(f"invalid answer: {qid!r} is missing")
        if answer.get("type") != "choice":
            raise SystemOneError(f"invalid answer: {qid!r} is not a choice answer")
        probs = answer.get("probabilities")
        p = probs.get("true") if isinstance(probs, dict) else None
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
            raise SystemOneError(f"invalid answer: {qid!r} has no probability of true in [0, 1]")
        probabilities[qid] = float(p)
    model = inner.get("model") if isinstance(inner, dict) else None
    served = model if isinstance(model, str) and _MODEL_RE.match(model) else fallback_model
    usage = inner.get("usage") if isinstance(inner, dict) else None
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    return Answers(probabilities, served, int(tokens) if isinstance(tokens, (int, float)) else None)


def _unwrap(raw: object) -> object:
    """The System One answer, or the one inside a Cloudflare envelope ``{result, success, errors}``."""
    if not isinstance(raw, dict) or "result" not in raw or "answers" in raw:
        return raw
    if raw.get("success") is False:
        errors = raw.get("errors")
        first = next((e for e in errors if isinstance(e, dict)), None) if isinstance(errors, list) else None
        message = f": {first.get('message')}" if first and isinstance(first.get("message"), str) else ""
        raise SystemOneError(f"invalid answer: the host reported an error{message}")
    return raw["result"]


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:200]
    except Exception:  # noqa: BLE001 - the status code is reported either way
        return ""


def _is_local(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}
