"""A small synthetic world for classifier mode, and a local System One stub.

Positives carry ``INTENT:bad``; a capability marker ``CAP:yes`` appears in both
classes, so capability questions separate badly and intent questions well, as
in the jev work. Some positives carry a ``LURE`` that only a lure question sees.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from skill_factory.classifier.dataset import Example
from skill_factory.classifier.probeset import ProbeSet
from skill_factory.classifier.questions import Question, QuestionBank, QuestionSpec, sha24

FRAMES = {"reviewer": "You review an item before it is used."}


def make_examples(n: int = 120) -> list[Example]:
    out = []
    for i in range(n):
        positive = i % 3 == 0
        cap = "CAP:yes" if i % 2 == 0 else "CAP:no"
        intent = "INTENT:bad" if positive else "INTENT:good"
        lure = " LURE" if positive and i % 9 == 0 else ""
        long = " pad" * 400 if i % 10 == 7 else ""  # some states are long (over a small budget)
        out.append(
            Example(
                id=f"ex{i:03d}",
                label=positive,
                state=f"item {i} {cap} {intent}{lure}{long}",
                group=f"g{i // 2}",
                source="src-b" if i >= n - 12 else "src-a",
                metadata={"findings": [{"severity": "high"}] if cap == "CAP:yes" else []},
            )
        )
    return out


def jitter(text: str, qid: str) -> float:
    """Deterministic noise in [0, 0.05)."""
    return int(sha24(text + qid)[:4], 16) / 0xFFFF * 0.05


def responder(state: str, qid: str, q: Question) -> float:
    text = q.instructions.lower()
    noise = jitter(state, qid)
    # A miss-prone intent reading: catches most positives, a few with low P.
    if "intent" in text:
        hit = "INTENT:bad" in state and not state.split()[1].endswith("6")
        return round(min(1.0, (0.9 if hit else 0.02) + noise), 2)
    if "lure" in text:
        return round(0.95 if "LURE" in state else noise / 5, 2)
    if "capability" in text:
        return round((0.8 if "CAP:yes" in state else 0.1) + noise, 2)
    return round(noise, 2)


def bank() -> QuestionBank:
    return QuestionBank(
        FRAMES,
        {
            "capability": QuestionSpec(
                true="Yes: it has the capability.", false="No: it lacks it.",
                instructions="Does this item have the capability?",
            ),
            "intent@reviewer": QuestionSpec(
                true="Yes: bad intent.", false="No: good intent.", frame="reviewer",
                question="Does the item show bad intent?",
            ),
            "lure": QuestionSpec(
                true="Yes: a lure.", false="No: no lure.", instructions="Does it contain a lure?",
            ),
            "noise": QuestionSpec(
                true="Yes: noise.", false="No: no noise.", instructions="Is it noisy?",
            ),
        },
    )


@pytest.fixture
def examples() -> list[Example]:
    return make_examples()


@pytest.fixture
def seed_probeset() -> ProbeSet:
    return ProbeSet("seed", bank().subset(["capability"]), "capability", None, model="fake-jev", state_budget=10_000)


class _Stub:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.fail_next: list[int] = []  # HTTP status codes to return before succeeding
        self.served = "fake-jev-1.0"


@pytest.fixture
def stub_server() -> Iterator[tuple[str, _Stub]]:
    """A local System One endpoint answering with ``responder``."""
    stub = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D401 - silence
            return

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["content-length"])).decode("utf-8"))
            stub.requests.append({"body": body, "auth": self.headers.get("authorization")})
            if stub.fail_next:
                code = stub.fail_next.pop(0)
                self.send_response(code)
                self.end_headers()
                self.wfile.write(b'{"error": "nope"}')
                return
            answers = {}
            for qid, q in body["questions"].items():
                question = Question(q["instructions"], q["criteria"]["true"], q["criteria"]["false"])
                p = responder(body["state"], qid, question)
                answers[qid] = {"type": "choice", "probabilities": {"true": p, "false": 1 - p}}
            payload = {"answers": answers, "model": stub.served,
                       "usage": {"input_tokens": len(body["state"]) // 4}}
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1/systemone", stub
    finally:
        server.shutdown()
        server.server_close()
