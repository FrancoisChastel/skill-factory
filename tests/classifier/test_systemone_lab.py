"""The System One client (over a local HTTP stub), the answer cache, and the lab."""

from __future__ import annotations

import json

import pytest

from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.dataset import Example
from skill_factory.classifier.lab import Lab, LabSettings
from skill_factory.classifier.questions import Question
from skill_factory.classifier.systemone import (
    FakeSystemOne,
    HttpSystemOne,
    SystemOneError,
    parse_answers,
)

from .conftest import bank, responder

Q = {"a": Question("A?", "Yes", "No"), "b": Question("B?", "Yes", "No")}


def _answer(p):
    return {"type": "choice", "probabilities": {"true": p, "false": 1 - p}}


# --- parsing -----------------------------------------------------------------

def test_parse_answers_reads_probabilities_model_and_usage():
    raw = {"answers": {"a": _answer(0.9), "b": _answer(0)}, "model": "jev-1.13.0", "usage": {"input_tokens": 12}}
    a = parse_answers(raw, ["a", "b"], fallback_model="jev-latest")
    assert a.probabilities == {"a": 0.9, "b": 0.0} and a.model == "jev-1.13.0" and a.input_tokens == 12


def test_parse_answers_unwraps_cloudflare_envelope_and_reports_its_error():
    raw = {"result": {"answers": {"a": _answer(0.5)}}, "success": True}
    assert parse_answers(raw, ["a"], fallback_model="m").probabilities["a"] == 0.5
    with pytest.raises(SystemOneError, match="quota"):
        parse_answers({"result": None, "success": False, "errors": [{"message": "quota"}]}, ["a"], fallback_model="m")


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"answers": {}},
        {"answers": {"a": {"type": "noul"}}},
        {"answers": {"a": {"type": "choice", "probabilities": {"true": 1.5}}}},
        {"answers": {"a": {"type": "choice", "probabilities": {"true": True}}}},
    ],
)
def test_parse_answers_refuses_partial_or_malformed(raw):
    with pytest.raises(SystemOneError, match="invalid answer"):
        parse_answers(raw, ["a"], fallback_model="m")


def test_untrusted_model_names_are_not_echoed():
    raw = {"answers": {"a": _answer(0.1)}, "model": "evil\nmodel name"}
    assert parse_answers(raw, ["a"], fallback_model="jev-latest").model == "jev-latest"


# --- HTTP client -------------------------------------------------------------

def test_http_client_round_trip(stub_server):
    url, stub = stub_server
    client = HttpSystemOne("custom", base_url=url, api_key="ts_secret", model="fake-jev")
    answers = client.ask("item 3 INTENT:bad", {"intent": Question("bad intent?", "Y", "N")})
    assert answers.probabilities["intent"] >= 0.9
    assert answers.model == "fake-jev-1.0"
    body = stub.requests[0]["body"]
    assert body["model"] == "fake-jev" and body["questions"]["intent"]["type"] == "choice"
    assert stub.requests[0]["auth"] == "Bearer ts_secret"


def test_http_client_retries_rate_limits_and_server_errors(stub_server):
    url, stub = stub_server
    stub.fail_next = [429, 503]
    client = HttpSystemOne("custom", base_url=url, max_retries=3, retry_delay=0)
    assert client.ask("s", Q).probabilities
    assert len(stub.requests) == 3


def test_http_client_fails_fast_on_client_errors_and_never_leaks_the_key(stub_server):
    url, stub = stub_server
    stub.fail_next = [401]
    client = HttpSystemOne("custom", base_url=url, api_key="ts_secret", max_retries=3, retry_delay=0)
    with pytest.raises(SystemOneError) as info:
        client.ask("s", Q)
    assert "401" in str(info.value) and "ts_secret" not in str(info.value)
    assert len(stub.requests) == 1


def test_http_client_configuration_errors(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("SKILL_FACTORY_SYSTEMONE_KEY", raising=False)
    with pytest.raises(ValueError, match="no key"):
        HttpSystemOne("typesafe")
    with pytest.raises(ValueError, match="unknown"):
        HttpSystemOne("nope")
    with pytest.raises(ValueError, match="base_url"):
        HttpSystemOne("custom")
    with pytest.raises(ValueError, match="plain HTTP"):
        HttpSystemOne("custom", base_url="http://example.com/v1/systemone")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError, match="account_id"):
        HttpSystemOne("cloudflare", api_key="k")
    cf = HttpSystemOne("cloudflare", api_key="k", account_id="acc")
    assert "/accounts/acc/ai/run/typesafe/jev" in cf.url
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_env")
    assert HttpSystemOne("typesafe").url.startswith("https://")
    assert HttpSystemOne("ollama").model == "nimble"


# --- cache ---------------------------------------------------------------------

def test_cache_round_trip_with_versions_and_replicates(tmp_path):
    path = tmp_path / "answers.jsonl"
    cache = AnswerCache(path)
    cache.put_many([{"s": "s1", "q": "q1", "m": "m", "p": 0.25, "v": "m-1"}, {"s": "s1", "q": "q1", "m": "m", "p": 0.5, "r": 1}])
    again = AnswerCache(path)
    assert again.get("s1", "q1", "m") == 0.25
    assert again.get("s1", "q1", "m", replicate=1) == 0.5
    assert again.version("s1", "q1", "m") == "m-1"
    assert again.models() == {"m"}
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert "r" not in lines[0] and lines[1]["r"] == 1  # the lab's format, extended only when needed


def test_cache_drops_a_cut_off_last_line_and_refuses_corruption(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"s": "a", "q": "b", "m": "m", "p": 0.1}\n{"s": "a", "q"')
    cache = AnswerCache(path)
    assert len(cache) == 1
    cache.put_many([{"s": "c", "q": "d", "m": "m", "p": 0.2}])
    assert len(AnswerCache(path)) == 2
    path.write_text('not json\n{"s": "a", "q": "b", "m": "m", "p": 0.1}\n')
    with pytest.raises(ValueError, match=":1: corrupt"):
        AnswerCache(path)


# --- lab -------------------------------------------------------------------------

def _examples(n=5, length=40):
    return [Example(id=f"e{i}", label=i % 2 == 0, state=f"item {i} " + "x" * length) for i in range(n)]


def test_lab_asks_only_what_the_cache_lacks_in_batches(tmp_path):
    client = FakeSystemOne(responder)
    lab = Lab(client, AnswerCache(tmp_path / "a.jsonl"), LabSettings(per_request=3, concurrency=2))
    questions = bank().compiled()  # 4 questions -> 2 requests per example at 3 per request
    report = lab.ask(_examples(), questions, budget=10_000)
    assert report.requests == 10 and report.answers == 20 and report.failed == 0
    assert all(len(ids) <= 3 for _, ids in client.requests)
    again = lab.ask(_examples(), questions, budget=10_000)
    assert again.planned == 0 and again.requests == 0
    table = lab.table(_examples(), questions, budget=10_000)
    assert set(table.of("e0")) == set(questions)
    assert table.versions == {"fake-jev": 20}


def test_lab_skips_over_budget_and_stateless_examples():
    exs = _examples(2) + [Example(id="long", label=True, state="x" * 500), Example(id="hash", label=False, state_hash="h")]
    lab = Lab(FakeSystemOne(responder), AnswerCache(None))
    report = lab.ask(exs, Q, budget=100)
    assert report.statuses == {"judged": 3, "over budget": 1}
    table = lab.table(exs, Q, budget=100)
    assert table.statuses["long"] == "over budget" and table.of("long") == {}


def test_lab_spend_cap_is_cumulative_and_stops_before_crossing():
    settings = LabSettings(per_request=40, concurrency=1, usd_per_million_input_tokens=1_000_000, max_usd=25.0)
    lab = Lab(FakeSystemOne(responder), AnswerCache(None), settings)
    first = lab.ask(_examples(3, length=36), Q, budget=10_000)  # ~12 tokens per request = ~$12
    assert first.requests == 2 and first.not_sent_over_cap == 1
    second = lab.ask(_examples(3, length=36), {"c": Question("C?", "Y", "N")}, budget=10_000)
    assert second.requests == 0 and second.not_sent_over_cap == 3
    assert lab.spent_usd <= 25.0


def test_lab_counts_failures_without_caching_them():
    client = FakeSystemOne(responder, fail_when=lambda s: "item 1 " in s)
    lab = Lab(client, AnswerCache(None))
    report = lab.ask(_examples(3), Q, budget=10_000)
    assert report.failed == 1 and report.requests == 2 and "fake failure" in report.errors[0]
    assert lab.table(_examples(3), Q, budget=10_000).of("e1") == {}


def test_offline_lab_scores_the_cache_and_refuses_to_ask():
    with pytest.raises(ValueError):
        Lab(None, AnswerCache(None))
    lab = Lab(None, AnswerCache(None), model="jev-latest", offline_reason="no key")
    with pytest.raises(RuntimeError, match="offline \\(no key\\)"):
        lab.ask(_examples(1), Q, budget=10_000)
    assert lab.table(_examples(1), Q, budget=10_000).of("e0") == {}


def test_cost_per_pass_counts_requests_per_example():
    lab = Lab(None, AnswerCache(None), LabSettings(per_request=2, usd_per_million_input_tokens=1.0), model="m")
    exs = [Example(id="a", label=True, state="x" * 4_000_000)]
    assert lab.cost_per_pass(exs, 3, budget=10**9) == pytest.approx(2.0)  # 2 requests x 1M tokens x $1/M
    with pytest.raises(ValueError):
        LabSettings(per_request=0)
