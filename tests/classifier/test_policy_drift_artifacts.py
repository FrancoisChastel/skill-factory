"""Policy replay, noise and drift tools, exporters and parity checks."""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess

import pytest

from skill_factory.classifier.artifacts import (
    check_artifact,
    check_production,
    export_probeset,
    format_of,
    load_production,
    write_export,
)
from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.drift import compare_tables, determinism, recalibrate, version_drift
from skill_factory.classifier.evaluation import evaluate_split, verdicts_for
from skill_factory.classifier.lab import Lab
from skill_factory.classifier.policy import THRESHOLD_POLICY, Policy, load_policy
from skill_factory.classifier.probeset import ProbeSet
from skill_factory.classifier.splits import SplitPlan
from skill_factory.classifier.systemone import FakeSystemOne

from .conftest import bank, make_examples, responder

SCORE = {"mean": [{"weighted": [[0.7, "intent@reviewer"], [0.3, "capability"]]},
                  {"max": [{"scale": ["lure", 0.5]}, {"min": ["noise", "lure"]}]}]}


def _ps(threshold=0.3, **kw) -> ProbeSet:
    return ProbeSet("demo", bank(), SCORE, threshold, model="fake-jev", calibration={"fpr_budget": 0.05}, **kw)


# --- policies -------------------------------------------------------------------

POLICY_FILE = '''
LEVELS = ("pass", "warn", "block")

def strict(example, answers, score, threshold):
    if score is None:
        return "pass"
    if score >= 0.8:
        return "block"
    return "warn" if score >= threshold else "pass"

def static(example, answers, score, threshold):
    return "warn" if example.metadata.get("findings") else "pass"
static.levels = ("pass", "warn")

NOT_CALLABLE = 3

def bad(example, answers, score, threshold):
    return "maybe"
'''


def test_load_policy_from_a_file_with_levels(tmp_path):
    (tmp_path / "policy.py").write_text(POLICY_FILE)
    strict = load_policy("policy.py:strict", tmp_path)
    assert strict.levels == ("pass", "warn", "block") and strict.detect_levels == ("warn", "block")
    assert load_policy("policy.py:static", tmp_path).levels == ("pass", "warn")
    with pytest.raises(ValueError, match="file.py:function"):
        load_policy("policy.py", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_policy("nope.py:x", tmp_path)
    with pytest.raises(ValueError, match="has no"):
        load_policy("policy.py:missing", tmp_path)
    with pytest.raises(ValueError, match="not callable"):
        load_policy("policy.py:NOT_CALLABLE", tmp_path)


def test_policy_replay_scores_each_level_and_rejects_unknown_verdicts(tmp_path, examples):
    (tmp_path / "policy.py").write_text(POLICY_FILE)
    by = SplitPlan().assign(examples)
    lab = Lab(FakeSystemOne(responder), AnswerCache(None))
    lab.ask(by["train"], bank().compiled(), budget=10_000)
    table = lab.table(by["train"], bank().compiled(), budget=10_000)
    rep = evaluate_split("train", by["train"], table, _ps(), load_policy("policy.py:strict", tmp_path))
    assert rep.rates("warn").recall >= rep.rates("block").recall
    static = evaluate_split("train", by["train"], table, _ps(), load_policy("policy.py:static", tmp_path))
    assert 0.3 < static.rates().fpr < 0.7  # capability findings fire on half of everything
    with pytest.raises(ValueError, match="returned 'maybe'"):
        verdicts_for(by["train"], table, _ps(), load_policy("policy.py:bad", tmp_path))
    with pytest.raises(ValueError):
        Policy("x", lambda *a: "pass", ("pass",))
    assert THRESHOLD_POLICY.at_least("flag", "flag")


# --- drift ---------------------------------------------------------------------

def test_determinism_counts_identical_answers_and_flipped_verdicts(examples):
    by = SplitPlan().assign(examples)
    calls = {"n": 0}

    def wobbly(state, qid, q):
        calls["n"] += 1
        base = responder(state, qid, q)
        return base if calls["n"] % 5 else min(1.0, base + 0.3)

    lab = Lab(FakeSystemOne(wobbly), AnswerCache(None))
    result = determinism(lab, _ps(), by["val"], replicate=1)
    assert 0.5 < result.identical_rate < 1.0
    assert result.max_abs_diff == pytest.approx(0.3, abs=0.01)
    assert "verdicts flipped" in result.summary()
    with pytest.raises(ValueError):
        determinism(lab, _ps(), by["val"], replicate=0)


def test_compare_tables_of_identical_answers():
    lab = Lab(FakeSystemOne(responder), AnswerCache(None))
    exs = SplitPlan().assign(make_examples(20))["train"]
    lab.ask(exs, bank().compiled(), budget=10_000)
    t = lab.table(exs, bank().compiled(), budget=10_000)
    d = compare_tables(t, t, _ps(), exs)
    assert d.identical_rate == 1.0 and not d.flipped


def test_recalibrate_refits_the_threshold_for_a_new_model(examples):
    by = SplitPlan().assign(examples)
    cache = AnswerCache(None)
    old = Lab(FakeSystemOne(responder, model="m1", served="m1.0"), cache)
    new = Lab(FakeSystemOne(lambda s, q, x: min(1.0, responder(s, q, x) * 0.5), model="m2", served="m2.0"), cache)
    old.ask(by["train"] + by["val"], bank().compiled(), budget=10_000)
    result = recalibrate(_ps(), old, new, by)
    assert result.new.model == "m2" and result.new.pinned_version == "m2.0"
    assert result.new.threshold < 0.3  # answers halved, so the threshold moves down
    assert result.new.calibration["recalibrated_from"]["model"] == "fake-jev"
    assert "threshold" in result.summary()
    with pytest.raises(ValueError, match="fpr_budget"):
        recalibrate(_ps().with_changes(calibration={}), old, new, by, ask=False)


def test_version_drift_warns_when_another_version_answered(examples):
    lab = Lab(FakeSystemOne(responder, served="jev-1.14.0"), AnswerCache(None))
    lab.ask(examples[:5], bank().compiled(), budget=10_000)
    table = lab.table(examples[:5], bank().compiled(), budget=10_000)
    assert "recalibrate" in version_drift(_ps(pinned_version="jev-1.13.0"), table)
    assert version_drift(_ps(pinned_version="jev-1.14.0"), table) is None
    assert version_drift(_ps(), table) is None


# --- exporters and parity ---------------------------------------------------------

def test_formats_and_unknown_format():
    assert format_of("probes.ts") == "ts" and format_of("p.py") == "py" and format_of("p.json") == "json"
    with pytest.raises(ValueError):
        format_of("p.txt")
    with pytest.raises(ValueError):
        export_probeset(_ps(), "yaml")


def test_python_export_scores_exactly_like_the_lab(tmp_path):
    ps = _ps()
    path = write_export(ps, tmp_path / "probes.py")
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102 - our own generated code
    assert namespace["PROBESET_SHA256"] == ps.content_hash()
    assert set(namespace["QUESTIONS"]) == set(bank().ids)
    assert namespace["QUESTIONS"]["intent@reviewer"]["instructions"] == bank().compiled()["intent@reviewer"].instructions
    rng = random.Random(0)
    for _ in range(200):
        p = {q: round(rng.random(), 2) for q in bank().ids}
        assert namespace["score"](p) == pytest.approx(ps.score_of(p))
        assert namespace["flags"](p) == ps.flags(p)
    assert math.isnan(namespace["score"]({"lure": 0.5}))  # unanswered: no verdict, as in the lab
    assert not namespace["flags"]({"lure": 0.5})


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_typescript_export_scores_exactly_like_the_lab(tmp_path):
    ps = _ps()
    src = export_probeset(ps, "ts")
    # Strip the type annotations so plain node can run it.
    js = (src.replace(": Readonly<Record<string, ChoiceQuestion>>", "").replace(": string | null", "")
          .replace("p: Readonly<Record<string, number>>", "p").replace("): number {", ") {")
          .replace("): boolean {", ") {").replace("export ", ""))
    js = js[: js.index("interface ChoiceQuestion")] + js[js.index("const QUESTIONS"):]
    inputs = [{q: round(random.Random(i).random(), 2) for q in bank().ids} for i in range(20)]
    (tmp_path / "run.js").write_text(js + f"\nconsole.log(JSON.stringify({json.dumps(inputs)}.map(score)));\n")
    out = subprocess.run(["node", str(tmp_path / "run.js")], capture_output=True, text=True, check=True)
    for got, p in zip(json.loads(out.stdout), inputs):
        assert got == pytest.approx(ps.score_of(p))


def test_artifact_parity_byte_for_byte(tmp_path):
    ps = _ps()
    for name in ("probes.ts", "probes.py", "probes.json"):
        shipped = write_export(ps, tmp_path / name)
        assert check_artifact(ps, shipped).ok, name
    ts = tmp_path / "probes.ts"
    ts.write_text(ts.read_text().replace("Does the item show bad intent?", "Does the item show bad intent ?"))
    res = check_artifact(ps, ts)
    assert not res.ok and "line" in res.detail and "MISMATCH" in res.summary()
    edited = _ps(threshold=0.31)
    assert not check_artifact(edited, tmp_path / "probes.json").ok
    (tmp_path / "junk.json").write_text("[]")
    assert not check_artifact(ps, tmp_path / "junk.json").ok
    with pytest.raises(FileNotFoundError):
        check_artifact(ps, tmp_path / "missing.ts")


def test_production_parity_within_tolerance(tmp_path, examples):
    by = SplitPlan().assign(examples)
    lab = Lab(FakeSystemOne(responder), AnswerCache(None))
    lab.ask(by["train"], bank().compiled(), budget=10_000)
    table = lab.table(by["train"], bank().compiled(), budget=10_000)
    ps = _ps()
    verdicts = verdicts_for(by["train"], table, ps)
    lab_detected = {k: v == "flag" for k, v in verdicts.items()}
    rows = [{"id": k, "verdict": v} for k, v in verdicts.items()]
    same = check_production(by["train"], lab_detected, rows)
    assert same.ok and same.agree == same.compared and "OK" in same.summary()
    flipped = [{"id": r["id"], "flagged": r["verdict"] != "flag"} for r in rows]
    assert not check_production(by["train"], lab_detected, flipped).ok
    scores = [{"id": ex.id, "score": ps.score_of(table.of(ex.id))} for ex in by["train"]]
    assert check_production(by["train"], lab_detected, scores, threshold=ps.threshold).ok
    with pytest.raises(ValueError, match="threshold"):
        check_production(by["train"], lab_detected, scores)
    partial = check_production(by["train"], lab_detected, rows[:3])
    assert partial.missing == len(by["train"]) - 3
    with pytest.raises(ValueError):
        check_production(by["train"], lab_detected, [{"verdict": "flag"}])
    with pytest.raises(ValueError, match="no verdict"):
        check_production(by["train"], lab_detected, [{"id": "x"}])
    path = tmp_path / "prod.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert len(load_production(path)) == len(rows)
    path.write_text("{bad\n")
    with pytest.raises(ValueError, match=":1:"):
        load_production(path)
