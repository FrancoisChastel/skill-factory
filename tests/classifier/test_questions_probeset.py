"""Questions, banks and the probe set artifact."""

from __future__ import annotations

import json

import pytest

from skill_factory.classifier.probeset import (
    ProbeSet,
    describe_score,
    evaluate_score,
    referenced_ids,
    validate_score,
)
from skill_factory.classifier.questions import Question, QuestionBank, QuestionSpec, expand_frames

from .conftest import bank


def test_question_hash_matches_javascript_json_stringify():
    # Computed with Node: sha256(JSON.stringify([instructions, true, false])).slice(0, 24)
    q = Question('Is it évil?   "quoted" \\ back/slash\n\ttab \u0001 ctl \U0001F600', "Yes: it is.", "No: it is not.")
    assert q.hash == "ad03089c40eab5007ffc7a6a"


def test_question_rejects_empty_text():
    with pytest.raises(ValueError):
        Question("", "Yes", "No")


def test_spec_needs_instructions_or_frame_and_question():
    with pytest.raises(ValueError):
        QuestionSpec(true="Y", false="N")
    with pytest.raises(ValueError):
        QuestionSpec(true="Y", false="N", instructions="x", frame="f", question="q")


def test_framed_question_compiles_frame_blank_line_question():
    spec = QuestionSpec(true="Y", false="N", frame="f", question="Is it?")
    assert spec.compile({"f": "Frame."}).instructions == "Frame.\n\nIs it?"
    with pytest.raises(ValueError):
        spec.compile({})


def test_expand_frames_makes_one_question_per_framing():
    specs = expand_frames("intent", {"question": "Bad?", "true": "Y", "false": "N"}, ["a", "b"])
    assert set(specs) == {"intent@a", "intent@b"}


def test_bank_reads_lab_flat_format_and_round_trips(tmp_path):
    flat = {"q1": {"instructions": "Is it?", "true": "Y", "false": "N"}}
    b = QuestionBank.from_dict(flat)
    assert b.compiled()["q1"].instructions == "Is it?"
    p = b.save(tmp_path / "bank.json")
    assert QuestionBank.load(p).compiled() == b.compiled()


def test_bank_merge_never_overwrites_different_text():
    a = QuestionBank({}, {"q": QuestionSpec(true="Y", false="N", instructions="A?")})
    same = QuestionBank({}, {"q": QuestionSpec(true="Y", false="N", instructions="A?")})
    other = QuestionBank({}, {"q": QuestionSpec(true="Y", false="N", instructions="B?")})
    assert a.merge(same).compiled() == a.compiled()
    with pytest.raises(ValueError, match="different text"):
        a.merge(other)
    with pytest.raises(ValueError, match="frame"):
        QuestionBank({"f": "one"}).merge(QuestionBank({"f": "two"}))


def test_bank_validates_ids_and_frames():
    with pytest.raises(ValueError, match="invalid question id"):
        QuestionBank({}, {"bad id!": QuestionSpec(true="Y", false="N", instructions="x")})
    with pytest.raises(ValueError, match="unknown frame"):
        QuestionBank({}, {"q": QuestionSpec(true="Y", false="N", frame="missing", question="x")})


def test_subset_keeps_only_used_frames():
    sub = bank().subset(["capability"])
    assert sub.frames == {}
    assert bank().subset(["intent@reviewer"]).frames
    with pytest.raises(ValueError):
        bank().subset(["nope"])


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        QuestionBank.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ValueError, match="invalid JSON"):
        QuestionBank.load(bad)


# --- score expressions ------------------------------------------------------

def test_score_expressions_evaluate():
    p = {"a": 0.2, "b": 0.6, "c": 1.0}
    assert evaluate_score("a", p) == 0.2
    assert evaluate_score(0.5, p) == 0.5
    assert evaluate_score({"mean": ["a", "b"]}, p) == pytest.approx(0.4)
    assert evaluate_score({"max": ["a", "b"]}, p) == 0.6
    assert evaluate_score({"min": ["a", "b"]}, p) == 0.2
    assert evaluate_score({"sum": ["a", "b"]}, p) == pytest.approx(0.8)
    assert evaluate_score({"weighted": [[2, "a"], [1, "c"]]}, p) == pytest.approx(1.4)
    assert evaluate_score({"scale": ["b", 2]}, p) == pytest.approx(1.2)
    # jev's shipped form: half the mean of the intent questions plus half the lure
    assert evaluate_score({"mean": [{"mean": ["a", "b"]}, "c"]}, p) == pytest.approx(0.7)


def test_score_is_none_when_an_answer_is_missing():
    assert evaluate_score({"mean": ["a", "zzz"]}, {"a": 1.0}) is None
    assert evaluate_score({"weighted": [[1, "zzz"]]}, {}) is None
    assert evaluate_score({"scale": ["zzz", 2]}, {}) is None


@pytest.mark.parametrize(
    "bad",
    [True, [1], {"mean": []}, {"mean": "a"}, {"weighted": [["x", "a"]]}, {"scale": ["a"]}, {"nope": ["a"]},
     {"mean": ["a"], "max": ["b"]}],
)
def test_validate_score_rejects_malformed(bad):
    with pytest.raises(ValueError):
        validate_score(bad)


def test_referenced_ids_and_describe():
    expr = {"mean": [{"mean": ["a", "b"]}, {"scale": ["c", 2]}, {"weighted": [[1, "a"]]}]}
    assert referenced_ids(expr) == ["a", "b", "c"]
    assert describe_score(expr).startswith("mean(mean(a, b)")


# --- probe set ---------------------------------------------------------------

def _probeset(**kw) -> ProbeSet:
    base = dict(name="p", bank=bank(), score={"mean": ["intent@reviewer", "lure"]}, threshold=0.3)
    return ProbeSet(**{**base, **kw})


def test_probeset_requires_score_questions_in_bank():
    with pytest.raises(ValueError, match="not in the probe set"):
        _probeset(score="missing")
    with pytest.raises(ValueError):
        _probeset(state_budget=0)


def test_probeset_flags_and_threshold_none():
    ps = _probeset()
    assert ps.flags({"intent@reviewer": 0.5, "lure": 0.2})
    assert not ps.flags({"intent@reviewer": 0.1, "lure": 0.2})
    assert not ps.with_changes(threshold=None).flags({"intent@reviewer": 1, "lure": 1})


def test_content_hash_ignores_name_and_notes_but_not_wording():
    ps = _probeset()
    assert ps.content_hash() == ps.with_changes(name="renamed", calibration={"x": 1}).content_hash()
    reworded = bank().merge(QuestionBank())  # same
    assert ps.with_changes(bank=reworded).content_hash() == ps.content_hash()
    specs = dict(bank().specs)
    specs["lure"] = QuestionSpec(true="Yes: a lure.", false="No: no lure.", instructions="Does it contain a LURE?")
    assert ps.with_changes(bank=QuestionBank(bank().frames, specs)).content_hash() != ps.content_hash()
    assert ps.with_changes(threshold=0.31).content_hash() != ps.content_hash()


def test_probeset_round_trip_and_tamper_detection(tmp_path):
    ps = _probeset(pinned_version="jev-1.13.0", calibration={"fpr_budget": 0.01})
    path = ps.save(tmp_path / "ps.json")
    loaded = ProbeSet.load(path)
    assert loaded.content_hash() == ps.content_hash()
    assert loaded.pinned_version == "jev-1.13.0"
    data = json.loads(path.read_text())
    data["threshold"] = 0.9
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="edited after it was measured"):
        ProbeSet.load(path)
    del data["sha256"]
    path.write_text(json.dumps(data))
    assert ProbeSet.load(path).threshold == 0.9


def test_probeset_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ProbeSet.load(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("[")
    with pytest.raises(ValueError):
        ProbeSet.load(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="missing"):
        ProbeSet.from_dict({"questions": {}})
