"""Reflection (digest and proposals) and the classifier optimizer, end to end with fakes."""

from __future__ import annotations

import json

import pytest

from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.dataset import Example
from skill_factory.classifier.lab import Lab
from skill_factory.classifier.objective import Constraint, Objective
from skill_factory.classifier.optimizer import ClassifierOptimizer
from skill_factory.classifier.probeset import ProbeSet
from skill_factory.classifier.questions import QuestionBank, QuestionSpec
from skill_factory.classifier.reflect import failure_digest, parse_proposal, proposal_prompt
from skill_factory.classifier.splits import SplitPlan
from skill_factory.classifier.systemone import FakeSystemOne
from skill_factory.llm.client import FakeLLMClient

from .conftest import bank, make_examples, responder

INTENT_PROPOSAL = {
    "rationale": "misses share bad intent, which capability does not see",
    "frames": {"careful": "Read the item carefully."},
    "questions": [
        {"id": "Intent Check", "question": "Does the item show bad intent?", "true": "Yes: bad.", "false": "No: fine.",
         "frames": ["careful", "missing-frame"]},
        {"id": "lure_q", "instructions": "Does it carry a lure?", "true": "Yes: lure.", "false": "No: none."},
        {"id": "broken"},
    ],
}


def _lab():
    return Lab(FakeSystemOne(responder), AnswerCache(None))


def _split(examples):
    by = SplitPlan().assign(examples)
    return {"train": by["train"], "val": by["val"]}


# --- proposals ---------------------------------------------------------------

def test_parse_proposal_expands_frames_and_drops_unusable_items():
    p = parse_proposal(json.dumps(INTENT_PROPOSAL), bank(), max_new=5, max_state_budget=None)
    assert set(p.new_ids) == {"intent_check@careful", "lure_q"}
    assert p.bank.frames == {"careful": "Read the item carefully."}
    assert "bad intent" in p.bank.compiled()["intent_check@careful"].instructions
    assert p.rationale.startswith("misses share")


def test_parse_proposal_keeps_ids_unique_and_skips_known_questions():
    existing = QuestionBank({}, {"q": QuestionSpec(true="Y", false="N", instructions="Old?")})
    reply = {"questions": [
        {"id": "q", "instructions": "New?", "true": "Y", "false": "N"},
        {"id": "q", "instructions": "Old?", "true": "Y", "false": "N"},
    ]}
    p = parse_proposal("```json\n" + json.dumps(reply) + "\n```", existing, max_new=5, max_state_budget=None)
    assert p.new_ids == ["q_2"]


def test_parse_proposal_pipeline_budget_is_clamped():
    reply = {"pipeline": {"state_budget": 10**9}}
    p = parse_proposal(json.dumps(reply), bank(), max_new=5, max_state_budget=96_000)
    assert p.pipeline == {"state_budget": 96_000}
    with pytest.raises(ValueError, match="no usable"):
        parse_proposal(json.dumps(reply), bank(), max_new=5, max_state_budget=None)


def test_parse_proposal_repairs_trailing_commas_and_comments():
    reply = """Here you go:
{
  // two new questions
  "rationale": "intent",
  "questions": [
    {"id": "a", "instructions": "A?", "true": "Yes: a.", "false": "No: a.",},
  ],
}"""
    assert parse_proposal(reply, bank(), max_new=5, max_state_budget=None).new_ids == ["a"]


@pytest.mark.parametrize("raw", ["", "no json here", "{broken", "[1, 2]"])
def test_parse_proposal_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_proposal(raw, bank(), max_new=5, max_state_budget=None)


def test_parse_proposal_refuses_to_redefine_a_frame():
    reply = {"frames": {"reviewer": "other text"}, "questions": []}
    with pytest.raises(ValueError, match="frame"):
        parse_proposal(json.dumps(reply), bank(), max_new=5, max_state_budget=None)


# --- digest --------------------------------------------------------------------

def test_digest_reports_system_causes_both_error_kinds_and_every_p(examples, seed_probeset):
    split = _split(examples)
    lab = _lab()
    small = seed_probeset.with_changes(threshold=0.5, state_budget=1000)
    questions = bank().compiled()
    lab.ask(split["train"], questions, budget=1000)
    table = lab.table(split["train"], questions, budget=1000)
    text = failure_digest(small, split["train"], table, max_state_budget=4000)
    assert "SYSTEM-LEVEL CAUSES" in text
    assert "over budget" in text and "would admit" in text
    assert "MISSED POSITIVES" in text and "FALSE FLAGS" in text
    assert "P(true): capability=" in text
    prompt = proposal_prompt(small, text, goal="find bad intent", frames=bank().frames, max_new=3,
                             max_state_budget=4000)
    assert "GOAL: find bad intent" in prompt and '"pipeline"' in prompt and "4,000" in prompt


def test_digest_with_nothing_missed():
    exs = [Example(id="a", label=True, state="INTENT:bad x"), Example(id="b", label=False, state="ok")]
    lab = _lab()
    ps = ProbeSet("p", bank(), "intent@reviewer", 0.5)
    lab.ask(exs, bank().compiled(), budget=10_000)
    table = lab.table(exs, bank().compiled(), budget=10_000)
    assert "none: every positive was detected" in failure_digest(ps, exs, table)


# --- optimizer -----------------------------------------------------------------

def test_selection_alone_finds_the_intent_question(examples, seed_probeset):
    opt = ClassifierOptimizer(_lab(), Objective(fpr_budget=0.05, constraints=(Constraint(max_fpr=0.1),)))
    result = opt.optimize(seed_probeset, _split(examples), pool=bank())
    assert result.history[0].change == "seed" and result.history[1].change == "selection"
    assert result.history[1].accepted
    assert "intent@reviewer" in result.best.score_ids
    assert result.best_score > result.baseline_score + 0.5
    assert result.best.calibration["fitted_on"] == "train"
    assert result.best.pinned_version == "fake-jev"
    assert result.stats and result.front


def test_reflection_adds_questions_that_the_gate_keeps(examples):
    seed = ProbeSet("seed", bank().subset(["capability", "noise"]), "capability", None,
                    model="fake-jev", state_budget=10_000)
    proposer = FakeLLMClient(responder=lambda prompt, system: json.dumps(INTENT_PROPOSAL))
    opt = ClassifierOptimizer(_lab(), Objective(fpr_budget=0.05, constraints=(Constraint(max_fpr=0.1),)),
                              proposer=proposer, rounds=2, patience=1, goal="bad intent")
    result = opt.optimize(seed, _split(examples))
    kept = [h for h in result.history if h.round >= 2 and h.accepted]
    assert kept and "intent_check@careful" in kept[0].new_questions
    assert "intent_check@careful" in result.pool
    assert any(q.startswith("intent_check") for q in result.best.score_ids)
    assert "GOAL: bad intent" in proposer.calls[0]["prompt"]


def test_a_pipeline_change_admits_skipped_examples(examples):
    # At a 1,000-character budget the long positives are never sent; raising the budget catches them.
    seed = ProbeSet("seed", bank().subset(["intent@reviewer", "lure"]), "intent@reviewer", None,
                    model="fake-jev", state_budget=1000)
    proposer = FakeLLMClient(responder=lambda p, s: json.dumps({"rationale": "too long", "pipeline": {"state_budget": 5000}}))
    opt = ClassifierOptimizer(_lab(), Objective(fpr_budget=0.05, constraints=()), proposer=proposer,
                              rounds=1, max_state_budget=8000, keep_questions=[])
    result = opt.optimize(seed, _split(examples))
    last = result.history[-1]
    assert "state budget 1,000 -> 5,000" in last.change
    assert last.accepted and result.best.state_budget == 5000


def test_constraints_refuse_a_better_recall_and_say_why(examples, seed_probeset):
    obj = Objective(fpr_budget=0.05, constraints=(Constraint(max_fpr=-1.0),))
    result = ClassifierOptimizer(_lab(), obj).optimize(seed_probeset, _split(examples), pool=bank())
    assert not result.history[1].accepted
    assert any("false-positive rate" in r for r in result.history[1].reasons)
    assert result.best.score == seed_probeset.score


def test_unusable_proposals_run_out_patience(examples, seed_probeset):
    proposer = FakeLLMClient(responses=["not json"])
    opt = ClassifierOptimizer(_lab(), Objective(fpr_budget=0.05), proposer=proposer, rounds=5, patience=2)
    result = opt.optimize(seed_probeset, _split(examples))
    unusable = [h for h in result.history if h.change == "proposal"]
    assert len(unusable) == 2 and all(not h.accepted for h in unusable)
    assert unusable[0].raw_reply == "not json" and unusable[0].to_dict()["raw_reply"] == "not json"


def test_optimizer_validates_inputs(examples, seed_probeset):
    with pytest.raises(ValueError):
        ClassifierOptimizer(_lab(), Objective(), patience=0)
    with pytest.raises(ValueError, match="train and val"):
        ClassifierOptimizer(_lab(), Objective()).optimize(seed_probeset, {"train": examples[:5], "val": []})


def test_round_records_serialize(examples, seed_probeset):
    result = ClassifierOptimizer(_lab(), Objective(fpr_budget=0.05)).optimize(seed_probeset, _split(examples), bank())
    data = [h.to_dict() for h in result.history]
    assert json.loads(json.dumps(data))[0]["reports"]["train"]["levels"]["flag"]["tp"] >= 0


def test_examples_without_state_can_only_be_rescored():
    exs = make_examples(30)
    hashed = [Example(id=e.id, label=e.label, state_hash=e.state_hash, chars=e.chars, group=e.group) for e in exs]
    lab = _lab()
    lab.ask(exs, bank().compiled(), budget=10_000)  # answers bought earlier, with the text
    offline = Lab(None, lab.cache, model="fake-jev")
    table = offline.table(hashed, bank().compiled(), budget=10_000)
    assert all(table.of(e.id) for e in hashed)
