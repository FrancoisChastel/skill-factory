"""Dataset-level metrics, selection and ensembling, and the constrained gate."""

from __future__ import annotations

import pytest

from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.evaluation import evaluate_split, fit_threshold, question_stats
from skill_factory.classifier.lab import Lab
from skill_factory.classifier.metrics import Rates, auc, pct, rates, threshold_at_fpr, widest_margin
from skill_factory.classifier.objective import Constraint, Objective
from skill_factory.classifier.policy import Policy
from skill_factory.classifier.probeset import ProbeSet, evaluate_score
from skill_factory.classifier.select import best_per_family, candidates, greedy_union, pareto, rank_by_auc
from skill_factory.classifier.splits import SplitPlan
from skill_factory.classifier.systemone import FakeSystemOne

from .conftest import bank, responder


def test_auc_counts_ties_as_half_and_handles_empty_sides():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
    assert auc([0.5, 0.9], [0.5, 0.1]) == pytest.approx(0.875)
    assert auc([], [0.1]) == 0.5


def test_threshold_at_fpr_matches_the_lab_definition():
    neg = [0.9, 0.5, 0.4, 0.1]
    assert threshold_at_fpr(neg, 0.0) == pytest.approx(0.9 + 1e-9)
    assert threshold_at_fpr(neg, 0.25) == pytest.approx(0.5 + 1e-9)  # one negative may pass
    # Unjudged negatives still count in the denominator: 1% of 200 allows two.
    assert threshold_at_fpr(neg, 0.01, n_neg_total=200) == pytest.approx(0.4 + 1e-9)
    assert threshold_at_fpr([0.3], 0.5, n_neg_total=10) == 0.0
    with pytest.raises(ValueError):
        threshold_at_fpr(neg, 1.0)


def test_widest_margin_never_spends_false_positives_that_buy_no_recall():
    pos, neg = [0.9, 0.8, 0.3], [0.5, 0.1, 0.05, 0.0]
    t = threshold_at_fpr(neg, 0.25)  # one negative allowed: just above 0.1, so 0.5 is flagged for nothing
    assert t == pytest.approx(0.1 + 1e-9)
    assert widest_margin(t, pos, neg) == pytest.approx(0.2)  # midway between 0.1 and the lowest catch, 0.3
    # Separable: the budget buys nothing, so the threshold sits mid-gap with no false positive.
    assert widest_margin(0.05 + 1e-9, [0.9, 0.7], [0.4, 0.05]) == pytest.approx(0.55)
    assert widest_margin(0.95, [0.9], [0.1]) == 0.95  # nothing caught: unchanged
    assert widest_margin(0.5, [0.9], [0.95]) == 0.5  # no negative below the catch: unchanged
    assert widest_margin(0.0, [0.23], [0.21]) == 0.22  # tidy, not 0.22000000000000003
    assert 0.1 < widest_margin(0.0, [0.1 + 2e-7], [0.1]) <= 0.1 + 2e-7  # too tight to round: exact midpoint


def test_rates_and_derived_metrics():
    r = rates([True, True, False, False, False], [True, False, True, False, False])
    assert (r.tp, r.fn, r.fp, r.tn) == (1, 1, 1, 2)
    assert r.recall == 0.5 and r.fpr == pytest.approx(1 / 3) and r.precision == 0.5
    assert r.f1 == pytest.approx(0.5) and r.balanced_accuracy == pytest.approx((0.5 + 2 / 3) / 2)
    assert Rates(0, 0, 0, 0).recall == 0.0
    assert pct(0.1234) == "12.3%"
    with pytest.raises(ValueError):
        rates([True], [])


@pytest.fixture
def world(examples):
    by = SplitPlan().assign(examples)
    lab = Lab(FakeSystemOne(responder), AnswerCache(None), model="fake-jev")
    tuning = by["train"] + by["val"]
    questions = bank().compiled()
    lab.ask(tuning, questions, budget=10_000)
    return by, lab.table(tuning, questions, budget=10_000)


def test_question_stats_separate_intent_from_capability(world):
    by, table = world
    stats = {s.qid: s for s in question_stats(bank().ids, by, table, 0.05)}
    assert stats["intent@reviewer"].auc["train"] > 0.9
    assert stats["capability"].auc["train"] < 0.7
    ranked = [q for q, _ in rank_by_auc(bank().ids, by["train"], table)]
    assert ranked[0] == "intent@reviewer"


def test_fit_threshold_holds_the_budget_on_train(world):
    by, table = world
    t = fit_threshold(by["train"], table, "intent@reviewer", 0.05)
    neg = [ex for ex in by["train"] if not ex.label]
    fp = sum(1 for ex in neg if (s := evaluate_score("intent@reviewer", table.of(ex.id))) is not None and s >= t)
    assert fp <= int(0.05 * len(neg))


def test_greedy_union_adds_the_lure_for_what_intent_misses(world):
    by, table = world
    union = greedy_union(bank().ids, by["train"], table, 0.05, max_questions=3)
    assert union is not None and union.family == "union"
    assert "intent@reviewer" in union.questions
    singles = {c.name: c for c in candidates(bank().ids, by["train"], table, 0.05) if c.family == "single"}
    assert union.train.recall >= singles["single:intent@reviewer"].train.recall


def test_candidates_cover_every_family_and_pareto_drops_dominated(world):
    by, table = world
    cands = candidates(bank().ids, by["train"], table, 0.05, max_questions=3)
    assert {c.family for c in cands} == {"single", "mean", "union", "mean+lure"}
    top = best_per_family(cands, k=2)
    assert len({c.family for c in top}) == 4 and len(top) <= 8
    front = pareto(cands)
    assert front and all(c.train.tp > 0 for c in front)
    assert candidates([], by["train"], table, 0.05) == []


def _reports(world, threshold: float, levels=("pass", "flag")):
    by, table = world
    ps = ProbeSet("p", bank(), "intent@reviewer", threshold)
    pol = Policy("t", lambda ex, a, s, t: levels[1] if s is not None and t is not None and s >= t else levels[0], levels)
    return {s: evaluate_split(s, by[s], table, ps, pol) for s in ("train", "val")}


def test_constraints_compare_with_the_incumbent(world):
    strict, loose = _reports(world, 0.5), _reports(world, 0.0)
    assert Constraint(max_fpr=0.01).check(strict, strict) is None
    assert "false-positive rate" in Constraint(max_fpr=0.01).check(loose, strict)
    assert "more false positives" in Constraint(max_added_false=0).check(loose, strict)
    assert "fewer true positives" in Constraint(max_lost_true=0).check(strict, loose)
    assert "recall" in Constraint(min_recall=1.01).check(strict, strict)
    assert Constraint(split="test", max_fpr=0).check(strict, strict) is None  # not scored here
    with pytest.raises(ValueError, match="levels"):
        Constraint(level="block", max_fpr=0).check(strict, strict)


def test_objective_gate_needs_a_strict_gain_and_every_constraint():
    obj = Objective.from_dict({"fpr_budget": 0.02})
    assert obj.constraints[0].max_fpr == pytest.approx(0.04)  # the default constraint


def test_objective_gate_decisions(world):
    good, weak = _reports(world, 0.5), _reports(world, 0.99)
    obj = Objective(fpr_budget=0.05)
    assert obj.gate(good, weak) == (True, [])
    ok, reasons = obj.gate(weak, good)
    assert not ok and "does not beat" in reasons[0]
    ok, reasons = Objective(constraints=(Constraint(max_fpr=-1),)).gate(good, weak)
    assert not ok and "false-positive rate" in reasons[0]


def test_objective_and_constraints_refuse_sealed_splits():
    with pytest.raises(ValueError):
        Objective(split="test")
    with pytest.raises(ValueError):
        Constraint.from_dict({"split": "held-out", "max_fpr": 0.1})
    with pytest.raises(ValueError, match="unknown"):
        Constraint.from_dict({"max_fp": 1})
    with pytest.raises(ValueError):
        Objective(fpr_budget=1.5)


def test_multi_level_policies_report_every_level(world):
    reports = _reports(world, 0.5, levels=("pass", "warn", "block"))
    assert set(reports["train"].by_level) == {"warn", "block"}
    assert Constraint(level="block", max_added_false=0).describe() == "val/block: at most 0 new false positives"
