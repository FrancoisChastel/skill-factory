"""Noise and drift: is a threshold of 0.075 meaningful, and does it survive a new model?

* ``determinism`` re-asks the same questions (a new replicate in the cache) and
  measures how many answers came back identical, the mean and largest change,
  and how many verdicts flipped at the threshold. jev asked twice about 505
  validation skills: 81% identical, mean difference 0.004, 3 verdicts flipped.
* ``recalibrate`` refits a probe set's threshold on train for another model (or
  model version) at the false-positive budget it was calibrated at, and reports
  how validation moved. Thresholds are pinned to the version that answered
  (``pinned_version``), and a lab answering as another version is a drift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from skill_factory.classifier.dataset import Example
from skill_factory.classifier.evaluation import SplitReport, evaluate_split, fit_threshold
from skill_factory.classifier.lab import AnswerTable, Lab
from skill_factory.classifier.policy import THRESHOLD_POLICY, Policy
from skill_factory.classifier.probeset import ProbeSet


@dataclass(frozen=True)
class Determinism:
    answers: int
    identical: int
    mean_abs_diff: float
    max_abs_diff: float
    examples: int
    flipped: tuple[str, ...]
    failed_requests: int = 0

    @property
    def identical_rate(self) -> float:
        return self.identical / self.answers if self.answers else 0.0

    def summary(self) -> str:
        return (
            f"{self.answers} answers re-asked: {self.identical_rate:.0%} identical, mean |diff| "
            f"{self.mean_abs_diff:.4f}, max {self.max_abs_diff:.2f}; {len(self.flipped)} of {self.examples} "
            "verdicts flipped at the threshold"
            + (f"; WARNING: {self.failed_requests} requests failed, so fewer answers were compared"
               if self.failed_requests else "")
        )


def determinism(
    lab: Lab, probeset: ProbeSet, examples: Sequence[Example], *, replicate: int = 1, ask: bool = True
) -> Determinism:
    """Compare replicate 0 (the cache) with ``replicate`` (asked now unless already cached)."""
    if replicate < 1:
        raise ValueError("replicate must be >= 1 (0 is the original answers)")
    questions = probeset.bank.compiled()
    budget = probeset.state_budget
    failed = 0
    if ask:
        failed += lab.ask(examples, questions, budget=budget).failed  # the originals, if any are missing
        failed += lab.ask(examples, questions, budget=budget, replicate=replicate).failed
    first = lab.table(examples, questions, budget=budget)
    again = lab.table(examples, questions, budget=budget, replicate=replicate)
    return replace(compare_tables(first, again, probeset, examples), failed_requests=failed)


def compare_tables(
    first: AnswerTable, again: AnswerTable, probeset: ProbeSet, examples: Sequence[Example]
) -> Determinism:
    diffs: list[float] = []
    flipped = []
    judged = 0
    for ex in examples:
        a, b = first.of(ex.id), again.of(ex.id)
        for qid in set(a) & set(b):
            diffs.append(abs(a[qid] - b[qid]))
        if a and b:
            judged += 1
            if probeset.flags(a) != probeset.flags(b):
                flipped.append(ex.id)
    return Determinism(
        answers=len(diffs),
        identical=sum(1 for d in diffs if d == 0),
        mean_abs_diff=sum(diffs) / len(diffs) if diffs else 0.0,
        max_abs_diff=max(diffs, default=0.0),
        examples=judged,
        flipped=tuple(flipped),
    )


@dataclass(frozen=True)
class Recalibration:
    old: ProbeSet
    new: ProbeSet
    before: Mapping[str, SplitReport]
    after: Mapping[str, SplitReport]

    def summary(self, level: str | None = None) -> str:
        lines = [f"threshold {self.old.threshold} ({self.old.model}) -> {self.new.threshold:.4f} ({self.new.model})"]
        for split in self.after:
            b, a = self.before[split].rates(level), self.after[split].rates(level)
            lines.append(f"{split}: recall {b.recall:.1%} -> {a.recall:.1%}, FPR {b.fpr:.2%} -> {a.fpr:.2%}")
        return "\n".join(lines)


def recalibrate(
    probeset: ProbeSet,
    old_lab: Lab,
    new_lab: Lab,
    by_split: Mapping[str, Sequence[Example]],
    *,
    fpr_budget: float | None = None,
    policy: Policy = THRESHOLD_POLICY,
    ask: bool = True,
) -> Recalibration:
    """Refit the threshold on train under ``new_lab``'s model at the calibrated budget."""
    budget_rate = fpr_budget if fpr_budget is not None else probeset.calibration.get("fpr_budget")
    if budget_rate is None:
        raise ValueError("the probe set has no calibration.fpr_budget; pass fpr_budget")
    train, val = list(by_split["train"]), list(by_split.get("val", []))
    questions = probeset.bank.compiled()
    examples = train + val
    if ask:
        report = new_lab.ask(examples, questions, budget=probeset.state_budget)
        if report.failed:
            # A threshold fitted on a partial cache would ship looser or stricter than measured.
            raise RuntimeError(
                f"{report.failed} requests to {new_lab.model} failed; not refitting on incomplete answers "
                f"(first error: {report.errors[0] if report.errors else 'unknown'}). Rerun to fill the cache."
            )
    old_table = old_lab.table(examples, questions, budget=probeset.state_budget)
    new_table = new_lab.table(examples, questions, budget=probeset.state_budget)
    threshold = fit_threshold(train, new_table, probeset.score, float(budget_rate))
    versions = sorted(new_table.versions, key=lambda v: -new_table.versions[v])
    new = probeset.with_changes(
        threshold=threshold,
        model=new_lab.model,
        pinned_version=versions[0] if versions else None,
        calibration={**probeset.calibration, "fpr_budget": float(budget_rate),
                     "recalibrated_from": {"model": probeset.model, "version": probeset.pinned_version,
                                           "threshold": probeset.threshold}},
    )
    splits = {"train": train, "val": val} if val else {"train": train}
    before = {s: evaluate_split(s, exs, old_table, probeset, policy) for s, exs in splits.items()}
    after = {s: evaluate_split(s, exs, new_table, new, policy) for s, exs in splits.items()}
    return Recalibration(probeset, new, before, after)


def version_drift(probeset: ProbeSet, table: AnswerTable) -> str | None:
    """A warning when cached answers came from a version other than the one the threshold is pinned to."""
    if not probeset.pinned_version or not table.versions:
        return None
    others = {v: n for v, n in table.versions.items() if v != probeset.pinned_version}
    if not others:
        return None
    seen = ", ".join(f"{v} ({n})" for v, n in sorted(others.items()))
    return (
        f"thresholds are pinned to {probeset.pinned_version}, but answers came from {seen}: "
        "run `skill-factory lab recalibrate` before trusting the threshold"
    )
