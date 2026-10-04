"""The gate: maximize recall at a false-positive budget, subject to constraints.

A scalar is not enough to keep a classifier honest. The jev rules were tuned
under "no new block on any split, at most 4 more benign warnings on training and
2 on validation"; a candidate that wins on recall and breaks one of those is
rejected, and the report says which constraint it broke.

    objective:
      fpr_budget: 0.005          # thresholds are fitted on train at this rate
      split: val                 # recall on this split decides
      min_delta: 0.0
      constraints:
        - {split: val, level: flag, max_fpr: 0.01}
        - {split: val, level: block, max_added_false: 0}
        - {split: train, level: warn, max_added_false: 4}

``max_added_false`` and ``max_lost_true`` compare with the incumbent (the current
best), so they read as "no new false blocks" and "lose no catch".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from skill_factory.classifier.evaluation import SplitReport


@dataclass(frozen=True)
class Constraint:
    split: str = "val"
    level: str | None = None
    max_fpr: float | None = None
    min_recall: float | None = None
    max_added_false: int | None = None
    max_lost_true: int | None = None

    def check(self, candidate: Mapping[str, SplitReport], incumbent: Mapping[str, SplitReport]) -> str | None:
        """None when satisfied, else why not."""
        if self.split not in candidate:
            return None  # the split is not scored in this run (sealed or empty)
        cand = _rates(candidate[self.split], self.level)
        where = f"{self.split}/{self.level or candidate[self.split].levels[1]}"
        if self.max_fpr is not None and cand.fpr > self.max_fpr + 1e-12:
            return f"{where}: false-positive rate {cand.fpr:.2%} > {self.max_fpr:.2%}"
        if self.min_recall is not None and cand.recall < self.min_recall - 1e-12:
            return f"{where}: recall {cand.recall:.2%} < {self.min_recall:.2%}"
        if self.split in incumbent:
            inc = _rates(incumbent[self.split], self.level)
            if self.max_added_false is not None and cand.fp - inc.fp > self.max_added_false:
                return f"{where}: {cand.fp - inc.fp} more false positives (at most {self.max_added_false})"
            if self.max_lost_true is not None and inc.tp - cand.tp > self.max_lost_true:
                return f"{where}: {inc.tp - cand.tp} fewer true positives (at most {self.max_lost_true})"
        return None

    def describe(self) -> str:
        parts = []
        if self.max_fpr is not None:
            parts.append(f"FPR <= {self.max_fpr:.2%}")
        if self.min_recall is not None:
            parts.append(f"recall >= {self.min_recall:.2%}")
        if self.max_added_false is not None:
            parts.append(f"at most {self.max_added_false} new false positives")
        if self.max_lost_true is not None:
            parts.append(f"at most {self.max_lost_true} lost catches")
        return f"{self.split}/{self.level or 'detect'}: " + ", ".join(parts)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Constraint":
        known = {"split", "level", "max_fpr", "min_recall", "max_added_false", "max_lost_true"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown constraint keys: {sorted(unknown)}")
        c = cls(
            split=str(data.get("split", "val")),
            level=data.get("level"),
            max_fpr=_opt_float(data.get("max_fpr")),
            min_recall=_opt_float(data.get("min_recall")),
            max_added_false=_opt_int(data.get("max_added_false")),
            max_lost_true=_opt_int(data.get("max_lost_true")),
        )
        if c.split in ("test", "held-out"):
            raise ValueError("constraints cannot read sealed splits (test, held-out)")
        return c


@dataclass(frozen=True)
class Objective:
    fpr_budget: float = 0.01
    split: str = "val"
    level: str | None = None
    min_delta: float = 0.0
    constraints: tuple[Constraint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.fpr_budget < 1.0:
            raise ValueError("fpr_budget must be in [0, 1)")
        if self.split in ("test", "held-out"):
            raise ValueError("the objective cannot read a sealed split")

    def value(self, reports: Mapping[str, SplitReport]) -> float:
        return _rates(reports[self.split], self.level).recall

    def gate(
        self, candidate: Mapping[str, SplitReport], incumbent: Mapping[str, SplitReport]
    ) -> tuple[bool, list[str]]:
        reasons = [r for c in self.constraints if (r := c.check(candidate, incumbent))]
        new, old = self.value(candidate), self.value(incumbent)
        if new <= old + self.min_delta:
            reasons.append(f"{self.split} recall {new:.1%} does not beat {old:.1%}")
        return (not reasons, reasons)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Objective":
        data = dict(data or {})
        fpr_budget = float(data.get("fpr_budget", 0.01))
        raw = data.get("constraints")
        if raw is None:
            # Without stated constraints, hold the validation false-positive rate to twice the budget:
            # a threshold fitted on train that does not carry to validation is the commonest overfit.
            constraints: tuple[Constraint, ...] = (Constraint(split="val", max_fpr=2 * fpr_budget),)
        else:
            constraints = tuple(Constraint.from_dict(c) for c in raw)
        return cls(
            fpr_budget=fpr_budget,
            split=str(data.get("split", "val")),
            level=data.get("level"),
            min_delta=float(data.get("min_delta", 0.0)),
            constraints=constraints,
        )


def _rates(report: SplitReport, level: str | None):
    if level is not None and level not in report.by_level:
        raise ValueError(f"level {level!r} is not one of the policy's levels {report.levels[1:]}")
    return report.rates(level)


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _opt_int(v: Any) -> int | None:
    return None if v is None else int(v)
