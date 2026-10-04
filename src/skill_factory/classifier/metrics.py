"""Dataset-level metrics: none of these is an average of per-example scores.

* AUC, the chance a positive outscores a negative (ties count half).
* The threshold that keeps the false-positive rate within a budget, and the
  recall it buys: "recall at 1% false flags".
* Confusion counts and rates at a threshold, per verdict level.

Pure functions over plain numbers, so the optimizer, the report and the tests
share one definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: Added to a negative's score to put the threshold just above it.
EPSILON = 1e-9


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Area under the ROC curve via ranks (Mann-Whitney U); 0.5 when either side is empty."""
    if not pos or not neg:
        return 0.5
    tagged = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = 0.0
    i = 0
    while i < len(tagged):
        j = i
        while j < len(tagged) and tagged[j][0] == tagged[i][0]:
            j += 1
        mid_rank = (i + 1 + j) / 2  # average rank of the tie block, 1-based
        rank_sum += mid_rank * sum(1 for k in range(i, j) if tagged[k][1] == 1)
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def threshold_at_fpr(neg_scores: Sequence[float], rate: float, n_neg_total: int | None = None) -> float:
    """The lowest threshold whose false-positive rate stays within ``rate``.

    ``n_neg_total`` counts negatives that have no score (never judged); they can
    never be flagged but they are still negatives. The threshold sits just above
    the first negative that would exceed the allowance. This is the definition
    skill-scanner's lab used, so its thresholds reproduce here.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("rate must be in [0, 1)")
    ranked = sorted(neg_scores, reverse=True)
    total = len(ranked) if n_neg_total is None else n_neg_total
    allowed = math.floor(rate * total)
    if allowed >= len(ranked):
        return 0.0
    return ranked[allowed] + EPSILON


def widest_margin(threshold: float, pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Move ``threshold`` to the middle of the gap it sits in, without losing a caught positive.

    ``threshold_at_fpr`` returns the lowest threshold the budget allows, so it spends
    false positives even when they buy no recall (on separable data it flags a negative
    for nothing). This puts the threshold halfway between the lowest positive caught at
    ``threshold`` and the highest negative below that positive: on the scores it was fitted
    on, recall can only rise and false positives can only fall, and new examples get a
    margin on both sides.
    """
    caught = [s for s in pos_scores if s >= threshold]
    if not caught:
        return threshold
    lowest_caught = min(caught)
    below = [s for s in neg_scores if s < lowest_caught]
    if not below:
        return threshold
    highest_below = max(below)
    mid = (highest_below + lowest_caught) / 2
    tidy = round(mid, 6)  # 0.22, not 0.22000000000000003, when that stays inside the gap
    return tidy if highest_below < tidy <= lowest_caught else mid


@dataclass(frozen=True)
class Rates:
    """Confusion counts and the rates derived from them."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.fp + self.tn

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.positives)

    @property
    def fpr(self) -> float:
        return _ratio(self.fp, self.negatives)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def specificity(self) -> float:
        return _ratio(self.tn, self.negatives)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _ratio(2 * p * r, p + r)

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2

    def to_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "recall": round(self.recall, 4), "fpr": round(self.fpr, 4), "precision": round(self.precision, 4),
        }


def rates(labels: Sequence[bool], flagged: Sequence[bool]) -> Rates:
    if len(labels) != len(flagged):
        raise ValueError("labels and flags differ in length")
    tp = sum(1 for y, f in zip(labels, flagged) if y and f)
    fn = sum(1 for y, f in zip(labels, flagged) if y and not f)
    fp = sum(1 for y, f in zip(labels, flagged) if not y and f)
    return Rates(tp=tp, fp=fp, fn=fn, tn=len(labels) - tp - fn - fp)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ratio(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den
