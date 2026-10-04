"""Score a probe set, through a policy, on each split, from cached answers.

Recall is over every positive in the split: an example the pipeline never sent
(over budget, failed, no state) counts as a miss, never as excluded. AUC is over
judged examples only, since an unjudged example has no score to rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from skill_factory.classifier.dataset import Example
from skill_factory.classifier.lab import AnswerTable
from skill_factory.classifier.metrics import Rates, auc, rates, threshold_at_fpr, widest_margin
from skill_factory.classifier.policy import THRESHOLD_POLICY, Policy
from skill_factory.classifier.probeset import ProbeSet, Score, evaluate_score


@dataclass(frozen=True)
class SplitReport:
    """One configuration on one split."""

    split: str
    positives: int
    negatives: int
    judged_positives: int
    judged_negatives: int
    levels: tuple[str, ...]
    by_level: Mapping[str, Rates]
    auc: float

    def rates(self, level: str | None = None) -> Rates:
        return self.by_level[level or self.levels[1]]

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "positives": self.positives,
            "negatives": self.negatives,
            "judged_positives": self.judged_positives,
            "judged_negatives": self.judged_negatives,
            "auc": round(self.auc, 4),
            "levels": {lvl: r.to_dict() for lvl, r in self.by_level.items()},
        }


def scores_for(examples: Sequence[Example], table: AnswerTable, expr: Score) -> dict[str, float | None]:
    return {ex.id: evaluate_score(expr, table.of(ex.id)) for ex in examples}


def verdicts_for(
    examples: Sequence[Example], table: AnswerTable, probeset: ProbeSet, policy: Policy = THRESHOLD_POLICY
) -> dict[str, str]:
    out = {}
    for ex in examples:
        answers = table.of(ex.id)
        out[ex.id] = policy.verdict(ex, answers, probeset.score_of(answers), probeset.threshold)
    return out


def evaluate_split(
    split: str,
    examples: Sequence[Example],
    table: AnswerTable,
    probeset: ProbeSet,
    policy: Policy = THRESHOLD_POLICY,
) -> SplitReport:
    verdicts = verdicts_for(examples, table, probeset, policy)
    scores = scores_for(examples, table, probeset.score)
    labels = [ex.label for ex in examples]
    by_level = {
        lvl: rates(labels, [policy.at_least(verdicts[ex.id], lvl) for ex in examples])
        for lvl in policy.detect_levels
    }
    pos = [s for ex in examples if ex.label and (s := scores[ex.id]) is not None]
    neg = [s for ex in examples if not ex.label and (s := scores[ex.id]) is not None]
    return SplitReport(
        split=split,
        positives=sum(labels),
        negatives=len(labels) - sum(labels),
        judged_positives=len(pos),
        judged_negatives=len(neg),
        levels=policy.levels,
        by_level=by_level,
        auc=auc(pos, neg),
    )


def fit_threshold(
    examples: Sequence[Example], table: AnswerTable, expr: Score, fpr_budget: float, *, margin: bool = True
) -> float:
    """The threshold for ``expr`` at ``fpr_budget`` false positives, fitted on ``examples`` (train).

    With ``margin`` (the default, for thresholds that ship), the budget is spent only where it
    buys recall and the threshold sits mid-gap (``widest_margin``). Without it, the threshold is
    the lowest the budget allows: skill-scanner's definition, kept for per-question diagnostics.
    """
    negatives = [ex for ex in examples if not ex.label]
    scored = [s for ex in negatives if (s := evaluate_score(expr, table.of(ex.id))) is not None]
    t = threshold_at_fpr(scored, fpr_budget, n_neg_total=len(negatives))
    if not margin:
        return t
    pos = [s for ex in examples if ex.label and (s := evaluate_score(expr, table.of(ex.id))) is not None]
    return widest_margin(t, pos, scored)


@dataclass(frozen=True)
class QuestionStat:
    """How well one question separates, per split, with its threshold fitted on train."""

    qid: str
    threshold: float
    auc: Mapping[str, float]
    recall: Mapping[str, float]
    fpr: Mapping[str, float]


def question_stats(
    qids: Sequence[str],
    by_split: Mapping[str, Sequence[Example]],
    table: AnswerTable,
    fpr_budget: float,
    splits: Sequence[str] = ("train", "val"),
) -> list[QuestionStat]:
    out = []
    train = by_split["train"]
    for qid in qids:
        t = fit_threshold(train, table, qid, fpr_budget, margin=False)
        aucs, recalls, fprs = {}, {}, {}
        for split in splits:
            exs = by_split.get(split, [])
            ps = [(ex.label, table.p(ex.id, qid)) for ex in exs]
            aucs[split] = auc([p for y, p in ps if y and p is not None], [p for y, p in ps if not y and p is not None])
            r = rates([y for y, _ in ps], [p is not None and p >= t for _, p in ps])
            recalls[split], fprs[split] = r.recall, r.fpr
        out.append(QuestionStat(qid, t, aucs, recalls, fprs))
    return out
