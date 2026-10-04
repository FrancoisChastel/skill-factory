"""Selection and ensembling: turn a pool of answered questions into a score and a threshold.

Generating questions is cheap once answers are cached; choosing among them is
where the gains are. Every candidate is fitted on train only, at the same
false-positive budget, so recall is comparable across them:

    single        one question, thresholded
    mean-top-k    the mean of the k best questions by training AUC, one framing each
    union         greedy forward selection: each step adds the question (and its
                  threshold) that catches the most new training positives while the
                  union stays within the budget (skill-scanner's select.py)
    mean+lure     a mean-top-k, averaged with the one question that adds the most:
                  the form jev shipped, half the mean of five intent questions plus
                  half the fake-prerequisite lure the others do not see

Single questions with a very low threshold often hold their false-positive rate
on train and lose it on validation; an averaged score moves less. That is why
the optimizer takes the best candidate of each family to the validation gate
instead of the best one overall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from skill_factory.classifier.dataset import Example
from skill_factory.classifier.evaluation import fit_threshold, scores_for
from skill_factory.classifier.lab import AnswerTable
from skill_factory.classifier.metrics import Rates, auc, rates
from skill_factory.classifier.probeset import Score

#: Thresholds of a union are expressed as max(p_i / t_i) >= 1; this absorbs float error.
UNION_THRESHOLD = 1.0 - 1e-9


@dataclass(frozen=True)
class Candidate:
    """A fitted score: expression, threshold, the questions it reads, and how it did on train."""

    name: str
    family: str
    score: Score
    threshold: float
    questions: tuple[str, ...]
    train: Rates
    train_auc: float

    @property
    def rank_key(self) -> tuple[float, int, float]:
        # More training recall; then fewer questions (cheaper, less to overfit); then better AUC.
        return (self.train.recall, -len(self.questions), self.train_auc)


def fit(
    name: str, family: str, expr: Score, questions: Sequence[str],
    train: Sequence[Example], table: AnswerTable, fpr_budget: float, threshold: float | None = None,
) -> Candidate:
    t = fit_threshold(train, table, expr, fpr_budget) if threshold is None else threshold
    scores = scores_for(train, table, expr)
    labels = [ex.label for ex in train]
    flagged = [(s := scores[ex.id]) is not None and s >= t for ex in train]
    pos = [s for ex in train if ex.label and (s := scores[ex.id]) is not None]
    neg = [s for ex in train if not ex.label and (s := scores[ex.id]) is not None]
    return Candidate(name, family, expr, t, tuple(questions), rates(labels, flagged), auc(pos, neg))


def rank_by_auc(qids: Sequence[str], train: Sequence[Example], table: AnswerTable) -> list[tuple[str, float]]:
    out = []
    for qid in qids:
        pos = [p for ex in train if ex.label and (p := table.p(ex.id, qid)) is not None]
        neg = [p for ex in train if not ex.label and (p := table.p(ex.id, qid)) is not None]
        out.append((qid, auc(pos, neg)))
    return sorted(out, key=lambda x: -x[1])


def greedy_union(
    qids: Sequence[str], train: Sequence[Example], table: AnswerTable, fpr_budget: float, max_questions: int
) -> Candidate | None:
    """Forward selection of (question, threshold) pairs whose union stays within the budget on train."""
    benign = [ex for ex in train if not ex.label]
    malicious = [ex for ex in train if ex.label]
    allowed = int(fpr_budget * len(benign))

    def p(ex: Example, qid: str) -> float:
        return table.p(ex.id, qid) or 0.0

    def hit(ex: Example, policy: Mapping[str, float]) -> bool:
        return any(p(ex, q) >= t for q, t in policy.items())

    def best_threshold(policy: Mapping[str, float], qid: str) -> float | None:
        already = {ex.id for ex in benign if hit(ex, policy)}
        best = None
        for t in sorted({round(p(ex, qid), 4) for ex in train} | {1.01}, reverse=True):
            fp = sum(1 for ex in benign if ex.id in already or p(ex, qid) >= t)
            if fp > allowed:
                break
            best = t
        return best

    policy: dict[str, float] = {}
    for _ in range(max_questions):
        choice: tuple[int, str, float] | None = None
        for qid in qids:
            if qid in policy:
                continue
            t = best_threshold(policy, qid)
            if t is None or t > 1 or t <= 0:
                continue
            tp = sum(1 for ex in malicious if hit(ex, {**policy, qid: t}))
            if choice is None or tp > choice[0]:
                choice = (tp, qid, t)
        current = sum(1 for ex in malicious if hit(ex, policy))
        if choice is None or (policy and choice[0] <= current):
            break
        policy[choice[1]] = choice[2]
    if not policy:
        return None
    expr: Score = {"max": [{"scale": [q, round(1.0 / t, 10)]} for q, t in policy.items()]}
    return fit(f"union-{len(policy)}", "union", expr, list(policy), train, table, fpr_budget, UNION_THRESHOLD)


def candidates(
    qids: Sequence[str],
    train: Sequence[Example],
    table: AnswerTable,
    fpr_budget: float,
    *,
    max_questions: int = 6,
) -> list[Candidate]:
    """Every candidate of every family, fitted on train at ``fpr_budget``."""
    if not qids:
        return []
    out = [fit(f"single:{q}", "single", q, [q], train, table, fpr_budget) for q in qids]
    # Means take one framing per question: two framings of one question mostly agree.
    distinct = _one_per_base([q for q, _ in rank_by_auc(qids, train, table)])
    means = []
    for k in range(2, min(max_questions, len(distinct)) + 1):
        top = distinct[:k]
        means.append(fit(f"mean-top{k}", "mean", {"mean": top}, top, train, table, fpr_budget))
    out.extend(means)
    union = greedy_union(qids, train, table, fpr_budget, max_questions)
    if union is not None:
        out.append(union)
    lured = [
        fit(f"{m.name}+lure:{q}", "mean+lure", {"mean": [m.score, q]}, [*m.questions, q], train, table, fpr_budget)
        for m in means
        if len(m.questions) < max_questions
        for q in qids
        if _base(q) not in {_base(x) for x in m.questions}
    ]
    out.extend(sorted(lured, key=lambda c: c.rank_key, reverse=True)[:3])
    return out


def _base(qid: str) -> str:
    return qid.split("@", 1)[0]


def _one_per_base(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for q in ranked:
        if _base(q) not in seen:
            seen.add(_base(q))
            out.append(q)
    return out


def best_per_family(cands: Sequence[Candidate], k: int = 1) -> list[Candidate]:
    """The ``k`` best candidates of each family by training rank, best first."""
    by_family: dict[str, list[Candidate]] = {}
    for c in cands:
        by_family.setdefault(c.family, []).append(c)
    out = [c for group in by_family.values() for c in sorted(group, key=lambda c: c.rank_key, reverse=True)[:k]]
    return sorted(out, key=lambda c: c.rank_key, reverse=True)


def pareto(cands: Sequence[Candidate]) -> list[Candidate]:
    """Candidates no other beats on all of: training recall, training false-positive rate, question count."""

    def dominates(a: Candidate, b: Candidate) -> bool:
        ge = (a.train.recall >= b.train.recall and a.train.fpr <= b.train.fpr
              and len(a.questions) <= len(b.questions))
        gt = (a.train.recall > b.train.recall or a.train.fpr < b.train.fpr
              or len(a.questions) < len(b.questions))
        return ge and gt

    useful = [c for c in cands if c.train.tp > 0]
    front = [c for c in useful if not any(dominates(o, c) for o in useful if o is not c)]
    return sorted(front, key=lambda c: (len(c.questions), -c.train.recall))
