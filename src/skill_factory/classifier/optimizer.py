"""The classifier optimizer: skill-factory's reflective loop, applied to a probe set.

    seed      score the seed probe set on train and validation (thresholds fitted on train)
    select    fit every candidate over the question pool (selection.py); the best of
              each family goes to the gate
    reflect   a proposer LLM reads the failure digest (system causes, separations,
              misses and false flags with every P) and proposes questions, framings
              or a pipeline change
    ask       the lab asks only the new questions (the rest is cached)
    gate      keep a candidate only if it beats the incumbent on validation recall
              and breaks no constraint. Validation picks among a few fitted candidates
              per round (the best of each family, at the budget and at half of it), so
              its numbers are optimistic; test and held-out are the unbiased ones.

Test and held-out examples are never passed in: the optimizer cannot see them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from skill_factory.classifier.dataset import Example
from skill_factory.classifier.evaluation import (
    QuestionStat,
    SplitReport,
    evaluate_split,
    fit_threshold,
    question_stats,
)
from skill_factory.classifier.lab import AnswerTable, AskReport, Lab
from skill_factory.classifier.objective import Objective
from skill_factory.classifier.policy import THRESHOLD_POLICY, Policy
from skill_factory.classifier.probeset import ProbeSet, describe_score
from skill_factory.classifier.reflect import (
    REFLECT_SYSTEM,
    failure_digest,
    parse_proposal,
    proposal_prompt,
)
from skill_factory.classifier.questions import QuestionBank
from skill_factory.classifier.select import Candidate, best_per_family, candidates, pareto
from skill_factory.llm.client import LLMClient

TUNING_SPLITS = ("train", "val")


@dataclass(frozen=True)
class RoundRecord:
    """One round: what changed, why, how it scored, and whether the gate kept it."""

    round: int
    change: str
    rationale: str
    candidate: str
    score: str
    threshold: float | None
    reports: Mapping[str, SplitReport]
    accepted: bool
    reasons: tuple[str, ...] = ()
    new_questions: tuple[str, ...] = ()
    state_budget: int | None = None
    usd: float = 0.0
    failed_requests: int = 0
    #: The proposer's reply, kept only when it could not be used, to see why.
    raw_reply: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "change": self.change,
            "rationale": self.rationale,
            "candidate": self.candidate,
            "score": self.score,
            "threshold": self.threshold,
            "state_budget": self.state_budget,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "new_questions": list(self.new_questions),
            "usd": round(self.usd, 6),
            "failed_requests": self.failed_requests,
            "reports": {s: r.to_dict() for s, r in self.reports.items()},
            **({"raw_reply": self.raw_reply} if self.raw_reply else {}),
        }


@dataclass(frozen=True)
class ClassifierResult:
    seed: ProbeSet
    best: ProbeSet
    before: Mapping[str, SplitReport]
    after: Mapping[str, SplitReport]
    history: tuple[RoundRecord, ...]
    pool: QuestionBank
    front: tuple[Candidate, ...] = ()
    stats: tuple[QuestionStat, ...] = ()
    usd: float = 0.0
    input_tokens: int = 0
    objective: Objective = field(default_factory=Objective)
    policy_name: str = "threshold"
    #: Requests that failed during the run: those answers are missing, and the examples they
    #: were about count as unanswered (misses), so the report says how many.
    failed_requests: int = 0
    errors: tuple[str, ...] = ()

    @property
    def baseline_score(self) -> float:
        return self.objective.value(self.before)

    @property
    def best_score(self) -> float:
        return self.objective.value(self.after)


class ClassifierOptimizer:
    name = "classifier"

    def __init__(
        self,
        lab: Lab,
        objective: Objective,
        *,
        proposer: LLMClient | None = None,
        policy: Policy = THRESHOLD_POLICY,
        rounds: int = 4,
        patience: int = 2,
        max_questions: int = 6,
        max_new_questions: int = 8,
        max_state_budget: int | None = None,
        goal: str = "",
        keep_questions: Sequence[str] | None = None,
        margins: Sequence[float] = (1.0, 0.5),
        gate_per_family: int = 3,
        log: Any = None,
    ):
        if rounds < 0 or patience < 1:
            raise ValueError("rounds must be >= 0 and patience >= 1")
        self.lab = lab
        self.objective = objective
        self.proposer = proposer
        self.policy = policy
        self.rounds = rounds
        self.patience = patience
        self.max_questions = max_questions
        self.max_new_questions = max_new_questions
        self.max_state_budget = max_state_budget
        self.goal = goal
        self.keep_questions = keep_questions
        # Each family is also fitted at a fraction of the budget: a threshold with slack on
        # train is likelier to hold its false-positive rate on validation.
        self.margins = tuple(margins)
        self.gate_per_family = gate_per_family
        self._log = log or (lambda msg: None)
        self._usd = 0.0
        self._tokens = 0
        self._failed = 0
        self._errors: list[str] = []

    def optimize(
        self, seed: ProbeSet, by_split: Mapping[str, Sequence[Example]], pool: QuestionBank | None = None
    ) -> ClassifierResult:
        train, val = list(by_split.get("train", [])), list(by_split.get("val", []))
        if not train or not val:
            raise ValueError("classifier optimization needs examples in both train and val")
        tuning = train + val
        keep = list(self.keep_questions) if self.keep_questions is not None else [
            q for q in seed.bank.ids if q not in seed.score_ids
        ]
        pool = seed.bank.merge(pool) if pool is not None else seed.bank
        budget = seed.state_budget

        table = self._ask_and_read(tuning, pool, budget)
        if seed.threshold is None:
            seed = seed.with_changes(threshold=fit_threshold(train, table, seed.score, self.objective.fpr_budget))
        best = seed
        best_reports = self._reports(best, by_split, table)
        before = best_reports
        history = [RoundRecord(0, "seed", "the seed probe set", "seed", describe_score(seed.score),
                               seed.threshold, best_reports, True, state_budget=budget)]
        self._log(f"seed: {self._fmt(best_reports)}")

        front: list[Candidate] = []
        best, best_reports, record, front = self._try_pool(
            1, "selection", "fit every candidate over the question pool", pool, keep, budget,
            table, by_split, best, best_reports, (),
        )
        history.append(record)

        stale = 0
        for r in range(2, self.rounds + 2 if self.proposer is not None else 2):
            stats = question_stats(pool.ids, by_split, table, self.objective.fpr_budget, splits=("train",))
            stats = sorted(stats, key=lambda s: -s.auc["train"])
            digest = failure_digest(best, train, table, policy=self.policy, stats=stats,
                                    max_state_budget=self.max_state_budget)
            prompt = proposal_prompt(best, digest, goal=self.goal, frames=pool.frames,
                                     max_new=self.max_new_questions, max_state_budget=self.max_state_budget)
            assert self.proposer is not None
            raw = self.proposer.generate(prompt, system=REFLECT_SYSTEM, temperature=0.4, max_tokens=4096)
            try:
                proposal = parse_proposal(raw, pool, max_new=self.max_new_questions,
                                          max_state_budget=self.max_state_budget)
            except ValueError as exc:
                self._log(f"round {r}: unusable proposal ({exc}); the reply is kept in history.json")
                history.append(RoundRecord(r, "proposal", f"unusable proposal: {exc}", "-", "-", None,
                                           best_reports, False, (str(exc),), raw_reply=(raw or "")[:4000]))
                stale += 1
                if stale >= self.patience:
                    break
                continue
            pool = pool.merge(proposal.bank)
            new_budget = int(proposal.pipeline.get("state_budget", budget))
            change = _describe_change(proposal.new_ids, budget, new_budget)
            spent_before, failed_before = self._usd, self._failed
            table = self._ask_and_read(tuning, pool, new_budget)
            best, best_reports, record, front = self._try_pool(
                r, change, proposal.rationale, pool, keep, new_budget, table, by_split,
                best, best_reports, tuple(proposal.new_ids), spent_before, failed_before,
            )
            history.append(record)
            if record.accepted:
                budget = new_budget
                stale = 0
            else:
                table = self._ask_and_read(tuning, pool, budget) if new_budget != budget else table
                stale += 1
            if stale >= self.patience:
                break

        final_table = self.lab.table(tuning, pool.compiled(), budget=best.state_budget)
        stats = question_stats(pool.ids, by_split, final_table, self.objective.fpr_budget, splits=TUNING_SPLITS)
        best = best.with_changes(
            pinned_version=_served_version(final_table) or best.pinned_version,
            calibration={
                "fpr_budget": self.objective.fpr_budget,
                "fitted_on": "train",
                "policy": self.policy.name,
                "train": best_reports["train"].rates(self.objective.level).to_dict(),
                "val": best_reports["val"].rates(self.objective.level).to_dict(),
            },
        )
        return ClassifierResult(
            seed=seed, best=best, before=before, after=best_reports, history=tuple(history), pool=pool,
            front=tuple(pareto(front)), stats=tuple(sorted(stats, key=lambda s: -s.auc["train"])),
            usd=self._usd, input_tokens=self._tokens, objective=self.objective, policy_name=self.policy.name,
            failed_requests=self._failed, errors=tuple(self._errors),
        )

    # --- steps ------------------------------------------------------------

    def _ask_and_read(self, examples: Sequence[Example], pool: QuestionBank, budget: int) -> AnswerTable:
        questions = pool.compiled()
        # Always through ask(): an offline lab answers from the cache, and raises when an
        # answer is missing that only a call could buy, instead of scoring it as a miss.
        report: AskReport = self.lab.ask(examples, questions, budget=budget)
        self._usd += report.usd
        self._tokens += report.input_tokens
        self._failed += report.failed
        self._errors.extend(report.errors[: max(0, 10 - len(self._errors))])
        if report.requests or report.failed or report.not_sent_over_cap:
            self._log(f"lab: {report.summary()}")
        if report.failed:
            self._log(f"lab: {report.failed} requests failed, e.g. {report.errors[0] if report.errors else '?'}")
        return self.lab.table(examples, questions, budget=budget)

    def _reports(self, probeset: ProbeSet, by_split: Mapping[str, Sequence[Example]], table: AnswerTable):
        return {s: evaluate_split(s, list(by_split[s]), table, probeset, self.policy) for s in TUNING_SPLITS}

    def _try_pool(
        self, round_no: int, change: str, rationale: str, pool: QuestionBank, keep: Sequence[str], budget: int,
        table: AnswerTable, by_split: Mapping[str, Sequence[Example]], best: ProbeSet,
        best_reports: Mapping[str, SplitReport], new_ids: tuple[str, ...], spent_before: float = 0.0,
        failed_before: int = 0,
    ) -> tuple[ProbeSet, Mapping[str, SplitReport], RoundRecord, list[Candidate]]:
        scored = [q for q in pool.ids if q not in keep]
        cands = []
        for margin in self.margins:
            for c in candidates(scored, list(by_split["train"]), table, self.objective.fpr_budget * margin,
                                max_questions=self.max_questions):
                cands.append(replace(c, name=c.name if margin == 1 else f"{c.name} @{margin:g}x budget"))
        tried: list[tuple[float, ProbeSet, Mapping[str, SplitReport], Candidate, list[str]]] = []
        for cand in best_per_family(cands, k=self.gate_per_family):
            probeset = best.with_changes(
                bank=pool.subset(set(cand.questions) | set(keep)),
                score=cand.score,
                threshold=cand.threshold,
                state_budget=budget,
            )
            reports = self._reports(probeset, by_split, table)
            ok, reasons = self.objective.gate(reports, best_reports)
            tried.append((self.objective.value(reports) if ok else -1.0, probeset, reports, cand, reasons))
        usd, failed = self._usd - spent_before, self._failed - failed_before
        if not tried:
            return best, best_reports, RoundRecord(round_no, change, rationale, "-", "-", None, best_reports,
                                                   False, ("no candidate",), new_ids, budget, usd, failed), cands
        tried.sort(key=lambda t: t[0], reverse=True)
        value, probeset, reports, cand, reasons = tried[0]
        accepted = value >= 0
        if not accepted:
            # Report the candidate that did best on train, with every reason it was refused.
            _, probeset, reports, cand, reasons = max(tried, key=lambda t: t[3].rank_key)
        record = RoundRecord(round_no, change, rationale, cand.name, describe_score(cand.score), cand.threshold,
                             reports, accepted, tuple(reasons), new_ids, budget, usd, failed)
        self._log(f"round {round_no} ({change}): {cand.name} {self._fmt(reports)} "
                  f"{'kept' if accepted else 'refused: ' + '; '.join(reasons)}")
        if accepted:
            return probeset, reports, record, cands
        return best, best_reports, record, cands

    def _fmt(self, reports: Mapping[str, SplitReport]) -> str:
        return "  ".join(
            f"{s} {r.rates(self.objective.level).recall:.1%}@{r.rates(self.objective.level).fpr:.2%}"
            for s, r in reports.items()
        )


def _describe_change(new_ids: Sequence[str], old_budget: int, new_budget: int) -> str:
    parts = []
    if new_ids:
        parts.append(f"+{len(new_ids)} question{'s' if len(new_ids) != 1 else ''}")
    if new_budget != old_budget:
        parts.append(f"state budget {old_budget:,} -> {new_budget:,}")
    return ", ".join(parts) or "no change"


def _served_version(table: AnswerTable) -> str | None:
    if not table.versions:
        return None
    return Counter(table.versions).most_common(1)[0][0]
