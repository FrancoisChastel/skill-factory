"""The lab: ask any set of questions about every example once, and rescore offline forever.

A System One request bills the state once and answers up to ``per_request``
questions about it, so 40 candidate questions cost about what one does. The lab
asks only what the cache lacks, appends each answer as it arrives, tracks tokens
and dollars, and stops scheduling before a spend cap is crossed. Every policy,
threshold and ensemble is then scored from the cache, without a call.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.dataset import Example
from skill_factory.classifier.questions import Question
from skill_factory.classifier.systemone import SystemOneClient, SystemOneError

_MAX_ERRORS_KEPT = 20


@dataclass(frozen=True)
class LabSettings:
    per_request: int = 40
    concurrency: int = 8
    usd_per_million_input_tokens: float = 0.042
    #: Dollars this lab may spend across all its calls; requests that would cross it are not sent.
    max_usd: float | None = None
    chars_per_token: int = 4

    def __post_init__(self) -> None:
        if self.per_request < 1 or self.concurrency < 1:
            raise ValueError("per_request and concurrency must be >= 1")

    def usd(self, tokens: float) -> float:
        return tokens / 1_000_000 * self.usd_per_million_input_tokens


@dataclass(frozen=True)
class Job:
    example: Example
    ids: tuple[str, ...]


@dataclass(frozen=True)
class AskReport:
    """What one ``Lab.ask`` did, including what it did not do and why."""

    planned: int
    requests: int
    failed: int
    not_sent_over_cap: int
    answers: int
    input_tokens: int
    tokens_estimated: bool
    usd: float
    statuses: Mapping[str, int]
    served: Mapping[str, int]
    errors: tuple[str, ...] = ()

    def summary(self) -> str:
        parts = [
            f"{self.requests}/{self.planned} requests sent, {self.failed} failed",
            f"{self.answers} answers cached",
            f"{self.input_tokens:,} input tokens{' (estimated)' if self.tokens_estimated else ''}",
            f"${self.usd:.4f}",
        ]
        if self.not_sent_over_cap:
            parts.append(f"{self.not_sent_over_cap} not sent: spend cap reached")
        status = ", ".join(f"{n} {s}" for s, n in sorted(self.statuses.items()))
        return "; ".join(parts) + (f"\nexamples: {status}" if status else "")


class Lab:
    """Ask questions through a System One client, through the answer cache."""

    def __init__(
        self,
        client: SystemOneClient | None,
        cache: AnswerCache,
        settings: LabSettings | None = None,
        *,
        model: str | None = None,
        offline_reason: str = "",
    ):
        if client is None and model is None:
            raise ValueError("an offline lab (no client) needs the model its cached answers were asked of")
        self.client = client
        self.cache = cache
        self.settings = settings or LabSettings()
        self.model = model or (client.model if client is not None else "")
        self.offline_reason = offline_reason
        #: Spent by this lab across calls; ``max_usd`` caps it.
        self.spent_usd = 0.0

    # --- planning ---------------------------------------------------------

    def plan(
        self, examples: Iterable[Example], questions: Mapping[str, Question], *, budget: int, replicate: int = 0
    ) -> list[Job]:
        hashes = {qid: q.hash for qid, q in questions.items()}
        jobs: list[Job] = []
        n = self.settings.per_request
        for ex in examples:
            if ex.status(budget) != "judged" or ex.state is None:
                continue
            missing = [
                qid for qid, h in hashes.items()
                if not self.cache.has(str(ex.state_hash), h, self.model, replicate)
            ]
            jobs.extend(Job(ex, tuple(missing[i : i + n])) for i in range(0, len(missing), n))
        return jobs

    def estimate_tokens(self, jobs: Sequence[Job], questions: Mapping[str, Question]) -> int:
        return sum(self._job_tokens(j, questions) for j in jobs)

    def cost_per_pass(self, examples: Iterable[Example], n_questions: int, *, budget: int) -> float:
        """Dollars to ask ``n_questions`` about every judged example once (the state dominates)."""
        requests_each = math.ceil(n_questions / self.settings.per_request)
        chars = sum(ex.chars or 0 for ex in examples if ex.status(budget) == "judged")
        return self.settings.usd(requests_each * chars / self.settings.chars_per_token)

    # --- asking -----------------------------------------------------------

    def ask(
        self,
        examples: Iterable[Example],
        questions: Mapping[str, Question],
        *,
        budget: int,
        replicate: int = 0,
        progress: Callable[[int, int], None] | None = None,
    ) -> AskReport:
        examples = list(examples)
        statuses = Counter(_status_bucket(ex.status(budget)) for ex in examples)
        jobs = self.plan(examples, questions, budget=budget, replicate=replicate)
        if jobs and self.client is None:
            raise RuntimeError(
                f"{len(jobs)} requests needed but the lab is offline"
                f" ({self.offline_reason or 'no System One client'}); "
                "configure systemone, or score only what is cached"
            )
        cap = self.settings.max_usd
        tally = _Tally(None if cap is None else max(0.0, cap - self.spent_usd))

        def run(job: Job) -> None:
            est_tokens = self._job_tokens(job, questions)
            estimate = self.settings.usd(est_tokens)
            if not tally.reserve(estimate):
                return
            assert self.client is not None and job.example.state is not None
            try:
                answers = self.client.ask(job.example.state, {qid: questions[qid] for qid in job.ids})
            except SystemOneError as exc:
                tally.fail(estimate, str(exc))
                return
            self.cache.put_many(
                {"s": job.example.state_hash, "q": questions[qid].hash, "m": self.model,
                 "p": answers.probabilities[qid], "v": answers.model, "r": replicate}
                for qid in job.ids
            )
            reported = answers.input_tokens
            tokens = reported if reported is not None else est_tokens
            tally.done(estimate, self.settings.usd(tokens), tokens, reported is None, answers.model, len(job.ids))
            if progress is not None:
                progress(tally.requests, len(jobs))

        with ThreadPoolExecutor(max_workers=self.settings.concurrency) as pool:
            list(pool.map(run, jobs))
        self.spent_usd += tally.spent
        return AskReport(
            planned=len(jobs),
            requests=tally.requests,
            failed=tally.failed,
            not_sent_over_cap=tally.over_cap,
            answers=tally.answers,
            input_tokens=tally.tokens,
            tokens_estimated=tally.estimated,
            usd=tally.spent,
            statuses=dict(statuses),
            served=dict(tally.served),
            errors=tuple(tally.errors),
        )

    # --- reading ----------------------------------------------------------

    def table(
        self, examples: Iterable[Example], questions: Mapping[str, Question], *, budget: int, replicate: int = 0
    ) -> "AnswerTable":
        return AnswerTable.build(examples, questions, self.cache, self.model, budget=budget, replicate=replicate)

    def _job_tokens(self, job: Job, questions: Mapping[str, Question]) -> int:
        chars = (job.example.chars or 0) + sum(
            len(questions[q].instructions) + len(questions[q].true) + len(questions[q].false) for q in job.ids
        )
        return math.ceil(chars / self.settings.chars_per_token)


@dataclass(frozen=True)
class AnswerTable:
    """Cached P(true) per example and question, plus why an example has no answers."""

    answers: Mapping[str, Mapping[str, float]]
    statuses: Mapping[str, str]
    versions: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        examples: Iterable[Example],
        questions: Mapping[str, Question],
        cache: AnswerCache,
        model: str,
        *,
        budget: int,
        replicate: int = 0,
    ) -> "AnswerTable":
        hashes = {qid: q.hash for qid, q in questions.items()}
        answers: dict[str, dict[str, float]] = {}
        statuses: dict[str, str] = {}
        versions: Counter = Counter()
        for ex in examples:
            status = ex.status(budget)
            statuses[ex.id] = status
            if status != "judged":
                answers[ex.id] = {}
                continue
            row = {}
            for qid, h in hashes.items():
                p = cache.get(str(ex.state_hash), h, model, replicate)
                if p is not None:
                    row[qid] = p
                    v = cache.version(str(ex.state_hash), h, model, replicate)
                    if v:
                        versions[v] += 1
            answers[ex.id] = row
        return cls(answers, statuses, dict(versions))

    def p(self, example_id: str, qid: str) -> float | None:
        return self.answers.get(example_id, {}).get(qid)

    def of(self, example_id: str) -> Mapping[str, float]:
        return self.answers.get(example_id, {})


class _Tally:
    """Spend and outcome counters shared by the worker threads."""

    def __init__(self, cap: float | None):
        self._lock = threading.Lock()
        self.cap = cap
        self.reserved = 0.0
        self.spent = 0.0
        self.requests = self.failed = self.over_cap = self.answers = self.tokens = 0
        self.estimated = False
        self.served: Counter = Counter()
        self.errors: list[str] = []

    def reserve(self, estimate: float) -> bool:
        with self._lock:
            if self.cap is not None and self.reserved + estimate > self.cap:
                self.over_cap += 1
                return False
            self.reserved += estimate
            return True

    def fail(self, estimate: float, error: str) -> None:
        with self._lock:
            self.reserved -= estimate
            self.failed += 1
            if len(self.errors) < _MAX_ERRORS_KEPT:
                self.errors.append(error[:300])

    def done(self, estimate: float, actual: float, tokens: int, estimated: bool, served: str, n: int) -> None:
        with self._lock:
            self.reserved += actual - estimate
            self.spent += actual
            self.requests += 1
            self.answers += n
            self.tokens += tokens
            self.estimated = self.estimated or estimated
            self.served[served] += 1


def _status_bucket(status: str) -> str:
    return "skipped" if status.startswith("skipped") else status
