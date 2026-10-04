"""A classifier workspace: config -> live objects, and the files a run leaves behind.

    <workspace>/answers.jsonl   the answer cache (every answer ever bought)
    <workspace>/pool.json       every question ever asked, so later runs can select from it
    <workspace>/probeset.json   the best probe set (the artifact to export and ship)
    <workspace>/seed.json       the seed, its threshold fitted on train (the "before" of reports)
    <workspace>/report.md       before/after per split, rounds, questions, Pareto front
    <workspace>/splits.svg      the before/after figure
    <workspace>/history.json    every round, machine-readable
    <workspace>/result.json     a summary the dashboard lists next to skill runs
    <workspace>/seal.json       the freeze record and every opening of the sealed splits
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from skill_factory.classifier.cache import AnswerCache
from skill_factory.classifier.config import ClassifierConfig
from skill_factory.classifier.dataset import Example, load_examples
from skill_factory.classifier.evaluation import SplitReport, evaluate_split
from skill_factory.classifier.lab import AskReport, Lab, LabSettings
from skill_factory.classifier.optimizer import ClassifierOptimizer, ClassifierResult
from skill_factory.classifier.policy import THRESHOLD_POLICY, Policy, load_policy
from skill_factory.classifier.probeset import ProbeSet
from skill_factory.classifier.questions import QuestionBank
from skill_factory.classifier.report import classifier_report, splits_svg
from skill_factory.classifier.splits import SEALED, Seal
from skill_factory.classifier.systemone import PROVIDERS, HttpSystemOne, SystemOneClient
from skill_factory.llm.client import LLMClient

_LAB_KEYS = {"per_request", "concurrency", "usd_per_million_input_tokens", "max_usd", "chars_per_token"}
_CLIENT_KEYS = {"provider", "model", "api_key", "base_url", "account_id", "timeout", "max_retries"}


def build_systemone(cfg: Mapping[str, Any]) -> HttpSystemOne:
    unknown = set(cfg) - _LAB_KEYS - _CLIENT_KEYS
    if unknown:
        raise ValueError(f"unknown systemone keys: {sorted(unknown)}")
    return HttpSystemOne(
        str(cfg.get("provider", "typesafe")),
        model=cfg.get("model"),
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        account_id=cfg.get("account_id"),
        timeout=float(cfg.get("timeout", 60.0)),
        max_retries=int(cfg.get("max_retries", 4)),
    )


def lab_settings(cfg: Mapping[str, Any]) -> LabSettings:
    max_usd = cfg.get("max_usd")
    return LabSettings(
        per_request=int(cfg.get("per_request", 40)),
        concurrency=int(cfg.get("concurrency", 8)),
        usd_per_million_input_tokens=float(cfg.get("usd_per_million_input_tokens", 0.042)),
        max_usd=float(max_usd) if max_usd is not None else None,
        chars_per_token=int(cfg.get("chars_per_token", 4)),
    )


@dataclass
class Workspace:
    config: ClassifierConfig
    client: SystemOneClient | None = None
    #: Every ask that had failed requests, so a command can say so and exit non-zero.
    failed_asks: list[AskReport] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return self.config.workspace

    def path(self, name: str) -> Path:
        return self.dir / name

    @cached_property
    def examples(self) -> list[Example]:
        return load_examples(self.config.dataset_path, positive=self.config.positive, negative=self.config.negative)

    @cached_property
    def by_split(self) -> dict[str, list[Example]]:
        return self.config.splits.assign(self.examples)

    @property
    def model(self) -> str:
        cfg = self.config.systemone
        return str(cfg.get("model") or PROVIDERS[str(cfg.get("provider", "typesafe"))].model)

    @cached_property
    def lab(self) -> Lab:
        """The lab, online when a client can be built; offline (cache only) otherwise, saying why."""
        cache = AnswerCache(self.path("answers.jsonl"))
        settings = lab_settings(self.config.systemone)
        client, reason = self.client, ""
        if client is None:
            try:
                client = build_systemone(self.config.systemone)
            except ValueError as exc:
                reason = str(exc)
        return Lab(client, cache, settings, model=self.model, offline_reason=reason)

    def seed(self) -> ProbeSet:
        return ProbeSet.load(self.config.probeset_path)

    def best(self) -> ProbeSet:
        """The workspace's best probe set, or the seed before any run."""
        p = self.path("probeset.json")
        return ProbeSet.load(p) if p.exists() else self.seed()

    def pool(self) -> QuestionBank:
        bank = self.seed().bank
        if self.config.pool_path is not None:
            bank = bank.merge(QuestionBank.load(self.config.pool_path))
        grown = self.path("pool.json")
        if grown.exists():
            bank = bank.merge(QuestionBank.load(grown))
        return bank

    def policy(self, ref: str | None = None) -> Policy:
        ref = ref or self.config.policy
        return load_policy(ref, self.config.base_dir) if ref else THRESHOLD_POLICY

    def seal(self) -> Seal:
        return Seal.load(self.dir)

    def statuses(self, budget: int) -> dict[str, dict[str, int]]:
        out = {}
        for split, exs in self.by_split.items():
            counts = Counter("skipped" if (s := ex.status(budget)).startswith("skipped") else s for ex in exs)
            if exs:
                out[split] = dict(counts)
        return out

    def baseline(self) -> ProbeSet:
        """The "before" of reports: the seed with its threshold fitted by the last run, else the seed."""
        p = self.path("seed.json")
        return ProbeSet.load(p) if p.exists() else self.seed()

    def compare(
        self, before: ProbeSet, after: ProbeSet, splits: Sequence[str], *, ask: bool = False
    ) -> tuple[dict[str, SplitReport], dict[str, SplitReport]]:
        """Score ``before`` and ``after`` on ``splits``, asking first when ``ask``.

        Sealed splits are opened once, through the seal: ``after`` must be the frozen probe
        set and ``before`` the baseline recorded with it.
        """
        wanted = [s for s in splits if self.by_split.get(s)]
        if set(wanted) & SEALED:
            self.seal().open(after.content_hash(), wanted, baseline=before.content_hash())
        return self._score(before, wanted, ask=ask), self._score(after, wanted, ask=ask)

    def _score(self, probeset: ProbeSet, splits: Sequence[str], *, ask: bool) -> dict[str, SplitReport]:
        examples = [ex for s in splits for ex in self.by_split[s]]
        questions = probeset.bank.compiled()
        if ask:
            report = self.lab.ask(examples, questions, budget=probeset.state_budget)
            if report.failed:
                self.failed_asks.append(report)
        table = self.lab.table(examples, questions, budget=probeset.state_budget)
        policy = self.policy()
        return {s: evaluate_split(s, self.by_split[s], table, probeset, policy) for s in splits}


def run_classifier(
    config: ClassifierConfig,
    *,
    client: SystemOneClient | None = None,
    proposer: LLMClient | None = None,
    log: Callable[[str], None] = print,
) -> tuple[ClassifierResult, Workspace]:
    """Optimize the configured probe set and write the workspace files."""
    ws = Workspace(config, client=client)
    if proposer is None and config.proposer is not None:
        from skill_factory.builder import build_client

        proposer = build_client(config.proposer)
    opt = config.optimizer
    optimizer = ClassifierOptimizer(
        ws.lab,
        config.objective,
        proposer=proposer,
        policy=ws.policy(),
        rounds=int(opt.get("rounds", 3)),
        patience=int(opt.get("patience", 2)),
        max_questions=int(opt.get("max_questions", 6)),
        max_new_questions=int(opt.get("max_new_questions", 8)),
        max_state_budget=int(opt["max_state_budget"]) if opt.get("max_state_budget") else None,
        goal=str(opt.get("goal", "")),
        keep_questions=config.keep_questions,
        margins=tuple(float(m) for m in opt.get("margins", (1.0, 0.5))),
        gate_per_family=int(opt.get("gate_per_family", 3)),
        log=log,
    )
    tuning = {s: ws.by_split[s] for s in ("train", "val")}
    result = optimizer.optimize(ws.seed(), tuning, ws.pool())
    result = replace(result, best=result.best.with_changes(name=config.name))
    save_classifier_run(result, ws)
    return result, ws


def save_classifier_run(result: ClassifierResult, ws: Workspace) -> dict[str, Path]:
    ws.dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed": result.seed.save(ws.path("seed.json")),  # threshold fitted on train: the report's "before"
        "probeset": result.best.save(ws.path("probeset.json")),
        "pool": result.pool.save(ws.path("pool.json")),
    }
    history = [h.to_dict() for h in result.history]
    paths["history"] = _write(ws.path("history.json"), json.dumps(history, indent=1))
    examples = [ex for s in ("train", "val") for ex in ws.by_split[s]]
    cost = ws.lab.cost_per_pass(examples, len(result.best.bank), budget=result.best.state_budget)
    report = classifier_report(
        result,
        seal_summary=ws.seal().summary(),
        cost_per_pass=cost,
        statuses=ws.statuses(result.best.state_budget),
        sealed_splits=[s for s in SEALED if ws.by_split.get(s)],
    )
    paths["report"] = _write(ws.path("report.md"), report)
    paths["figure"] = _write(ws.path("splits.svg"), splits_svg(result.before, result.after, result.objective.level))
    paths["result"] = _write(ws.path("result.json"), json.dumps(_dashboard_payload(result), indent=1))
    return paths


def _dashboard_payload(result: ClassifierResult) -> dict:
    """The shape skill runs use, so ``skill-factory ui`` lists classifier runs too."""
    level = result.objective.level

    def body(probeset: ProbeSet) -> str:
        return json.dumps(probeset.to_dict(), indent=1, ensure_ascii=False)

    return {
        "kind": "classifier",
        "optimizer": "classifier",
        "baseline_score": round(result.baseline_score, 4),
        "best_score": round(result.best_score, 4),
        "improvement": round(result.best_score - result.baseline_score, 4),
        "candidates": len(result.history),
        "accepted": sum(1 for h in result.history if h.accepted),
        "metadata": {"usd": round(result.usd, 6), "input_tokens": result.input_tokens,
                     "sha256": result.best.content_hash(), "failed_requests": result.failed_requests},
        "best_skill": {"name": result.best.name, "description": "probe set", "body": body(result.best),
                       "markdown": body(result.best)},
        "history": [
            {
                "iteration": h.round,
                "train_score": round(h.reports["train"].rates(level).recall, 4),
                "val_score": round(h.reports["val"].rates(level).recall, 4),
                "accepted": h.accepted,
                "note": f"{h.change}: {h.candidate}",
                "skill_name": result.best.name,
                "body": body(result.seed) if h.round == 0 else h.score,
            }
            for h in result.history
        ],
    }


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path
