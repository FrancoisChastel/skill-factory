"""Self-contained reflective optimizer.

Implements the same loop SkillOpt and GEPA use, with no heavy dependencies:

    rollout  -> run the current skill on a train minibatch
    reflect  -> an optimizer LLM reads the low-scoring rollouts + feedback and
                proposes an improved skill body
    gate     -> the candidate is accepted only if it beats the current best on
                the held-out validation split

This is the default backend and the reference implementation of the Optimizer
protocol. It only needs an :class:`LLMClient` for the reflection step.
"""

from __future__ import annotations

import random
import re

from skill_factory.core.result import CandidateRecord, OptimizationResult
from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset
from skill_factory.evaluate import evaluate_skill
from skill_factory.harness.base import Harness
from skill_factory.llm.client import LLMClient
from skill_factory.metrics.base import Metric
from skill_factory.optimizers.base import OptimizerConfig, register_optimizer

_BODY_RE = re.compile(r"<NEW_SKILL_BODY>\s*(.*?)\s*</NEW_SKILL_BODY>", re.DOTALL)
_RATIONALE_RE = re.compile(r"<RATIONALE>\s*(.*?)\s*</RATIONALE>", re.DOTALL)
_MAX_FIELD_CHARS = 800

_REFLECT_SYSTEM = (
    "You are a prompt/skill optimizer. You improve a reusable agent SKILL "
    "document so that an agent following it produces higher-scoring outputs. "
    "You make targeted, generalizable edits — you never overfit to specific "
    "examples, never hardcode answers, and keep the skill portable across "
    "harnesses. You preserve the skill's intent while fixing the failure modes "
    "revealed by the feedback."
)


class LLMLoopOptimizer:
    """Reflective, validation-gated skill optimizer."""

    name = "llm_loop"

    def __init__(
        self,
        optimizer_client: LLMClient,
        *,
        config: OptimizerConfig | None = None,
        goal: str = "",
        min_delta: float = 0.0,
    ):
        self._client = optimizer_client
        self._config = config or OptimizerConfig()
        self._goal = goal
        self._min_delta = min_delta

    def optimize(
        self,
        seed: Skill,
        trainset: Dataset,
        valset: Dataset,
        harness: Harness,
        metric: Metric,
    ) -> OptimizationResult:
        cfg = self._config
        workers = cfg.max_workers

        baseline_eval = evaluate_skill(seed, valset, harness, metric, max_workers=workers)
        baseline = baseline_eval.score

        best_skill = seed
        best_val = baseline
        history: list[CandidateRecord] = [
            CandidateRecord(0, seed, train_score=baseline, val_score=baseline,
                            accepted=True, note="seed (baseline)")
        ]

        train_tasks = list(trainset)
        rng = random.Random(cfg.seed)
        no_improve = 0

        for round_idx in range(1, cfg.rounds + 1):
            minibatch = _sample(train_tasks, cfg.minibatch_size, rng)
            train_eval = evaluate_skill(best_skill, minibatch, harness, metric, max_workers=workers)

            proposals = self._propose(best_skill, train_eval, cfg.max_edits_per_round)
            if not proposals:
                history.append(
                    CandidateRecord(round_idx, best_skill, train_eval.score, best_val,
                                    accepted=False, note="no valid proposal from optimizer LLM")
                )
                no_improve += 1
                if no_improve >= cfg.patience:
                    break
                continue

            round_best: tuple[float, Skill, str] | None = None
            for body, rationale in proposals:
                if body.strip() == best_skill.body.strip():
                    continue
                candidate = best_skill.with_body(body)
                cand_eval = evaluate_skill(candidate, valset, harness, metric, max_workers=workers)
                if round_best is None or cand_eval.score > round_best[0]:
                    round_best = (cand_eval.score, candidate, rationale)

            if round_best is None:
                history.append(
                    CandidateRecord(round_idx, best_skill, train_eval.score, best_val,
                                    accepted=False, note="proposal identical to current skill")
                )
                no_improve += 1
            else:
                cand_val, candidate, rationale = round_best
                accepted = cand_val > best_val + self._min_delta
                note = (rationale or "").strip()[:200] or "edit"
                history.append(
                    CandidateRecord(round_idx, candidate, train_eval.score, cand_val,
                                    accepted=accepted, note=note)
                )
                if accepted:
                    best_skill, best_val = candidate, cand_val
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= cfg.patience:
                break

        return OptimizationResult(
            best_skill=best_skill,
            baseline_score=baseline,
            best_score=best_val,
            history=history,
            optimizer=self.name,
            metadata={"rounds_run": history[-1].iteration, "valset_size": len(valset)},
        )

    def _propose(self, skill: Skill, train_eval, n: int) -> list[tuple[str, str]]:
        prompt = self._reflection_prompt(skill, train_eval)
        proposals: list[tuple[str, str]] = []
        for i in range(max(1, n)):
            # Slightly raise temperature for additional candidates to diversify.
            temperature = 0.2 if i == 0 else min(1.0, 0.4 + 0.2 * i)
            raw = self._client.generate(
                prompt, system=_REFLECT_SYSTEM, temperature=temperature, max_tokens=4096
            )
            body = _extract(_BODY_RE, raw)
            if body:
                rationale = _extract(_RATIONALE_RE, raw) or ""
                proposals.append((body, rationale))
        return proposals

    def _reflection_prompt(self, skill: Skill, train_eval) -> str:
        goal = self._goal or (
            f"Improve this skill named {skill.name!r} so an agent following it "
            "scores higher on the evaluation metric."
        )
        examples = []
        for item in train_eval.worst(self._config.minibatch_size):
            examples.append(
                "\n".join(
                    [
                        f"--- example (score {item.result.score:.2f}) ---",
                        f"INPUT: {_truncate(item.task.input)}",
                        f"SKILL OUTPUT: {_truncate(item.rollout.output)}",
                        *(
                            [f"EXPECTED: {_truncate(item.task.expected)}"]
                            if item.task.expected
                            else []
                        ),
                        f"FEEDBACK: {_truncate(item.result.feedback)}",
                    ]
                )
            )
        return "\n".join(
            [
                f"GOAL: {goal}",
                f"CURRENT MEAN SCORE ON THIS BATCH: {train_eval.score:.3f}",
                "",
                "CURRENT SKILL BODY:",
                "<CURRENT_SKILL_BODY>",
                skill.body.strip(),
                "</CURRENT_SKILL_BODY>",
                "",
                "LOWEST-SCORING ROLLOUTS AND THEIR FEEDBACK:",
                *examples,
                "",
                "Rewrite the skill body to fix the failure modes above. Rules:",
                "- Keep it a general, reusable SKILL.md body (no example-specific answers).",
                "- Prefer clear, imperative instructions; add rules/steps/format guidance",
                "  that would have prevented the failures.",
                "- Keep it concise; do not pad.",
                "",
                "Respond in EXACTLY this format:",
                "<RATIONALE>one or two sentences on what you changed and why</RATIONALE>",
                "<NEW_SKILL_BODY>",
                "...the full improved skill body in markdown...",
                "</NEW_SKILL_BODY>",
            ]
        )


def _sample(tasks: list, k: int, rng: random.Random) -> list:
    if k >= len(tasks):
        return list(tasks)
    return rng.sample(tasks, k)


def _extract(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _truncate(text: str | None, limit: int = _MAX_FIELD_CHARS) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


register_optimizer("llm_loop", lambda config, **kw: LLMLoopOptimizer(config=config, **kw))
