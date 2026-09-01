"""DSPy GEPA optimizer adapter.

GEPA (Genetic-Pareto) is a reflective optimizer: it mutates a program's
instruction using an LM that reads the metric's *textual feedback*, keeping a
Pareto frontier across validation tasks.

This adapter maps the factory's primitives onto DSPy:

    Skill.body   -> the instruction of a single-predictor DSPy program
    Dataset      -> dspy.Example list
    Metric       -> a GEPA metric returning dspy.Prediction(score, feedback)

GEPA performs the *search*; the resulting instruction becomes the new skill body,
which is then scored with the factory's own harness + metric so the reported
baseline/best are directly comparable with the other backends.

Requires: pip install 'skill-factory[dspy]'
"""

from __future__ import annotations

from typing import Any

from skill_factory.core.result import CandidateRecord, OptimizationResult
from skill_factory.core.rollout import Rollout
from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task
from skill_factory.evaluate import evaluate_skill
from skill_factory.harness.base import Harness
from skill_factory.metrics.base import Metric
from skill_factory.optimizers.base import OptimizerConfig, register_optimizer


class DspyGepaOptimizer:
    """Optimize a skill's instruction text with DSPy GEPA."""

    name = "dspy_gepa"

    def __init__(
        self,
        *,
        config: OptimizerConfig | None = None,
        lm: Any = None,
        reflection_lm: Any = None,
        auto: str = "light",
        goal: str = "",
    ):
        try:
            import dspy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "DspyGepaOptimizer requires DSPy. Install with: "
                "pip install 'skill-factory[dspy]'"
            ) from exc
        self._config = config or OptimizerConfig()
        self._auto = auto
        self._goal = goal
        self._lm = lm
        self._reflection_lm = reflection_lm

    def optimize(
        self,
        seed: Skill,
        trainset: Dataset,
        valset: Dataset,
        harness: Harness,
        metric: Metric,
    ) -> OptimizationResult:
        import dspy

        lm = _as_lm(self._lm) if self._lm is not None else dspy.settings.lm
        if lm is None:
            raise ValueError(
                "DspyGepaOptimizer needs a DSPy LM. Pass lm='openai/gpt-...' or "
                "call dspy.configure(lm=...) beforehand."
            )
        dspy.configure(lm=lm)
        reflection_lm = _as_lm(self._reflection_lm) if self._reflection_lm else lm

        signature = dspy.Signature("task_input -> output").with_instructions(seed.body)
        program = _SkillProgram(signature)

        gepa_metric = _make_gepa_metric(metric)
        optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto=self._auto,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            skip_perfect_score=True,
            use_merge=True,
            num_threads=max(1, self._config.max_workers),
            track_stats=True,
            seed=self._config.seed,
        )
        optimized = optimizer.compile(
            student=program,
            trainset=_to_examples(trainset),
            valset=_to_examples(valset),
        )

        new_body = _extract_instructions(optimized) or seed.body
        candidate = seed.with_body(new_body)

        workers = self._config.max_workers
        baseline = evaluate_skill(seed, valset, harness, metric, max_workers=workers).score
        cand_score = evaluate_skill(candidate, valset, harness, metric, max_workers=workers).score

        accepted = cand_score >= baseline
        best_skill = candidate if accepted else seed
        best_score = cand_score if accepted else baseline

        history = [
            CandidateRecord(0, seed, baseline, baseline, accepted=True, note="seed (baseline)"),
            CandidateRecord(1, candidate, cand_score, cand_score, accepted=accepted,
                            note="GEPA-optimized instruction"),
        ]
        return OptimizationResult(
            best_skill=best_skill,
            baseline_score=baseline,
            best_score=best_score,
            history=history,
            optimizer=self.name,
            metadata=_gepa_stats(optimized),
        )


def _make_gepa_metric(metric: Metric):
    """Wrap a factory Metric as a GEPA metric returning dspy.Prediction."""
    import dspy

    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        task = _example_to_task(gold)
        output = getattr(pred, "output", "") or ""
        rollout = Rollout(task=task, output=str(output))
        result = metric.evaluate(task, rollout)
        # GEPA requires a Prediction (a dict can fail inside GEPA).
        return dspy.Prediction(score=result.score, feedback=result.feedback)

    return gepa_metric


def _to_examples(dataset: Dataset) -> list:
    import dspy

    examples = []
    for task in dataset:
        ex = dspy.Example(
            id=task.id,
            task_input=task.input,
            expected=task.expected or "",
        ).with_inputs("task_input")
        examples.append(ex)
    return examples


def _example_to_task(example) -> Task:
    return Task(
        id=str(getattr(example, "id", "0")),
        input=str(getattr(example, "task_input", "")),
        expected=(getattr(example, "expected", "") or None),
    )


def _extract_instructions(program) -> str:
    """Pull the optimized instruction out of a compiled DSPy program."""
    predict = getattr(program, "predict", None)
    signature = getattr(predict, "signature", None) or getattr(program, "signature", None)
    return (getattr(signature, "instructions", "") or "").strip()


def _gepa_stats(program) -> dict[str, Any]:
    results = getattr(program, "detailed_results", None)
    if results is None:
        return {"backend": "dspy_gepa"}
    scores = getattr(results, "val_aggregate_scores", None)
    return {
        "backend": "dspy_gepa",
        "val_aggregate_scores": list(scores) if scores is not None else None,
    }


def _as_lm(lm: Any):
    import dspy

    if isinstance(lm, str):
        return dspy.LM(lm)
    return lm


try:  # pragma: no cover - only defined when dspy is importable
    import dspy as _dspy

    class _SkillProgram(_dspy.Module):
        """Single-predictor program whose instruction is the skill body."""

        def __init__(self, signature):
            super().__init__()
            self.predict = _dspy.Predict(signature)

        def forward(self, task_input):
            return self.predict(task_input=task_input)

    register_optimizer("dspy_gepa", lambda config, **kw: DspyGepaOptimizer(config=config, **kw))
except ImportError:  # pragma: no cover - dspy not installed
    pass
