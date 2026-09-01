"""Run a skill across a dataset and score it — the shared evaluation step.

Every optimizer calls :func:`evaluate_skill` to measure a candidate. Rollouts
are IO-bound (LLM calls), so an optional thread pool parallelizes them; a single
worker keeps evaluation deterministic for tests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from skill_factory.core.result import Evaluation, TaskEvaluation
from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task
from skill_factory.harness.base import Harness
from skill_factory.metrics.base import Metric


def evaluate_skill(
    skill: Skill,
    dataset: Dataset | Iterable[Task],
    harness: Harness,
    metric: Metric,
    *,
    max_workers: int = 1,
) -> Evaluation:
    """Score ``skill`` on every task in ``dataset`` using ``harness`` + ``metric``."""
    tasks = list(dataset)
    if not tasks:
        raise ValueError("cannot evaluate on an empty dataset")

    def run_one(task: Task) -> TaskEvaluation:
        rollout = harness.run(skill, task)
        result = metric.evaluate(task, rollout)
        return TaskEvaluation(task=task, rollout=rollout, result=result)

    if max_workers <= 1:
        items = [run_one(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            items = list(pool.map(run_one, tasks))

    # Preserve dataset order regardless of completion order.
    order = {task.id: i for i, task in enumerate(tasks)}
    items.sort(key=lambda it: order[it.task.id])
    return Evaluation(items=items)
