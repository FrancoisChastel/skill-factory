# Contributing to Skill Factory

Thanks for your interest in improving Skill Factory! This project turns skill
authoring into a measurable, reproducible process — contributions that keep it
honest, well-tested, and provider-agnostic are very welcome.

## Ground rules

- **Keep the core dependency-light.** `skill_factory.core`, `metrics`, and
  `optimizers.llm_loop` must not require heavy or optional packages. Backend SDKs
  (`anthropic`, `openai`, `dspy`, `skillopt`) are imported lazily.
- **Everything testable offline.** New logic should be verifiable without a
  network or API key. Use `FakeLLMClient` and `CallableHarness` (see `tests/`).
- **Small, cohesive files.** Prefer many focused modules over large ones.
- **Immutability.** Skills and results are frozen dataclasses; "editing" a skill
  returns a new `Skill`.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # should be all green
```

## Development workflow

1. Create a branch off `main`.
2. Write a test first (TDD encouraged), then the implementation.
3. Keep coverage at or above the current level:
   ```bash
   pytest --cov
   ```
4. Run the whole suite before opening a PR.

## Adding a new optimizer backend

Implement the `Optimizer` protocol (`optimize(seed, trainset, valset, harness,
metric) -> OptimizationResult`) and self-register:

```python
from skill_factory.optimizers.base import register_optimizer
register_optimizer("my_backend", lambda config, **kw: MyOptimizer(config=config, **kw))
```

Score the produced skill through the given `harness` + `metric` so results stay
comparable with the other backends.

## Adding a new provider

Add an entry to `skill_factory/llm/factory.py`. If it speaks the OpenAI Chat
Completions API, just add its default `base_url` to `_OPENAI_COMPATIBLE`.

## Adding a metric

Implement the `Metric` protocol (`evaluate(task, rollout) -> MetricResult`).
Return **textual feedback**, not just a score — reflective optimizers depend on it.
Raise `MetricNotApplicable` when a metric can't score a task.

## Commit messages

Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`,
`perf:`, `ci:`.

## Code of Conduct

Be kind and constructive. Harassment or discrimination of any kind is not
tolerated.
