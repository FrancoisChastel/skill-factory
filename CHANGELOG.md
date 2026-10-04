# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-04

### Added

- **Classifier mode** (`kind: classifier`, `skill-factory optimize --kind classifier`):
  optimize a probe set (System One questions under framings, a JSON score
  expression, a threshold, a state budget, a policy) for a decision model such as
  jev, instead of a `SKILL.md`.
  - A **lab** that caches every answer by (state, question, model) in skill-scanner's
    `answers.jsonl` format, batches up to 40 questions per request, runs requests
    concurrently, tracks tokens and dollars, and stops before a spend cap.
  - **Dataset-level metrics** (AUC, recall at a false-positive budget) and a gate with
    **constraints** checked against the incumbent (`max_fpr`, `max_added_false`, ...).
  - **Honest splits**: group-aware hashing, held-out sources and groups,
    read-before-split examples, and a **seal** that keeps test and held-out unreadable
    until `lab freeze`, logging every opening.
  - **Reflection** with system-level causes first (over budget, failed, no state),
    every question's separation, and misses and false flags with every P(true);
    proposals may add questions, framings, or a state-budget change.
  - **Selection and ensembling**: single questions, means of the best distinct
    questions, greedy unions, mean plus a lure; a Pareto front in the report.
  - **Policy replay** from a Python file (`lab simulate`), **re-ask determinism**
    (`lab reask`), **version pinning** and `lab recalibrate`.
  - **Exporters** to JSON, TypeScript and Python with the probe set's SHA-256, and
    `lab parity` for byte-for-byte artifact checks and production-versus-lab verdicts.
  - `lab import-scanner` turns a skill-scanner judge lab into a workspace; its
    published jev numbers reproduce from the cache.
  - Per-run `report.md` with a before/after table per split and `splits.svg`.
- **`metric.systemone`**: yes/no rubric checks answered by a System One model, cached
  per rollout, as a cheap alternative to an LLM judge in skill mode.
- Example `examples/pr-mismatch`: flag pull requests whose diff does something the
  description does not disclose; verified live against jev (`jev-1.13.0`), with a
  `config.lmstudio.yaml` variant that adds reflective rounds from a local model.
- Fitted thresholds never spend the false-positive budget where it buys no recall; they
  sit mid-gap between the lowest caught positive and the highest negative below it.
- README **Results** section with light and dark charts for the jev case study, skill
  mode and the pr-mismatch live run; `docs/media/charts/build_charts.py` rebuilds them.

## [0.1.0] - 2026-09-02

Initial release.

### Added

- **CLI harnesses (optimize *in situ*)**: `harness.type: claude_code` runs every
  rollout through the real Claude Code CLI (`claude -p --append-system-prompt`),
  and `harness.type: subprocess` runs any argv template with `{skill_body}` /
  `{skill_file}` / `{input}` / `{input_file}` placeholders; failures, non-zero
  exits, and timeouts become scored-zero rollouts instead of aborting the run.
- **Skill gallery** (`gallery/`): optimized skills with reproducible numbers and
  full candidate histories, guarded by tests; contribution checklist included.
- **PyPI packaging**: complete project metadata, `twine`-validated sdist+wheel,
  and a trusted-publishing release workflow (publishes when a GitHub release goes
  out — no tokens stored in the repo).
- **Explainer v2**: crossfade transitions (`@remotion/transitions`) and an
  optional narrated cut (macOS TTS), rendered as separate silent/narrated MP4s.

- **Core primitives**: `Skill` (portable SKILL.md), `Dataset`/`Task`, `Harness`,
  `Metric`, `Rollout`, and result/report types — all dependency-light and immutable.
- **Metrics** (composable by weight): golden-set (`exact` / `normalized` /
  `token_f1` / `json_equal` / `numeric`), programmatic checks (JSON validity,
  required keys, regex, contains, length…), and an LLM-as-judge rubric.
- **Optimizers** behind one `Optimizer` protocol and a lazy registry:
  - `llm_loop` — self-contained reflective, validation-gated loop (default, no
    heavy deps).
  - `dspy_gepa` — DSPy GEPA reflective/evolutionary search adapter.
  - `skillopt` — Microsoft SkillOpt subprocess bridge (consumes `best_skill.md`).
- **Providers** via a factory: `anthropic` plus any OpenAI-compatible endpoint
  (`openai`, `azure`, `openrouter`, `together`, `groq`, `deepseek`, `vllm`,
  `lmstudio`, `ollama`). Anthropic client is compatible with `anthropic>=1.x`
  (auto-detects removed `temperature` param) and fails fast on non-retryable errors.
- **`npx skills` export** — emit optimized skills as `<slug>/SKILL.md` directories.
- **Optional web dashboard** (`skill-factory ui`) built on the Python stdlib —
  browse runs, score timelines, and seed↔best skill diffs; zero extra dependencies.
- **CLI**: `optimize`, `evaluate`, `export`, `list`, `ui`.
- **Examples**: `invoice-extractor` (JSON) and `ticket-classifier` (classification),
  with configs verified live against a local LM Studio model
  (0.559→1.000 and 0.151→1.000 respectively).
- **Tests**: fully offline suite (fake LLM + callable harness), ~80% coverage.

[Unreleased]: https://github.com/FrancoisChastel/skill-factory/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/FrancoisChastel/skill-factory/releases/tag/v0.2.0
[0.1.0]: https://github.com/FrancoisChastel/skill-factory/releases/tag/v0.1.0
