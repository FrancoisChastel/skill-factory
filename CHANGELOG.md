# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/FrancoisChastel/skill-factory/releases/tag/v0.1.0
