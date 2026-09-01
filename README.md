# Skill Factory

**Scientifically optimize portable agent skills.** Instead of hand-tweaking a
`SKILL.md` and hoping it got better, Skill Factory treats a skill as a *trainable
parameter*: you define what "good" means (a metric) and a set of tasks, and an
optimizer measurably improves the skill against held-out validation data.

This is the same idea behind [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt)
and [DSPy GEPA](https://dspy.ai) — Skill Factory gives you one clean interface over
all of them, plus a self-contained optimizer that needs no heavy dependencies.

Optimized skills are emitted in the open [`npx skills`](https://github.com/vercel-labs/skills)
layout, so they drop straight into Claude Code, Codex, Cursor, Gemini CLI, and more.

---

## The scientific loop

Every optimizer backend reduces to the same five primitives and the same loop:

```
        ┌──────────────────────────────────────────────────────┐
        │                                                        │
   Skill (SKILL.md, the trainable text)                         │
        │                                                        │
        ▼                                                        │
   Harness ──run(skill, task)──▶ Rollout ──▶ Metric ──▶ score + feedback
        ▲                                                        │
        │                                                        ▼
        └────────── Optimizer (reflect on feedback, edit skill, ─┘
                    accept only if it beats validation)
```

| Primitive  | What it is                                   | Where |
| ---------- | -------------------------------------------- | ----- |
| `Skill`    | A `SKILL.md` document (frontmatter + body)   | `core/skill.py` |
| `Dataset`  | Tasks with optional gold answers             | `core/task.py` |
| `Harness`  | Runs a skill on a task → `Rollout`           | `harness/` |
| `Metric`   | Scores a rollout **+ returns feedback**      | `metrics/` |
| `Optimizer`| Mutates the skill to maximize the metric     | `optimizers/` |

The textual feedback is load-bearing: reflective optimizers read *why* a rollout
scored low to decide how to edit the skill.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[anthropic]"        # default Claude backend
# or: pip install -e ".[openai]"     # OpenAI-compatible providers
# or: pip install -e ".[dspy]"       # DSPy GEPA backend
# or: pip install -e ".[dev]"        # tests
```

Copy `.env.example` to `.env` and set your key(s).

---

## Quickstart

```bash
# 1. Optimize the example invoice-extractor skill
skill-factory optimize -c examples/invoice-extractor/config.yaml

# 2. Inspect the report + emitted skill
cat runs/invoice-extractor/report.md
cat dist/invoice-extractor/SKILL.md

# 3. Install the optimized skill into any agent
npx skills add dist/invoice-extractor
```

Other commands:

```bash
skill-factory evaluate -c config.yaml            # score a skill, no optimization
skill-factory export --skill SKILL.md -o dist    # emit npx skills layout
skill-factory list                               # registered optimizers + providers
skill-factory ui --runs runs                     # optional web dashboard
```

---

## Metrics — combine all three

Configured under `metric:` in your run config and merged by weight. A metric that
can't apply to a task (e.g. golden with no label) is dropped, not penalized.

```yaml
metric:
  golden:                 # compare to a labeled expected answer
    mode: json_equal      # exact | normalized | token_f1 | json_equal | numeric
    weight: 2.0
  programmatic:           # deterministic, objective checks
    weight: 1.0
    checks:
      - is_valid_json
      - {json_has_keys: [vendor, date, total, currency]}
      - {matches_regex: "^\\{"}
  judge:                  # rubric-based LLM-as-judge
    weight: 1.0
    criteria:
      - {name: correctness, description: "all fields correct?", weight: 3}
      - {name: format, description: "strict JSON, no prose?", weight: 2}
```

---

## Optimizers — pluggable backends

Select with `optimizer.name`. All share `rounds`, `minibatch_size`,
`max_edits_per_round`, `patience`, `seed`.

| Name        | Needs            | Notes |
| ----------- | ---------------- | ----- |
| `llm_loop`  | just an LLM key  | **Default.** Self-contained reflective loop: rollout → reflect → validation-gated edit. |
| `dspy_gepa` | `[dspy]`         | DSPy GEPA reflective/evolutionary search over the instruction. |
| `skillopt`  | your SkillOpt setup | Bridges to [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt) via a configured command that emits `best_skill.md`. |

All backends report a comparable `baseline → best` validation score because the
final skill is always scored through the same harness + metric.

### Using DSPy GEPA

```yaml
optimizer:
  name: dspy_gepa
  auto: medium            # light | medium | heavy
  lm: anthropic/claude-sonnet-5          # optional; defaults from optimizer_provider
  reflection_lm: anthropic/claude-opus-4-8
```

### Using Microsoft SkillOpt

```yaml
optimizer:
  name: skillopt
  params:
    # Placeholders {seed} {train} {val} {out_dir} are substituted before running.
    command: "python scripts/train.py --config configs/mine.yaml --out_root {out_dir}"
    cwd: /path/to/SkillOpt
    output_skill: best_skill.md
```

---

## Providers — bring any model

Set per role: the **target** runs the skill, the **optimizer** proposes edits, the
**judge** scores. Any of these providers works for each:

`anthropic`, `openai`, `azure`, `openrouter`, `together`, `groq`, `deepseek`,
`vllm`, `lmstudio`, `ollama`, or any `openai-compatible` endpoint.

```yaml
provider:            {name: anthropic, model: claude-sonnet-5}    # target
optimizer_provider:  {name: anthropic, model: claude-opus-4-8}    # proposes edits
judge_provider:      {name: openai,    model: gpt-4.1, base_url: https://...}  # judge

# Local models:
# provider: {name: ollama, model: llama3.1}
```

### Run it free, fully local

Point every role at a local OpenAI-compatible server (LM Studio, Ollama, vLLM) and
optimize at zero API cost. The `*.lmstudio.yaml` example configs do exactly this and
are **verified live** against `gemma-4-31b-it-mlx`:

| Example | Task | Baseline → Best |
| ------- | ---- | --------------- |
| `invoice-extractor/config.lmstudio.yaml` | JSON extraction | **0.559 → 1.000** (+44 pts) |
| `ticket-classifier/config.lmstudio.yaml` | classification  | **0.151 → 1.000** (+85 pts) |

```bash
skill-factory optimize -c examples/ticket-classifier/config.lmstudio.yaml
```

In both cases a single validation-gated reflective edit rewrote a vague seed into a
structured skill (explicit schema, output-format constraints, normalization rules),
and the loop correctly rejected later edits that couldn't beat a perfect score.

---

## Optional web UI

A zero-dependency dashboard (Python stdlib only) to browse runs, view the
score-improvement timeline, compare seed vs. optimized skill, and launch runs:

```bash
skill-factory ui --runs runs        # → http://127.0.0.1:8765
```

---

## Project layout

```
src/skill_factory/
├── core/          skill · task · rollout · result  (dependency-light primitives)
├── metrics/       golden · programmatic · llm_judge · composite
├── harness/       anthropic_api · callable  (runs a skill on a task)
├── optimizers/    llm_loop · dspy_gepa · skillopt  (+ registry)
├── llm/           client · openai_compat · factory  (multi-provider)
├── evaluate.py    run a skill across a dataset → Evaluation
├── builder.py     config → live objects → run
├── export.py      emit npx skills-compatible directories
├── persistence.py save/load runs (report.md, best_skill.md, result.json)
├── ui/            optional stdlib web dashboard
├── config.py      declarative YAML run config
└── cli.py         skill-factory <command>
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # 66 tests, fully offline (fake LLM + callable harness)
pytest --cov              # ~80% coverage
```

The whole loop is testable without any network or API key: a `FakeLLMClient` and a
`CallableHarness` stand in for real models, so the optimization logic is verified
deterministically.

## Credits & references

- [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt) — agent skills as trainable parameters
- [Microsoft PromptWizard](https://github.com/microsoft/PromptWizard) — feedback-driven prompt optimization
- [DSPy / GEPA](https://dspy.ai) — reflective/evolutionary program optimization
- [npx skills](https://github.com/vercel-labs/skills) — the open agent-skills format
