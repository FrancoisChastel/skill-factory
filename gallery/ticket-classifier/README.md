# ticket-classifier

Route a customer support message to exactly one of `billing`, `bug`,
`feature_request`, `account`, `other` — output is a single bare lowercase label.

## Result

| | |
| --- | --- |
| **Baseline → Best (validation)** | **0.151 → 1.000** (+84.9 pts) |
| Optimizer | `llm_loop` (reflective, validation-gated) |
| Target / optimizer model | `gemma-4-31b-it-mlx` via LM Studio (fully local) |
| Metric | golden `normalized` (w2) + label-regex check (w1) + LLM judge (w1) |
| Rounds | 4 (best found in round 1; later edits gated out) |

Candidate history from the live run:

| iter | val | gate | what the optimizer did |
| ---: | --: | :--: | :--- |
| 0 | 0.151 | ✓ | seed (baseline) |
| 1 | 1.000 | ✓ | replaced vague instructions with a closed label set, per-category definitions, and strict bare-label formatting |
| 2 | 1.000 | · | rejected — no improvement over best |
| 3 | 1.000 | · | rejected — no improvement over best |

## Reproduce

Seed, dataset (15 labeled tickets), and config live in
[`examples/ticket-classifier/`](../../examples/ticket-classifier/):

```bash
skill-factory optimize -c examples/ticket-classifier/config.lmstudio.yaml
```

## Artifact

[`optimized_skill.md`](optimized_skill.md) — the winning skill exactly as emitted
by the run (closed category list with boundaries, strict output constraints).
