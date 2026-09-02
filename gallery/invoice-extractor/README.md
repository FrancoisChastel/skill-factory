# invoice-extractor

Extract `vendor`, `date`, `total`, `currency` from messy invoice/receipt text as
strict JSON — normalizing localized dates, decimal separators, and currency symbols.

## Result

| | |
| --- | --- |
| **Baseline → Best (validation)** | **0.559 → 1.000** (+44.1 pts) |
| Optimizer | `llm_loop` (reflective, validation-gated) |
| Target / optimizer model | `gemma-4-31b-it-mlx` via LM Studio (fully local) |
| Metric | golden `json_equal` (w2) + programmatic JSON checks (w1) + LLM judge (w1) |
| Rounds | 3 (best found in round 1; later edits gated out) |

Candidate history from the live run:

| iter | val | gate | what the optimizer did |
| ---: | --: | :--: | :--- |
| 0 | 0.559 | ✓ | seed (baseline) |
| 1 | 1.000 | ✓ | fixed key naming, banned code fences/prose, added ISO-8601 + numeric normalization rules |
| 2 | 1.000 | · | rejected — no improvement over best |
| 3 | 1.000 | · | rejected — no improvement over best |

## Reproduce

Seed, dataset (12 labeled invoices across locales/currencies), and config live in
[`examples/invoice-extractor/`](../../examples/invoice-extractor/):

```bash
skill-factory optimize -c examples/invoice-extractor/config.lmstudio.yaml
```

## Artifact

[`optimized_skill.md`](optimized_skill.md) — the winning skill exactly as emitted
by the run (schema, output-format constraints, extraction rules).
