# Skill Gallery

Optimized skills produced with Skill Factory, with the numbers to back them up.
Every entry links its seed, config, and dataset so the result is **reproducible**,
and ships the optimized `SKILL.md` as a reference artifact you can install today.

## Entries

| Skill | Task | Optimizer | Target model | Baseline → Best |
| ----- | ---- | --------- | ------------ | --------------- |
| [invoice-extractor](invoice-extractor/) | JSON extraction | `llm_loop` | gemma-4-31b-it-mlx (local) | **0.559 → 1.000** (+44 pts) |
| [ticket-classifier](ticket-classifier/) | classification | `llm_loop` | gemma-4-31b-it-mlx (local) | **0.151 → 1.000** (+85 pts) |

Install any entry (export renames it to the `SKILL.md` layout `npx skills` expects):

```bash
skill-factory export --skill gallery/ticket-classifier/optimized_skill.md -o dist
npx skills add dist/ticket-classifier
```

## Contributing an entry

PRs adding gallery entries are very welcome. An entry must include:

1. **`README.md`** — what the skill does, the metric mixture, optimizer + models
   used, and the measured `baseline → best` on the validation split.
2. **`optimized_skill.md`** — the winning skill (valid SKILL.md frontmatter:
   `name` + `description`).
3. **Reproducibility** — either link a config + dataset under `examples/`, or
   include `config.yaml` + `dataset.jsonl` (≥ 10 tasks) in the entry itself.
4. **Honest numbers** — report the validation score from `runs/<name>/report.md`,
   not a hand-picked subset. Include the optimizer's candidate history if
   interesting.

Checklist before opening the PR:

- [ ] `pytest` passes (the gallery guard test validates your entry parses)
- [ ] Dataset contains no secrets or personal data
- [ ] The skill body is general — no answers to specific dataset tasks baked in
