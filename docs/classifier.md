# Classifier mode

Skill mode trains one `SKILL.md`, scored rollout by rollout. Classifier mode trains
what a decision model needs to classify well: a **probe set**. A probe set is a few
yes/no questions (each possibly under several framings), a score that combines
their answers, a threshold, the state budget, and the policy that turns a score
into a verdict. The model is a System One decision model such as
[jev](https://typesafe.ai), which returns a calibrated P(true) per question.

It is the workflow that took skill-scanner's jev judge from catching 16% of
malicious skills to 78% at under 0.5% false flags, for $1.27 of tuning,
packaged so it can be repeated on any classification task.

```bash
skill-factory optimize -c examples/pr-mismatch/config.yaml   # kind: classifier
skill-factory lab freeze -c examples/pr-mismatch/config.yaml
skill-factory lab report -c examples/pr-mismatch/config.yaml --ask
skill-factory lab export --probeset runs/pr-mismatch/probeset.json --out src/probes.ts
```

## How it differs from skill mode

| | Skill mode | Classifier mode |
|---|---|---|
| Trainable artifact | `SKILL.md` text | probe set: questions, framings, score, threshold, budget |
| Unit of cost | one rollout per task per candidate | one request per example answers up to 40 questions |
| Metric | mean of per-rollout scores | recall at a false-positive budget, AUC (dataset-level) |
| Gate | beats validation | beats validation **and** breaks no constraint |
| Splits | random train/val | group-aware train/val/test, held-out groups, sealed test |
| Output | `best_skill.md` | `probeset.json`, per-split report, figure, exports with a hash |

## The pieces

### 1. The probe set (`probeset.json`)

```json
{
 "name": "pr-mismatch",
 "model": "jev-latest",
 "pinned_version": "jev-1.13.0",
 "state_budget": 24000,
 "frames": {"reviewer": "You are reviewing a pull request before it is merged. ..."},
 "questions": {
  "undisclosed_change@reviewer": {"frame": "reviewer", "question": "Does the diff contain a change ...?",
                                  "true": "Yes: ...", "false": "No: ..."}
 },
 "score": {"mean": [{"mean": ["q1", "q2", "q3"]}, "lure"]},
 "threshold": 0.075,
 "sha256": "..."
}
```

A question is sent as its frame, a blank line, then the question. The same question
under two frames is two questions to the model (`id@frame`) and is scored
separately. Score expressions are JSON: a question id, a number, `mean`, `max`,
`min`, `sum`, `weighted` (`[[w, expr], ...]`) or `scale` (`[expr, k]`). jev's
shipped score is `{"mean": [{"mean": [five intent questions]}, "lure"]}` at 0.075.

`sha256` covers what decides a verdict (compiled questions, score, threshold, model,
budget). Editing a measured probe set by hand makes it fail to load until you remove
the hash, so an unmeasured edit cannot pass for a measured one.

### 2. The lab: batch, cache, rescore offline

The state is billed once per request, so 40 candidate questions cost about what one
costs. The lab caches every answer by (state hash, question hash, model) in
`answers.jsonl`, asks only what is missing, appends answers as they arrive, tracks
tokens and dollars, and stops scheduling before `systemone.max_usd`. Everything
else (thresholds, ensembles, policies, reports) is scored from the cache for free.

The cache format and question hash are skill-scanner's, so a lab cached there is
usable here: `skill-factory lab import-scanner <lab-dir> --out <dir> --questions q.json`.

### 3. Dataset-level metrics and constraints

The objective is recall at a false-positive budget: the threshold is fitted on
train at `objective.fpr_budget`, then recall is read on validation. Constraints
are checked against the incumbent:

```yaml
objective:
  fpr_budget: 0.005
  constraints:
    - {split: val, max_fpr: 0.01}
    - {split: val, level: block, max_added_false: 0}   # no new false blocks
    - {split: train, level: warn, max_added_false: 4}  # at most 4 more warnings
```

Without constraints, validation FPR is held to twice the budget. Recall counts
every positive: one the pipeline never sent (over budget, failed, no state) is a
miss, not an exclusion.

### 4. Honest splits

Examples carry a `group` (a template, a repository) and a `source` (a corpus). The
split is a hash of the group, so near-duplicates never straddle train and test.
Whole sources or groups can be held out; examples read before the split was chosen
go to train (`read_before_split`). A group found in two splits is an error.

Test and held-out are **sealed**. The optimizer never receives them, `lab ask`,
`score` and `simulate` refuse them, and `lab report` refuses them until `lab freeze`
records the probe set's hash. Every opening is logged in `seal.json`; refreezing
after a look needs `--force`, and the report then says the test set was seen under
an earlier configuration.

Every run writes `report.md` and `splits.svg`: before and after on each split, so a
gain that holds on train and vanishes on validation is visible at a glance.

### 5. Reflection beyond rewording

Each reflective round shows the proposer LLM a failure digest that starts with
system-level causes, because jev's biggest gains were not rewordings:

```
SYSTEM-LEVEL CAUSES (misses the questions never had a chance on):
- 26 of 48 misses were never judged by the model.
  - 26: over budget
  - over-budget sizes: 25,414 to 1,233,351 characters
  - a state_budget of 48,000 would admit 6 of them
  - a state_budget of 96,000 would admit 12 of them
```

(That is the real digest of jev's shipped questions on skill-scanner's training
split at its original 24,000-character budget, computed from the imported lab.)

then every question's training AUC and recall at the budget (including pool
questions the probe set does not use), then misses **and** false flags with every
question's P(true). The proposer answers with new questions, new framings, and
optionally `{"pipeline": {"state_budget": N}}` (capped by `optimizer.max_state_budget`).

### 6. Selection and ensembling

Before any reflection, and after each, every candidate is fitted on train at the
same budget: each single question; the mean of the top k distinct questions by
AUC; a greedy union of (question, threshold) pairs; and a mean plus the one lure
question that adds the most. The best few of each family, at the budget and at
half of it, go to the validation gate. Single questions with very low thresholds
tend to hold their false-positive rate on train and lose it on validation; the
gate's constraints are what catch that.

A fitted threshold never spends the false-positive budget where it buys no recall:
it starts at the lowest threshold the budget allows, then moves to the middle of
the gap between the lowest positive it catches and the highest negative below it.
On the training scores recall can only rise and false flags only fall, and new
examples get a margin on both sides. (The per-question table keeps the lowest
threshold, skill-scanner's definition, so its numbers compare with the jev work.)

### 7. Offline policy simulation

A policy is a Python function `(example, answers, score, threshold) -> verdict`
with ordered `LEVELS`. `example.metadata` carries whatever the pipeline knows
(static findings, severities), so a whole judge can be replayed:

```bash
skill-factory lab simulate -c config.yaml --policy policy.py:doubt_add --policy policy.py:doubt_confirm_add
```

A policy file is Python and runs with your permissions, like the config naming it.

### 8. Noise and drift

- `lab reask --split val` asks again into a separate replicate of the cache and
  reports identical answers, mean and largest change, and verdicts flipped at the
  threshold.
- The probe set records the version that answered (`pinned_version`). Scoring
  answers from another version warns, and `lab recalibrate --model <new>` refits the
  threshold on train at the calibrated budget and shows how validation moved.
- `lab status` and every report give the cost of one pass of the probe set.

### 9. Exporters and parity

```bash
skill-factory lab export --probeset probeset.json --out src/judge/probes.ts   # or .py, .json
skill-factory lab parity --probeset probeset.json --artifact src/judge/probes.ts
skill-factory lab parity --probeset probeset.json --production prod.jsonl -c config.yaml --split train
```

Exports carry the hash, threshold, model, budget, every question exactly as sent,
and a `score()` compiled from the expression (unanswered questions give NaN, so no
verdict). The artifact check regenerates the export and compares byte for byte;
the production check compares production's verdicts on a split with the lab's,
within a tolerance on recall and false-positive rate.

## The reverse direction: jev as a skill metric

Rubric checks like "is the output strict JSON?" don't need an LLM judge writing
prose. A System One model answers them with calibrated probabilities, cached per
rollout. Live, on an invoice-extraction output: the correct JSON scored 0.99; the same
JSON wrapped in prose and a code fence got `strict_json` P = 0.00; JSON with a vendor
and total the input never mentioned got `grounded` P = 0.00.

```yaml
metric:
  golden: {mode: json_equal, weight: 2}
  systemone:
    provider: typesafe
    checks:
      - {name: strict_json, question: "Is the OUTPUT strict JSON, with no prose or fences?",
         true: "Yes: strict JSON only.", false: "No: prose, fences, or invalid JSON."}
```

## Live run: the pr-mismatch example

`examples/pr-mismatch` flags pull requests whose diff does something the title and
description do not disclose: 44 synthetic PRs in 16 template groups, seeded with four
capability questions (does it touch CI? delete code? change access? change a value?)
and a pool of six mismatch questions under two framings. Run against jev
(`jev-1.13.0`):

- **The capability questions separate at chance or worse** (train AUC 0.35 to 0.52 for
  CI, access and deletion), because the consistent PRs are scary on purpose: they
  delete files, edit CI and change permissions, and say so. Fitted at a 10%
  false-positive budget, the seed flags nothing.
- **Every mismatch question separates** (train AUC 1.000, validation 0.94 to 1.00).
  Selection keeps one, "Does the diff contain a change that the title and description
  do not mention?", at >= 0.22: training 100% at 0% false flags, validation 80% at 0%.
- **The sealed test split, opened once after `lab freeze`:** 3 of 3 caught, 0 of 9
  consistent PRs flagged. A re-ask of validation: 70% of answers identical, mean
  change 0.004, no verdict flipped.
- **Cost:** 32 requests, 60,700 input tokens, $0.0025, three seconds.
- **Reflective rounds** (`config.lmstudio.yaml`, a local `gpt-oss-120b` proposing):
  six new questions cost $0.0020 to ask; the best reached validation AUC 1.000 but did
  not beat the incumbent, so the gate kept it.

It is a small dataset (3 test positives), so read it as the workflow working end to
end, not as a benchmark.

## Case study: jev in skill-scanner, rescored here

The skill-scanner lab's cached answers (1,666 skills, 47 candidate questions, 70k
answers) import as-is. Everything below was recomputed in skill-factory from that
cache, without a single call:

| What | Training | Validation |
|---|---:|---:|
| Released judge: four capability probes at P >= 0.85 (96,000-character budget) | 19.0% at 1.52% false flags | 29.0% at 1.28% |
| Shipped score: mean of 5 intent questions + lure, >= 0.075 | 78.8% at 0.33% | 79.6% at 0.64% |
| `lab select`, greedy union of 4 (as skill-scanner's `select.py`) | 83.7% at 0.43% | 87.1% at **1.50%** (refused: FPR) |
| `optimize`, selection only: mean of 5 distinct intent questions + lure, >= 0.0755 | 81.5% at 0.43% | 83.9% at 0.43% |

The optimizer found the shipped structure on its own in three seconds: an average
of five intent questions plus the fake-prerequisite lure, at a threshold of 0.0755
against the 0.075 chosen by hand. It also refused the union,
whose false flags tripled from train to validation, the overfit the jev work
found by hand. Its validation number is optimistic, because validation chose
among a few candidates; skill-scanner's sealed splits are the unbiased check
(published: 73.2% on the test split and 69.9% on held-out corpora).

The same import reproduces the rest of the jev write-up from cache: per-question
AUCs (0.940 / 0.952 for the best intent question, 0.441 / 0.526 for "does it send
data out?"); the policy replay, where static rules flag 44.0% / 54.8% and static
plus jev flags 85.3% / 86.0%, and confirming adds a false block on each split; and
the re-ask run, with 81% identical answers, a mean change of 0.0035, and 3 of 505
verdicts flipped at the threshold.

```bash
skill-factory lab import-scanner <skill-scanner lab dir> --out jev --questions q-all.json --budget 96000
# write jev/config.yaml: kind classifier, dataset dataset.jsonl, labels malicious/benign, workspace .
skill-factory lab score -c jev/config.yaml --probeset shipped.json --questions
skill-factory lab simulate -c jev/config.yaml --policy judge_policy.py:static --policy judge_policy.py:doubt_add
skill-factory optimize -c jev/config.yaml
```

## Config reference

| Key | Meaning |
|---|---|
| `kind: classifier` | selects this mode (`optimize --kind classifier` also works) |
| `dataset` | JSONL: `id`, `state` (or `input`), `label`, `group`, `source`, optional `split`, `skipped`, `metadata` |
| `labels` | `{positive, negative}` names of the two classes |
| `probeset` | the seed probe set; `threshold: null` means fit it on train |
| `pool` | more questions to select from (`{frames, questions}` or the lab's flat format) |
| `workspace` | where the cache, reports and seal live |
| `splits` | `fractions`, `held_out_sources`, `held_out_groups`, `read_before_split`, `salt` |
| `systemone` | `provider` (typesafe, openrouter, vercel, cloudflare, ollama, custom), `model`, `base_url`, `api_key`, `per_request`, `concurrency`, `usd_per_million_input_tokens`, `max_usd` |
| `objective` | `fpr_budget`, `split`, `level`, `min_delta`, `constraints` |
| `policy` | `file.py:function`, default: flag at the threshold |
| `optimizer` | `rounds`, `patience`, `max_questions`, `max_new_questions`, `max_state_budget`, `goal`, `margins`, `gate_per_family` |
| `optimizer_provider` | the LLM that proposes questions; omit it for selection only |
| `keep_questions` | questions kept for the policy even when the score does not read them |

Keys: `SKILL_FACTORY_SYSTEMONE_KEY`, or the provider's own (`TYPESAFE_API_KEY`,
`OPENROUTER_API_KEY`, `AI_GATEWAY_API_KEY`, `CLOUDFLARE_API_TOKEN`). A state is never
sent over plain HTTP except to localhost, and a key never appears in an error.
