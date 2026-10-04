"""Import a skill-scanner judge lab (``scripts/judge-lab.ts``) as a classifier workspace.

The lab wrote ``bundles-<budget>.jsonl`` (one skill per line: directory, corpus,
label, split, state hash, static findings) and ``answers.jsonl`` (the answer
cache, same format as ours). Importing turns the bundles into examples that
carry their state hash but not their text: every cached answer can be rescored,
selected and replayed through a policy, and the splits are the lab's own.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from skill_factory.classifier.dataset import Example, save_examples
from skill_factory.classifier.questions import QuestionBank

# skill-scanner's reason for a state over the judge's budget (src/judge/state.ts).
_OVER_BUDGET = re.compile(r"skill text is (\d+) characters, over the judge's (\d+) budget")


@dataclass(frozen=True)
class ImportResult:
    dataset: Path
    answers: Path
    pool: Path | None
    examples: int


def import_scanner_lab(
    lab_dir: str | Path,
    out_dir: str | Path,
    *,
    budget: int = 96_000,
    questions: str | Path | None = None,
    positive: str = "malicious",
    negative: str = "benign",
) -> ImportResult:
    lab = Path(lab_dir)
    bundles_path = lab / f"bundles-{budget}.jsonl"
    answers_path = lab / "answers.jsonl"
    for p in (bundles_path, answers_path):
        if not p.exists():
            raise FileNotFoundError(f"not a skill-scanner lab: {p} is missing")
    rows = []
    for lineno, line in enumerate(bundles_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{bundles_path}:{lineno}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{bundles_path} has no bundles")
    root = os.path.commonpath([r["dir"] for r in rows]) if len(rows) > 1 else str(Path(rows[0]["dir"]).parent)

    examples = []
    for r in rows:
        rel = os.path.relpath(r["dir"], root)
        skipped = r.get("skipped") or (None if r.get("stateHash") else "not sent by the lab")
        chars = r.get("chars")
        over = _OVER_BUDGET.search(skipped or "")
        if over:
            # Keep the size, not the reason: the budget it is compared with is the workspace's.
            chars, skipped = int(over.group(1)), None
        examples.append(
            Example(
                id=rel,
                label=r["label"] == positive,
                state_hash=r.get("stateHash"),
                chars=chars,
                group=rel,
                source=str(r.get("corpus", "")),
                split=r.get("split"),
                skipped=skipped,
                metadata={"findings": r.get("findings", []), "kind": r.get("kind", "")},
            )
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = save_examples(examples, out / "dataset.jsonl", positive=positive, negative=negative)
    answers = out / "answers.jsonl"
    shutil.copyfile(answers_path, answers)  # a copy: the workspace appends to its own cache
    pool = None
    if questions is not None:
        pool = QuestionBank.load(questions).save(out / "pool.json")
    return ImportResult(dataset, answers, pool, len(examples))
