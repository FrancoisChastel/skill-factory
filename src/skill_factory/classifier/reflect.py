"""Reflection beyond rewording: show the proposer what failed, at every level.

jev's two biggest gains were not edits of existing wording. Round 1 changed what
was asked (intent and mismatch instead of capability: AUC 0.39-0.77 to 0.97).
Round 2's main finding was a pipeline cause: 26 of 42 misses had never been
sent, because their state was over the budget. A proposer that only sees "here
are your questions, reword them" finds neither, so the digest carries:

* system-level causes first: misses never sent (over budget, skipped, failed),
  with the budget that would admit them;
* every question's separation on train (AUC, recall at the budget), including
  pool questions the probe set does not use;
* misses *and* false flags, each with every question's P(true).

The proposer answers in JSON: new questions (optionally under several framings),
new framings, and pipeline changes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from skill_factory.classifier.dataset import Example
from skill_factory.classifier.evaluation import QuestionStat, verdicts_for
from skill_factory.classifier.lab import AnswerTable
from skill_factory.classifier.policy import THRESHOLD_POLICY, Policy
from skill_factory.classifier.probeset import ProbeSet, describe_score
from skill_factory.classifier.questions import QuestionBank, QuestionSpec, expand_frames

_SAFE_ID = re.compile(r"[^a-z0-9_]+")

REFLECT_SYSTEM = (
    "You improve a classifier made of yes/no questions that a decision model answers with "
    "calibrated probabilities. You propose NEW questions and framings that separate the "
    "classes, and pipeline changes when failures are not about wording at all. You never "
    "write questions that name specific examples, and you prefer asking about intent and "
    "about mismatch between what something claims and what it does over asking whether it "
    "has a capability that benign examples also have. Reply with one JSON object only."
)


@dataclass(frozen=True)
class Proposal:
    rationale: str
    bank: QuestionBank
    pipeline: Mapping[str, Any] = field(default_factory=dict)

    @property
    def new_ids(self) -> list[str]:
        return self.bank.ids


def failure_digest(
    probeset: ProbeSet,
    train: Sequence[Example],
    table: AnswerTable,
    *,
    policy: Policy = THRESHOLD_POLICY,
    stats: Sequence[QuestionStat] = (),
    max_examples: int = 10,
    excerpt_chars: int = 1200,
    max_state_budget: int | None = None,
) -> str:
    """The training failures of ``probeset``, from the pipeline down to each answer."""
    verdicts = verdicts_for(train, table, probeset, policy)
    detect = policy.levels[1]
    misses = [ex for ex in train if ex.label and not policy.at_least(verdicts[ex.id], detect)]
    false_flags = [ex for ex in train if not ex.label and policy.at_least(verdicts[ex.id], detect)]
    positives = sum(1 for ex in train if ex.label)
    shown_ids = _shown_questions(probeset, stats)

    lines = [
        f"PROBE SET: score = {describe_score(probeset.score)}, threshold {probeset.threshold}, "
        f"state budget {probeset.state_budget:,} characters",
        f"TRAINING: {positives - len(misses)}/{positives} positives detected; "
        f"{len(false_flags)}/{len(train) - positives} negatives falsely flagged.",
        "",
        "SYSTEM-LEVEL CAUSES (misses the questions never had a chance on):",
        *_system_causes(misses, table, probeset, max_state_budget),
        "",
        "QUESTION SEPARATION ON TRAINING (AUC; recall at the false-positive budget, threshold fitted on train):",
        *(
            f"- {s.qid}{' [in probe set]' if s.qid in probeset.bank else ''}: "
            f"AUC {s.auc.get('train', 0.5):.3f}, recall {s.recall.get('train', 0.0):.0%} at t={s.threshold:.3f}"
            for s in stats
        ),
        "",
        f"MISSED POSITIVES THAT WERE JUDGED ({sum(1 for e in misses if table.statuses.get(e.id) == 'judged')}):",
        *_examples(misses, table, shown_ids, max_examples, excerpt_chars),
        "",
        f"FALSE FLAGS ({len(false_flags)}):",
        *_examples(false_flags, table, shown_ids, max(1, max_examples // 2), excerpt_chars),
    ]
    return "\n".join(lines)


def _system_causes(
    misses: Sequence[Example], table: AnswerTable, probeset: ProbeSet, max_budget: int | None
) -> list[str]:
    if not misses:
        return ["- none: every positive was detected"]
    causes = Counter()
    over = []
    for ex in misses:
        status = table.statuses.get(ex.id, "judged")
        if status == "judged" and probeset.score_of(table.of(ex.id)) is None:
            status = "not answered (a request failed or a question was never asked)"
        causes[status.split(":")[0] if status.startswith("skipped") else status] += 1
        if status == "over budget":
            over.append(ex.chars or 0)
    out = []
    unsent = sum(n for s, n in causes.items() if s != "judged")
    out.append(f"- {unsent} of {len(misses)} misses were never judged by the model.")
    for status, n in causes.most_common():
        if status != "judged":
            out.append(f"  - {n}: {status}")
    if over:
        over.sort()
        out.append(f"  - over-budget sizes: {over[0]:,} to {over[-1]:,} characters")
        for budget in _budget_steps(probeset.state_budget, max_budget):
            admitted = sum(1 for c in over if c <= budget)
            if admitted:
                out.append(f"  - a state_budget of {budget:,} would admit {admitted} of them")
    return out


def _budget_steps(current: int, cap: int | None) -> list[int]:
    steps = [current * 2, current * 4]
    if cap is not None:
        steps = [min(s, cap) for s in steps] + [cap]
    return sorted({s for s in steps if s > current})


def _shown_questions(probeset: ProbeSet, stats: Sequence[QuestionStat], extra: int = 5) -> list[str]:
    ids = list(probeset.score_ids)
    for s in stats:
        if len(ids) >= len(probeset.score_ids) + extra:
            break
        if s.qid not in ids:
            ids.append(s.qid)
    return ids


def _examples(
    exs: Sequence[Example], table: AnswerTable, qids: Sequence[str], limit: int, excerpt_chars: int
) -> list[str]:
    judged = [ex for ex in exs if table.statuses.get(ex.id) == "judged"]
    if not judged:
        return ["- none"]
    out = []
    for ex in judged[:limit]:
        probs = ", ".join(
            f"{q}={p:.2f}" if (p := table.p(ex.id, q)) is not None else f"{q}=?" for q in qids
        )
        out.append(f"--- {ex.id} (group {ex.group}) ---")
        if ex.state:
            text = ex.state.strip()
            out.append(text[:excerpt_chars] + (" ...[truncated]" if len(text) > excerpt_chars else ""))
        else:
            out.append("(state text not available; only cached answers)")
        out.append(f"P(true): {probs}")
    if len(judged) > limit:
        out.append(f"... and {len(judged) - limit} more")
    return out


def proposal_prompt(
    probeset: ProbeSet,
    digest: str,
    *,
    goal: str,
    frames: Mapping[str, str],
    max_new: int,
    max_state_budget: int | None,
) -> str:
    frame_lines = [f'- "{name}": {text}' for name, text in frames.items()] or ["- (none yet)"]
    current = [f"- {qid}: {q.instructions.splitlines()[-1]}" for qid, q in probeset.bank.compiled().items()]
    budget_note = (
        f'You may set "pipeline": {{"state_budget": N}} with N <= {max_state_budget:,} if misses were never sent.'
        if max_state_budget else "The state budget is fixed."
    )
    return "\n".join([
        f"GOAL: {goal or 'Separate positives from negatives at a low false-positive rate.'}",
        "",
        "EXISTING FRAMINGS (a framing is the preamble sent before a question):",
        *frame_lines,
        "",
        "CURRENT QUESTIONS (last line of each):",
        *current,
        "",
        digest,
        "",
        f"Propose up to {max_new} new questions. Each is a yes/no question with a 'true' and a 'false' "
        "criterion written as full sentences ('Yes: ...', 'No: ...'). Ask what the failures above have "
        "in common, not about any one example. You may add framings and ask a question under several.",
        budget_note,
        "",
        "Reply with exactly one JSON object:",
        '{"rationale": "what the failures share and why these questions catch it",',
        ' "frames": {"<new_frame_name>": "<framing text>"},',
        ' "questions": [{"id": "<snake_case>", "question": "<question>", "true": "Yes: ...", '
        '"false": "No: ...", "frames": ["<frame name>"]}],',
        ' "pipeline": {}}',
    ])


def parse_proposal(
    raw: str,
    existing: QuestionBank,
    *,
    max_new: int,
    max_state_budget: int | None,
) -> Proposal:
    """Parse and validate a proposer reply. Raises ValueError with the reason it is unusable."""
    data = _json_object(raw)
    frames = {str(k): str(v) for k, v in (data.get("frames") or {}).items() if str(v).strip()}
    for name, text in frames.items():
        if name in existing.frames and existing.frames[name] != text:
            raise ValueError(f"frame {name!r} already exists with different text")
    all_frames = {**existing.frames, **frames}
    specs: dict[str, QuestionSpec] = {}
    for item in list(data.get("questions") or [])[:max_new]:
        if not isinstance(item, Mapping) or not item.get("true") or not item.get("false"):
            continue
        base = _SAFE_ID.sub("_", str(item.get("id") or "q").lower()).strip("_") or "q"
        wanted = [str(f) for f in item.get("frames") or [] if str(f) in all_frames]
        if item.get("question") and wanted:
            new = expand_frames(base, item, wanted)
        elif item.get("instructions") or item.get("question"):
            text = str(item.get("instructions") or item.get("question"))
            new = {base: QuestionSpec(true=str(item["true"]), false=str(item["false"]), instructions=text)}
        else:
            continue
        for qid, spec in new.items():
            final = _unique(qid, spec, existing, specs, all_frames)
            if final not in existing:  # an identical question already asked is not new
                specs[final] = spec
    used = {s.frame for s in specs.values() if s.frame}
    bank = QuestionBank({k: v for k, v in all_frames.items() if k in used}, specs)
    pipeline: dict[str, Any] = {}
    budget = (data.get("pipeline") or {}).get("state_budget") if isinstance(data.get("pipeline"), Mapping) else None
    if budget is not None and max_state_budget:
        pipeline["state_budget"] = max(1, min(int(budget), max_state_budget))
    if not specs and not pipeline:
        raise ValueError("the proposal has no usable question and no pipeline change")
    return Proposal(str(data.get("rationale", "")).strip(), bank, pipeline)


def _unique(qid: str, spec: QuestionSpec, existing: QuestionBank, taken: Mapping, frames: Mapping[str, str]) -> str:
    """Keep the id if free or identical; otherwise suffix it, never overwrite another question."""
    candidate, n = qid, 2
    while True:
        same = candidate in existing and existing.specs[candidate].compile(existing.frames) == spec.compile(frames)
        if (candidate not in existing and candidate not in taken) or same:
            return candidate
        base, _, frame = qid.partition("@")
        candidate = f"{base}_{n}" + (f"@{frame}" if frame else "")
        n += 1


def _json_object(raw: str) -> dict:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    body = text[start : end + 1]
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        try:
            obj = json.loads(_repair(body))  # the two slips models make most: trailing commas, // comments
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON in the reply: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("the reply is not a JSON object")
    return obj


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_LINE_COMMENT = re.compile(r'^(\s*)//[^\n]*$', re.MULTILINE)


def _repair(body: str) -> str:
    return _TRAILING_COMMA.sub(r"\1", _LINE_COMMENT.sub(r"\1", body))
