"""Labeled examples for classifier mode: (state, label) pairs with group ids.

    {"id": "...", "state": "...", "label": "malicious", "group": "template-12", "source": "corpus-a"}

``state`` is the text the model reads (``input`` is accepted as an alias). An
example may come without its state but with ``state_hash``: its answers can be
rescored from the cache, but no new question can be asked about it (that is what
an imported skill-scanner lab looks like). ``skipped`` records why a pipeline
never sent an example, which the failure digest reports as a system-level cause.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from skill_factory.classifier.questions import sha24

_TRUE = {"1", "true", "yes", "positive"}
_FALSE = {"0", "false", "no", "negative"}


@dataclass(frozen=True)
class Example:
    """One labeled example.

    Attributes:
        id: Stable identifier.
        label: True for a positive (the class the classifier flags).
        state: The text sent to the model, or None when only its hash is known.
        state_hash: Cache key of the state; computed from ``state`` when absent.
        chars: Length of the state, used against the state budget.
        group: Near-duplicates share a group (one template, one repository) and
            always land in the same split. Defaults to ``id``.
        source: The corpus it came from; a whole source can be held out.
        split: An explicit split, which overrides the computed one.
        skipped: Why the pipeline never sent it, when it did not.
        metadata: Anything a policy needs (static findings, severity...).
    """

    id: str
    label: bool
    state: str | None = None
    state_hash: str | None = None
    chars: int | None = None
    group: str = ""
    source: str = ""
    split: str | None = None
    skipped: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("Example.id must be non-empty")
        if self.state is not None:
            object.__setattr__(self, "state_hash", self.state_hash or sha24(self.state))
            object.__setattr__(self, "chars", len(self.state))
        if not self.group:
            object.__setattr__(self, "group", self.id)

    def status(self, budget: int) -> str:
        """``judged`` when it can be sent within ``budget``, else the reason it is not."""
        if self.skipped:
            return f"skipped: {self.skipped}"
        if self.chars is not None and self.chars > budget:
            return "over budget"
        if self.state_hash is None:
            return "no state"
        return "judged"


def parse_label(value: Any, positive: str, negative: str) -> bool:
    """A label as a bool. Strings must name one of the two classes exactly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip()
    if text == positive:
        return True
    if text == negative:
        return False
    low = text.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"label {value!r} is neither {positive!r} nor {negative!r}")


def examples_from_records(
    records: Sequence[Mapping[str, Any]], *, positive: str = "positive", negative: str = "negative"
) -> list[Example]:
    out: list[Example] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        if "id" not in rec:
            raise ValueError(f"record {i} has no 'id'")
        if "label" not in rec:
            raise ValueError(f"record {rec['id']!r} has no 'label'")
        ex_id = str(rec["id"])
        if ex_id in seen:
            raise ValueError(f"duplicate example id {ex_id!r}")
        seen.add(ex_id)
        state = rec.get("state", rec.get("input"))
        if state is not None and not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False, indent=2)
        out.append(
            Example(
                id=ex_id,
                label=parse_label(rec["label"], positive, negative),
                state=state,
                state_hash=rec.get("state_hash"),
                chars=_opt_int(rec.get("chars")),
                group=str(rec.get("group") or ""),
                source=str(rec.get("source") or ""),
                split=rec.get("split"),
                skipped=rec.get("skipped"),
                metadata=dict(rec.get("metadata") or {}),
            )
        )
    if not out:
        raise ValueError("the dataset is empty")
    return out


def load_examples(path: str | Path, *, positive: str = "positive", negative: str = "negative") -> list[Example]:
    """Load examples from JSONL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")
    records = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
    return examples_from_records(records, positive=positive, negative=negative)


def save_examples(examples: Sequence[Example], path: str | Path, *, positive: str, negative: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ex in examples:
        row: dict[str, Any] = {"id": ex.id, "label": positive if ex.label else negative}
        if ex.state is not None:
            row["state"] = ex.state
        else:
            row["state_hash"] = ex.state_hash
            row["chars"] = ex.chars
        row.update({"group": ex.group, "source": ex.source})
        for key in ("split", "skipped"):
            if getattr(ex, key):
                row[key] = getattr(ex, key)
        if ex.metadata:
            row["metadata"] = dict(ex.metadata)
        lines.append(json.dumps(row, ensure_ascii=False))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)
