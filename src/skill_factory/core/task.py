"""The Task/Dataset primitives: the examples a skill is trained and validated on."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class Task:
    """A single evaluation example.

    Attributes:
        id: Stable identifier (used for deterministic splitting and logging).
        input: The user-facing input handed to the skill (string or serializable).
        expected: Optional gold answer, required only by golden-set metrics.
        metadata: Free-form tags (difficulty, category, source, ...).
    """

    id: str
    input: str
    expected: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not str(self.id).strip():
            raise ValueError("Task.id must be a non-empty string")
        if self.input is None:
            raise ValueError(f"Task {self.id!r} has no input")


class Dataset:
    """An ordered, immutable collection of tasks with deterministic splitting."""

    def __init__(self, tasks: Sequence[Task]):
        if not tasks:
            raise ValueError("Dataset must contain at least one task")
        ids = [t.id for t in tasks]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Dataset has duplicate task ids: {dupes}")
        self._tasks: tuple[Task, ...] = tuple(tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    def __getitem__(self, index: int) -> Task:
        return self._tasks[index]

    @property
    def tasks(self) -> tuple[Task, ...]:
        return self._tasks

    def split(self, val_fraction: float = 0.3, seed: int = 0) -> tuple["Dataset", "Dataset"]:
        """Deterministically split into (train, val).

        Splitting is stable across runs given the same ``seed`` because tasks are
        first sorted by id, then shuffled with a seeded RNG. A validation set is
        mandatory for gated optimization, so both sides always get >= 1 task.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between 0 and 1")
        if len(self) < 2:
            raise ValueError("Cannot split a dataset with fewer than 2 tasks")

        ordered = sorted(self._tasks, key=lambda t: t.id)
        rng = random.Random(seed)
        rng.shuffle(ordered)

        n_val = max(1, round(len(ordered) * val_fraction))
        n_val = min(n_val, len(ordered) - 1)  # keep at least one train task
        val = ordered[:n_val]
        train = ordered[n_val:]
        return Dataset(train), Dataset(val)

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> "Dataset":
        """Build a Dataset from plain dicts (``id``/``input``/``expected``/``metadata``)."""
        tasks = []
        for i, rec in enumerate(records):
            if "input" not in rec:
                raise ValueError(f"Record {i} is missing required 'input' field")
            tasks.append(
                Task(
                    id=str(rec.get("id", i)),
                    input=_coerce_input(rec["input"]),
                    expected=_coerce_optional(rec.get("expected")),
                    metadata=dict(rec.get("metadata", {})),
                )
            )
        return cls(tasks)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        """Load a Dataset from a JSONL file (one task record per line)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        records = []
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
        return cls.from_records(records)


def _coerce_input(value: Any) -> str:
    """Inputs are stored as strings; dicts/lists are JSON-encoded for prompt use."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _coerce_optional(value: Any) -> str | None:
    if value is None:
        return None
    return _coerce_input(value)
