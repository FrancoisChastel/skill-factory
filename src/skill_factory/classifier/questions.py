"""Questions: the text parameters of a probe set.

A question is a System One ``choice`` question with two criteria, ``true`` and
``false``; the model returns P(true). A question can be written whole
(``instructions``) or as a ``question`` under a named ``frame``, the shared
preamble that tells the model how to read the state. The same question under two
framings is two questions to the model, so they are compiled to two ids,
``<id>@<frame>``, and scored separately.

The hash of a compiled question is the cache key of its answers. It is the hash
skill-scanner's lab uses (``scripts/judge-lab.ts``), so answers cached there are
answers here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@:-]{0,99}$")


def sha24(text: str) -> str:
    """First 24 hex characters of SHA-256, the lab's key length."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class Question:
    """A compiled question, exactly as sent to the model."""

    instructions: str
    true: str
    false: str

    def __post_init__(self) -> None:
        for name in ("instructions", "true", "false"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Question.{name} must be non-empty")

    @property
    def hash(self) -> str:
        # Same bytes as JSON.stringify([instructions, true, false]) in the lab.
        payload = json.dumps(
            [self.instructions, self.true, self.false], separators=(",", ":"), ensure_ascii=False
        )
        return sha24(payload)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": {"true": self.true, "false": self.false},
        }


@dataclass(frozen=True)
class QuestionSpec:
    """A question as written: whole, or a ``question`` under a ``frame``."""

    true: str
    false: str
    instructions: str | None = None
    frame: str | None = None
    question: str | None = None

    def __post_init__(self) -> None:
        whole = self.instructions is not None
        framed = self.frame is not None and self.question is not None
        if whole == framed:
            raise ValueError("a question needs either 'instructions' or both 'frame' and 'question'")

    def compile(self, frames: Mapping[str, str]) -> Question:
        if self.instructions is not None:
            return Question(self.instructions, self.true, self.false)
        frame = self.frame or ""
        if frame not in frames:
            raise ValueError(f"unknown frame {frame!r}")
        # The framing, a blank line, then the question: the layout jev was tuned on.
        return Question(f"{frames[frame]}\n\n{self.question}", self.true, self.false)

    def to_dict(self) -> dict[str, str]:
        if self.instructions is not None:
            return {"instructions": self.instructions, "true": self.true, "false": self.false}
        return {"frame": str(self.frame), "question": str(self.question), "true": self.true, "false": self.false}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuestionSpec":
        if not isinstance(data, Mapping):
            raise ValueError(f"a question must be a mapping, got {type(data).__name__}")
        missing = [k for k in ("true", "false") if not data.get(k)]
        if missing:
            raise ValueError(f"question is missing {', '.join(missing)}")
        return cls(
            true=str(data["true"]),
            false=str(data["false"]),
            instructions=_opt(data.get("instructions")),
            frame=_opt(data.get("frame")),
            question=_opt(data.get("question")),
        )


@dataclass(frozen=True)
class QuestionBank:
    """Named framings plus question specs: a pool of candidates, or a probe set's questions."""

    frames: Mapping[str, str] = field(default_factory=dict)
    specs: Mapping[str, QuestionSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for qid in self.specs:
            if not ID_RE.match(qid):
                raise ValueError(f"invalid question id {qid!r}")
        for qid, spec in self.specs.items():
            if spec.frame is not None and spec.frame not in self.frames:
                raise ValueError(f"question {qid!r} uses unknown frame {spec.frame!r}")

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, qid: object) -> bool:
        return qid in self.specs

    @property
    def ids(self) -> list[str]:
        return list(self.specs)

    def compiled(self) -> dict[str, Question]:
        return {qid: spec.compile(self.frames) for qid, spec in self.specs.items()}

    def subset(self, ids: list[str] | set[str]) -> "QuestionBank":
        unknown = sorted(set(ids) - set(self.specs))
        if unknown:
            raise ValueError(f"unknown question ids: {unknown}")
        specs = {qid: spec for qid, spec in self.specs.items() if qid in ids}
        used = {s.frame for s in specs.values() if s.frame is not None}
        return QuestionBank({k: v for k, v in self.frames.items() if k in used}, specs)

    def merge(self, other: "QuestionBank") -> "QuestionBank":
        """Union of two banks. The same id or frame with different text is an error, never an overwrite."""
        frames = dict(self.frames)
        for name, text in other.frames.items():
            if name in frames and frames[name] != text:
                raise ValueError(f"frame {name!r} already exists with different text")
            frames[name] = text
        specs = dict(self.specs)
        for qid, spec in other.specs.items():
            if qid in specs and specs[qid].compile(frames) != spec.compile(frames):
                raise ValueError(f"question {qid!r} already exists with different text")
            specs[qid] = spec
        return QuestionBank(frames, specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": dict(self.frames),
            "questions": {qid: spec.to_dict() for qid, spec in self.specs.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuestionBank":
        """Read ``{"frames", "questions"}``, or the lab's flat ``{id: {instructions, true, false}}``."""
        if not isinstance(data, Mapping):
            raise ValueError("a question bank must be a mapping")
        if "questions" in data and isinstance(data["questions"], Mapping):
            frames = {str(k): str(v) for k, v in (data.get("frames") or {}).items()}
            raw = data["questions"]
        else:
            frames, raw = {}, data
        return cls(frames, {str(qid): QuestionSpec.from_dict(spec) for qid, spec in raw.items()})

    @classmethod
    def load(cls, path: str | Path) -> "QuestionBank":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"question file not found: {p}")
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}: invalid JSON: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return p


def expand_frames(
    base_id: str, spec: Mapping[str, Any], frame_names: list[str]
) -> dict[str, QuestionSpec]:
    """One written question under several framings: ``{"<id>@<frame>": spec}``."""
    out = {}
    for frame in frame_names:
        out[f"{base_id}@{frame}"] = QuestionSpec(
            true=str(spec["true"]), false=str(spec["false"]), frame=frame, question=str(spec["question"])
        )
    return out


def with_frame_text(bank: QuestionBank, name: str, text: str) -> QuestionBank:
    """A new bank with one more framing (an error if the name exists with other text)."""
    return bank.merge(QuestionBank({name: text}, {}))


def _opt(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ID_RE",
    "Question",
    "QuestionBank",
    "QuestionSpec",
    "expand_frames",
    "sha24",
    "with_frame_text",
]
