"""The Skill primitive: a portable SKILL.md document treated as a trainable parameter.

A skill is immutable. Every "edit" during optimization produces a *new* Skill via
:meth:`Skill.with_body`, so a candidate can never silently mutate its parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

_FRONTMATTER_DELIMITER = "---"


@dataclass(frozen=True)
class Skill:
    """A single agent skill.

    Attributes:
        name: Short skill identifier (SKILL.md frontmatter ``name``).
        description: When-to-use description (SKILL.md frontmatter ``description``).
        body: The markdown instruction body — this is the text being optimized.
        extra: Any additional frontmatter keys, preserved on round-trip.
    """

    name: str
    description: str
    body: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Skill.name must be a non-empty string")
        if not isinstance(self.body, str):
            raise TypeError("Skill.body must be a string")

    def with_body(self, body: str) -> "Skill":
        """Return a new Skill with a replaced body (the optimizer's core operation)."""
        return replace(self, body=body)

    def with_metadata(self, **changes: Any) -> "Skill":
        """Return a new Skill with replaced top-level fields (name/description)."""
        return replace(self, **changes)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (~4 chars/token) of the full document."""
        return max(1, len(self.to_markdown()) // 4)

    def to_markdown(self) -> str:
        """Serialize back to a canonical SKILL.md string with YAML frontmatter."""
        frontmatter: dict[str, Any] = {"name": self.name, "description": self.description}
        frontmatter.update(dict(self.extra))
        fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
        body = self.body.strip()
        return f"{_FRONTMATTER_DELIMITER}\n{fm_yaml}\n{_FRONTMATTER_DELIMITER}\n\n{body}\n"

    @classmethod
    def from_markdown(cls, text: str) -> "Skill":
        """Parse a SKILL.md string. Frontmatter is optional but recommended.

        Raises:
            ValueError: if frontmatter is present but malformed, or ``name`` is missing.
        """
        stripped = text.lstrip("﻿")  # tolerate a leading BOM
        if stripped.startswith(_FRONTMATTER_DELIMITER):
            frontmatter, body = _split_frontmatter(stripped)
        else:
            frontmatter, body = {}, stripped

        name = frontmatter.pop("name", None)
        description = frontmatter.pop("description", "")
        if not name:
            raise ValueError(
                "SKILL.md is missing a frontmatter 'name'. Add:\n---\nname: my-skill\n"
                "description: ...\n---"
            )
        return cls(
            name=str(name),
            description=str(description),
            body=body.strip(),
            extra=frontmatter,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Skill":
        """Load a Skill from a SKILL.md file on disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Skill file not found: {p}")
        return cls.from_markdown(p.read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        """Write the Skill to disk as SKILL.md and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")
        return p


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``---\\n...\\n---\\n<body>`` into (frontmatter dict, body string)."""
    lines = text.splitlines()
    # lines[0] is the opening delimiter; find the closing one.
    closing_index = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIMITER:
            closing_index = i
            break
    if closing_index is None:
        raise ValueError("SKILL.md frontmatter opened with '---' but never closed")

    fm_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])
    try:
        parsed = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    return parsed, body
