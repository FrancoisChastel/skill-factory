"""Emit optimized skills in the open ``npx skills`` layout.

The open agent-skills ecosystem (``npx skills``, used by Claude Code, Codex,
Cursor, Gemini CLI, Copilot, OpenCode, ...) installs each skill as a directory
containing a ``SKILL.md`` whose frontmatter has a ``name`` and ``description``.
A repo can hold many skills under a ``skills/`` folder.

    my-skill/SKILL.md                # single skill (installable directly)
    skills/my-skill/SKILL.md         # inside a collection repo
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from skill_factory.core.skill import Skill

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ExportResult:
    """Where a skill was written."""

    skill_dir: Path
    skill_md: Path


def slugify(name: str) -> str:
    """Turn a skill name into a filesystem/URL-safe slug."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"skill name {name!r} produced an empty slug")
    return slug


def export_skill(
    skill: Skill,
    out_dir: str | Path,
    *,
    as_collection: bool = False,
    extra_files: Mapping[str, str] | None = None,
) -> ExportResult:
    """Write ``skill`` as an ``npx skills``-compatible directory.

    Args:
        skill: The skill to emit (must have name + description for npx skills).
        out_dir: Destination root.
        as_collection: If True, nest under ``out_dir/skills/<slug>`` so the
            repo can hold multiple skills; otherwise ``out_dir/<slug>``.
        extra_files: Optional ``{relative_path: content}`` bundled resources
            (e.g. ``reference.md``, ``scripts/run.py``).

    Raises:
        ValueError: if the skill lacks a description (required by npx skills).
    """
    if not skill.description or not skill.description.strip():
        raise ValueError(
            "npx skills requires a non-empty frontmatter 'description'; "
            f"skill {skill.name!r} has none"
        )

    slug = slugify(skill.name)
    root = Path(out_dir)
    skill_dir = (root / "skills" / slug) if as_collection else (root / slug)
    skill_dir.mkdir(parents=True, exist_ok=True)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(skill.to_markdown(), encoding="utf-8")

    for rel_path, content in (extra_files or {}).items():
        target = skill_dir / rel_path
        if not _is_within(skill_dir, target):
            raise ValueError(f"extra_files path escapes skill dir: {rel_path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return ExportResult(skill_dir=skill_dir, skill_md=skill_md)


def install_hint(result: ExportResult) -> str:
    """A copy-pasteable hint for installing the emitted skill locally."""
    return (
        f"Emitted skill → {result.skill_md}\n"
        f"Install into an agent with:\n"
        f"  npx skills add {result.skill_dir}\n"
        f"or copy the folder into ~/.claude/skills, ~/.codex/skills, etc."
    )


def _is_within(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
