"""Guard the gallery: every entry must be complete, valid, and indexed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from skill_factory.core.skill import Skill

_GALLERY = Path(__file__).resolve().parents[1] / "gallery"
_ENTRIES = sorted(p for p in _GALLERY.iterdir() if p.is_dir())

_SCORE_RE = re.compile(r"\d\.\d{3}\s*→\s*\d\.\d{3}")


def test_gallery_has_entries():
    assert len(_ENTRIES) >= 2


@pytest.mark.parametrize("entry", _ENTRIES, ids=lambda p: p.name)
def test_entry_is_complete(entry: Path):
    readme = entry / "README.md"
    artifact = entry / "optimized_skill.md"
    assert readme.exists(), f"{entry.name}: missing README.md"
    assert artifact.exists(), f"{entry.name}: missing optimized_skill.md"

    # Artifact must parse as a valid SKILL.md with the npx skills essentials.
    skill = Skill.from_markdown(artifact.read_text(encoding="utf-8"))
    assert skill.name
    assert skill.description.strip(), f"{entry.name}: artifact needs a description"
    assert skill.body.strip()

    # README must report a measured baseline → best score.
    text = readme.read_text(encoding="utf-8")
    assert _SCORE_RE.search(text), f"{entry.name}: README lacks 'baseline → best' scores"
    assert "Reproduce" in text, f"{entry.name}: README lacks a Reproduce section"


@pytest.mark.parametrize("entry", _ENTRIES, ids=lambda p: p.name)
def test_entry_is_indexed(entry: Path):
    index = (_GALLERY / "README.md").read_text(encoding="utf-8")
    assert f"({entry.name}/)" in index, f"{entry.name}: not listed in gallery/README.md"
