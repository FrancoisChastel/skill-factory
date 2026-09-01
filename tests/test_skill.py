"""Tests for the Skill primitive."""

from __future__ import annotations

import pytest

from skill_factory.core.skill import Skill

_MARKDOWN = """---
name: my-skill
description: Does a thing.
license: MIT
---

# Heading

Body text here.
"""


def test_from_markdown_parses_frontmatter_and_body():
    skill = Skill.from_markdown(_MARKDOWN)
    assert skill.name == "my-skill"
    assert skill.description == "Does a thing."
    assert skill.extra["license"] == "MIT"
    assert skill.body.startswith("# Heading")


def test_roundtrip_markdown_preserves_fields():
    skill = Skill.from_markdown(_MARKDOWN)
    reparsed = Skill.from_markdown(skill.to_markdown())
    assert reparsed.name == skill.name
    assert reparsed.description == skill.description
    assert reparsed.extra.get("license") == "MIT"
    assert reparsed.body == skill.body


def test_missing_name_raises():
    with pytest.raises(ValueError, match="name"):
        Skill.from_markdown("---\ndescription: x\n---\nbody")


def test_unclosed_frontmatter_raises():
    with pytest.raises(ValueError, match="never closed"):
        Skill.from_markdown("---\nname: x\nbody without close")


def test_with_body_is_immutable():
    skill = Skill(name="s", description="d", body="original")
    updated = skill.with_body("changed")
    assert skill.body == "original"  # original untouched
    assert updated.body == "changed"
    assert updated.name == "s"


def test_body_without_frontmatter_needs_name():
    with pytest.raises(ValueError):
        Skill.from_markdown("just a plain body, no frontmatter")


def test_save_and_load(tmp_path):
    skill = Skill(name="s", description="d", body="hello")
    path = skill.save(tmp_path / "SKILL.md")
    loaded = Skill.load(path)
    assert loaded.body == "hello"
    assert loaded.name == "s"
