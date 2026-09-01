"""Shared fixtures."""

from __future__ import annotations

import pytest

from skill_factory.core.skill import Skill
from skill_factory.core.task import Dataset, Task


@pytest.fixture
def seed_skill() -> Skill:
    return Skill(
        name="echo",
        description="Echo the input back.",
        body="Return the input unchanged.",
    )


@pytest.fixture
def small_dataset() -> Dataset:
    return Dataset(
        [
            Task(id=f"t{i}", input=f"input {i}", expected=f"MAGIC {i}")
            for i in range(6)
        ]
    )
