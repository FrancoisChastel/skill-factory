"""Harnesses run a skill against a task and return a Rollout.

A harness is the boundary between the portable SKILL.md text and an actual
execution environment. The default :class:`AnthropicHarness` runs the skill as a
system prompt via the Claude API; other harnesses can shell out to the Claude
Code / Codex CLIs so skills are optimized in the same environment they deploy to.
"""

from skill_factory.harness.base import Harness
from skill_factory.harness.anthropic_api import AnthropicHarness
from skill_factory.harness.callable_harness import CallableHarness

__all__ = ["Harness", "AnthropicHarness", "CallableHarness"]
