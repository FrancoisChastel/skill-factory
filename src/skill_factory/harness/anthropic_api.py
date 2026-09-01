"""Default harness: run a skill as a system prompt through an LLM client."""

from __future__ import annotations

import time

from skill_factory.core.rollout import Rollout
from skill_factory.core.skill import Skill
from skill_factory.core.task import Task
from skill_factory.llm.client import LLMClient

# The skill body becomes the system prompt. A tiny framing line keeps the model
# anchored on the task input without editorializing the skill itself.
_SYSTEM_TEMPLATE = "{body}"


class AnthropicHarness:
    """Runs a skill by injecting its body as the system prompt of an LLM call.

    This is the "direct chat" evaluation mode from the SkillOpt paper: the skill
    is the system prompt, the task input is the user message, the output is the
    completion. Works with any :class:`LLMClient` (real Claude or a fake).
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def run(self, skill: Skill, task: Task) -> Rollout:
        system = _SYSTEM_TEMPLATE.format(body=skill.body.strip())
        started = time.perf_counter()
        try:
            output = self._client.generate(
                task.input,
                system=system,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a failed rollout, never crash the epoch
            return Rollout.failed(task, f"{type(exc).__name__}: {exc}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        return Rollout(
            task=task,
            output=output,
            metadata={"latency_ms": round(elapsed_ms, 1)},
        )
