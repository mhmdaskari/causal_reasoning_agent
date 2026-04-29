"""
evaluations/common/planner_factory.py

One-line wiring of an env-customised ``Planner`` for evaluation runners.

The factory reads the three optional hooks declared on ``GameEnvironment``:

- ``env.system_prompt()``    — game-specific system prompt.
- ``env.tools(agent_id)``    — game-specific tool registry (or None).
- ``env.preview``            — read-only counterfactual hook (or default None).

…and constructs a ``Planner`` configured accordingly. Per-game runners
should call this instead of building a ``Planner`` directly so that any
new env automatically benefits from the right prompt + tools + preview.
"""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from causal_agent import Planner
from causal_agent.acting import GameAction
from causal_agent.llm import BaseLLM

if TYPE_CHECKING:
    from games.base import GameEnvironment


def build_planner(
    env: "GameEnvironment",
    llm: BaseLLM,
    agent_id: str = "Agent",
    *,
    simulate_before_plan: bool = False,
    max_tool_iterations: int = 8,
) -> Planner:
    """
    Build a ``Planner`` configured from the env's optional hooks.

    Parameters
    ----------
    env
        The active ``GameEnvironment``.
    llm
        The backend the planner should call.
    agent_id
        Identifier of the acting agent (default ``"Agent"``).
    simulate_before_plan
        Forwarded to ``Planner``. Defaults to ``False`` for evaluations —
        Kripke intervention summaries add token cost without value for
        full-information puzzles. Override per-game when running on
        deduction or social-deduction games.
    max_tool_iterations
        Cap on ReAct iterations when the env registers tools.

    Returns
    -------
    Planner
        Configured to use ``env.system_prompt()``, ``env.tools(agent_id)``,
        and ``env.preview`` when each is provided.
    """
    system = env.system_prompt()
    registry = env.tools(agent_id)

    def _preview(
        agent: str,
        action_type: str,
        parameters: Mapping[str, Any],
    ) -> dict | None:
        action = GameAction(
            action_type=action_type,
            payload=dict(parameters),
            agent_id=agent,
        )
        return env.preview(agent, action)

    # Only pass the preview callable if the env meaningfully implements it.
    # The base implementation returns None; we still forward it cheaply, but
    # a runner that wants to opt out can pass a stripped-down env.
    return Planner(
        llm,
        simulate_before_plan=simulate_before_plan,
        system=system,
        tools=registry,
        preview_callable=_preview,
        max_tool_iterations=max_tool_iterations,
    )
