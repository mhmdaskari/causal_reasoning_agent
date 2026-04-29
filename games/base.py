"""
games/base.py

Abstract base class for all game environments.

Every game the framework supports must implement GameEnvironment.
The only causal_agent type it needs to know about is GameAction —
that is the sole coupling between the games layer and the agent layer.

Optional hooks
--------------
Three optional methods let an env tune the planner without touching
framework code:

- ``system_prompt()``  – game-specific system prompt for the reactive
  Planner. Default: ``REACTIVE_SYSTEM`` (works for hidden-information
  games such as Werewolf). Override to give the LLM concrete strategy
  guidance for puzzles, deduction games, etc.
- ``tools(agent_id)``  – a ``ToolRegistry`` of game-specific tools the
  planner exposes to the LLM (e.g. ``simulate_move``, ``score_board``,
  ``filter_candidates``). Default: ``None`` (no tools).
- ``preview(agent_id, action)`` – read-only counterfactual: what would
  ``step(action)`` produce, without committing? Default: ``None``
  (env can't preview cheaply). Used by the planner to embed per-action
  consequences in the prompt — the cheapest way to give the model
  one-step lookahead.

See ``games/AUTHORING.md`` for the contract a new env should follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from causal_agent.actions import ActionSpec

if TYPE_CHECKING:
    from causal_agent.acting import GameAction
    from causal_agent.kripke import KripkeModel
    from causal_agent.tools import ToolRegistry


class GameEnvironment(ABC):
    """
    Minimal interface a game environment must satisfy.

    Methods
    -------
    observe(agent_id)
        Return a raw dict describing what `agent_id` can perceive this turn.
        The dict is passed to FeedbackProcessor.process(); its schema is
        game-specific but should include at minimum "kind", "source", "content".

    step(agent_id, action)
        Apply `action` for `agent_id` and advance the game state.
        Return a raw feedback dict (same schema as observe).

    action_specs(agent_id)
        Return the legal ActionSpec objects for `agent_id` right now. This is
        the canonical contract used by planning and acting.

    valid_actions(agent_id)
        Compatibility helper returning the legal action_type strings.

    is_terminal
        True when the game has reached an end condition.

    initial_kripke(agent_id)
        Construct the KripkeModel for `agent_id` at game start, encoding
        what that agent initially knows and doesn't know.

    Optional hooks
    --------------
    system_prompt()
        Game-specific reactive system prompt. Default: REACTIVE_SYSTEM.

    tools(agent_id)
        Game-specific ToolRegistry for the planner. Default: None.

    preview(agent_id, action)
        Read-only counterfactual outcome of `action`. Default: None.
    """

    @abstractmethod
    def observe(self, agent_id: str) -> dict:
        """Current percept for agent_id."""
        ...

    @abstractmethod
    def step(self, agent_id: str, action: "GameAction") -> dict:
        """Apply action and return raw feedback dict."""
        ...

    @abstractmethod
    def action_specs(self, agent_id: str) -> list[ActionSpec]:
        """Legal structured actions for agent_id this turn."""
        ...

    def valid_actions(self, agent_id: str) -> list[str]:
        """Legal action type names for legacy callers."""
        return [spec.action_type for spec in self.action_specs(agent_id)]

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """True when the game is over."""
        ...

    def initial_kripke(self, agent_id: str) -> "KripkeModel":
        """
        Build the agent's initial epistemic model.

        Games with hidden information can override this. Fully observable
        games get a trivial one-world model by default, making the Kripke
        module optional for puzzles such as 2048.
        """
        from causal_agent.kripke import KripkeModel, World

        return KripkeModel(worlds=[World.from_dict("actual", {})])

    # ------------------------------------------------------------------
    # Optional hooks (override to specialise the planner per game)
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        """
        Game-specific system prompt for the reactive Planner.

        Default: ``REACTIVE_SYSTEM`` (hidden-information / epistemic
        framing). Override for puzzles or deduction games where the
        epistemic-worlds language is unhelpful.
        """
        from causal_agent.prompts import REACTIVE_SYSTEM

        return REACTIVE_SYSTEM

    def tools(self, agent_id: str) -> "ToolRegistry | None":
        """
        Game-specific tools the planner makes available to the LLM.

        Return ``None`` (default) for envs that do not register tools.
        Return a populated ``ToolRegistry`` to switch the planner into
        a bounded ReAct loop using ``BaseLLM.complete_with_tools()``.
        """
        return None

    def preview(self, agent_id: str, action: "GameAction") -> dict | None:
        """
        Read-only counterfactual: what would ``step(action)`` produce?

        Implementations must NOT mutate any env state. Returning a dict
        of observable consequences (e.g. ``{"gained": 8, "empty_after":
        6, "max_tile_after": 64}``) lets the planner embed concrete
        per-action outcomes in the prompt without invoking tool-calling.

        Default: ``None`` — preview not supported.
        """
        return None
