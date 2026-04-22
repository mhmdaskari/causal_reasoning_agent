"""
games/base.py

Abstract base class for all game environments.

Every game the framework supports must implement GameEnvironment.
The only causal_agent type it needs to know about is GameAction —
that is the sole coupling between the games layer and the agent layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from causal_agent.acting import GameAction
    from causal_agent.kripke import KripkeModel


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

    valid_actions(agent_id)
        Return the list of action_type strings legal for `agent_id` right now.

    is_terminal
        True when the game has reached an end condition.

    initial_kripke(agent_id)
        Construct the KripkeModel for `agent_id` at game start, encoding
        what that agent initially knows and doesn't know.
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
    def valid_actions(self, agent_id: str) -> list[str]:
        """Legal action types for agent_id this turn."""
        ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """True when the game is over."""
        ...

    @abstractmethod
    def initial_kripke(self, agent_id: str) -> "KripkeModel":
        """
        Build the agent's initial epistemic model.

        Worlds correspond to all role/state assignments consistent with
        what agent_id is told at game start.  Accessibility is derived
        from which facts each other player has been told privately.
        """
        ...
