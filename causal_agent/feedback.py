"""
causal_agent/feedback.py

Normalises raw environment output into typed FeedbackEvents.

Responsibility split
--------------------
The *environment* emits raw dicts in whatever schema it likes.
FeedbackProcessor translates those into a canonical FeedbackEvent, which
is what Memory and Orchestration know about.  This keeps environment-specific
parsing entirely out of the planning and acting modules.

FeedbackEvent also carries a `facts` dict — the propositions that can be
used to update the KripkeModel via update_with_facts().

Signal taxonomy (FeedbackKind)
------------------------------
OBSERVATION   – new information about game state (phase, who said what).
REWARD        – scalar outcome (win/loss/points).
PHASE_CHANGE  – the game moved to a new phase (day→night, etc.).
SOCIAL        – another player's speech act that updates our beliefs.
ILLEGAL_MOVE  – our last action was invalid; need to replan.
TERMINAL      – the game has ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# FeedbackKind
# ---------------------------------------------------------------------------

class FeedbackKind(str, Enum):
    OBSERVATION  = "observation"
    REWARD       = "reward"
    PHASE_CHANGE = "phase_change"
    SOCIAL       = "social"
    ILLEGAL_MOVE = "illegal_move"
    TERMINAL     = "terminal"


# ---------------------------------------------------------------------------
# FeedbackEvent
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEvent:
    """
    Canonical feedback record produced by FeedbackProcessor.

    Attributes
    ----------
    kind     : classification of this signal.
    turn     : game turn at which it occurred.
    source   : originator — "env", a player name, "self", etc.
    content  : plain-text description (for memory / logging).
    facts    : propositions to assert into the KripkeModel.
    reward   : scalar value; 0.0 when not a reward signal.
    terminal : True when the game session has ended.
    """
    kind: FeedbackKind
    turn: int
    source: str
    content: str
    facts: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    terminal: bool = False

    def __str__(self) -> str:
        parts = [f"[{self.kind.value}|t={self.turn}|{self.source}] {self.content}"]
        if self.facts:
            parts.append(f"  facts={self.facts}")
        if self.reward:
            parts.append(f"  reward={self.reward}")
        if self.terminal:
            parts.append("  TERMINAL")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# FeedbackProcessor
# ---------------------------------------------------------------------------

class FeedbackProcessor:
    """
    Converts a raw environment dict into a FeedbackEvent.

    The raw dict is expected to carry at minimum:
        "kind"    : str  — one of the FeedbackKind values (or a game alias).
        "source"  : str  — who emitted this signal.
        "content" : str  — description.

    Optional keys that are forwarded if present:
        "facts"    : dict  — KripkeModel update payload.
        "reward"   : float
        "terminal" : bool
        "phase"    : str   — convenience shorthand; translated to facts["phase"].

    Subclass and override `process` to add game-specific logic (e.g. parsing
    speech acts into structured facts for Werewolf claims).
    """

    # Map environment-specific kind strings to canonical FeedbackKind.
    _KIND_ALIASES: dict[str, FeedbackKind] = {
        "obs"          : FeedbackKind.OBSERVATION,
        "observe"      : FeedbackKind.OBSERVATION,
        "observation"  : FeedbackKind.OBSERVATION,
        "reward"       : FeedbackKind.REWARD,
        "score"        : FeedbackKind.REWARD,
        "phase"        : FeedbackKind.PHASE_CHANGE,
        "phase_change" : FeedbackKind.PHASE_CHANGE,
        "chat"         : FeedbackKind.SOCIAL,
        "speech"       : FeedbackKind.SOCIAL,
        "social"       : FeedbackKind.SOCIAL,
        "illegal"      : FeedbackKind.ILLEGAL_MOVE,
        "illegal_move" : FeedbackKind.ILLEGAL_MOVE,
        "invalid"      : FeedbackKind.ILLEGAL_MOVE,
        "done"         : FeedbackKind.TERMINAL,
        "terminal"     : FeedbackKind.TERMINAL,
        "end"          : FeedbackKind.TERMINAL,
    }

    def process(self, raw: dict, turn: int) -> FeedbackEvent:
        """
        Parameters
        ----------
        raw  : env-emitted dict.
        turn : current game turn (injected by Orchestration).

        Returns
        -------
        FeedbackEvent
        """
        kind_str = str(raw.get("kind", "observation")).lower()
        kind = self._KIND_ALIASES.get(kind_str, FeedbackKind.OBSERVATION)

        facts: dict[str, Any] = dict(raw.get("facts", {}))

        # Convenience: if the raw dict carries a top-level "phase" key,
        # propagate it into facts so the KripkeModel can be updated.
        if "phase" in raw and "phase" not in facts:
            facts["phase"] = raw["phase"]

        # Convenience: eliminated players update the symbolic state too.
        if "eliminated" in raw and "eliminated" not in facts:
            facts["eliminated"] = raw["eliminated"]

        terminal = bool(raw.get("terminal", False))
        if kind == FeedbackKind.TERMINAL:
            terminal = True

        return FeedbackEvent(
            kind=kind,
            turn=turn,
            source=str(raw.get("source", "env")),
            content=str(raw.get("content", "")),
            facts=facts,
            reward=float(raw.get("reward", 0.0)),
            terminal=terminal,
        )

    def batch_process(
        self, raws: list[dict], turn: int
    ) -> list[FeedbackEvent]:
        """Process multiple raw signals from a single env step."""
        return [self.process(r, turn) for r in raws]
