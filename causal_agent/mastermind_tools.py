"""
causal_agent/mastermind_tools.py

Game-specific tools for Mastermind.

The fundamental object in Mastermind reasoning is the *candidate set*: every
secret code still consistent with all feedback observed so far. Each guess
partitions the candidate set across the 14-ish possible feedback responses;
the optimal guess is the one whose worst-case partition is smallest (Knuth's
minimax) or whose expected partition has highest information (entropy).

This toolset gives the LLM the same primitives a hand-coded solver would
use, exposed as native function calls:

- candidate_count()                       – size of the candidate set.
- enumerate_candidates(limit?)            – list (a sample of) candidates.
- filter_candidates(guess, exact, partial)
                                          – counterfactual: how many
                                            candidates would survive if the
                                            given feedback came back?
- expected_information(guess)             – Shannon entropy of the
                                            guess-induced partition over
                                            the current candidate set.
- score_guess(guess)                      – exact/partial against an
                                            arbitrary code (for sanity).

The toolset operates on a *getter* for the env so it always sees the
current history, plus a static list of all valid codes (computed once at
construction since the search space is independent of game state).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import permutations, product
from typing import Any, Callable, Sequence

from causal_agent.tools import ToolDefinition, ToolRegistry


Code = tuple[str, ...]


def score_guess(guess: Sequence[str], code: Sequence[str]) -> tuple[int, int]:
    """Standard Mastermind feedback for guess against code."""
    exact = sum(g == c for g, c in zip(guess, code))
    rem_g = [g for g, c in zip(guess, code) if g != c]
    rem_c = [c for g, c in zip(guess, code) if g != c]
    gc, cc = Counter(rem_g), Counter(rem_c)
    partial = sum(min(n, cc.get(s, 0)) for s, n in gc.items())
    return exact, partial


def all_codes(
    colors: Sequence[str],
    code_length: int,
    duplicates_allowed: bool,
) -> list[Code]:
    if duplicates_allowed:
        return list(product(colors, repeat=code_length))
    return list(permutations(colors, code_length))


class MastermindToolset:
    """
    Build and register Mastermind-specific tools.

    Parameters
    ----------
    get_env : zero-arg callable returning the live ``MastermindEnv``.
    duplicates_allowed : whether the secret may repeat colors. The env's
                         own configuration does not directly expose this;
                         pass it explicitly from the runner.
    enumerate_limit_default : default cap on enumerated candidates.
    """

    def __init__(
        self,
        get_env: Callable[[], Any],
        duplicates_allowed: bool = True,
        enumerate_limit_default: int = 20,
    ) -> None:
        self._get = get_env
        self._dups = duplicates_allowed
        self._default_limit = enumerate_limit_default
        env = get_env()
        self._all_codes: list[Code] = all_codes(
            env._colors, env._code_length, duplicates_allowed
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_all(self, registry: ToolRegistry) -> None:
        for defn, fn in self._all_tools():
            registry.register(defn, fn)

    def _all_tools(self) -> list[tuple[ToolDefinition, Callable]]:
        return [
            (self._defn_candidate_count(),       self._candidate_count),
            (self._defn_enumerate_candidates(),  self._enumerate_candidates),
            (self._defn_filter_candidates(),     self._filter_candidates),
            (self._defn_expected_information(),  self._expected_information),
            (self._defn_score_guess(),           self._score_guess),
        ]

    # ------------------------------------------------------------------
    # Tool: candidate_count
    # ------------------------------------------------------------------

    def _defn_candidate_count(self) -> ToolDefinition:
        return ToolDefinition(
            name="candidate_count",
            description=(
                "Return the number of secret codes still consistent with "
                "all guesses seen so far. Smaller is closer to the answer."
            ),
            parameters={"type": "object", "properties": {}},
        )

    def _candidate_count(self) -> int:
        return len(self._consistent_candidates())

    # ------------------------------------------------------------------
    # Tool: enumerate_candidates
    # ------------------------------------------------------------------

    def _defn_enumerate_candidates(self) -> ToolDefinition:
        return ToolDefinition(
            name="enumerate_candidates",
            description=(
                "List secret codes still consistent with all guesses so "
                "far. The result is capped (default 20) to keep the "
                "response compact; enumerating large candidate sets is "
                "rarely useful — call candidate_count first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max codes to return. Default 20.",
                    },
                },
            },
        )

    def _enumerate_candidates(self, limit: int | None = None) -> dict:
        cands = self._consistent_candidates()
        cap = min(limit or self._default_limit, self._default_limit)
        return {
            "total": len(cands),
            "shown": [list(c) for c in cands[:cap]],
            "truncated": len(cands) > cap,
        }

    # ------------------------------------------------------------------
    # Tool: filter_candidates
    # ------------------------------------------------------------------

    def _defn_filter_candidates(self) -> ToolDefinition:
        return ToolDefinition(
            name="filter_candidates",
            description=(
                "Counterfactual: if you guessed `guess` and received feedback "
                "(exact, partial), how many of the currently consistent "
                "candidates would still be consistent? Use this to estimate "
                "the worst-case partition size before submitting a guess."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "guess": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The guess code (list of color symbols).",
                    },
                    "exact": {
                        "type": "integer",
                        "description": "Hypothetical exact-match count.",
                    },
                    "partial": {
                        "type": "integer",
                        "description": "Hypothetical partial-match count.",
                    },
                },
                "required": ["guess", "exact", "partial"],
            },
        )

    def _filter_candidates(self, guess: list[str], exact: int, partial: int) -> dict:
        cands = self._consistent_candidates()
        target = (int(exact), int(partial))
        survivors = [c for c in cands if score_guess(guess, c) == target]
        return {
            "total_before": len(cands),
            "survivors": len(survivors),
            "fraction": round(len(survivors) / max(1, len(cands)), 4),
        }

    # ------------------------------------------------------------------
    # Tool: expected_information
    # ------------------------------------------------------------------

    def _defn_expected_information(self) -> ToolDefinition:
        return ToolDefinition(
            name="expected_information",
            description=(
                "Compute the Shannon entropy (in bits) of the partition "
                "induced by `guess` over the current candidate set. "
                "Higher entropy = more expected information = better early "
                "guess. Also returns the worst-case partition size (Knuth's "
                "minimax target)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "guess": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The guess code (list of color symbols).",
                    },
                },
                "required": ["guess"],
            },
        )

    def _expected_information(self, guess: list[str]) -> dict:
        cands = self._consistent_candidates()
        if not cands:
            return {"entropy_bits": 0.0, "worst_partition": 0, "candidates": 0}
        partitions: dict[tuple[int, int], int] = defaultdict(int)
        for c in cands:
            partitions[score_guess(guess, c)] += 1
        n = len(cands)
        entropy = -sum(
            (count / n) * math.log2(count / n)
            for count in partitions.values()
            if count > 0
        )
        worst = max(partitions.values())
        return {
            "entropy_bits": round(entropy, 4),
            "worst_partition": int(worst),
            "candidates": int(n),
            "num_buckets": len(partitions),
        }

    # ------------------------------------------------------------------
    # Tool: score_guess (utility)
    # ------------------------------------------------------------------

    def _defn_score_guess(self) -> ToolDefinition:
        return ToolDefinition(
            name="score_guess",
            description=(
                "Compute exact/partial feedback for `guess` against an "
                "arbitrary `code`. Use this to verify your understanding "
                "of the scoring rules; it does not query the secret."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "guess": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "code": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["guess", "code"],
            },
        )

    def _score_guess(self, guess: list[str], code: list[str]) -> dict:
        exact, partial = score_guess(guess, code)
        return {"exact": int(exact), "partial": int(partial)}

    # ------------------------------------------------------------------
    # Internal: candidate filtering against history
    # ------------------------------------------------------------------

    def _consistent_candidates(self) -> list[Code]:
        env = self._get()
        cands: list[Code] = list(self._all_codes)
        for record in env.history:
            g = tuple(record["guess"])
            target = (int(record["exact"]), int(record["partial"]))
            cands = [c for c in cands if score_guess(g, c) == target]
        return cands
