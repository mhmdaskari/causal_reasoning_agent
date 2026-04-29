"""
causal_agent/game_2048_tools.py

Game-specific tools for 2048.

Wraps the env's deterministic move simulation as ToolDefinitions registrable
in a ToolRegistry. The planner can then expose these tools to the LLM, which
calls them during a ReAct loop to evaluate candidate slides before
committing.

Tools registered
----------------
- simulate_move(direction)   – noop-on-the-env preview of a slide.
- score_board(board?)        – heuristic score for the current or given board.
- count_empty_cells()        – how many empty cells right now.
- max_tile()                 – largest tile currently on the board.
- legal_directions()         – which slides are currently legal.

The ``score_board`` heuristic combines four classical 2048 features:

  monotonicity – rewards rows/cols that increase or decrease consistently;
                 penalises non-monotonic shapes (anti-snake).
  smoothness   – rewards adjacent tiles whose log2 values differ by 1 or 0;
                 penalises sharp jumps (un-mergeable neighbours).
  empty_cells  – more empty cells = more flexibility = far longer survival.
  corner_bonus – the max tile sitting in any corner is worth a flat bonus.

The exact weights are not load-bearing; they're picked to be reasonable and
to give the LLM a meaningful comparison signal across moves. The LLM does
not need to trust the heuristic — it can call it for sanity-checking and
combine it with its own reasoning.

Pattern note
------------
The toolset is constructed with a *getter callable* rather than a direct
env reference, so updates to the env state during the turn (e.g. preview
recomputation) always reflect the current board.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from causal_agent.tools import ToolDefinition, ToolRegistry


Board = list[list[int]]


class Game2048Toolset:
    """
    Build and register 2048-specific inspection / simulation tools.

    Parameters
    ----------
    get_env : zero-arg callable returning the live ``Game2048Env`` so the
              tools always see the current board.
    """

    def __init__(self, get_env: Callable[[], Any]) -> None:
        self._get = get_env

    # ------------------------------------------------------------------
    # Public registration entry point
    # ------------------------------------------------------------------

    def register_all(self, registry: ToolRegistry) -> None:
        for defn, fn in self._all_tools():
            registry.register(defn, fn)

    def _all_tools(self) -> list[tuple[ToolDefinition, Callable]]:
        return [
            (self._defn_simulate_move(),    self._simulate_move),
            (self._defn_score_board(),      self._score_board),
            (self._defn_count_empty(),      self._count_empty),
            (self._defn_max_tile(),         self._max_tile),
            (self._defn_legal_directions(), self._legal_directions),
        ]

    # ------------------------------------------------------------------
    # Tool: simulate_move
    # ------------------------------------------------------------------

    def _defn_simulate_move(self) -> ToolDefinition:
        return ToolDefinition(
            name="simulate_move",
            description=(
                "Counterfactually slide the current board one direction and "
                "return the resulting board, score gained, empty cell count, "
                "max tile, and merge count — without modifying the live "
                "game state. Use this to compare candidate slides before "
                "committing via submit_plan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Which direction to simulate.",
                    },
                },
                "required": ["direction"],
            },
        )

    def _simulate_move(self, direction: str) -> dict:
        env = self._get()
        legal = env._legal_directions()
        if direction not in legal:
            return {
                "direction": direction,
                "legal": False,
                "legal_directions": legal,
                "note": "Direction is not currently legal; pick from legal_directions.",
            }
        moved, gained = env._move(env.board, direction)
        return {
            "direction": direction,
            "legal": True,
            "gained": int(gained),
            "board_after": moved,
            "empty_after": _count_empty(moved),
            "max_tile_after": _max_tile(moved),
            "merges": int(_count_merges(env.board, moved)),
            "heuristic_score_after": round(_heuristic_score(moved), 3),
        }

    # ------------------------------------------------------------------
    # Tool: score_board
    # ------------------------------------------------------------------

    def _defn_score_board(self) -> ToolDefinition:
        return ToolDefinition(
            name="score_board",
            description=(
                "Compute a heuristic score for the current board (or one "
                "supplied via the optional `board` argument). Combines "
                "monotonicity, smoothness, empty-cell count, and a "
                "corner bonus. Higher is better. Use this to break ties "
                "between candidate slides after simulate_move."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "board": {
                        "type": "array",
                        "description": (
                            "Optional 2D list of tile values to score. If "
                            "omitted, the live board is scored."
                        ),
                    },
                },
            },
        )

    def _score_board(self, board: Board | None = None) -> dict:
        b = board if board is not None else self._get().board
        return {
            "heuristic_score":   round(_heuristic_score(b), 3),
            "monotonicity":      round(_monotonicity(b), 3),
            "smoothness":        round(_smoothness(b), 3),
            "empty_cells":       _count_empty(b),
            "max_tile":          _max_tile(b),
            "max_in_corner":     _max_in_corner(b),
        }

    # ------------------------------------------------------------------
    # Tool: count_empty_cells
    # ------------------------------------------------------------------

    def _defn_count_empty(self) -> ToolDefinition:
        return ToolDefinition(
            name="count_empty_cells",
            description=(
                "Return the number of empty cells on the current board. "
                "Lower means the board is dense and one bad slide could "
                "end the game."
            ),
            parameters={"type": "object", "properties": {}},
        )

    def _count_empty(self) -> int:
        return _count_empty(self._get().board)

    # ------------------------------------------------------------------
    # Tool: max_tile
    # ------------------------------------------------------------------

    def _defn_max_tile(self) -> ToolDefinition:
        return ToolDefinition(
            name="max_tile",
            description="Return the largest tile value currently on the board.",
            parameters={"type": "object", "properties": {}},
        )

    def _max_tile(self) -> int:
        return _max_tile(self._get().board)

    # ------------------------------------------------------------------
    # Tool: legal_directions
    # ------------------------------------------------------------------

    def _defn_legal_directions(self) -> ToolDefinition:
        return ToolDefinition(
            name="legal_directions",
            description=(
                "Return the slide directions that are currently legal "
                "(i.e. would move at least one tile)."
            ),
            parameters={"type": "object", "properties": {}},
        )

    def _legal_directions(self) -> list[str]:
        return list(self._get()._legal_directions())


# ---------------------------------------------------------------------------
# Pure helpers used by the heuristic and the simulate_move tool
# ---------------------------------------------------------------------------

def _count_empty(board: Board) -> int:
    return sum(1 for row in board for v in row if v == 0)


def _max_tile(board: Board) -> int:
    return max(max(row) for row in board)


def _max_in_corner(board: Board) -> bool:
    if not board:
        return False
    n = len(board)
    m = _max_tile(board)
    corners = (board[0][0], board[0][-1], board[n - 1][0], board[n - 1][-1])
    return m in corners and m > 0


def _count_merges(before: Board, after: Board) -> int:
    """
    Approximate the number of tiles that merged in this move by comparing
    non-zero counts. Each merge eliminates one tile (two collapse into one).
    """
    nz_before = sum(1 for row in before for v in row if v != 0)
    nz_after = sum(1 for row in after for v in row if v != 0)
    # 2048 also adds one new tile per move, but `_move` does not add it; the
    # caller only calls `_move` and not `_add_tile`, so non-zero count drops
    # by exactly the number of merges.
    return max(0, nz_before - nz_after)


def _monotonicity(board: Board) -> float:
    """
    Reward rows/columns whose log2 values are monotonic. Returns the
    negative of the worst-direction inversion mass per row/col, summed,
    then normalised by board size. Higher is better; max 0.
    """
    if not board:
        return 0.0
    n = len(board)

    def _row_score(row: list[int]) -> float:
        # Compare adjacent log2 values; penalise the smaller of left- and
        # right-monotonicity violations.
        logs = [math.log2(v) if v > 0 else 0.0 for v in row]
        inc = sum(max(0.0, logs[i] - logs[i + 1]) for i in range(len(logs) - 1))
        dec = sum(max(0.0, logs[i + 1] - logs[i]) for i in range(len(logs) - 1))
        return -min(inc, dec)

    rows = sum(_row_score(row) for row in board)
    cols = sum(_row_score([board[r][c] for r in range(n)]) for c in range(n))
    return (rows + cols) / max(1, n)


def _smoothness(board: Board) -> float:
    """
    Reward boards whose adjacent non-zero tiles have similar log2 values.
    Returns the negative sum of |log2(a) - log2(b)| over horizontally and
    vertically adjacent non-zero pairs. Higher (less negative) is better.
    """
    if not board:
        return 0.0
    n = len(board)
    score = 0.0
    for r in range(n):
        for c in range(n):
            v = board[r][c]
            if v == 0:
                continue
            if c + 1 < n and board[r][c + 1] != 0:
                score -= abs(math.log2(v) - math.log2(board[r][c + 1]))
            if r + 1 < n and board[r + 1][c] != 0:
                score -= abs(math.log2(v) - math.log2(board[r + 1][c]))
    return score


def _heuristic_score(board: Board) -> float:
    """
    Weighted combination of the four classical 2048 heuristics.

    Weights are picked to be reasonable in practice; tuning them is left
    for future work and should not change the *direction* of rankings on
    typical boards.
    """
    if not board:
        return 0.0
    return (
        1.0 * _monotonicity(board)
        + 0.1 * _smoothness(board)
        + 2.7 * _count_empty(board)
        + (1.0 if _max_in_corner(board) else 0.0)
    )
