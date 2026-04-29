"""
causal_agent/kripke_tools.py

Wraps KripkeModel operations as ToolDefinition + callable pairs,
registrable in a ToolRegistry for LLM function-calling.

Rather than pre-computing all intervention simulations and dumping them
into the prompt, this lets the LLM actively explore the world space
during planning — querying only the hypotheticals it finds relevant.
The LLM drives an epistemic search; each tool call narrows or informs
its belief state before it commits to a plan.

Usage
-----
    from causal_agent.kripke_tools import KripkeToolset

    # model_ref is a one-element list so the toolset always sees the
    # current model even after update_with_facts() returns a new instance.
    model_ref = [initial_kripke]
    toolset = KripkeToolset(lambda: model_ref[0])
    toolset.register_all(registry)

    # When the orchestrator updates the model:
    model_ref[0] = model_ref[0].update_with_facts(new_facts)
"""

from __future__ import annotations

import json
from typing import Any, Callable

from causal_agent.tools import ToolDefinition, ToolRegistry
from causal_agent.kripke import KripkeModel


class KripkeToolset:
    """
    Produces a set of Kripke-inspection tools backed by a live KripkeModel.

    Parameters
    ----------
    get_model : zero-argument callable that returns the *current* KripkeModel.
                Use a lambda over a mutable container so the toolset always
                reflects the latest model state.
    max_worlds_returned : cap on how many worlds are included in list responses.
    """

    def __init__(
        self,
        get_model: Callable[[], KripkeModel],
        max_worlds_returned: int = 20,
    ) -> None:
        self._get = get_model
        self._max = max_worlds_returned

    # ------------------------------------------------------------------
    # Public: register everything into a ToolRegistry at once
    # ------------------------------------------------------------------

    def register_all(self, registry: ToolRegistry) -> None:
        """Register all Kripke tools into `registry`."""
        for defn, fn in self._all_tools():
            registry.register(defn, fn)

    def _all_tools(self) -> list[tuple[ToolDefinition, Callable]]:
        return [
            (self._defn_certain_facts(),       self._certain_facts),
            (self._defn_count_worlds(),         self._count_worlds),
            (self._defn_enumerate_worlds(),     self._enumerate_worlds),
            (self._defn_inspect_world(),        self._inspect_world),
            (self._defn_simulate_intervention(),self._simulate_intervention),
            (self._defn_compare_interventions(),self._compare_interventions),
            (self._defn_worlds_reaching_goal(), self._worlds_reaching_goal),
        ]

    # ------------------------------------------------------------------
    # Tool: certain_facts
    # ------------------------------------------------------------------

    def _defn_certain_facts(self) -> ToolDefinition:
        return ToolDefinition(
            name="kripke_certain_facts",
            description=(
                "Return the facts that are identical across every currently "
                "possible world — i.e. what the agent knows for certain right now. "
                "Call this to understand what is already settled before planning."
            ),
            parameters={"type": "object", "properties": {}},
        )

    def _certain_facts(self) -> str:
        facts = self._get().certain_facts()
        if not facts:
            return "No facts are certain yet — all propositions vary across possible worlds."
        lines = [f"  {k}: {v}" for k, v in sorted(facts.items())]
        return "Certain facts (true in every possible world):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool: count_worlds
    # ------------------------------------------------------------------

    def _defn_count_worlds(self) -> ToolDefinition:
        return ToolDefinition(
            name="kripke_count_worlds",
            description=(
                "Count how many possible worlds currently exist, optionally "
                "filtered to those matching specific facts. "
                "Use this for a quick size check before enumerating."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": (
                            "Optional dict of {proposition: value} constraints. "
                            "Only worlds where every listed fact matches are counted. "
                            "Omit or pass {} to count all worlds."
                        ),
                    }
                },
            },
        )

    def _count_worlds(self, filter: dict | None = None) -> str:
        model = self._get()
        if filter:
            matching = [w for w in model.worlds if w.matches(filter)]
            return (
                f"{len(matching)} of {len(model.worlds)} possible worlds "
                f"match the filter {filter}."
            )
        return f"{len(model.worlds)} possible worlds remain."

    # ------------------------------------------------------------------
    # Tool: enumerate_worlds
    # ------------------------------------------------------------------

    def _defn_enumerate_worlds(self) -> ToolDefinition:
        return ToolDefinition(
            name="kripke_enumerate_worlds",
            description=(
                "List possible worlds with their uncertain propositions, "
                "optionally filtered to those matching given facts. "
                "Use this to inspect which specific scenarios are still consistent "
                "with what the agent knows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": (
                            "Optional {proposition: value} constraints. "
                            "Only worlds matching all constraints are listed."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max worlds to return. Defaults to 20.",
                    },
                },
            },
        )

    def _enumerate_worlds(
        self,
        filter: dict | None = None,
        limit: int | None = None,
    ) -> str:
        model = self._get()
        uncertain = model.uncertain_props()
        worlds = model.worlds

        if filter:
            worlds = [w for w in worlds if w.matches(filter)]

        cap = min(limit or self._max, self._max)
        shown = worlds[:cap]

        if not shown:
            return "No worlds match the given filter."

        lines = [f"{len(worlds)} world(s) match" + (f" filter {filter}" if filter else "") + ":"]
        for w in shown:
            uncertain_vals = {k: v for k, v in w.to_dict().items() if k in uncertain}
            lines.append(f"  {w.id}: {uncertain_vals}")
        if len(worlds) > cap:
            lines.append(f"  ... and {len(worlds) - cap} more (increase limit to see).")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool: inspect_world
    # ------------------------------------------------------------------

    def _defn_inspect_world(self) -> ToolDefinition:
        return ToolDefinition(
            name="kripke_inspect_world",
            description=(
                "Return the complete fact assignment for a single world by ID. "
                "Use this after enumerate_worlds to examine a specific scenario "
                "in full detail."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "world_id": {
                        "type": "string",
                        "description": "The world ID as returned by kripke_enumerate_worlds.",
                    }
                },
                "required": ["world_id"],
            },
        )

    def _inspect_world(self, world_id: str) -> str:
        w = self._get().world_by_id(world_id)
        if w is None:
            return f"World '{world_id}' not found in the current model."
        facts = w.to_dict()
        if not facts:
            return f"World {world_id}: (no propositions assigned)"
        lines = [f"World {world_id}:"] + [f"  {k}: {v}" for k, v in sorted(facts.items())]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool: simulate_intervention
    # ------------------------------------------------------------------

    def _defn_simulate_intervention(self) -> ToolDefinition:
        return ToolDefinition(
            name="kripke_simulate_intervention",
            description=(
                "Hypothetically assert a set of facts and compute the epistemic "
                "effect: how many worlds survive, how many are eliminated, and "
                "what new facts become certain. Does NOT modify the actual model. "
                "Use this to evaluate a candidate action or assumption before "
                "committing to it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "object",
                        "description": (
                            "The hypothetical {proposition: value} facts to assert. "
                            "Worlds inconsistent with any fact will be eliminated."
                        ),
                    }
                },
                "required": ["facts"],
            },
        )

    def _simulate_intervention(self, facts: dict) -> str:
        model = self._get()
        before = len(model.worlds)
        certain_before = set(model.certain_facts().keys())

        after_model = model.simulate_intervention(facts)
        after = len(after_model.worlds)
        eliminated = before - after

        certain_after = after_model.certain_facts()
        newly_certain = {
            k: v for k, v in certain_after.items()
            if k not in certain_before
        }

        lines = [
            f"Simulating intervention: {facts}",
            f"  Worlds: {before} → {after} ({eliminated} eliminated, "
            f"{after / before * 100:.0f}% survive)" if before else "  No worlds to eliminate.",
        ]
        if newly_certain:
            lines.append(f"  Newly certain: {newly_certain}")
        else:
            lines.append("  No new facts become certain.")
        if after == 0:
            lines.append("  WARNING: This intervention eliminates ALL worlds (contradiction).")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool: compare_interventions
    # ------------------------------------------------------------------

    def _defn_compare_interventions(self) -> ToolDefinition:
        return ToolDefinition(
            name="kripke_compare_interventions",
            description=(
                "Compare two hypothetical interventions side by side. "
                "Returns surviving world counts and newly certain facts for each, "
                "so you can weigh the epistemic cost/benefit of two candidate actions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "facts_a": {
                        "type": "object",
                        "description": "First hypothetical fact set.",
                    },
                    "facts_b": {
                        "type": "object",
                        "description": "Second hypothetical fact set.",
                    },
                },
                "required": ["facts_a", "facts_b"],
            },
        )

    def _compare_interventions(self, facts_a: dict, facts_b: dict) -> str:
        model = self._get()
        before = len(model.worlds)
        certain_before = set(model.certain_facts().keys())

        def _summarise(label: str, facts: dict) -> list[str]:
            after_model = model.simulate_intervention(facts)
            after = len(after_model.worlds)
            eliminated = before - after
            newly_certain = {
                k: v for k, v in after_model.certain_facts().items()
                if k not in certain_before
            }
            pct = f"{after / before * 100:.0f}%" if before else "n/a"
            return [
                f"  {label}: {facts}",
                f"    worlds {before} → {after} ({eliminated} eliminated, {pct} survive)",
                f"    newly certain: {newly_certain if newly_certain else '(none)'}",
            ]

        lines = [f"Comparing interventions ({before} worlds before):"]
        lines += _summarise("A", facts_a)
        lines += _summarise("B", facts_b)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool: worlds_reaching_goal
    # ------------------------------------------------------------------

    def _defn_worlds_reaching_goal(self) -> ToolDefinition:
        return ToolDefinition(
            name="kripke_worlds_reaching_goal",
            description=(
                "Check how many of the current possible worlds are consistent "
                "with the goal state (i.e. already satisfy the desired facts). "
                "Use this to gauge how close the current belief state is to the "
                "goal, and to inspect which specific worlds would achieve it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "object",
                        "description": (
                            "The desired end-state as {proposition: value} pairs. "
                            "A world 'reaches the goal' if it satisfies all of these."
                        ),
                    },
                    "show_worlds": {
                        "type": "boolean",
                        "description": "If true, list the IDs of worlds that satisfy the goal.",
                    },
                },
                "required": ["goal"],
            },
        )

    def _worlds_reaching_goal(self, goal: dict, show_worlds: bool = False) -> str:
        model = self._get()
        matching = [w for w in model.worlds if w.matches(goal)]
        total = len(model.worlds)

        lines = [
            f"Goal: {goal}",
            f"  {len(matching)} of {total} possible worlds satisfy the goal "
            f"({len(matching) / total * 100:.0f}%)." if total else "  No worlds in model.",
        ]

        if show_worlds and matching:
            lines.append("  Worlds satisfying goal:")
            for w in matching[:self._max]:
                lines.append(f"    {w.id}")
            if len(matching) > self._max:
                lines.append(f"    ... and {len(matching) - self._max} more.")
        elif not matching:
            lines.append("  No current world satisfies the goal — "
                         "consider which interventions would increase this count.")

        return "\n".join(lines)
