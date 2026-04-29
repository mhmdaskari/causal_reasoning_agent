"""
causal_agent/planning.py

Reactive, per-turn planner.

The Planner produces a structured ``Plan`` for one decision. It has two
modes:

1. **One-shot structured output** (default).
   The planner builds a single prompt and calls
   ``BaseLLM.complete_structured(...)``. Suitable for hidden-information
   games where the LLM only needs the static observation plus the Kripke
   summary.

2. **Bounded ReAct loop with tools** (when tools are registered).
   If the env (or caller) provides a non-empty ``ToolRegistry``, the
   planner runs ``BaseLLM.complete_with_tools(...)`` in a bounded loop.
   The model can call game-specific tools (e.g. ``simulate_move``,
   ``score_board``, ``filter_candidates``, ``kripke_*``) to inspect and
   reason, and submits its final decision via the auto-registered
   ``submit_plan`` tool. This is what lets the agent actually try moves
   before committing — the cheapest way to beat a random baseline on
   puzzles such as 2048.

In both modes the env can optionally provide a ``preview`` callable
(``env.preview(agent_id, action)``) used to embed per-action consequences
in the prompt. With or without tools, this gives the LLM concrete
one-step lookahead at zero token cost beyond the prompt extension.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

from causal_agent.actions import (
    ActionSchemaError,
    ActionSpec,
    action_spec_by_type,
    action_type_names,
    coerce_action_specs,
    format_action_specs_for_prompt,
    structured_plan_schema,
)
from causal_agent.kripke import KripkeModel
from causal_agent.memory import MemoryStore
from causal_agent.llm import BaseLLM
from causal_agent.prompts import REACTIVE_SYSTEM
from causal_agent.tools import (
    LLMResponse,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
)

if TYPE_CHECKING:
    pass


log = logging.getLogger("causal_agent.planning")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """
    Structured intent produced by the Planner.

    Attributes
    ----------
    intent           : high-level goal for this step (natural language).
    action_type      : which action category to take (e.g. "speak", "vote").
    parameters       : action-specific payload (message text, target player…).
    reasoning        : brief public rationale from the LLM.
    supporting_worlds: world ids from the KripkeModel that support this plan.
    intervention_notes: summaries of any counterfactual simulations run.
    tool_calls       : ordered log of tool calls made during planning, when
                       the ReAct mode was used. Empty in one-shot mode.
    """
    intent: str
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    supporting_worlds: list[str] = field(default_factory=list)
    intervention_notes: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Plan(intent={self.intent!r}, "
            f"action_type={self.action_type!r}, "
            f"params={self.parameters})"
        )


class PlanParseError(ValueError):
    """Raised when the planner cannot parse a valid structured plan."""


PreviewCallable = Callable[[str, str, Mapping[str, Any]], "dict[str, Any] | None"]
"""Signature: ``preview(agent_id, action_type, parameters) -> dict | None``."""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Per-turn reactive planner.

    Parameters
    ----------
    llm
        Any ``BaseLLM`` backend.
    simulate_before_plan
        If True, run intervention simulations on the Kripke frame for each
        legal action and embed the summaries in the prompt. This is useful
        for hidden-information games and harmless (but token-spending) for
        puzzles. Default: True.
    max_parse_retries
        How many times to retry on a malformed structured response in
        one-shot mode. Default: 1.
    system
        System prompt override. Defaults to ``REACTIVE_SYSTEM``. Override
        per-game via ``GameEnvironment.system_prompt()``.
    tools
        Optional ``ToolRegistry`` of game-specific tools. When non-empty,
        the planner switches to a bounded ReAct loop (see module docstring).
    preview_callable
        Optional ``preview(agent_id, action_type, parameters) -> dict | None``
        used to compute per-action consequences and embed them in the
        prompt. Wire ``GameEnvironment.preview`` through here.
    max_tool_iterations
        Hard cap on ReAct iterations when tools are present. Default: 20.
        Each iteration is one round-trip with the model; some backends
        (notably DeepSeek) do not always return parallel tool calls, so a
        small budget is exhausted quickly on games with rich tool usage
        such as Mastermind. The planner emits a final budget warning to
        the model on the second-to-last iteration so it can commit cleanly.
    """

    def __init__(
        self,
        llm: BaseLLM,
        simulate_before_plan: bool = True,
        max_parse_retries: int = 1,
        *,
        system: str | None = None,
        tools: ToolRegistry | None = None,
        preview_callable: PreviewCallable | None = None,
        max_tool_iterations: int = 20,
    ) -> None:
        self._llm = llm
        self._simulate = simulate_before_plan
        self._max_parse_retries = max_parse_retries
        self._system = system if system is not None else REACTIVE_SYSTEM
        self._tools = tools if tools is not None and len(tools) > 0 else None
        self._preview = preview_callable
        self._max_tool_iter = max_tool_iterations

    # Backwards-compat: some callers/tests still read this attribute.
    @property
    def SYSTEM_PROMPT(self) -> str:  # noqa: N802 (keep historical name)
        return self._system

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        kripke: KripkeModel,
        memory: MemoryStore,
        goal: str,
        agent_id: str,
        valid_actions: list[str] | None = None,
        action_specs: Sequence[ActionSpec | str] | None = None,
    ) -> Plan:
        """
        Produce a Plan given the current epistemic state and memory.
        """
        specs = coerce_action_specs(action_specs or valid_actions or [])
        action_types = action_type_names(specs)

        if not specs:
            raise ValueError("Planner.plan() requires at least one legal action spec.")

        intervention_notes: list[str] = []
        if self._simulate and action_types:
            for action in action_types:
                note = self.evaluate_intervention(
                    kripke, {"last_action_type": action}, agent_id
                )
                intervention_notes.append(f"[{action}]: {note}")

        preview_notes = self._build_preview_notes(agent_id, specs)

        prompt = self._build_prompt(
            kripke=kripke,
            memory=memory,
            goal=goal,
            agent_id=agent_id,
            action_specs=specs,
            intervention_notes=intervention_notes,
            preview_notes=preview_notes,
            tools_present=self._tools is not None,
        )

        if self._tools is not None:
            plan = self._plan_with_tools(prompt, specs)
        else:
            plan = self._plan_one_shot(prompt, specs)

        plan.intervention_notes = intervention_notes
        plan.supporting_worlds = [
            w.id for w in kripke.worlds
            if w.matches({"last_action_type": plan.action_type})
        ] or [w.id for w in kripke.worlds[:3]]

        return plan

    def evaluate_intervention(
        self,
        kripke: KripkeModel,
        hypothetical_facts: dict[str, Any],
        agent_id: str,
    ) -> str:
        """
        Symbolically evaluate the epistemic effect of asserting hypothetical_facts.
        """
        before = len(kripke.worlds)
        after_model = kripke.simulate_intervention(hypothetical_facts)
        after = len(after_model.worlds)
        new_certain = after_model.certain_facts()
        eliminated = before - after

        lines = [
            f"Asserting {hypothetical_facts}:",
            f"  worlds {before} → {after} ({eliminated} eliminated)",
            f"  newly certain: {new_certain if new_certain else '(none)'}",
        ]
        return " | ".join(lines)

    # ------------------------------------------------------------------
    # One-shot path (no tools)
    # ------------------------------------------------------------------

    def _plan_one_shot(
        self,
        prompt: str,
        specs: Sequence[ActionSpec],
    ) -> Plan:
        plan_schema = structured_plan_schema(specs)
        last_error = ""
        plan: Plan | None = None

        for _ in range(self._max_parse_retries + 1):
            retry_prompt = prompt
            if last_error:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response was invalid: {last_error}\n"
                    "Return a corrected JSON object using the same action schemas."
                )
            try:
                raw = self._llm.complete_structured(
                    retry_prompt,
                    schema=plan_schema,
                    system=self._system,
                )
                plan = self._parse_response(raw, specs)
                break
            except (ActionSchemaError, PlanParseError, ValueError) as exc:
                last_error = str(exc)

        if plan is None:
            plan = self._fallback_plan(specs, last_error)
        return plan

    # ------------------------------------------------------------------
    # ReAct path (tools present)
    # ------------------------------------------------------------------

    def _plan_with_tools(
        self,
        prompt: str,
        specs: Sequence[ActionSpec],
    ) -> Plan:
        """
        Run a bounded ReAct loop using the supplied tool registry plus an
        auto-registered ``submit_plan`` tool that captures the structured
        decision and stops the loop.
        """
        # Build a per-call working registry: the env's tools + submit_plan.
        registry = ToolRegistry()
        for defn in self._tools.definitions():  # type: ignore[union-attr]
            _, fn = self._tools._entries[defn.name]  # type: ignore[union-attr]
            registry.register(defn, fn)

        captured: dict[str, Any] = {}
        tool_call_log: list[dict[str, Any]] = []

        action_types = [s.action_type for s in specs]
        by_type = action_spec_by_type(specs)

        def _submit_plan(
            intent: str = "",
            action_type: str = "",
            parameters: dict | None = None,
            public_rationale: str = "",
        ) -> str:
            if action_type not in by_type:
                return (
                    f"Error: action_type {action_type!r} is not legal. "
                    f"Choose one of: {action_types}."
                )
            if parameters is None or not isinstance(parameters, dict):
                return "Error: parameters must be a JSON object."
            try:
                validated = by_type[action_type].validate_payload(parameters)
            except ActionSchemaError as exc:
                return f"Error validating parameters: {exc}"
            captured["plan"] = Plan(
                intent=str(intent or "planned action"),
                action_type=action_type,
                parameters=validated,
                reasoning=str(public_rationale or ""),
            )
            return "Plan recorded. End your turn now — do not call any further tools."

        registry.register(_submit_plan_definition(action_types), _submit_plan)

        # Seed the conversation with the prompt as a single user message.
        budget_intro = (
            f"\n\nYou have a tool-call budget of {self._max_tool_iter} "
            "iterations for this turn. Use research tools sparingly and "
            "call submit_plan as soon as you have enough information to "
            "decide. If submit_plan validation fails, the error message "
            "is returned to you and you can call it again with corrected "
            "arguments."
        )
        messages: list[dict] = [{"role": "user", "content": prompt + budget_intro}]

        for iteration in range(1, self._max_tool_iter + 1):
            try:
                response: LLMResponse = self._llm.complete_with_tools(
                    messages=messages,
                    registry=registry,
                    system=self._system,
                )
            except NotImplementedError:
                log.warning(
                    "Backend %r does not implement complete_with_tools(); "
                    "falling back to one-shot structured planning.",
                    self._llm,
                )
                return self._plan_one_shot(prompt, specs)

            if response.is_final:
                # Model gave a final text response without calling submit_plan.
                # Try to parse the text as a structured plan as a fallback.
                if "plan" not in captured:
                    try:
                        captured["plan"] = self._parse_response(
                            response.content or "", specs
                        )
                    except (PlanParseError, ActionSchemaError, ValueError):
                        pass
                break

            # Append the assistant turn (tool_calls) and dispatch each call.
            messages.append(_assistant_message(response.tool_calls))
            for tc in response.tool_calls:
                result = registry.dispatch(tc)
                tool_call_log.append({
                    "call": {"name": tc.name, "arguments": tc.arguments},
                    "result": result.content,
                })
                messages.append(result.to_openai_message())
                if tc.name == "submit_plan" and "plan" in captured:
                    # Tight termination once a valid plan has been captured.
                    break

            if "plan" in captured:
                break

            # On the second-to-last iteration, nudge the model to commit.
            # Inject the warning as a synthetic user message so it's visible
            # before the next assistant turn.
            remaining = self._max_tool_iter - iteration
            if remaining == 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "BUDGET WARNING: only 1 tool-call iteration remains. "
                        "Call submit_plan now with your best decision — if "
                        "you do not, a fallback action will be selected for "
                        "you."
                    ),
                })

        if "plan" not in captured:
            log.warning(
                "Planner ReAct loop ended without a valid plan after %d iterations; "
                "falling back to first legal action.",
                self._max_tool_iter,
            )
            plan = self._fallback_plan(
                specs,
                "ReAct loop ended without calling submit_plan with valid arguments.",
            )
        else:
            plan = captured["plan"]

        plan.tool_calls = tool_call_log
        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_preview_notes(
        self,
        agent_id: str,
        specs: Sequence[ActionSpec],
    ) -> list[str]:
        """
        For each legal action with a single example payload, ask the env
        what would happen. Returns concise strings suitable for the prompt.

        We use ``spec.examples[0]`` as the canonical "this action with these
        parameters" representative when an action_type has multiple legal
        parameter shapes (e.g. one ``slide`` per legal direction). For a
        well-formed env the example should already match the legal payload.
        """
        if self._preview is None:
            return []
        notes: list[str] = []
        for spec in specs:
            for example in spec.examples or [{}]:
                try:
                    result = self._preview(agent_id, spec.action_type, example)
                except Exception as exc:  # pragma: no cover — env bug guard
                    notes.append(
                        f"[{spec.action_type} {example}]: preview error: {exc}"
                    )
                    continue
                if result is None:
                    continue
                notes.append(
                    f"[{spec.action_type} {example}]: {json.dumps(result, sort_keys=True, default=str)}"
                )
        return notes

    def _build_prompt(
        self,
        kripke: KripkeModel,
        memory: MemoryStore,
        goal: str,
        agent_id: str,
        action_specs: Sequence[ActionSpec],
        intervention_notes: list[str],
        preview_notes: list[str],
        tools_present: bool,
    ) -> str:
        sections: list[str] = []

        sections.append(f"=== AGENT: {agent_id} | GOAL: {goal} ===")

        sections.append("--- EPISTEMIC STATE ---")
        sections.append(kripke.summary())

        sections.append("--- RECENT MEMORY ---")
        sections.append(memory.short_term_context(k=15))

        if intervention_notes:
            sections.append("--- INTERVENTION SIMULATIONS ---")
            sections.append(
                "For each valid action, here is the epistemic effect of asserting it:"
            )
            sections.extend(intervention_notes)

        if preview_notes:
            sections.append("--- ACTION PREVIEWS ---")
            sections.append(
                "Result of taking each candidate action (read-only "
                "counterfactual; no commitment):"
            )
            sections.extend(preview_notes)

        sections.append("--- LEGAL ACTION SCHEMAS ---")
        sections.append(format_action_specs_for_prompt(action_specs))

        if tools_present:
            sections.append(
                "--- YOUR TASK ---\n"
                "You have tools available. Use them to inspect the state and "
                "evaluate hypotheticals before deciding. When you are ready, "
                "call submit_plan(intent, action_type, parameters, "
                "public_rationale) — this is mandatory and ends your turn. "
                "Choose action_type from the legal action schemas and make "
                "parameters match that schema."
            )
        else:
            sections.append(
                "--- YOUR TASK ---\n"
                "Output a JSON object with keys: intent, action_type, "
                "parameters, public_rationale.\n"
                "Choose action_type from the legal action schemas and make "
                "parameters match that schema."
            )

        return "\n\n".join(sections)

    def _parse_response(
        self,
        raw: str | Mapping[str, Any],
        action_specs: Sequence[ActionSpec | str],
    ) -> Plan:
        """
        Parse and validate LLM output into a Plan.
        """
        data = self._coerce_response_dict(raw)
        by_type = action_spec_by_type(action_specs)
        action_type = str(data.get("action_type", ""))

        if action_type not in by_type:
            raise PlanParseError(
                f"LLM chose unknown action_type {action_type!r}; "
                f"expected one of {list(by_type)}."
            )

        parameters = data.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise PlanParseError("LLM parameters must be a JSON object.")

        validated_parameters = by_type[action_type].validate_payload(parameters)
        public_rationale = data.get("public_rationale", data.get("reasoning", ""))

        return Plan(
            intent=str(data.get("intent", "unknown")),
            action_type=action_type,
            parameters=validated_parameters,
            reasoning=str(public_rationale),
        )

    def _coerce_response_dict(self, raw: str | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)

        cleaned = raw.strip()
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise PlanParseError(f"LLM did not return JSON: {raw!r}")
            data = json.loads(match.group())

        if not isinstance(data, dict):
            raise PlanParseError("LLM plan response must be a JSON object.")
        return data

    def _fallback_plan(
        self,
        action_specs: Sequence[ActionSpec | str],
        reason: str,
    ) -> Plan:
        specs = coerce_action_specs(action_specs)
        spec = specs[0]
        return Plan(
            intent="fallback legal action",
            action_type=spec.action_type,
            parameters=spec.fallback_payload(),
            reasoning=f"Fell back after invalid structured output: {reason}",
        )


# ---------------------------------------------------------------------------
# submit_plan tool definition (auto-registered in ReAct mode)
# ---------------------------------------------------------------------------

def _submit_plan_definition(action_types: list[str]) -> ToolDefinition:
    return ToolDefinition(
        name="submit_plan",
        description=(
            "Submit your final structured decision for this turn. Call this "
            "as your LAST tool call — it ends your turn. Use the legal "
            "action_type values and provide parameters matching that "
            "action's payload schema (shown under LEGAL ACTION SCHEMAS in "
            "the prompt)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Your high-level goal for this step.",
                },
                "action_type": {
                    "type": "string",
                    "enum": action_types,
                    "description": "Must be one of the legal action types.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Payload matching the chosen action's schema.",
                },
                "public_rationale": {
                    "type": "string",
                    "description": "Brief reasoning safe to log.",
                },
            },
            "required": ["intent", "action_type", "parameters"],
        },
    )


def _assistant_message(tool_calls: list[ToolCall]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ],
    }
