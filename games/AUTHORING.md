# Authoring a New Game

This document describes the contract a new `GameEnvironment` should follow
to make full use of the framework — tool-calling, game-specific prompts,
counterfactual previews, and the evaluation runner infrastructure.

The minimum required to add a game is short. Doing it _well_ — so an LLM
agent reliably beats trivial baselines — requires picking up four optional
hooks and wiring an evaluation runner. This guide covers both.

## TL;DR

1. Subclass `GameEnvironment` (`games/base.py`) and implement the four
   abstract methods.
2. Override the **three optional hooks** that match your game's structure
   (decision tree below).
3. Add a runner under `evaluations/<game>/eval.py` that uses
   `evaluations.common.build_planner(env, llm)`.
4. Add a smoke test under `tests/`.
5. Read the checklist at the bottom before opening a PR.

---

## Required interface

`GameEnvironment` is the only abstraction the agent talks to. Five
methods/properties are required:

| Member | Purpose |
|---|---|
| `observe(agent_id) -> dict` | Per-turn percept. Schema is game-specific but should include `kind`, `source`, `content`, optionally `facts`, `reward`, `terminal`. Consumed by `FeedbackProcessor`. |
| `step(agent_id, action) -> dict` | Apply `action`, advance state, return a feedback dict in the same schema as `observe`. |
| `action_specs(agent_id) -> list[ActionSpec]` | Legal structured actions, each with a Pydantic payload model. The agent's planner uses these to constrain LLM output. **Filter no-op actions here** — random sampling from the spec list is your best defense against the LLM picking obviously-wasteful moves. |
| `is_terminal -> bool` | Game-over flag. |
| `initial_kripke(agent_id) -> KripkeModel` | The agent's initial epistemic state. **Override this only if your game has hidden information.** Fully-observable games (puzzles) use the default trivial one-world model. |

`valid_actions(agent_id)` is provided by the base class and just returns
the action_type strings — keep using it for legacy/back-compat callers.

---

## Optional hooks (this is where the leverage is)

Three optional methods on `GameEnvironment` let you specialise the
planner without touching framework code. They are how 2048 and Mastermind
beat the random baseline.

### `system_prompt() -> str`

Game-specific reactive system prompt. **Default**: `REACTIVE_SYSTEM` from
`causal_agent/prompts.py`, which is written for hidden-information /
epistemic games (Werewolf-like).

**Override when:**
- Your game is fully observable (epistemic-worlds language is misleading).
- Optimal play requires concrete heuristics the LLM should follow
  (corner anchoring in 2048; entropy maximisation in Mastermind).

Add your prompt to `causal_agent/prompts.py` as a constant
(e.g. `MY_GAME_SYSTEM`) and return it from `system_prompt()`. Keep the
output-format paragraph from `REACTIVE_SYSTEM` (the JSON schema for the
plan) — the planner relies on it.

### `tools(agent_id) -> ToolRegistry | None`

Game-specific tools the planner exposes to the LLM. When non-empty, the
planner switches from one-shot structured output to a **bounded ReAct
loop** (`BaseLLM.complete_with_tools(...)`) and the LLM finishes its
turn by calling the auto-registered `submit_plan` tool.

**Default**: `None` (no tools; planner stays one-shot).

**Override when:**
- Your game admits cheap counterfactual evaluation (deterministic puzzle:
  expose `simulate_move`).
- The state space has structure the LLM benefits from inspecting
  programmatically (Mastermind: expose `candidate_count`,
  `enumerate_candidates`, `expected_information`).
- The action space is small but the consequence space is large (2048: 4
  directions, but each leads to a different post-merge board).

Build the registry with `causal_agent.tools.ToolRegistry`. For tools
that read live env state, the canonical pattern is a class with
`register_all(registry)` and a `get_env` callable so the toolset always
sees the latest state — see `causal_agent/game_2048_tools.py` for a
worked example.

**Do NOT register tools here that need agent-side state** (e.g. the
running Kripke model owned by the orchestrator). See
"Runner-level tool registration" below.

### `preview(agent_id, action) -> dict | None`

Read-only counterfactual: what would `step(action)` produce, without
committing? **Default**: `None` (env can't preview cheaply).

When provided, the planner calls it for each legal action's example
payload before each LLM call and embeds the result in the prompt as an
`ACTION PREVIEWS` section. This gives the LLM **free one-step
lookahead** — typically the cheapest single fix to make an LLM agent
beat random on deterministic games.

**Override when:**
- Your game is deterministic (or close to it) and a single-step outcome
  is well-defined and cheap.
- The action's effect on the observation is large but easy to compute
  (2048 slide; Sokoban push; sliding-puzzle move).

**Do not override when:**
- Stepping is expensive (many simulation cycles).
- Outcomes are stochastic and a one-shot preview is misleading.
- Hidden information is involved (the env can't reveal it).

`preview()` MUST NOT mutate any env state. Run a copy of the inner state
or use a pure helper.

---

## Decision tree

```
                                   ┌─ Hidden information? ─┐
                                   │                       │
                                  yes                      no
                                   │                       │
                ┌──────────────────┘                       └──────────────────┐
                ▼                                                              ▼
   Override initial_kripke()                              initial_kripke() = default (1 world)
   to encode possible worlds.                             Don't pretend Kripke matters here.
   Consider Kripke tools at the
   *runner* level (see below).
                │                                                              │
                ▼                                                              ▼
   ┌─ Cheap to simulate one step? ─┐                          ┌─ Cheap to simulate one step? ─┐
   │                                │                          │                                │
   yes                               no                          yes                               no
   │                                │                          │                                │
   ▼                                ▼                          ▼                                ▼
   Override preview().              Skip preview.              Override preview().              Skip preview.
   Register simulate_*              Lean on intervention      Register simulate_* /            Lean on system_prompt()
   tools where useful.              simulations through       score_* tools (this is           heuristics; the LLM is
                                    the Kripke model.          where puzzles win).              flying blind.
                │                                                              │
                ▼                                                              ▼
   ┌─ Optimal play has concrete heuristics? ────────────────────────────────────┐
   │                                                                            │
   yes                                                                          no
   │                                                                            │
   ▼                                                                            ▼
   Override system_prompt() and add your prompt to                              system_prompt() = default
   causal_agent/prompts.py. Be concrete: name the                              (REACTIVE_SYSTEM is fine).
   strategy, list the priorities, give the action ordering.
```

---

## Three worked examples

### Werewolf — hidden info, no preview, no env-level tools

- `initial_kripke()`: one world per consistent role assignment
  (`games/werewolf/env.py:initial_kripke`).
- `system_prompt()`: default `REACTIVE_SYSTEM` (epistemic-game language
  is the right framing).
- `tools()`: returns `None`. Werewolf's natural tools are the Kripke
  inspection tools (`KripkeToolset`), but those need the agent's
  *running* Kripke model — owned by the orchestrator, not the env.
  Register them at the runner level (see below).
- `preview()`: returns `None`. There is no useful read-only preview
  for a vote / kill / speech act in a hidden-info social game.

### Mastermind — hidden info, deduction, env-level tools

- `initial_kripke()`: one world per candidate code
  (`games/mastermind/env.py:initial_kripke`). Updates via
  `update_with_facts(...)` as feedback arrives.
- `system_prompt()`: `MASTERMIND_SYSTEM` — describes information-theoretic
  guess selection (high-entropy partitions early, consistent guesses
  late).
- `tools()`: returns a `MastermindToolset` registry — `candidate_count`,
  `enumerate_candidates`, `filter_candidates`, `expected_information`,
  `score_guess` (`causal_agent/mastermind_tools.py`). All read live
  env history.
- `preview()`: not overridden. A single guess's outcome depends on the
  hidden secret; previewing isn't cheap or honest.

### 2048 — full info, deterministic, env-level tools, preview

- `initial_kripke()`: default trivial single-world model (no hidden
  state). Kripke is decorative here.
- `system_prompt()`: `GAME_2048_SYSTEM` — corner anchoring, monotonic
  gradient, never-up rule.
- `tools()`: returns a `Game2048Toolset` registry — `simulate_move`,
  `score_board`, `count_empty_cells`, `max_tile`, `legal_directions`
  (`causal_agent/game_2048_tools.py`).
- `preview()`: returns `{gained, empty_after, max_tile_after}` for each
  candidate slide (`games/game_2048/env.py:preview`). The planner's
  `ACTION PREVIEWS` section embeds these in every prompt, giving free
  one-step lookahead.

---

## Runner-level tool registration

Some tools — specifically the canonical Kripke inspection tools — need
to read the agent's *running* state, which is owned by the orchestrator
or evaluation runner, not by the env. If you have such tools:

1. Don't register them in `env.tools()`.
2. In your runner, build a `ToolRegistry` whose definitions close over a
   getter pointing at the runner's live state. For Kripke tools, that
   means a getter that returns the orchestrator's current `KripkeModel`.
3. Construct the `Planner` directly (instead of via `build_planner`),
   passing both your registry and the env-level registry merged
   together. Or: extend `env.tools()` to return a registry that the
   runner *augments* with runner-level entries before passing to
   `Planner`.

Today only Werewolf falls into this case, and the
`examples/run_werewolf.py` flow uses `Orchestrator` (which doesn't yet
expose a tool-registration hook). When that's needed, follow the
pattern above.

---

## Adding an evaluation runner

Convention: `evaluations/<game>/eval.py`. Reuse the helpers in
`evaluations/common/`:

| Helper | Use |
|---|---|
| `add_llm_args(parser)` | Adds `--model`, `--temperature`, etc. |
| `build_llm(args, mock_responses)` | Returns the configured `BaseLLM`. |
| `build_planner(env, llm, agent_id)` | Returns a `Planner` already wired to the env's `system_prompt()`, `tools()`, and `preview()`. **Use this** unless you need runner-level tools (above). |
| `TraceLogger(log_dir, filename)` | JSONL writer for per-episode traces. |
| `write_summary(log_dir, policy, summary)` | Writes the aggregate JSON. |
| `dataclass_to_dict(obj)` | Serialises an `EpisodeResult` dataclass. |

A new runner is typically ~250 lines; see
`evaluations/game_2048/eval.py` and `evaluations/mastermind/eval.py` for
templates. Keep at least these policies:

- `random` (sanity baseline; the LLM must beat this).
- One game-specific deterministic baseline (`greedy` for 2048, `knuth`
  or `candidate` for Mastermind).
- `interactive` (a humans-in-the-loop policy for debugging).
- `llm` (uses `build_planner`).

---

## Authoring checklist

Before opening a PR for a new game, confirm each of:

- [ ] Subclass of `GameEnvironment` is in `games/<game>/env.py`.
- [ ] All four required abstract methods are implemented.
- [ ] `action_specs()` filters no-op / illegal actions; the Pydantic
      payload model encodes any per-turn legal subset (e.g. enum of
      legal directions, length-bounded list of legal symbols).
- [ ] `is_terminal` is correctly set whenever the env transitions to a
      terminal state.
- [ ] `initial_kripke()` is overridden **only** if your game has hidden
      information; otherwise rely on the default.
- [ ] `system_prompt()` is overridden if generic epistemic framing is
      misleading or if optimal play has concrete heuristics; the prompt
      is added to `causal_agent/prompts.py`.
- [ ] `tools()` is overridden when the game admits useful env-level
      simulation or inspection; the toolset lives in
      `causal_agent/<game>_tools.py` and follows the
      `register_all(registry)` pattern.
- [ ] `preview()` is overridden for deterministic games where a
      one-step counterfactual is cheap; it does NOT mutate state.
- [ ] An evaluation runner is added under `evaluations/<game>/eval.py`
      using `build_planner(env, llm)`.
- [ ] At least one smoke test is added under `tests/` (use
      `tests/test_game_smoke.py` as a template).
- [ ] The LLM policy beats the random baseline on a small (≥10 episode)
      run with a real backend (DeepSeek, Anthropic, OpenAI). If it
      doesn't: the system prompt or the toolset is the most likely
      culprit, in that order.

---

## What you can rely on (do not rewrite)

- `causal_agent/tools.py` — `ToolRegistry`, `ToolDefinition`, `ToolCall`,
  `LLMResponse`. All backends already implement
  `complete_with_tools(messages, registry, ...)`.
- `causal_agent/kripke_tools.py` — `KripkeToolset` (template for
  game-specific toolsets; copy the pattern).
- `causal_agent/research_planner.py` — full ReAct loop for *planning
  phase* (research, web tools, plan_complete). The reactive `Planner`
  has its own bounded ReAct loop for per-turn decisions; pick the right
  one.
- `causal_agent/llm.py` — backends (`MockLLM`, `OpenAILLM`,
  `AnthropicLLM`, `GeminiLLM`, `DeepSeekLLM`). Each implements both
  `complete()` and `complete_with_tools()`.
- `evaluations/common/{llm,logging,planner_factory,types}.py` — the
  helpers above.
- `causal_agent/log_config.py` — `setup_logging` / `get_logger` for any
  diagnostics you add to your env or toolset.
