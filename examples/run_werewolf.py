"""
examples/run_werewolf.py

End-to-end demo: one AI agent playing Werewolf with MockLLM (no API key needed).

Swap MockLLM for OpenAILLM, AnthropicLLM, GeminiLLM, or DeepSeekLLM to use a
real backend.

Usage
-----
    python -m examples.run_werewolf
    python -m examples.run_werewolf --model openai --openai-key sk-...
    python -m examples.run_werewolf --model anthropic --anthropic-key sk-ant-...
    python -m examples.run_werewolf --model gemini --gemini-key AIza...
    python -m examples.run_werewolf --model deepseek --deepseek-key <key>
"""

from __future__ import annotations

import argparse
import os
import sys

# Load .env before any SDK imports so keys are in the environment.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; keys can still come from the shell environment

# Make the repo root importable when running as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causal_agent import (
    AgentConfig,
    Actor,
    AnthropicLLM,
    DeepSeekLLM,
    FeedbackProcessor,
    GeminiLLM,
    MemoryStore,
    MockLLM,
    OpenAILLM,
    Orchestrator,
    Planner,
    setup_logging,
)
from games.werewolf import WerewolfEnv


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def build_llm(args: argparse.Namespace):
    if args.model == "openai":
        print(f"[llm] Using OpenAI ({args.openai_model})")
        return OpenAILLM(model=args.openai_model, api_key=args.openai_key)
    if args.model == "anthropic":
        print(f"[llm] Using Anthropic ({args.anthropic_model})")
        return AnthropicLLM(model=args.anthropic_model, api_key=args.anthropic_key)
    if args.model == "gemini":
        print(f"[llm] Using Gemini ({args.gemini_model})")
        return GeminiLLM(model=args.gemini_model, api_key=args.gemini_key)
    if args.model == "deepseek":
        print(f"[llm] Using DeepSeek ({args.deepseek_model})")
        return DeepSeekLLM(model=args.deepseek_model, api_key=args.deepseek_key)
    print("[llm] Using MockLLM (no API calls)")
    return MockLLM()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    setup_logging("INFO")

    players = ["Agent", "Alice", "Bob", "Charlie", "Dave"]
    agent_id = "Agent"

    print("=" * 60)
    print("  Causal Reasoning Agent — Werewolf Demo")
    print("=" * 60)
    print(f"Players : {players}")
    print(f"Agent   : {agent_id}")
    print()

    # ── Environment ──────────────────────────────────────────────────
    env = WerewolfEnv(
        players=players,
        agent_id=agent_id,
        n_werewolves=1,
        seed=args.seed,
    )
    print(f"[env] Roles assigned. Agent role: {env._players[agent_id].role}")
    print()

    # ── Kripke model (built by the env) ──────────────────────────────
    kripke = env.initial_kripke(agent_id)
    print(kripke.summary())
    print()

    # ── Modules ──────────────────────────────────────────────────────
    llm               = build_llm(args)
    planner           = Planner(llm, simulate_before_plan=True)
    actor             = Actor(post_processors=[Actor.truncate_message(250)])
    feedback_proc     = FeedbackProcessor()
    memory            = MemoryStore(max_short_term=60)

    config = AgentConfig(
        agent_id=agent_id,
        goal=(
            "Win the game. If you are a villager, identify and eliminate the werewolf. "
            "If you are the werewolf, eliminate villagers without being caught."
        ),
        max_turns=50,
        verbose=True,
    )

    # ── Orchestrator ─────────────────────────────────────────────────
    orchestrator = Orchestrator(
        env=env,
        planner=planner,
        actor=actor,
        feedback_processor=feedback_proc,
        memory=memory,
        kripke=kripke,
        config=config,
    )

    print("Starting session…\n")
    result = orchestrator.run_session()

    # ── Results ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(result.summary())
    print(f"Winner : {env._winner}")
    print(f"Actions taken by agent: {len(result.actions)}")
    print()
    print("--- Final epistemic state ---")
    print(f"  Kripke snapshots recorded : {len(memory.kripke_history())}")
    snap = memory.last_kripke_snapshot()
    if snap:
        print(f"  Final snapshot            : {snap}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Werewolf demo for causal_agent framework.")
    p.add_argument("--model", choices=["mock", "openai", "anthropic", "gemini", "deepseek"], default="mock")
    p.add_argument("--openai-key", default=None)
    p.add_argument("--openai-model", default="gpt-4o")
    p.add_argument("--anthropic-key", default=None)
    p.add_argument("--anthropic-model", default="claude-3-5-sonnet-20241022")
    p.add_argument("--gemini-key", default=None)
    p.add_argument("--gemini-model", default="gemini-2.0-flash")
    p.add_argument("--deepseek-key", default=None)
    p.add_argument("--deepseek-model", default="deepseek-chat")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
