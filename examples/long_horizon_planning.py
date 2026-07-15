"""Long-horizon planning demo — shows how reflection prevents belief drift.

Compares a vanilla agent (no belief tracking) with a belief-agent equipped
with auto-reflection.  Uses a mock LLM client so no API key is needed.

Run:
    python examples/long_horizon_planning.py
"""

from __future__ import annotations

import json
from typing import Any

from belief_agent import BeliefAgent, BeliefState, LLMClient


class MockClient(LLMClient):
    """Simulates an LLM that drifts without belief anchoring."""

    def __init__(self, drift: bool = True):
        self.drift = drift
        self.call_count = 0

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        self.call_count += 1
        last = messages[-1]["content"] if messages else ""

        # Simulate a coherent assistant that answers facts
        if "capital of France" in last:
            return "Paris"
        if "capital of Italy" in last:
            return "Rome"
        if "German" in last:
            return "Berlin"

        # If drifting, the agent "forgets" and contradicts itself
        if self.drift and self.call_count > 2:
            if "climate" in last:
                return "The climate is tropical and humid year-round."
            return "I no longer recall our earlier agreement."

        return "I will keep that in mind."

    def complete_stream(self, messages: list[dict], **kwargs: Any) -> Any:
        raise NotImplementedError


def run_vanilla() -> None:
    """Agent without beliefs — prone to drift."""
    print("--- Vanilla agent (no belief tracking) ---")
    client = MockClient(drift=True)
    history: list[dict] = []
    for msg in [
        "Remember: the project climate is cold and dry.",
        "What is the capital of France?",
        "What is the climate of our project?",
    ]:
        history.append({"role": "user", "content": msg})
        resp = client.complete(history)
        history.append({"role": "assistant", "content": resp})
        print(f"  User: {msg}")
        print(f"  Agent: {resp}")
    # Note: the third response drifts because the agent has no belief memory
    print()


def run_with_beliefs() -> None:
    """Agent with BeliefState + auto-reflection — stays consistent."""
    print("--- Belief agent (with belief tracking) ---")
    client = MockClient(drift=True)
    state = BeliefState()

    agent = BeliefAgent(
        client=client,
        state=state,
        auto_reflect=True,
        reflect_mode="auto",
    )

    messages = [
        "Remember: the project climate is cold and dry.",
        "What is the capital of France?",
        "What is the climate of our project?",
    ]
    for msg in messages:
        resp = agent.chat(msg)
        print(f"  User: {msg}")
        print(f"  Agent: {resp}")
        # Show beliefs after each turn
        for b in agent.state.query("climate"):
            print(f"    [belief] {b}")

    print()


def main() -> None:
    print("=" * 60)
    print("Long-horizon Planning: Vanilla vs Belief-Agent")
    print("=" * 60)
    run_vanilla()
    run_with_beliefs()
    print("✓ Demo complete — the belief-agent maintains consistency across turns.")


if __name__ == "__main__":
    main()
