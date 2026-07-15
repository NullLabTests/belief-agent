"""Multi-objective negotiation demo — resolves conflicting goals using
structured trade-off analysis.

Uses a mock LLM client so no API key is needed.

Run:
    python examples/multi_objective_negotiation.py
"""

from __future__ import annotations

import json
from typing import Any

from belief_agent import LLMClient, negotiate_goals


class MockNegotiator(LLMClient):
    """Simulates an LLM that returns structured negotiation output."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        self.call_count += 1
        return json.dumps(
            {
                "conflicts": [
                    {
                        "goal_a": "Minimize cost",
                        "goal_b": "Maximize safety",
                        "tension": "Higher safety requires more expensive materials and testing.",
                        "severity": 0.8,
                    },
                    {
                        "goal_a": "Ship by end of month",
                        "goal_b": "Maximize safety",
                        "tension": "Rushing compromises thorough safety validation.",
                        "severity": 0.6,
                    },
                ],
                "ranked_goals": [
                    {"goal": "Maximize safety", "priority": 1, "feasibility": 0.7, "notes": "Non-negotiable for production"},
                    {"goal": "Minimize cost", "priority": 2, "feasibility": 0.5, "notes": "Can use cheaper materials in non-critical areas"},
                    {"goal": "Ship by end of month", "priority": 3, "feasibility": 0.4, "notes": "May need to slip by 2 weeks for safety validation"},
                ],
                "compromise": "Prioritize safety above all. Use cost-effective materials for non-critical "
                "components. Extend timeline by 2 weeks to complete safety validation. "
                "This satisfies the core of all three goals.",
                "recommended_beliefs": [
                    "Safety is the top priority for this project",
                    "Cost savings can be found in non-critical areas",
                    "The timeline needs a 2-week extension for safety validation",
                ],
            }
        )

    def complete_stream(self, messages: list[dict], **kwargs: Any) -> Any:
        raise NotImplementedError


def main() -> None:
    print("=" * 60)
    print("Multi-Objective Negotiation Demo")
    print("=" * 60)

    goals = [
        {"goal": "Minimize cost", "importance": 9, "stakeholders": "Finance"},
        {"goal": "Maximize safety", "importance": 10, "stakeholders": "Engineering, Legal"},
        {"goal": "Ship by end of month", "importance": 7, "stakeholders": "Product, Sales"},
    ]

    print("\nGoals:")
    for g in goals:
        print(f"  - {g['goal']} (importance: {g['importance']})")

    client = MockNegotiator()
    result = negotiate_goals(goals, client)

    print("\n--- Detected Conflicts ---")
    for c in result.get("conflicts", []):
        print(f"  {c['goal_a']} <-> {c['goal_b']}")
        print(f"    Tension: {c['tension']}")
        print(f"    Severity: {c['severity']:.0%}")

    print("\n--- Ranked Goals ---")
    for rg in result.get("ranked_goals", []):
        print(f"  #{rg['priority']}: {rg['goal']} (feasibility: {rg['feasibility']:.0%})")
        print(f"    {rg['notes']}")

    print(f"\n--- Proposed Compromise ---\n  {result.get('compromise', 'N/A')}")

    print("\n--- Recommended Beliefs ---")
    for b in result.get("recommended_beliefs", []):
        print(f"  - {b}")

    print("\n✓ Negotiation demo complete.")


if __name__ == "__main__":
    main()
