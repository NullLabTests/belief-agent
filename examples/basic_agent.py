"""Basic agent example — demonstrates belief state without an LLM.

Run:
    python examples/basic_agent.py

This example exercises BeliefState directly (no API key needed) to show
how beliefs are added, queried, updated, serialized, and checked for
contradictions.
"""

from __future__ import annotations

from belief_agent import Belief, BeliefState


def main() -> None:
    print("=" * 60)
    print("belief-agent: Basic Belief State Demo")
    print("=" * 60)

    # --- Create a fresh state ---
    state = BeliefState()
    print(f"\nCreated: {state}")

    # --- Add beliefs ---
    b1 = state.add_belief(
        "The user prefers Python over JavaScript",
        confidence=0.9,
        source="user",
        tags=["preference", "language"],
    )
    b2 = state.add_belief(
        "The sky is blue",
        confidence=0.95,
        source="observation",
        tags=["fact"],
    )
    b3 = state.add_belief(
        "The sky is not blue",  # will trigger contradiction
        confidence=0.3,
        source="user",
        tags=["claim"],
    )
    print(f"\nAdded 3 beliefs. State now has {len(state)} beliefs.")
    print(state)

    # --- Query ---
    print("\n--- Query: 'sky' ---")
    for b in state.query("sky"):
        print(f"  {b}")

    # --- Contradictions ---
    print("\n--- Contradictions ---")
    contradictions = state.get_contradictions()
    if contradictions:
        for a, b in contradictions:
            print(f"  {a.fact}  <-->  {b.fact}")
    else:
        print("  None detected")

    # --- Update a belief ---
    state.update_belief(b2.id, confidence=1.0)
    print(f"\nUpdated '{b2.fact}' confidence to 1.0.")
    print(f"  -> {state.query('sky is blue')[0]}")

    # --- Serialization round-trip ---
    json_str = state.serialize(indent=2)
    print(f"\nSerialized JSON ({len(json_str)} chars):")
    print(json_str[:300] + "...")

    restored = BeliefState.deserialize(json_str)
    assert len(restored) == len(state)
    print(f"\nDeserialized back: {len(restored)} beliefs — round-trip OK")

    # --- Merge ---
    other = BeliefState()
    other.add_belief("The user enjoys hiking", confidence=0.7, source="inference")
    added = state.merge(other)
    print(f"\nMerged another state: {added} new belief(s) added.")
    print(state)

    print("\n✓ Basic demo complete.")


if __name__ == "__main__":
    main()
