"""Reflection loop with human-in-the-loop."""

import json

from belief_agent import BeliefAgent, BeliefState
from belief_agent.reflection import reflect, human_review


def critique_llm(messages):
    return json.dumps({
        "critique": "The belief seems reasonable but lacks strong evidence.",
        "new_evidence": ["Additional source confirms"],
        "new_contradictions": ["Alternative interpretation exists"],
        "updated_confidence": 0.55,
    })


agent = BeliefAgent(llm_call=critique_llm, name="CritiquedAgent")
agent.adopt(BeliefState(
    "AI will replace all programmers",
    confidence=0.8,
    evidence=["Automation is advancing rapidly"],
))

print("Before reflection:")
print(f"  {agent.get_belief('AI will replace all programmers')}")

results = reflect(agent, critique_llm)

print("\nReflection results:")
for r in results:
    print(f"  {r.proposition}: {r.original_confidence:.2f} -> {r.updated_confidence:.2f}")
    print(f"  Critique: {r.critique}")
    print(f"  Accepted: {r.accepted}")

# simulate human override
print("\n--- Human override ---")
result = results[0]
result = human_review(result, "I disagree with the lowered confidence", accepted=False)
print(f"After human review: accepted={result.accepted}")
print(f"Critique now includes: {result.critique}")

print("\nFinal belief:")
print(f"  {agent.get_belief('AI will replace all programmers')}")
