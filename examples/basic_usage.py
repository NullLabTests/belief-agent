"""Minimal example: create an agent with a fake LLM and observe beliefs."""

import json

from belief_agent import BeliefAgent, BeliefState


def fake_llm(messages):
    last = messages[-1]["content"]
    if "reflect" in last.lower():
        return json.dumps({
            "proposition": "I should reflect",
            "confidence": 0.6,
            "contradictions": ["but I am a fake LLM"],
            "evidence": ["reflection is useful"],
        })
    return json.dumps({
        "proposition": f"response to: {last[:40]}",
        "confidence": 0.85,
    })


agent = BeliefAgent(llm_call=fake_llm, system_prompt="You are a demo agent.")

# adopt beliefs manually
agent.adopt(BeliefState("Python is a great language", confidence=0.9, source="developer"))
agent.adopt(BeliefState("TypeScript is also great", confidence=0.7, source="developer"))

print("Beliefs after manual adoption:")
for b in agent.list_beliefs():
    print(f"  - {b}")

# interact
print("\nUser: What do you think about Rust?")
reply = agent("What do you think about Rust?")
print(f"Agent: {reply}")
print(f"Beliefs now: {len(agent.beliefs)}")

# reflect
print("\n--- Reflection ---")
reflection = agent.reflect()
print(f"Reflection result: {reflection}")

print("\nFinal belief state:")
for b in agent.list_beliefs():
    print(f"  - {b}")
