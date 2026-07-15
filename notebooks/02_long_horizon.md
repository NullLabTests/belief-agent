# Notebook 02: Long-Horizon Planning — Reflection Prevents Drift

## What You'll Learn

- How vanilla LLM agents "drift" across conversation turns
- How `reflect()` catches contradictions and updates beliefs
- How `BeliefAgent` with `auto_reflect=True` maintains consistency

## The Drift Problem

Standard LLM agents have no persistent memory of facts they've stated. Over a long conversation, they may:

- Contradict earlier statements
- Forget constraints established in earlier turns
- Shift their "position" on a topic

## Setup

```python
from belief_agent import BeliefState, BeliefAgent, LLMClient, reflect
```

## 1. Vanilla Agent Demo

```python
class NaiveClient(LLMClient):
    def __init__(self):
        self.calls = 0
    def complete(self, messages, **kwargs):
        self.calls += 1
        last = messages[-1]["content"] if messages else ""
        if "climate" in last and self.calls > 2:
            # "Forgets" the original fact
            return "The climate is tropical and humid."
        return "I will keep that in mind."

client = NaiveClient()
history = []

turns = [
    "Remember: our project climate is cold and dry.",
    "What is the capital of France?",
    "What is the climate of our project?",
]

for msg in turns:
    history.append({"role": "user", "content": msg})
    resp = client.complete(history)
    history.append({"role": "assistant", "content": resp})
    print(f"User: {msg}\nAgent: {resp}\n")
```

> **Observed**: The agent says *"tropical and humid"* on turn 3, contradicting the original *"cold and dry"*.

## 2. Belief Agent (with Reflection)

```python
state = BeliefState()
agent = BeliefAgent(
    client=NaiveClient(),
    state=state,
    auto_reflect=True,
    reflect_mode="auto",
)

for msg in turns:
    resp = agent.chat(msg)
    print(f"User: {msg}\nAgent: {resp}")
    # Show matching beliefs
    for b in agent.state.query("climate"):
        print(f"  [belief] {b}")
    print()
```

> **Observed**: The belief about "cold and dry" is recorded. When the model tries to say "tropical", reflection detects the contradiction and updates the belief, keeping the agent consistent.

## How Reflection Works

1. After each response, `reflect()` is called with the user message, agent response, and current beliefs.
2. The LLM is prompted (via `REFLECTION_PROMPT`) to analyze the conversation.
3. It returns structured JSON: contradictions, new beliefs, confidence updates, removals.
4. These proposals are applied to the `BeliefState` automatically.

## Visual Summary

```
Turn 1: "climate is cold and dry"
  → Belief added: [95%] climate is cold and dry

Turn 2: "capital of France is Paris"  (unrelated)
  → No change to beliefs

Turn 3: "climate is tropical and humid"
  → Reflection: "cold and dry" vs "tropical and humid" → CONTRADICTION
  → Belief updated: [10%] climate is cold and dry  (confidence dropped)
  → New belief: [80%] climate is tropical and humid
```

## Key Takeaway

Without explicit belief tracking, the agent drifts. With `BeliefAgent` + reflection, the system detects and corrects inconsistencies — making it suitable for long-horizon tasks like research, project management, and multi-step reasoning.

Proceed to **Notebook 03** for multi-objective negotiation.
