# Notebook 01: Quickstart — Belief Management

## What You'll Learn

- Creating a `BeliefState`
- Adding, querying, updating, and removing beliefs
- Contradiction detection
- Serialization round-trip

## Setup

```bash
pip install belief-agent
```

```python
from belief_agent import BeliefState, Belief
```

## 1. Create a Fresh State

```python
state = BeliefState()
print(state)
# BeliefState(empty)
```

## 2. Add Beliefs

```python
state.add_belief("Python is dynamically typed", confidence=0.95, source="knowledge")
state.add_belief("The sky is blue", confidence=0.9, source="observation")
state.add_belief("The sky is not blue", confidence=0.2, source="user")  # contradiction
print(state)
```

## 3. Query

```python
# Substring match
for b in state.query("sky"):
    print(b)
    # [90%] The sky is blue (source: observation)
    # [20%] The sky is not blue (source: user)

# By tag
state.add_belief("Fast", tags=["performance"])
print(state.query_by_tag("performance"))
```

## 4. Contradictions

```python
pairs = state.get_contradictions()
for a, b in pairs:
    print(f"Contradiction: {a.fact}  <->  {b.fact}")
```

## 5. Update

```python
b = state.query("sky is blue")[0]
state.update_belief(b.id, confidence=1.0)
print(b)
# [100%] The sky is blue (source: observation)
```

## 6. JSON Round-Trip

```python
json_str = state.serialize(indent=2)
restored = BeliefState.deserialize(json_str)
assert len(restored) == len(state)
print("Round-trip OK")
```

## 7. Merge

```python
other = BeliefState()
other.add_belief("The user likes hiking", confidence=0.7, source="inference")
added = state.merge(other)
print(f"Added {added} new belief(s)")
print(state)
```

## Summary

You now know the core `BeliefState` API. Proceed to **Notebook 02** to see how reflection prevents drift over long conversations.
