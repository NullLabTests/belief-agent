# Notebook 03: Multi-Objective Negotiation

## What You'll Learn

- Defining conflicting goals
- Using `negotiate_goals()` for structured trade-off analysis
- Interpreting conflict severity, feasibility scores, and compromise proposals
- Integrating recommended beliefs into your `BeliefState`

## The Problem

Real-world agents often juggle conflicting goals:

- **Cost** vs **Quality**
- **Speed** vs **Safety**
- **User A**'s preferences vs **User B**'s preferences
- Short-term **shipping deadline** vs Long-term **maintainability**

Humans negotiate these trade-offs implicitly. LLMs need explicit structure.

## Setup

```python
from belief_agent import negotiate_goals, score_tradeoffs, propose_compromise
```

## 1. Define Conflicting Goals

```python
goals = [
    {"goal": "Minimize infrastructure cost",     "importance": 9, "stakeholders": "Finance"},
    {"goal": "Maximize system reliability",       "importance": 10, "stakeholders": "Engineering"},
    {"goal": "Ship the MVP by end of quarter",    "importance": 7, "stakeholders": "Product"},
]
```

## 2. Run Negotiation

```python
result = negotiate_goals(goals, client)  # client is any LLMClient
```

## 3. Inspect the Result

```python
print("=== Conflicts ===")
for c in result["conflicts"]:
    print(f"{c['goal_a']} <-> {c['goal_b']}")
    print(f"  Tension: {c['tension']}")
    print(f"  Severity: {c['severity']:.0%}")

print("\n=== Ranked Goals ===")
for rg in result["ranked_goals"]:
    print(f"#{rg['priority']}: {rg['goal']} (feasibility: {rg['feasibility']:.0%})")
    print(f"  {rg['notes']}")

print(f"\n=== Compromise ===")
print(result["compromise"])

print(f"\n=== Recommended Beliefs ===")
for b in result["recommended_beliefs"]:
    print(f"  - {b}")
```

## 4. Integrate with BeliefState

```python
from belief_agent import BeliefState

state = BeliefState()
for belief_text in result["recommended_beliefs"]:
    state.add_belief(belief_text, confidence=0.8, source="negotiation", tags=["goal"])
```

## Sample Output

```
=== Conflicts ===
Minimize infrastructure cost <-> Maximize system reliability
  Tension: Reliability requires redundant systems which increase cost.
  Severity: 85%

=== Ranked Goals ===
#1: Maximize system reliability (feasibility: 70%)
  Non-negotiable for production
#2: Minimize infrastructure cost (feasibility: 50%)
  Use spot instances for non-critical workloads
#3: Ship by end of quarter (feasibility: 40%)
  Reduce scope to core features

=== Compromise ===
Use spot instances for non-critical workloads to reduce cost. Keep redundant
systems for critical paths to maintain reliability. Ship a reduced-scope MVP
on time. This satisfies the spirit of all three goals.

=== Recommended Beliefs ===
  - System reliability is the top priority
  - Cost savings can come from non-critical workloads
  - Scope reduction enables on-time shipping
```

## API Reference

| Function | Returns | Description |
|---|---|---|
| `negotiate_goals(goals, client)` | `dict` | Full analysis: conflicts, ranking, compromise, beliefs |
| `score_tradeoffs(goals, client)` | `dict` | Alias for `negotiate_goals` |
| `propose_compromise(goals, client)` | `str` | Just the compromise text |

## When to Use This

- **Product management**: Resolving stakeholder conflicts
- **Multi-agent systems**: Agents with different objectives need to find common ground
- **Safety-critical systems**: Balancing capability with constraints
- **Resource allocation**: Budget, time, and quality trade-offs

## Summary

`negotiate_goals()` provides structured, explainable trade-off analysis for any set of conflicting objectives. Combined with `BeliefState`, you can persist the resulting priorities and use them to guide future agent behavior.
