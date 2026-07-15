"""Multi-agent negotiation example."""

import json

from belief_agent import BeliefAgent, BeliefState
from belief_agent.negotiation import Goal, negotiate, rank_goals


def _rank_response(goals):
    return json.dumps([{"index": i, "rank": i + 1} for i in range(len(goals))])


def engineer_llm(messages):
    prompt = messages[-1]["content"]
    if "Rank the following goals" in prompt:
        return _rank_response([g for g in prompt.split("\n") if '"' in g])
    return json.dumps({
        "goal_index": 0,
        "action": "Use a microservices architecture",
        "satisfaction": 0.6,
        "tradeoffs": ["higher operational cost", "more complexity"],
    })


def pm_llm(messages):
    prompt = messages[-1]["content"]
    if "Rank the following goals" in prompt:
        return _rank_response([g for g in prompt.split("\n") if '"' in g])
    return json.dumps({
        "goal_index": 0,
        "action": "Ship a monolith fast, then refactor",
        "satisfaction": 0.9,
        "tradeoffs": ["technical debt"],
    })


engineer = BeliefAgent(llm_call=engineer_llm, name="Engineer")
pm = BeliefAgent(llm_call=pm_llm, name="ProductManager")

engineer.adopt(BeliefState("Microservices scale better", confidence=0.9))
pm.adopt(BeliefState("Speed to market is critical", confidence=0.95))

goals = [
    Goal("Scalability", priority=0.8),
    Goal("Speed to market", priority=0.9),
    Goal("Low operational cost", priority=0.5),
]

print("Ranked goals:", [g.description for g in rank_goals(goals)])

print("\n--- Negotiation ---")
result = negotiate([engineer, pm], "Architecture decision", goals, engineer_llm)

print(f"Consensus action: {result.consensus_action}")
print(f"Satisfaction: {result.consensus_satisfaction:.2f}")
print(f"Unresolved goals: {result.unresolved_goals}")
