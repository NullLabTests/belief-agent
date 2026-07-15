import json

from belief_agent.agent import BeliefAgent
from belief_agent.belief_state import BeliefState
from belief_agent.negotiation import (
    Goal,
    NegotiationProposal,
    find_tradeoffs,
    negotiate,
    negotiate_goals,
    rank_goals,
    suggest_compromise,
)


def _llm(messages):
    prompt = messages[-1]["content"]
    if "Rank the following goals" in prompt:
        return json.dumps([{"index": 0, "rank": 1}, {"index": 1, "rank": 2}])
    return json.dumps({"goal_index": 0, "action": "do X", "satisfaction": 0.7, "tradeoffs": ["cost"]})


def _llm_high(messages):
    prompt = messages[-1]["content"]
    if "Rank the following goals" in prompt:
        return json.dumps([{"index": 0, "rank": 1}])
    return json.dumps({"goal_index": 0, "action": "do Y", "satisfaction": 0.9, "tradeoffs": []})


def _llm_tradeoffs(messages):
    return json.dumps({"tradeoffs": ["split resources", "alternate days"]})


def _llm_compromise(messages):
    return json.dumps({"compromise": "hybrid approach"})


class MockClient:
    def __init__(self, return_value=None):
        self.return_value = return_value or json.dumps({})

    def complete(self, messages, **kwargs):
        return self.return_value


class TestNegotiation:
    def test_goal_defaults(self):
        g = Goal(description="test")
        assert g.priority == 1.0
        assert g.constraints == []
        assert g.tradeoffs == []

    def test_rank_goals_by_priority(self):
        goals = [
            Goal("low", priority=0.3),
            Goal("high", priority=0.9),
            Goal("mid", priority=0.6),
        ]
        ranked = rank_goals(goals)
        assert [g.description for g in ranked] == ["high", "mid", "low"]

    def test_rank_goals_with_llm(self):
        goals = [Goal("a"), Goal("b")]
        ranked = rank_goals(goals, _llm)
        assert len(ranked) == 2

    def test_negotiate_basic(self):
        agent_a = BeliefAgent(llm_call=_llm, name="A")
        agent_b = BeliefAgent(llm_call=_llm, name="B")
        goals = [Goal("speed", priority=0.9), Goal("cost", priority=0.5)]
        result = negotiate([agent_a, agent_b], "build vs buy", goals, _llm)
        assert result.issue == "build vs buy"
        assert result.consensus_action is not None
        assert len(result.proposals) > 0

    def test_negotiate_early_consensus(self):
        agent_a = BeliefAgent(llm_call=_llm_high, name="A")
        agent_b = BeliefAgent(llm_call=_llm_high, name="B")
        goals = [Goal("speed", priority=0.9)]
        result = negotiate([agent_a, agent_b], "pick tool", goals, _llm_high)
        assert result.consensus_satisfaction >= 0.8

    def test_negotiate_with_beliefs(self):
        agent = BeliefAgent(llm_call=_llm, name="A")
        agent.adopt("tool X is fast", confidence=0.9)
        goals = [Goal("performance")]
        result = negotiate([agent], "tool choice", goals, _llm)
        assert result.consensus_action is not None

    def test_find_tradeoffs(self):
        agent = BeliefAgent(llm_call=_llm_tradeoffs, name="A")
        ga = Goal("speed")
        gb = Goal("cost")
        tradeoffs = find_tradeoffs(agent, ga, gb, _llm_tradeoffs)
        assert len(tradeoffs) > 0

    def test_suggest_compromise(self):
        proposals = [
            NegotiationProposal(Goal("speed"), "optimize", 0.6, []),
            NegotiationProposal(Goal("cost"), "minimize", 0.7, []),
        ]
        compromise = suggest_compromise(proposals, _llm_compromise)
        assert compromise == "hybrid approach"

    def test_negotiate_goals_v1_compat(self):
        client = MockClient(return_value=json.dumps({
            "conflicts": [],
            "ranked_goals": [{"goal": "speed", "priority": 1, "feasibility": 0.8, "notes": ""}],
            "compromise": "optimize for speed",
            "recommended_beliefs": ["speed matters most"],
        }))
        result = negotiate_goals([{"goal": "speed", "importance": 9}], client)
        assert result["compromise"] == "optimize for speed"
