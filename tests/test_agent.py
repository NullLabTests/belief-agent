import json

import pytest

from belief_agent.agent import BeliefAgent
from belief_agent.belief_state import BeliefState


def dummy_llm(messages):
    return json.dumps({"proposition": "test belief", "confidence": 0.8})


def _make_llm(response_data):
    def llm(messages):
        return json.dumps(response_data)
    return llm


class TestBeliefAgent:
    def test_init(self):
        agent = BeliefAgent(llm_call=dummy_llm, name="test-agent")
        assert agent.name == "test-agent"
        assert len(agent.state) == 0

    def test_adopt_and_get_belief(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("p", confidence=0.5)
        b = agent.get_belief("p")
        assert b is not None
        assert b.confidence == 0.5

    def test_adopt_merge(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("p", confidence=0.6, evidence=["e1"])
        agent.adopt("p", confidence=0.4, evidence=["e2"])
        b = agent.get_belief("p")
        assert b.confidence == 0.5
        assert len(b.evidence) == 2

    def test_update_belief(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("p", confidence=0.5)
        agent.update_belief("p", confidence=0.9)
        assert agent.get_belief("p").confidence == 0.9

    def test_update_belief_nonexistent(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        assert agent.update_belief("nope") is False

    def test_remove_belief(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("p")
        assert agent.remove_belief("p") is True
        assert agent.get_belief("p") is None
        assert agent.remove_belief("p") is False

    def test_list_beliefs_filter(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("a", confidence=0.9)
        agent.adopt("b", confidence=0.3)
        agent.adopt("c", confidence=0.6)
        c = agent.get_belief("c")
        if c:
            c.contradictions.append("x")
        assert len(agent.list_beliefs(min_confidence=0.5)) == 2
        assert len(agent.list_beliefs(only_contradicted=True)) == 1

    def test_chat(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        reply = agent.chat("hello")
        assert reply is not None

    def test_call(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        reply = agent("hello")
        assert reply is not None

    def test_extracts_belief_from_reply(self):
        agent = BeliefAgent(llm_call=_make_llm({"fact": "extracted belief", "confidence": 0.7}))
        agent.chat("hello")
        assert agent.get_belief("extracted belief") is not None

    def test_serialization(self):
        agent = BeliefAgent(llm_call=dummy_llm, name="ser")
        agent.adopt("p", confidence=0.7)
        d = agent.to_dict()
        assert d["name"] == "ser"
        assert len(d["state"]["beliefs"]) == 1

    def test_to_json(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("p")
        raw = agent.to_json()
        d = json.loads(raw)
        assert "name" in d

    def test_reset_history(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.chat("hi")
        assert len(agent.message_history) > 0
        agent.reset_history()
        assert len(agent.message_history) == 0

    def test_reset_conversation(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("p")
        agent.chat("hi")
        agent.reset_conversation(keep_beliefs=True)
        assert agent.get_belief("p") is not None
        agent.reset_conversation(keep_beliefs=False)
        assert agent.get_belief("p") is None

    def test_reflect_method(self):
        agent = BeliefAgent(llm_call=dummy_llm)
        agent.adopt("test belief", confidence=0.5)
        result = agent.reflect()
        assert result is not None
