import json

from belief_agent.agent import BeliefAgent
from belief_agent.belief_state import BeliefState
from belief_agent.reflection import (
    ReflectionResult,
    auto_update,
    human_review,
    reflect,
    reflect_on_beliefs,
)


def _llm_accept(messages):
    return json.dumps({
        "critique": "Looks solid",
        "new_evidence": ["more data"],
        "new_contradictions": [],
        "updated_confidence": 0.9,
    })


def _llm_garbage(messages):
    return "not json"


class MockClient:
    def __init__(self, return_value=None):
        self.return_value = return_value or json.dumps([])

    def complete(self, messages, **kwargs):
        return self.return_value


class TestReflect:
    def test_reflect_empty_state(self):
        state = BeliefState()
        client = MockClient()
        result = reflect(state, "hello", "world", client)
        assert result == []

    def test_reflect_with_proposals(self):
        state = BeliefState()
        state.add_belief("p", confidence=0.5)
        client = MockClient(return_value=json.dumps([
            {"type": "confidence_update", "fact": "p", "confidence": 0.9, "reason": "updated"}
        ]))
        result = reflect(state, "hello", "world", client)
        assert len(result) == 1
        assert state.query("p")[0].confidence == 0.9

    def test_reflect_human_mode(self):
        state = BeliefState()
        state.add_belief("p", confidence=0.5)
        client = MockClient(return_value=json.dumps([
            {"type": "confidence_update", "fact": "p", "confidence": 0.9, "reason": "updated"}
        ]))
        callback_called = False
        def callback(proposals):
            nonlocal callback_called
            callback_called = True
            return False
        result = reflect(state, "hello", "world", client, mode="human", callback=callback)
        assert callback_called
        assert state.query("p")[0].confidence == 0.5  # not applied


class TestReflectOnBeliefs:
    def test_reflect_on_beliefs_updates_confidence(self):
        state = BeliefState()
        state.add_belief("p", confidence=0.5, evidence=["e"])
        results = reflect_on_beliefs(state, _llm_accept)
        assert len(results) == 1
        assert results[0].proposition == "p"
        assert results[0].original_confidence == 0.5
        assert state.query("p")[0].confidence == 0.9
        assert "more data" in state.query("p")[0].evidence

    def test_reflect_on_multiple_beliefs(self):
        state = BeliefState()
        state.add_belief("a", confidence=0.5)
        state.add_belief("b", confidence=0.5)
        results = reflect_on_beliefs(state, _llm_accept)
        assert len(results) == 2

    def test_reflect_depth(self):
        state = BeliefState()
        state.add_belief("p", confidence=0.5)
        results = reflect_on_beliefs(state, _llm_accept, depth=2)
        assert len(results) == 2

    def test_reflect_empty_state(self):
        state = BeliefState()
        results = reflect_on_beliefs(state, _llm_accept)
        assert results == []

    def test_reflect_depth_zero(self):
        state = BeliefState()
        state.add_belief("p")
        results = reflect_on_beliefs(state, _llm_accept, depth=0)
        assert results == []

    def test_garbage_llm_output(self):
        state = BeliefState()
        state.add_belief("p", confidence=0.5)
        results = reflect_on_beliefs(state, _llm_garbage)
        assert len(results) == 1
        assert results[0].accepted is False
        assert results[0].updated_confidence == 0.5

    def test_auto_update_low_confidence(self):
        state = BeliefState()
        state.add_belief("high", confidence=0.9)
        state.add_belief("low", confidence=0.2)
        results = auto_update(state, _llm_accept, threshold=0.3)
        assert len(results) == 1
        assert results[0].proposition == "low"

    def test_auto_update_no_low_confidence(self):
        state = BeliefState()
        state.add_belief("p", confidence=0.9)
        results = auto_update(state, _llm_accept, threshold=0.3)
        assert results == []

    def test_human_review(self):
        result = ReflectionResult(
            proposition="p",
            original_confidence=0.5,
            updated_confidence=0.8,
            critique="ok",
        )
        result = human_review(result, "Actually wrong", accepted=False)
        assert "Actually wrong" in result.critique
        assert result.accepted is False
