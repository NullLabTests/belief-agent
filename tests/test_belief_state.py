import json

import pytest

from belief_agent.belief_state import Belief, BeliefState


class TestBelief:
    def test_defaults(self):
        b = Belief(fact="The sky is blue")
        assert b.fact == "The sky is blue"
        assert b.confidence == 0.5
        assert b.source == "user"
        assert len(b.id) == 12

    def test_confidence_clamped(self):
        b = Belief(fact="p", confidence=1.5)
        assert b.confidence == 1.0
        b = Belief(fact="p", confidence=-0.5)
        assert b.confidence == 0.0

    def test_eq_by_id(self):
        a = Belief(fact="p")
        b = Belief(fact="p")
        assert a != b  # different ids

    def test_str_repr(self):
        b = Belief(fact="hello", confidence=0.9)
        assert "90%" in str(b)
        assert "hello" in repr(b)


class TestBeliefState:
    def test_empty(self):
        bs = BeliefState()
        assert len(bs) == 0
        assert "empty" in str(bs)

    def test_add_belief(self):
        bs = BeliefState()
        b = bs.add_belief("The sky is blue", confidence=0.9, source="observation")
        assert len(bs) == 1
        assert b.fact == "The sky is blue"
        assert bs.version == 2

    def test_update_belief(self):
        bs = BeliefState()
        b = bs.add_belief("p", confidence=0.5)
        bs.update_belief(b.id, confidence=0.9)
        assert bs.get(b.id).confidence == 0.9

    def test_update_belief_nonexistent(self):
        bs = BeliefState()
        assert bs.update_belief("nope", confidence=0.9) is None

    def test_remove_belief(self):
        bs = BeliefState()
        b = bs.add_belief("p")
        assert bs.remove_belief(b.id) is True
        assert len(bs) == 0
        assert bs.remove_belief(b.id) is False

    def test_query(self):
        bs = BeliefState()
        bs.add_belief("Python is great", confidence=0.9)
        bs.add_belief("Java is also fine", confidence=0.5)
        assert len(bs.query("Python")) == 1
        assert len(bs.query("java")) == 1
        assert len(bs.query("nonexistent")) == 0

    def test_query_by_tag(self):
        bs = BeliefState()
        bs.add_belief("p", tags=["math"])
        bs.add_belief("q", tags=["science"])
        assert len(bs.query_by_tag("math")) == 1

    def test_query_by_source(self):
        bs = BeliefState()
        bs.add_belief("p", source="user")
        bs.add_belief("q", source="model")
        assert len(bs.query_by_source("user")) == 1

    def test_high_confidence(self):
        bs = BeliefState()
        bs.add_belief("a", confidence=0.9)
        bs.add_belief("b", confidence=0.3)
        assert len(bs.high_confidence(threshold=0.8)) == 1

    def test_low_confidence(self):
        bs = BeliefState()
        bs.add_belief("a", confidence=0.9)
        bs.add_belief("b", confidence=0.2)
        assert len(bs.low_confidence(threshold=0.3)) == 1

    def test_list_beliefs(self):
        bs = BeliefState()
        bs.add_belief("a", confidence=0.9)
        bs.add_belief("b", confidence=0.3)
        bs.add_belief("c", confidence=0.6, tags=["x"])
        c = bs.query("c")[0]
        c.contradictions.append("some-id")
        assert len(bs.list_beliefs(min_confidence=0.5)) == 2
        assert len(bs.list_beliefs(only_contradicted=True)) == 1

    def test_is_confident(self):
        bs = BeliefState()
        bs.add_belief("p", confidence=0.8)
        assert bs.is_confident("p") is True
        assert bs.is_confident("nonexistent") is False

    def test_is_contradicted(self):
        bs = BeliefState()
        b = bs.add_belief("p", confidence=0.8)
        b.contradictions.append("other-id")
        assert bs.is_contradicted("p") is True
        assert bs.is_contradicted("nonexistent") is False

    def test_support(self):
        bs = BeliefState()
        bs.add_belief("p", confidence=0.5)
        bs.support("p", "new evidence")
        assert bs.query("p")[0].confidence == 0.6
        assert "new evidence" in bs.query("p")[0].evidence

    def test_contradict(self):
        bs = BeliefState()
        bs.add_belief("p", confidence=0.8)
        bs.contradict("p", "counterpoint")
        assert bs.query("p")[0].confidence == 0.4

    def test_contradiction_detection(self):
        bs = BeliefState()
        bs.add_belief("The sky is blue", confidence=0.9, detect_contradictions=False)
        bs.add_belief("The sky is not blue", confidence=0.5)
        pairs = bs.get_contradictions()
        assert len(pairs) == 1

    def test_merge(self):
        bs1 = BeliefState()
        bs1.add_belief("p", confidence=0.8, evidence=["e1"])
        bs2 = BeliefState()
        bs2.add_belief("p", confidence=0.4, evidence=["e2"])
        bs1.merge(bs2)
        b = bs1.query("p")[0]
        assert b.confidence == pytest.approx(0.6)
        assert len(b.evidence) == 2

    def test_serialize_roundtrip(self):
        bs = BeliefState()
        bs.add_belief("p", confidence=0.7, evidence=["e1"], source="test")
        raw = bs.serialize(indent=2)
        restored = BeliefState.deserialize(raw)
        assert len(restored) == 1
        assert restored.query("p")[0].confidence == 0.7
        assert restored.query("p")[0].evidence == ["e1"]

    def test_to_json_roundtrip(self):
        bs = BeliefState()
        bs.add_belief("p", confidence=0.5)
        raw = bs.to_json()
        restored = BeliefState.from_json(raw)
        assert len(restored) == 1

    def test_to_dict_roundtrip(self):
        bs = BeliefState()
        bs.add_belief("p", confidence=0.7)
        d = bs.to_dict()
        restored = BeliefState.from_dict(d)
        assert len(restored) == 1

    def test_iter_and_getitem(self):
        bs = BeliefState()
        bs.add_belief("a")
        bs.add_belief("b")
        assert len(list(iter(bs))) == 2
        assert bs[0].fact == "a"

    def test_get_all(self):
        bs = BeliefState()
        bs.add_belief("a")
        assert len(bs.get_all()) == 1
