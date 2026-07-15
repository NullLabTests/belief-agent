"""Tests for the BeliefState class."""

from __future__ import annotations

import json

import pytest

from belief_agent import Belief, BeliefState


class TestBelief:
    def test_create(self):
        b = Belief(fact="The sky is blue", confidence=0.9, source="observation")
        assert b.fact == "The sky is blue"
        assert b.confidence == 0.9
        assert b.source == "observation"
        assert len(b.id) == 12

    def test_confidence_clamped(self):
        b = Belief(fact="test", confidence=99.0)
        assert b.confidence == 1.0
        b2 = Belief(fact="test", confidence=-5)
        assert b2.confidence == 0.0

    def test_eq_by_id(self):
        b1 = Belief(fact="A", id="same")
        b2 = Belief(fact="B", id="same")
        assert b1 == b2

    def test_hash(self):
        b = Belief(fact="test")
        assert hash(b) == hash(b.id)


class TestBeliefState:
    def test_empty(self):
        bs = BeliefState()
        assert len(bs) == 0
        assert str(bs) == "BeliefState(empty)"

    def test_add_belief(self):
        bs = BeliefState()
        b = bs.add_belief("Hello world", confidence=0.8, source="test")
        assert len(bs) == 1
        assert b.fact == "Hello world"
        assert bs.version > 1

    def test_query(self):
        bs = BeliefState()
        bs.add_belief("Python is great")
        bs.add_belief("I like Java")
        results = bs.query("python")
        assert len(results) == 1
        assert "Python" in results[0].fact

    def test_query_case_sensitive(self):
        bs = BeliefState()
        bs.add_belief("Python")
        bs.add_belief("python")
        assert len(bs.query("python")) == 2
        assert len(bs.query("python", case_sensitive=True)) == 1

    def test_query_by_tag(self):
        bs = BeliefState()
        bs.add_belief("A", tags=["x"])
        bs.add_belief("B", tags=["y"])
        assert len(bs.query_by_tag("x")) == 1
        assert len(bs.query_by_tag("z")) == 0

    def test_query_by_source(self):
        bs = BeliefState()
        bs.add_belief("A", source="user")
        bs.add_belief("B", source="model")
        assert len(bs.query_by_source("user")) == 1

    def test_update_belief(self):
        bs = BeliefState()
        b = bs.add_belief("Old fact", confidence=0.5)
        updated = bs.update_belief(b.id, fact="New fact", confidence=0.9)
        assert updated is not None
        assert updated.fact == "New fact"
        assert updated.confidence == 0.9

    def test_update_nonexistent(self):
        bs = BeliefState()
        assert bs.update_belief("nonexistent", fact="x") is None

    def test_remove_belief(self):
        bs = BeliefState()
        b = bs.add_belief("Remove me")
        assert bs.remove_belief(b.id) is True
        assert len(bs) == 0

    def test_remove_nonexistent(self):
        bs = BeliefState()
        assert bs.remove_belief("nope") is False

    def test_get(self):
        bs = BeliefState()
        b = bs.add_belief("Get me")
        assert bs.get(b.id) is b
        assert bs.get("nope") is None

    def test_high_low_confidence(self):
        bs = BeliefState()
        bs.add_belief("Sure", confidence=0.9)
        bs.add_belief("Unsure", confidence=0.2)
        bs.add_belief("Mid", confidence=0.5)
        assert len(bs.high_confidence(0.8)) == 1
        assert len(bs.low_confidence(0.3)) == 1

    def test_contradiction_detection(self):
        bs = BeliefState()
        bs.add_belief("The sky is blue")
        bs.add_belief("The sky is not blue")
        contradictions = bs.get_contradictions()
        assert len(contradictions) == 1
        a, b = contradictions[0]
        assert "blue" in a.fact and "blue" in b.fact
        assert a.id in b.contradictions
        assert b.id in a.contradictions

    def test_no_false_positive_contradiction(self):
        bs = BeliefState()
        bs.add_belief("Python is great")
        bs.add_belief("JavaScript is also great")
        assert len(bs.get_contradictions()) == 0

    def test_merge(self):
        bs1 = BeliefState()
        bs1.add_belief("Fact A", confidence=0.5, source="user")
        bs2 = BeliefState()
        bs2.add_belief("Fact B", confidence=0.9, source="user")
        bs2.add_belief("Fact A", confidence=0.8, source="user")  # higher confidence
        added = bs1.merge(bs2)
        assert added == 1  # only Fact B is new
        assert bs1.query("Fact A")[0].confidence == 0.8  # updated

    def test_serialize_deserialize(self):
        bs = BeliefState()
        bs.add_belief("Keep me", confidence=1.0)
        json_str = bs.serialize(indent=2)
        restored = BeliefState.deserialize(json_str)
        assert len(restored) == 1
        assert restored[0].fact == "Keep me"

    def test_to_dict_from_dict(self):
        bs = BeliefState()
        bs.add_belief("Test", tags=["a"])
        d = bs.to_dict()
        restored = BeliefState.from_dict(d)
        assert len(restored) == 1
        assert restored[0].tags == ["a"]

    def test_iter(self):
        bs = BeliefState()
        bs.add_belief("A")
        bs.add_belief("B")
        facts = [b.fact for b in bs]
        assert facts == ["A", "B"]

    def test_get_all_returns_copy(self):
        bs = BeliefState()
        bs.add_belief("A")
        all_b = bs.get_all()
        all_b.clear()
        assert len(bs) == 1

    def test_merge_same_belief_lower_confidence(self):
        bs1 = BeliefState()
        bs1.add_belief("Fact", confidence=0.9, source="user")
        bs2 = BeliefState()
        bs2.add_belief("Fact", confidence=0.3, source="user")
        bs1.merge(bs2)
        assert bs1.query("Fact")[0].confidence == 0.9  # not lowered


if __name__ == "__main__":
    pytest.main([__file__])
