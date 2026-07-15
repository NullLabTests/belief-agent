"""Tests for the Reflection module."""

from __future__ import annotations

import json
from typing import Any

from belief_agent import BeliefState, LLMClient, reflect


class ReflectionMockClient(LLMClient):
    """Returns a canned reflection analysis."""

    def __init__(self, response: str | None = None) -> None:
        self.response = response

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        if self.response is not None:
            return self.response
        return json.dumps(
            [
                {
                    "type": "contradiction",
                    "fact": "The sky is blue",
                    "reason": "User said the sky is not blue",
                    "confidence": 0.0,
                },
                {
                    "type": "new_belief",
                    "fact": "The user is testing contradiction detection",
                    "reason": "Inferred from conversation",
                    "confidence": 0.6,
                },
            ]
        )

    def complete_stream(self, messages: list[dict], **kwargs: Any) -> Any:
        raise NotImplementedError


class TestReflection:
    def test_reflect_contradiction(self):
        state = BeliefState()
        state.add_belief("The sky is blue", confidence=0.9, source="user")
        client = ReflectionMockClient()
        updates = reflect(state, "The sky is not blue!", "You're right, my mistake.", client, mode="auto")
        assert len(updates) >= 1
        # The contradiction proposal should reduce confidence
        belief = state.query("The sky is blue")[0]
        assert belief.confidence < 0.9

    def test_reflect_new_belief(self):
        state = BeliefState()
        client = ReflectionMockClient()
        reflect(state, "I like Python", "Great choice!", client, mode="auto")
        new_beliefs = state.query("testing contradiction detection")
        assert len(new_beliefs) == 1
        assert new_beliefs[0].confidence == 0.6
        assert new_beliefs[0].source == "reflection"

    def test_reflect_human_mode_skip(self):
        state = BeliefState()
        state.add_belief("Fact A", confidence=0.9)
        client = ReflectionMockClient()

        # callback returns False → skip
        def reject(_proposals: list[dict]) -> bool:
            return False

        reflect(state, "test", "response", client, mode="human", callback=reject)
        assert state.query("testing contradiction detection") == []

    def test_reflect_human_mode_apply(self):
        state = BeliefState()
        client = ReflectionMockClient()

        def accept(_proposals: list[dict]) -> bool:
            return True

        reflect(state, "test", "response", client, mode="human", callback=accept)
        assert len(state.query("testing contradiction detection")) == 1

    def test_reflect_no_updates(self):
        state = BeliefState()
        state.add_belief("Fact", confidence=0.5)
        client = ReflectionMockClient(response=json.dumps([]))
        updates = reflect(state, "Hello", "Hi", client, mode="auto")
        assert updates == []
        # nothing changed
        assert state.query("Fact")[0].confidence == 0.5

    def test_reflect_malformed_json(self):
        state = BeliefState()
        client = ReflectionMockClient(response="not json at all")
        updates = reflect(state, "Hello", "Hi", client, mode="auto")
        assert updates == []

    def test_reflect_with_markdown_fence(self):
        state = BeliefState()
        client = ReflectionMockClient(
            response="""```json
[{"type": "new_belief", "fact": "Inferred from fence", "reason": "test", "confidence": 0.8}]
```"""
        )
        reflect(state, "test", "resp", client, mode="auto")
        assert state.query("Inferred from fence")[0].confidence == 0.8
