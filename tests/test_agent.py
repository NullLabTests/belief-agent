"""Tests for the Agent wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from belief_agent import BeliefAgent, BeliefState, LLMClient


class EchoClient(LLMClient):
    """Returns a canned response for testing."""

    def __init__(self, resp: str = "Hello") -> None:
        self.resp = resp
        self.last_messages: list[dict] = []

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        self.last_messages = messages
        return self.resp

    def complete_stream(self, messages: list[dict], **kwargs: Any) -> Any:
        raise NotImplementedError


class TestBeliefAgent:
    def test_init(self):
        agent = BeliefAgent(client=EchoClient())
        assert isinstance(agent.state, BeliefState)
        assert len(agent.state) == 0

    def test_chat_adds_to_history(self):
        agent = BeliefAgent(client=EchoClient("Hello!"))
        resp = agent.chat("Hi")
        assert resp == "Hello!"
        assert len(agent.message_history) == 3  # system, user, assistant

    def test_chat_injects_system_prompt(self):
        client = EchoClient("ok")
        agent = BeliefAgent(client=client, state=BeliefState())
        agent.state.add_belief("Test belief")
        agent.chat("Hello")
        system_msg = client.last_messages[0]
        assert system_msg["role"] == "system"
        assert "Test belief" in system_msg["content"]

    def test_complete_stateless(self):
        client = EchoClient("response")
        agent = BeliefAgent(client=client)
        resp = agent.complete("prompt")
        assert resp == "response"
        # stateless — no history kept
        assert len(agent.message_history) == 0

    def test_reset_conversation(self):
        agent = BeliefAgent(client=EchoClient())
        agent.state.add_belief("Keep me")
        agent.chat("test")
        agent.reset_conversation(keep_beliefs=True)
        assert len(agent.message_history) == 0
        assert len(agent.state) == 1

    def test_reset_conversation_clear_beliefs(self):
        agent = BeliefAgent(client=EchoClient())
        agent.state.add_belief("Lost")
        agent.reset_conversation(keep_beliefs=False)
        assert len(agent.state) == 0

    def test_system_prompt_style_json(self):
        client = EchoClient("ok")
        agent = BeliefAgent(client=client, system_prompt_style="json")
        agent.state.add_belief("Fact")
        agent.chat("Hi")
        content = client.last_messages[0]["content"]
        assert '"fact"' in content

    def test_auto_reflect_disabled_by_default(self):
        client = EchoClient("ok")
        agent = BeliefAgent(client=client)
        assert agent.auto_reflect is False

    def test_message_history_returns_copy(self):
        agent = BeliefAgent(client=EchoClient())
        agent.chat("hi")
        hist = agent.message_history
        hist.append("x")  # mutating the copy doesn't affect the agent
        assert len(agent.message_history) == 3  # unchanged


class TestWithoutLitellm:
    def test_no_client_raises(self):
        import belief_agent.agent as agent_mod

        agent_mod.HAS_LITELLM = False
        with pytest.raises(ValueError):
            BeliefAgent(client=None)
