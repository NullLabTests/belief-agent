"""belief-agent: explicit belief states, structured reflection, and human
negotiation for LLM agents.

Exports
-------
Belief, BeliefState
    Core data model for storing structured beliefs.
BeliefAgent, LLMClient, LiteLLMClient
    Agent wrapper that injects beliefs into LLM conversations.
reflect
    Analyze LLM output against current beliefs and propose updates.
negotiate_goals, score_tradeoffs, propose_compromise
    Multi-objective negotiation helpers.
build_system_prompt, format_beliefs_bullet_points, format_beliefs_json,
format_beliefs_natural
    Utilities for formatting beliefs into prompts.
"""

from __future__ import annotations

from .agent import BeliefAgent, LLMClient, LiteLLMClient
from .belief_state import Belief, BeliefState
from .negotiation import negotiate_goals, propose_compromise, score_tradeoffs
from .reflection import reflect
from .utils import (
    build_system_prompt,
    format_beliefs_bullet_points,
    format_beliefs_json,
    format_beliefs_natural,
)

__all__ = [
    "Belief",
    "BeliefState",
    "BeliefAgent",
    "LLMClient",
    "LiteLLMClient",
    "reflect",
    "negotiate_goals",
    "score_tradeoffs",
    "propose_compromise",
    "build_system_prompt",
    "format_beliefs_bullet_points",
    "format_beliefs_json",
    "format_beliefs_natural",
]

__version__ = "0.1.0"
