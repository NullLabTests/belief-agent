from __future__ import annotations

from .agent import BeliefAgent, LLMClient, LiteLLMClient
from .belief_state import Belief, BeliefState
from .negotiation import (
    Goal,
    NegotiationProposal,
    NegotiationResult,
    find_tradeoffs,
    negotiate,
    negotiate_goals,
    propose_compromise,
    rank_goals,
    score_tradeoffs,
    suggest_compromise,
)
from .reflection import (
    ReflectionResult,
    auto_update,
    human_review,
    reflect,
    reflect_on_beliefs,
)
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
    "Goal",
    "NegotiationProposal",
    "NegotiationResult",
    "ReflectionResult",
    "reflect",
    "reflect_on_beliefs",
    "auto_update",
    "human_review",
    "negotiate",
    "negotiate_goals",
    "rank_goals",
    "find_tradeoffs",
    "suggest_compromise",
    "propose_compromise",
    "score_tradeoffs",
    "build_system_prompt",
    "format_beliefs_bullet_points",
    "format_beliefs_json",
    "format_beliefs_natural",
]

__version__ = "2.0.0"
