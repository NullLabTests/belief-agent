from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .belief_state import BeliefState

Message = dict[str, str]

NEGOTIATION_PROMPT = """You are a neutral negotiation and trade-off analysis system. The user has provided multiple \
goals that may conflict with each other.

=== Goals ===
{goals_text}

--- Task ---
Analyze the goals above. For each pair of goals that conflict, explain the tension. \
Then produce a recommendation that:
1. Ranks the goals by priority (1 = highest).
2. Scores each goal's feasibility (0-1).
3. Proposes a concrete compromise that keeps all goals partially satisfied.
4. Lists any new beliefs the agent should adopt based on this analysis.

Return a JSON object with this exact structure - and nothing else:
{{
  "conflicts": [
    {{"goal_a": <str>, "goal_b": <str>, "tension": <str>, "severity": <float 0-1>}}
  ],
  "ranked_goals": [
    {{"goal": <str>, "priority": <int>, "feasibility": <float 0-1>, "notes": <str>}}
  ],
  "compromise": <str>,
  "recommended_beliefs": [<str>]
}}
"""


@dataclass
class Goal:
    description: str
    priority: float = 1.0
    constraints: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)


@dataclass
class NegotiationProposal:
    goal: Goal
    proposed_action: str
    estimated_satisfaction: float
    tradeoffs: list[str] = field(default_factory=list)


@dataclass
class NegotiationResult:
    issue: str
    proposals: list[NegotiationProposal] = field(default_factory=list)
    consensus_action: str | None = None
    consensus_satisfaction: float = 0.0
    unresolved_goals: list[str] = field(default_factory=list)


def rank_goals(
    goals: list[Goal],
    llm_call: Callable[[list[Message]], str] | None = None,
) -> list[Goal]:
    if llm_call:
        prompt = (
            "Rank the following goals by importance (1 = most important).\n"
            + "\n".join(
                f'  {i}. "{g.description}" (priority={g.priority})'
                for i, g in enumerate(goals)
            )
            + '\n\nReturn as JSON array: [{"index":0,"rank":1}, ...]'
        )
        raw = llm_call([{"role": "user", "content": prompt}])
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise TypeError("expected JSON array")
            indices = {item["index"]: item["rank"] for item in data}
            return sorted(goals, key=lambda g: indices.get(goals.index(g), 999))
        except (json.JSONDecodeError, KeyError):
            pass
    return sorted(goals, key=lambda g: g.priority, reverse=True)


def negotiate(
    agents: list[Any],
    issue: str,
    goals: list[Goal],
    llm_call: Callable[[list[Message]], str],
    max_rounds: int = 3,
) -> NegotiationResult:
    ranked = rank_goals(goals, llm_call)
    result = NegotiationResult(issue=issue)

    for round_num in range(1, max_rounds + 1):
        proposals: list[NegotiationProposal] = []

        for agent in agents:
            prompt = (
                f"Negotiation round {round_num}/{max_rounds} on: {issue}\n\n"
                f"Your name: {agent.name}\n\n"
                "Goals (ranked by priority):\n"
                + "\n".join(
                    f'  {i+1}. "{g.description}" (priority={g.priority})'
                    for i, g in enumerate(ranked)
                )
                + "\n\n"
                + f"Your current beliefs: {agent.state}\n\n"
                "Propose an action that satisfies as many high-priority "
                "goals as possible, with tradeoffs noted.\n"
                "Return as JSON: "
                '{"goal_index":0,"action":"...","satisfaction":0.X,"tradeoffs":["..."]}'
            )
            raw = agent._call_llm([{"role": "user", "content": prompt}])
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            gi = data.get("goal_index", 0)
            goal = ranked[gi] if gi < len(ranked) else ranked[0]
            proposals.append(
                NegotiationProposal(
                    goal=goal,
                    proposed_action=data.get("action", ""),
                    estimated_satisfaction=data.get("satisfaction", 0.5),
                    tradeoffs=data.get("tradeoffs", []),
                )
            )

        if proposals:
            best = max(proposals, key=lambda p: p.estimated_satisfaction)
            result.proposals = proposals
            result.consensus_action = best.proposed_action
            result.consensus_satisfaction = best.estimated_satisfaction

        if result.consensus_satisfaction >= 0.8:
            break

    if result.consensus_action:
        resolved = {p.goal.description for p in result.proposals}
        for g in ranked:
            if g.description not in resolved:
                result.unresolved_goals.append(g.description)

    return result


def find_tradeoffs(
    agent: Any,
    goal_a: Goal,
    goal_b: Goal,
    llm_call: Callable[[list[Message]], str],
) -> list[str]:
    prompt = (
        f"Find tradeoffs between these two goals:\n\n"
        f'Goal A: "{goal_a.description}" (priority={goal_a.priority})\n'
        f'Goal B: "{goal_b.description}" (priority={goal_b.priority})\n\n'
        "Suggest specific tradeoffs that could partially satisfy both.\n"
        'Return as JSON: {"tradeoffs":["...","..."]}'
    )
    raw = llm_call([{"role": "user", "content": prompt}])
    try:
        data = json.loads(raw)
        return data.get("tradeoffs", [])
    except json.JSONDecodeError:
        return []


def suggest_compromise(
    proposals: list[NegotiationProposal],
    llm_call: Callable[[list[Message]], str],
) -> str | None:
    prompt = (
        "Given these negotiation proposals, suggest a compromise that "
        "combines elements from each:\n\n"
        + "\n".join(
            f'  - Goal: "{p.goal.description}", Action: "{p.proposed_action}", '
            f"Satisfaction: {p.estimated_satisfaction:.2f}, Tradeoffs: {p.tradeoffs}"
            for p in proposals
        )
        + '\n\nReturn as JSON: {"compromise":"..."}'
    )
    raw = llm_call([{"role": "user", "content": prompt}])
    try:
        data = json.loads(raw)
        return data.get("compromise")
    except json.JSONDecodeError:
        return None


# ---- v1-compatible wrappers ----

def negotiate_goals(
    goals: list[dict],
    client: Any,
) -> dict[str, Any]:
    goals_text = _format_goals(goals)
    prompt = NEGOTIATION_PROMPT.format(goals_text=goals_text)
    response = client.complete([{"role": "user", "content": prompt}])
    return _parse_negotiation_response(response)


def score_tradeoffs(
    goals: list[dict],
    client: Any,
) -> dict[str, Any]:
    return negotiate_goals(goals, client)


def propose_compromise(
    goals: list[dict],
    client: Any,
) -> str:
    result = negotiate_goals(goals, client)
    return result.get("compromise", "Unable to propose a compromise.")


def _format_goals(goals: list[dict]) -> str:
    lines = []
    for i, g in enumerate(goals, 1):
        imp = g.get("importance", "")
        stake = g.get("stakeholders", "")
        constraints = g.get("constraints", "")
        parts = [f"Goal {i}: {g.get('goal', '(unnamed)')}"]
        if imp:
            parts.append(f"  Importance: {imp}")
        if stake:
            parts.append(f"  Stakeholders: {stake}")
        if constraints:
            parts.append(f"  Constraints: {constraints}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _parse_negotiation_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "conflicts": [],
            "ranked_goals": [],
            "compromise": "Could not parse negotiation result.",
            "recommended_beliefs": [],
        }
