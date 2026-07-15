"""Multi-objective negotiation — handle conflicting goals with structured
trade-off analysis."""

from __future__ import annotations

from typing import Any

NEGOTIATION_PROMPT = """You are a neutral negotiation and trade-off analysis system. The user has provided multiple \
goals that may conflict with each other.

=== Goals ===
{goals_text}

--- Task ---
Analyze the goals above. For each pair of goals that conflict, explain the tension. \
Then produce a recommendation that:
1. Ranks the goals by priority (1 = highest).
2. Scores each goal's feasibility (0–1).
3. Proposes a concrete compromise that keeps all goals partially satisfied.
4. Lists any new beliefs the agent should adopt based on this analysis.

Return a JSON object with this exact structure — and nothing else:
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


def negotiate_goals(
    goals: list[dict],
    client: Any,
) -> dict[str, Any]:
    """Analyze a list of potentially conflicting goals and produce a
    structured trade-off analysis.

    Parameters
    ----------
    goals:
        Each dict should have at least a ``"goal"`` key (the description)
        and may have optional keys such as ``"importance"`` (1-10),
        ``"constraints"``, ``"stakeholders"``.
    client:
        An LLM client with a ``.complete(messages)`` method.

    Returns
    -------
    A dictionary with keys ``conflicts``, ``ranked_goals``, ``compromise``,
    and ``recommended_beliefs``.

    Examples
    --------
    >>> from unittest.mock import MagicMock
    >>> import json
    >>> mock = MagicMock()
    >>> mock.complete.return_value = json.dumps({
    ...     "conflicts": [],
    ...     "ranked_goals": [],
    ...     "compromise": "No conflict detected.",
    ...     "recommended_beliefs": []
    ... })
    >>> result = negotiate_goals([
    ...     {"goal": "Minimize cost", "importance": 9},
    ...     {"goal": "Maximize safety", "importance": 10},
    ... ], mock)
    >>> result["compromise"]
    'No conflict detected.'
    """
    goals_text = _format_goals(goals)
    prompt = NEGOTIATION_PROMPT.format(goals_text=goals_text)
    response = client.complete([{"role": "user", "content": prompt}])
    return _parse_negotiation_response(response)


def score_tradeoffs(
    goals: list[dict],
    client: Any,
) -> dict[str, Any]:
    """Alias / convenience wrapper around :func:`negotiate_goals`.

    Returns the same structured result.
    """
    return negotiate_goals(goals, client)


def propose_compromise(
    goals: list[dict],
    client: Any,
) -> str:
    """Return only the compromise text from a negotiation analysis.

    Parameters
    ----------
    goals:
        Conflicting goals.
    client:
        An LLM client.

    Returns
    -------
    A plain-text compromise recommendation.
    """
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
    import json

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
