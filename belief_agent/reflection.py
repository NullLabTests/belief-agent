"""Reflection logic — detect contradictions and propose belief updates."""

from __future__ import annotations

import json
from typing import Any, Callable

from .belief_state import BeliefState


REFLECTION_PROMPT = """You are a careful belief-analysis system. Your job is to compare a user message and your \
response against a set of current beliefs, then propose updates.

=== Current Beliefs ===
{beliefs_text}

=== User Message ===
{user_message}

=== Your Response ===
{response}

--- Task ---
Analyze the conversation. For each issue you find, output one JSON object with these keys:
- "type": one of "contradiction", "new_belief", "confidence_update", "removal"
- "fact": the belief statement that should be added / updated / removed
- "reason": a short justification
- "confidence" (optional): new confidence level (0-1) if type is "new_belief" or "confidence_update"

Return a JSON list of these objects. If no updates are needed, return an empty list.
Do NOT output anything except the JSON array.
"""


def reflect(
    state: BeliefState,
    user_message: str,
    response: str,
    client: Any,
    mode: str = "auto",
    callback: Callable[[list[dict]], bool] | None = None,
) -> list[dict]:
    """Analyze a conversation turn and propose belief-state updates.

    Parameters
    ----------
    state:
        The current belief state (mutated in place when updates are applied).
    user_message:
        The user's input that triggered *response*.
    response:
        The LLM's output to analyze.
    client:
        An LLM client with a ``.complete(messages)`` method.
    mode:
        ``"auto"`` — apply all proposed updates without confirmation.
        ``"human"`` — call *callback* with the proposed updates and only
        apply if the callback returns ``True``.
    callback:
        A function called with the list of proposed update dicts when
        *mode* is ``"human"``.  Must return ``True`` to apply, ``False``
        to skip.

    Returns
    -------
    The list of proposed updates (whether applied or not).

    Examples
    --------
    >>> bs = BeliefState()
    >>> bs.add_belief("The capital of France is London", confidence=0.9, source="user")
    >>> # ^ intentionally wrong — reflection will catch this
    >>> from unittest.mock import MagicMock
    >>> mock_client = MagicMock()
    >>> mock_client.complete.return_value = json.dumps([
    ...     {"type": "confidence_update", "fact": "The capital of France is London",
    ...      "reason": "Paris is the capital, not London", "confidence": 0.0}
    ... ])
    >>> updates = reflect(bs, "What is the capital of France?", "London", mock_client, mode="auto")
    >>> bs.query("capital of France")[0].confidence
    0.0
    """
    beliefs_text = _format_beliefs_for_reflection(state)
    prompt = REFLECTION_PROMPT.format(
        beliefs_text=beliefs_text,
        user_message=user_message,
        response=response,
    )

    llm_response = client.complete([{"role": "user", "content": prompt}])
    proposals = _parse_proposals(llm_response)

    if not proposals:
        return proposals

    should_apply = True
    if mode == "human" and callback is not None:
        should_apply = callback(proposals)

    if should_apply:
        _apply_proposals(state, proposals)

    return proposals


def _format_beliefs_for_reflection(state: BeliefState) -> str:
    lines = []
    for b in state.beliefs:
        contra = f" [contradicts: {', '.join(b.contradictions)}]" if b.contradictions else ""
        lines.append(f"- [{b.confidence:.0%}] {b.fact} (source={b.source}, id={b.id}){contra}")
    return "\n".join(lines) if lines else "(none)"


def _parse_proposals(raw: str) -> list[dict]:
    """Parse the LLM's JSON response into a list of proposal dicts."""
    raw = raw.strip()
    # Try to extract JSON array from markdown fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        proposals = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(proposals, list):
        return []
    return proposals


def _apply_proposals(state: BeliefState, proposals: list[dict]) -> None:
    """Apply a list of proposals to a belief state (mutates in place)."""
    for p in proposals:
        typ = p.get("type", "")
        fact = p.get("fact", "")
        confidence = p.get("confidence")
        reason = p.get("reason", "")

        if typ == "contradiction" or typ == "confidence_update":
            # Find matching belief and update its confidence
            matches = state.query(fact)
            if matches:
                for b in matches:
                    if confidence is not None:
                        b.confidence = confidence
                    b.timestamp = __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat()
            elif confidence is not None:
                state.add_belief(
                    fact=fact,
                    confidence=confidence,
                    source="reflection",
                    assumptions=["inferred by reflection"],
                    tags=["reflection"],
                )

        elif typ == "new_belief":
            state.add_belief(
                fact=fact,
                confidence=confidence or 0.7,
                source="reflection",
                assumptions=["inferred by reflection"],
                tags=["reflection"],
            )

        elif typ == "removal":
            matches = state.query(fact)
            for b in matches:
                state.remove_belief(b.id)

    state.version += 1
