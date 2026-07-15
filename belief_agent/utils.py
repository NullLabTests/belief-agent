"""Utility functions for formatting, prompt construction, and serialization."""

from __future__ import annotations

from typing import Any

from .belief_state import BeliefState


def format_beliefs_bullet_points(state: BeliefState) -> str:
    """Render beliefs as a bullet-point list for inclusion in prompts.

    Parameters
    ----------
    state:
        The belief state to format.

    Returns
    -------
    A plain-text bullet list.

    Examples
    --------
    >>> bs = BeliefState()
    >>> bs.add_belief("The user prefers Python", confidence=0.9, source="chat")
    >>> print(format_beliefs_bullet_points(bs))
    - [90%] The user prefers Python (source: chat)
    """
    if not state.beliefs:
        return "(no beliefs recorded)"
    lines = ["Current beliefs:"]
    for b in state.beliefs:
        tag_str = f" [{', '.join(b.tags)}]" if b.tags else ""
        contra_str = f" *contradicts: {len(b.contradictions)} other belief(s)*" if b.contradictions else ""
        lines.append(
            f"- [{b.confidence:.0%}] {b.fact} "
            f"(source: {b.source}{tag_str}){contra_str}"
        )
    return "\n".join(lines)


def format_beliefs_json(state: BeliefState, indent: int = 2) -> str:
    """Render beliefs as a JSON string."""
    return state.serialize(indent=indent)


def format_beliefs_natural(state: BeliefState) -> str:
    """Render beliefs in a natural-language paragraph."""
    if not state.beliefs:
        return "I have no recorded beliefs yet."
    parts: list[str] = []
    for b in state.beliefs:
        confidence_word = (
            "am certain that"
            if b.confidence >= 0.9
            else "believe that"
            if b.confidence >= 0.6
            else "think it is possible that"
            if b.confidence >= 0.3
            else "am uncertain whether"
        )
        parts.append(f"I {confidence_word} {b.fact.lower().rstrip('.')}.")
    return " ".join(parts)


def format_beliefs_for_system_prompt(
    state: BeliefState,
    style: str = "bullet",
) -> str:
    """Format beliefs into a section suitable for a system prompt.

    Parameters
    ----------
    state:
        The belief state to inject.
    style:
        One of ``"bullet"``, ``"json"``, or ``"natural"``.

    Returns
    -------
    A formatted string.
    """
    fmt = {"bullet": format_beliefs_bullet_points, "json": format_beliefs_json, "natural": format_beliefs_natural}
    fn = fmt.get(style)
    if fn is None:
        raise ValueError(f"Unknown style {style!r}. Choose from: {', '.join(fmt)}")
    return fn(state)


SYSTEM_PROMPT_CORE = """You are a helpful AI assistant that maintains explicit beliefs about the world and your \
conversation with the user. You strive to be truthful, acknowledge uncertainty, and update your beliefs when new \
information contradicts old assumptions.

=== Current Beliefs ===
{beliefs_section}

=== Guidelines ===
- Be honest about what you know and don't know.
- If new information contradicts a previously held belief, acknowledge the contradiction and update your belief.
- Always distinguish between facts, assumptions, and speculation.
- When uncertain, state your confidence level explicitly.
"""


def build_system_prompt(
    state: BeliefState,
    style: str = "bullet",
    extra_instructions: str = "",
) -> str:
    """Build a complete system prompt that includes the current belief state.

    Parameters
    ----------
    state:
        Belief state to inject.
    style:
        Formatting style for beliefs.
    extra_instructions:
        Additional instructions appended before the closing guidelines.

    Returns
    -------
    A ready-to-use system prompt string.
    """
    beliefs_section = format_beliefs_for_system_prompt(state, style=style)
    prompt = SYSTEM_PROMPT_CORE.format(beliefs_section=beliefs_section)
    if extra_instructions:
        prompt += f"\n\n=== Additional Instructions ===\n{extra_instructions}"
    return prompt


LENIENT_JSON_ENCODER_KWARGS: dict[str, Any] = {
    "default": str,
    "ensure_ascii": False,
    "allow_nan": True,
}
