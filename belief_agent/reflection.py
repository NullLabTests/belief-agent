from __future__ import annotations

import json
from dataclasses import dataclass, field
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


@dataclass
class ReflectionResult:
    proposition: str
    original_confidence: float
    updated_confidence: float
    critique: str
    new_evidence: list[str] = field(default_factory=list)
    new_contradictions: list[str] = field(default_factory=list)
    accepted: bool = True


def reflect(
    state: BeliefState,
    user_message: str,
    response: str,
    client: Any,
    mode: str = "auto",
    callback: Callable[[list[dict]], bool] | None = None,
) -> list[dict]:
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


def reflect_on_beliefs(
    state: BeliefState,
    llm_call: Callable[[list[dict]], str],
    depth: int = 1,
) -> list[ReflectionResult]:
    results: list[ReflectionResult] = []
    if depth < 1:
        return results

    for belief in list(state.beliefs):
        result = _reflect_on_single_belief(belief, llm_call)
        if result.accepted:
            result.accepted = _decide_acceptance(result, llm_call)
        if result.accepted:
            for b in state.beliefs:
                if b.id == belief.id:
                    b.confidence = result.updated_confidence
                    for e in result.new_evidence:
                        if e not in b.evidence:
                            b.evidence.append(e)
                    for c in result.new_contradictions:
                        if c not in b.contradictions:
                            b.contradictions.append(c)
                    b.timestamp = __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat()
                    break
            state.version += 1
        results.append(result)

    if depth > 1:
        results.extend(reflect_on_beliefs(state, llm_call, depth=depth - 1))

    return results


def _reflect_on_single_belief(
    belief: Any,
    llm_call: Callable[[list[dict]], str],
) -> ReflectionResult:
    prompt = (
        f"Critique the following belief held with confidence {belief.confidence:.2f}:\n"
        f'  "{belief.fact}"\n\n'
        f"Evidence:\n"
        + ("\n".join(f"  - {e}" for e in belief.evidence) or "  (none)")
        + "\n\nContradictions:\n"
        + ("\n".join(f"  - {c}" for c in belief.contradictions) or "  (none)")
        + "\n\nProvide:\n"
        "1. A brief critique.\n"
        "2. Any new evidence (one item).\n"
        "3. Any new contradictions (one item).\n"
        "4. An updated confidence (0.0-1.0).\n\n"
        "Return as JSON: "
        '{"critique":"...","new_evidence":["..."],"new_contradictions":["..."],"updated_confidence":0.X}'
    )
    raw = llm_call([{"role": "user", "content": prompt}])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ReflectionResult(
            proposition=belief.fact,
            original_confidence=belief.confidence,
            updated_confidence=belief.confidence,
            critique="Could not parse LLM reflection output.",
            accepted=False,
        )
    if not isinstance(data, dict):
        return ReflectionResult(
            proposition=belief.fact,
            original_confidence=belief.confidence,
            updated_confidence=belief.confidence,
            critique="LLM returned non-dict JSON.",
            accepted=False,
        )
    return ReflectionResult(
        proposition=belief.fact,
        original_confidence=belief.confidence,
        updated_confidence=data.get("updated_confidence", belief.confidence),
        critique=data.get("critique", ""),
        new_evidence=data.get("new_evidence", []),
        new_contradictions=data.get("new_contradictions", []),
    )


def _decide_acceptance(
    result: ReflectionResult,
    llm_call: Callable[[list[dict]], str],
) -> bool:
    prompt = (
        f"Should the following reflection be accepted?\n\n"
        f"Proposition: {result.proposition}\n"
        f"Original confidence: {result.original_confidence:.2f}\n"
        f"Updated confidence: {result.updated_confidence:.2f}\n"
        f"Critique: {result.critique}\n\n"
        'Answer "yes" or "no" as JSON: {"accept": true/false}'
    )
    raw = llm_call([{"role": "user", "content": prompt}])
    try:
        data = json.loads(raw)
        return data.get("accept", True)
    except (json.JSONDecodeError, KeyError):
        return True


def auto_update(
    state: BeliefState,
    llm_call: Callable[[list[dict]], str],
    threshold: float = 0.3,
) -> list[ReflectionResult]:
    low = {b.fact for b in state.beliefs if b.confidence < threshold}
    if not low:
        return []
    results: list[ReflectionResult] = []
    for belief in state.beliefs:
        if belief.fact not in low:
            continue
        result = _reflect_on_single_belief(belief, llm_call)
        if result.accepted:
            result.accepted = _decide_acceptance(result, llm_call)
        if result.accepted:
            belief.confidence = result.updated_confidence
            for e in result.new_evidence:
                if e not in belief.evidence:
                    belief.evidence.append(e)
            for c in result.new_contradictions:
                if c not in belief.contradictions:
                    belief.contradictions.append(c)
            state.version += 1
        results.append(result)
    return results


def human_review(
    result: ReflectionResult,
    human_feedback: str,
    accepted: bool | None = None,
) -> ReflectionResult:
    result.critique += f"\n[Human feedback: {human_feedback}]"
    if accepted is not None:
        result.accepted = accepted
    return result


# ---- internal helpers ----

def _format_beliefs_for_reflection(state: BeliefState) -> str:
    lines = []
    for b in state.beliefs:
        contra = f" [contradicts: {', '.join(b.contradictions)}]" if b.contradictions else ""
        lines.append(f"- [{b.confidence:.0%}] {b.fact} (source={b.source}, id={b.id}){contra}")
    return "\n".join(lines) if lines else "(none)"


def _parse_proposals(raw: str) -> list[dict]:
    raw = raw.strip()
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
    for p in proposals:
        typ = p.get("type", "")
        fact = p.get("fact", "")
        confidence = p.get("confidence")

        if typ in ("contradiction", "confidence_update"):
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
