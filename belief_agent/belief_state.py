from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Belief(BaseModel):
    fact: str = Field(..., description="The belief statement")
    confidence: float = Field(
        default=0.5,
        description="Confidence in this belief (0 = none, 1 = certain)",
    )
    source: str = Field(
        default="user",
        description="Origin of the belief (user, model, reflection, tool, ...)",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Conditions or assumptions this belief depends on",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="IDs of beliefs that this belief contradicts",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence items",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for grouping / retrieval",
    )
    id: str = Field(default_factory=lambda: uuid4().hex[:12], description="Unique ID")

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Belief):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return f"[{self.confidence:.0%}] {self.fact} (source: {self.source})"

    def __repr__(self) -> str:
        return (
            f"Belief(fact={self.fact!r}, confidence={self.confidence:.2f}, "
            f"source={self.source!r}, id={self.id!r})"
        )


_NEGATION_PATTERNS = ("not ", "no ", "never ", "without ", "isn't ", "aren't ", "don't ", "doesn't ")
_OPPOSITES = [("true", "false"), ("yes", "no"), ("on", "off"), ("enable", "disable")]


class BeliefState(BaseModel):
    beliefs: list[Belief] = Field(default_factory=list)
    version: int = Field(default=1)

    # ---- add / update / remove ----

    def add_belief(
        self,
        fact: str,
        confidence: float = 0.5,
        source: str = "user",
        assumptions: list[str] | None = None,
        tags: list[str] | None = None,
        evidence: list[str] | None = None,
        detect_contradictions: bool = True,
    ) -> Belief:
        b = Belief(
            fact=fact,
            confidence=confidence,
            source=source,
            assumptions=assumptions or [],
            tags=tags or [],
            evidence=evidence or [],
        )
        if detect_contradictions:
            self._detect_contradictions_for(b)
        self.beliefs.append(b)
        self.version += 1
        return b

    def update_belief(
        self,
        belief_id: str,
        fact: str | None = None,
        confidence: float | None = None,
        source: str | None = None,
        assumptions: list[str] | None = None,
        tags: list[str] | None = None,
        evidence: list[str] | None = None,
        detect_contradictions: bool = True,
    ) -> Belief | None:
        for b in self.beliefs:
            if b.id == belief_id:
                if fact is not None:
                    b.fact = fact
                if confidence is not None:
                    b.confidence = confidence
                if source is not None:
                    b.source = source
                if assumptions is not None:
                    b.assumptions = assumptions
                if tags is not None:
                    b.tags = tags
                if evidence is not None:
                    b.evidence = evidence
                b.timestamp = datetime.now(timezone.utc).isoformat()
                if detect_contradictions:
                    self._detect_contradictions_for(b)
                self.version += 1
                return b
        return None

    def remove_belief(self, belief_id: str) -> bool:
        before = len(self.beliefs)
        self.beliefs = [b for b in self.beliefs if b.id != belief_id]
        removed = len(self.beliefs) < before
        if removed:
            self.version += 1
        return removed

    # ---- support / contradict ----

    def support(self, fact: str, evidence_item: str, delta: float = 0.1) -> Belief | None:
        matches = self.query(fact)
        if not matches:
            return None
        for b in matches:
            if evidence_item not in b.evidence:
                b.evidence.append(evidence_item)
            b.confidence = min(1.0, b.confidence + delta)
            b.timestamp = datetime.now(timezone.utc).isoformat()
        self.version += 1
        return matches[0]

    def contradict(self, fact: str, statement: str) -> Belief | None:
        matches = self.query(fact)
        if not matches:
            return None
        for b in matches:
            if statement not in b.contradictions:
                b.contradictions.append(statement)
            b.confidence *= 0.5
            b.timestamp = datetime.now(timezone.utc).isoformat()
        self.version += 1
        return matches[0]

    # ---- query ----

    def query(self, text: str, case_sensitive: bool = False) -> list[Belief]:
        if case_sensitive:
            return [b for b in self.beliefs if text in b.fact]
        text_lower = text.lower()
        return [b for b in self.beliefs if text_lower in b.fact.lower()]

    def query_by_tag(self, tag: str) -> list[Belief]:
        return [b for b in self.beliefs if tag in b.tags]

    def query_by_source(self, source: str) -> list[Belief]:
        return [b for b in self.beliefs if b.source == source]

    def get_all(self) -> list[Belief]:
        return list(self.beliefs)

    def get(self, belief_id: str) -> Belief | None:
        for b in self.beliefs:
            if b.id == belief_id:
                return b
        return None

    def list_beliefs(self, min_confidence: float = 0.0, only_contradicted: bool = False) -> list[Belief]:
        result = list(self.beliefs)
        if only_contradicted:
            result = [b for b in result if b.contradictions]
        if min_confidence > 0.0:
            result = [b for b in result if b.confidence >= min_confidence]
        return result

    def high_confidence(self, threshold: float = 0.8) -> list[Belief]:
        return [b for b in self.beliefs if b.confidence >= threshold]

    def low_confidence(self, threshold: float = 0.3) -> list[Belief]:
        return [b for b in self.beliefs if b.confidence <= threshold]

    def is_confident(self, fact: str, threshold: float = 0.7) -> bool:
        matches = self.query(fact)
        return bool(matches) and matches[0].confidence >= threshold

    def is_contradicted(self, fact: str) -> bool:
        matches = self.query(fact)
        return bool(matches) and len(matches[0].contradictions) > 0

    # ---- contradictions ----

    def get_contradictions(self) -> list[tuple[Belief, Belief]]:
        pairs: list[tuple[Belief, Belief]] = []
        for i, a in enumerate(self.beliefs):
            for b in self.beliefs[i + 1:]:
                if a.id in b.contradictions or b.id in a.contradictions:
                    pairs.append((a, b))
        return pairs

    def _detect_contradictions_for(self, belief: Belief) -> None:
        for other in self.beliefs:
            if other.id == belief.id:
                continue
            if self._texts_contradict(belief.fact, other.fact):
                if belief.id not in other.contradictions:
                    other.contradictions.append(belief.id)
                if other.id not in belief.contradictions:
                    belief.contradictions.append(other.id)

    @staticmethod
    def _texts_contradict(a: str, b: str) -> bool:
        a_lower = a.lower().strip().rstrip(".").lstrip(".")
        b_lower = b.lower().strip().rstrip(".").lstrip(".")
        if a_lower == b_lower:
            return False

        def _remove_negation(text: str) -> str | None:
            for pat in _NEGATION_PATTERNS:
                if pat in text:
                    candidate = text.replace(pat, "").strip()
                    while "  " in candidate:
                        candidate = candidate.replace("  ", " ")
                    return candidate
            return None

        cleaned_a = _remove_negation(a_lower)
        cleaned_b = _remove_negation(b_lower)
        if cleaned_a is not None and cleaned_a == b_lower:
            return True
        if cleaned_b is not None and cleaned_b == a_lower:
            return True
        for x, y in _OPPOSITES:
            if x in a_lower and y in b_lower:
                return True
        return False

    # ---- merge ----

    def merge(self, other: BeliefState) -> int:
        added = 0
        for incoming in other.beliefs:
            match = self._find_match(incoming)
            if match is None:
                self.beliefs.append(incoming.model_copy(deep=True))
                added += 1
            else:
                match.confidence = (match.confidence + incoming.confidence) / 2.0
                for e in incoming.evidence:
                    if e not in match.evidence:
                        match.evidence.append(e)
                for c in incoming.contradictions:
                    if c not in match.contradictions:
                        match.contradictions.append(c)
                match.timestamp = datetime.now(timezone.utc).isoformat()
        if added > 0:
            self.version += 1
        return added

    def _find_match(self, belief: Belief) -> Belief | None:
        for b in self.beliefs:
            if b.fact.strip().lower() == belief.fact.strip().lower():
                return b
        return None

    # ---- serialization ----

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefState:
        return cls.model_validate(data)

    def serialize(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def deserialize(cls, raw: str) -> BeliefState:
        return cls.model_validate(json.loads(raw))

    def to_json(self, indent: int = 2) -> str:
        return self.serialize(indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> BeliefState:
        return cls.deserialize(raw)

    # ---- convenience ----

    def __len__(self) -> int:
        return len(self.beliefs)

    def __iter__(self):
        return iter(self.beliefs)

    def __getitem__(self, index: int) -> Belief:
        return self.beliefs[index]

    def __str__(self) -> str:
        if not self.beliefs:
            return "BeliefState(empty)"
        lines = [f"BeliefState (v{self.version}, {len(self.beliefs)} beliefs):"]
        for i, b in enumerate(self.beliefs, 1):
            lines.append(f"  {i}. {b}")
        return "\n".join(lines)
