from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class Belief(BaseModel):
    """A single belief held by the agent.

    Each belief stores a factual claim along with metadata about its
    origin, certainty, and relationship to other beliefs.
    """

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
        description="ISO-8601 timestamp of when this belief was added/updated",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Conditions or assumptions this belief depends on",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Facts that this belief explicitly contradicts",
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


class BeliefState(BaseModel):
    """Container for a collection of :class:`Belief` objects.

    Provides add / update / query / merge operations and automated
    contradiction detection.
    """

    beliefs: list[Belief] = Field(default_factory=list, description="All stored beliefs")
    version: int = Field(default=1, description="Monotonic version counter (bumped on every mutation)")

    # ------------------------------------------------------------------
    # Add / update
    # ------------------------------------------------------------------

    def add_belief(
        self,
        fact: str,
        confidence: float = 0.5,
        source: str = "user",
        assumptions: list[str] | None = None,
        tags: list[str] | None = None,
        detect_contradictions: bool = True,
    ) -> Belief:
        """Add a new belief.

        Parameters
        ----------
        fact:
            The belief statement.
        confidence:
            How certain the agent is (0-1).
        source:
            Origin of the belief.
        assumptions:
            Conditional assumptions this belief depends on.
        tags:
            Optional tags for grouping.
        detect_contradictions:
            If True, automatically scan existing beliefs for contradictions.

        Returns
        -------
        The newly created :class:`Belief`.

        Examples
        --------
        >>> bs = BeliefState()
        >>> b = bs.add_belief("The sky is blue", confidence=0.9, source="observation")
        >>> b.fact
        'The sky is blue'
        """
        b = Belief(
            fact=fact,
            confidence=confidence,
            source=source,
            assumptions=assumptions or [],
            tags=tags or [],
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
        detect_contradictions: bool = True,
    ) -> Belief | None:
        """Update an existing belief by ID.

        Only the fields provided are changed; ``None`` fields are left untouched.

        Returns
        -------
        The updated :class:`Belief`, or ``None`` if no belief matched *belief_id*.
        """
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
                b.timestamp = datetime.now(timezone.utc).isoformat()
                if detect_contradictions:
                    self._detect_contradictions_for(b)
                self.version += 1
                return b
        return None

    def remove_belief(self, belief_id: str) -> bool:
        """Remove a belief by ID.

        Returns
        -------
        True if a belief was removed, False otherwise.
        """
        before = len(self.beliefs)
        self.beliefs = [b for b in self.beliefs if b.id != belief_id]
        removed = len(self.beliefs) < before
        if removed:
            self.version += 1
        return removed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str, case_sensitive: bool = False) -> list[Belief]:
        """Return all beliefs whose fact contains *text*.

        Performs substring matching.
        """
        if case_sensitive:
            return [b for b in self.beliefs if text in b.fact]
        text_lower = text.lower()
        return [b for b in self.beliefs if text_lower in b.fact.lower()]

    def query_by_tag(self, tag: str) -> list[Belief]:
        """Return all beliefs tagged with *tag*."""
        return [b for b in self.beliefs if tag in b.tags]

    def query_by_source(self, source: str) -> list[Belief]:
        """Return all beliefs from a given source."""
        return [b for b in self.beliefs if b.source == source]

    def get_all(self) -> list[Belief]:
        """Return a shallow copy of all beliefs."""
        return list(self.beliefs)

    def get(self, belief_id: str) -> Belief | None:
        """Retrieve a single belief by ID."""
        for b in self.beliefs:
            if b.id == belief_id:
                return b
        return None

    def high_confidence(self, threshold: float = 0.8) -> list[Belief]:
        """Return beliefs with confidence >= *threshold*."""
        return [b for b in self.beliefs if b.confidence >= threshold]

    def low_confidence(self, threshold: float = 0.3) -> list[Belief]:
        """Return beliefs with confidence <= *threshold*."""
        return [b for b in self.beliefs if b.confidence <= threshold]

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------

    def get_contradictions(self) -> list[tuple[Belief, Belief]]:
        """Return all pairs of beliefs that list each other as contradictions."""
        pairs: list[tuple[Belief, Belief]] = []
        for i, a in enumerate(self.beliefs):
            for b in self.beliefs[i + 1:]:
                if a.id in b.contradictions or b.id in a.contradictions:
                    pairs.append((a, b))
        return pairs

    def _detect_contradictions_for(self, belief: Belief) -> None:
        """Scan every existing belief and cross-link contradictions."""
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
        """Cheap heuristic: treat negated statements as contradictions.

        For example ``"The sky is blue"`` and ``"The sky is not blue"`` will
        be flagged.  This is deliberately simple; a production system should
        use an LLM-based judge for semantic contradiction detection.
        """
        a_lower = a.lower().strip().rstrip(".").lstrip(".")
        b_lower = b.lower().strip().rstrip(".").lstrip(".")

        # If both statements match exactly, they are not contradictory
        if a_lower == b_lower:
            return False

        # Check if stripping a negation word from one yields the other
        negation_patterns = ("not ", "no ", "never ", "without ", "isn't ", "aren't ", "don't ", "doesn't ")

        def _remove_negation(text: str) -> str | None:
            """If *text* contains a negation word, return the text without it."""
            for pat in negation_patterns:
                if pat in text:
                    candidate = text.replace(pat, "").strip()
                    # Collapse repeated spaces
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

        # Opposite claims
        opposites = [("true", "false"), ("yes", "no"), ("on", "off"), ("enable", "disable")]
        for x, y in opposites:
            if x in a_lower and y in b_lower:
                return True

        return False

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge(self, other: BeliefState) -> int:
        """Merge another ``BeliefState`` into this one.

        Duplicate beliefs (same fact and source) are skipped unless they have
        higher confidence — in which case the existing belief is updated.

        Parameters
        ----------
        other:
            Another belief state to merge in.

        Returns
        -------
        Number of new beliefs added.
        """
        added = 0
        for incoming in other.beliefs:
            match = self._find_match(incoming)
            if match is None:
                self.beliefs.append(incoming.model_copy(deep=True))
                added += 1
            elif incoming.confidence > match.confidence:
                match.confidence = incoming.confidence
                match.source = incoming.source
                match.timestamp = datetime.now(timezone.utc).isoformat()
        self.version += 1
        return added

    def _find_match(self, belief: Belief) -> Belief | None:
        """Return a belief that shares the same fact and source, or None."""
        for b in self.beliefs:
            if b.fact.strip().lower() == belief.fact.strip().lower() and b.source == belief.source:
                return b
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export state as a JSON-compatible dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefState:
        """Build a ``BeliefState`` from a dictionary."""
        return cls.model_validate(data)

    def serialize(self, **kwargs: Any) -> str:
        """Export state as a JSON string.

        Parameters
        ----------
        **kwargs:
            Passed through to ``json.dumps`` (e.g. ``indent=2``).
        """
        return self.model_dump_json(**kwargs)

    @classmethod
    def deserialize(cls, raw: str) -> BeliefState:
        """Parse a JSON string back into a ``BeliefState``."""
        return cls.model_validate(json.loads(raw))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

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
