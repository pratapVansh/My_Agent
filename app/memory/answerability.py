"""
Whether the evidence in hand is enough to answer.

The distinction this module exists to protect is between *nothing is stored*
and *the lookup failed*. They arrive at the same place — an empty list — and
they warrant opposite responses:

    NO_DATA    → "I don't have your CGPA on file."      (a true statement)
    TOOL_ERROR → "I couldn't reach your records."       (also true)

Saying the first when the second happened is a false statement about the user's
own data, and it is the exact state in which a language model, told there is no
information, invents some. `RetrievalResult` already carries the distinction at
the store boundary; this carries it up to the decision about what to say.

The second thing it protects is the answer-first policy. Clarification is a
*state*, not a mood: it is reachable only from AMBIGUOUS and
ACTION_MISSING_PARAMETER. Low planner confidence is not on that list, and
neither is an empty store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.memory.retrieval_result import RetrievalResult, RetrievalStatus
from app.memory.sources import MemorySource, QueryCategory, RetrievedMemory


class Answerability(str, Enum):
    """What the retrieved evidence permits us to do."""

    ANSWERABLE = "ANSWERABLE"
    """Enough evidence to answer the question as asked."""

    PARTIALLY_ANSWERABLE = "PARTIALLY_ANSWERABLE"
    """Some of it is known. Say what is known and name what is not — never
    withhold a partial answer behind a question."""

    NO_DATA = "NO_DATA"
    """Retrieval succeeded and there is genuinely nothing. A fact, statable."""

    AMBIGUOUS = "AMBIGUOUS"
    """Several candidates and no way to choose between them."""

    ACTION_MISSING_PARAMETER = "ACTION_MISSING_PARAMETER"
    """An action cannot execute without something no store holds."""

    TOOL_ERROR = "TOOL_ERROR"
    """The lookup failed. Absence is NOT established."""


# The only two states from which asking the user a question is correct.
CLARIFIABLE_STATES = frozenset({
    Answerability.AMBIGUOUS,
    Answerability.ACTION_MISSING_PARAMETER,
})


@dataclass
class Assessment:
    """The verdict plus why, for logging and for the response policy."""

    state: Answerability
    evidence_count: int = 0
    sources_hit: List[MemorySource] = field(default_factory=list)
    sources_expected: List[MemorySource] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def may_clarify(self) -> bool:
        return self.state in CLARIFIABLE_STATES

    @property
    def should_answer(self) -> bool:
        """Whether to produce an answer rather than a question."""
        return self.state in (
            Answerability.ANSWERABLE,
            Answerability.PARTIALLY_ANSWERABLE,
        )

    def guidance(self) -> str:
        """
        The sentence handed to the model about how to treat this evidence.

        Deliberately prescriptive. A model told only "no results" will fill the
        gap; told what the emptiness *means*, it reports it.
        """
        if self.state is Answerability.TOOL_ERROR:
            return (
                "Retrieval failed for this question. Do NOT say the information "
                "is missing or that you have nothing on file — that is not "
                "established. Say you could not look it up right now."
            )
        if self.state is Answerability.NO_DATA:
            return (
                "Retrieval succeeded and returned nothing. State plainly that "
                "you don't have that information. Do not guess a value, and do "
                "not ask the user to clarify a question that was already clear."
            )
        if self.state is Answerability.PARTIALLY_ANSWERABLE:
            missing = ", ".join(self.missing) if self.missing else "some details"
            return (
                f"Partial evidence is available. Answer with what was found and "
                f"say only that {missing} is not on file. Do not withhold the "
                f"partial answer behind a clarifying question."
            )
        if self.state is Answerability.AMBIGUOUS:
            return (
                "Several candidates matched and the conversation does not single "
                "one out. Ask which one — this is the one case where a question "
                "is the right response."
            )
        if self.state is Answerability.ACTION_MISSING_PARAMETER:
            missing = ", ".join(self.missing) if self.missing else "a required detail"
            return f"This action cannot run without {missing}. Ask for it, once."
        return "Sufficient evidence was retrieved. Answer the question directly."

    def summary(self) -> Dict[str, Any]:
        """Structured form for logs — counts and provenance, never content."""
        return {
            "answerability": self.state.value,
            "results": self.evidence_count,
            "sources_hit": [s.value for s in self.sources_hit],
            "missing": list(self.missing),
        }


def assess(
    evidence: Sequence[RetrievedMemory],
    *,
    category: Optional[QueryCategory] = None,
    expected_sources: Optional[Iterable[MemorySource]] = None,
    errored_sources: Optional[Iterable[MemorySource]] = None,
    missing_parameters: Optional[Sequence[str]] = None,
    ambiguous_candidates: int = 0,
) -> Assessment:
    """
    Decide what the retrieved evidence permits.

    `errored_sources` is what keeps NO_DATA honest: a source that raised is not
    a source that came back empty, and if any expected source failed while none
    produced evidence, the correct verdict is TOOL_ERROR.
    """
    expected = list(expected_sources or [])
    errored = list(errored_sources or [])
    missing = list(missing_parameters or [])
    hits = list(dict.fromkeys(item.source for item in evidence))

    if missing:
        return Assessment(
            state=Answerability.ACTION_MISSING_PARAMETER,
            evidence_count=len(evidence),
            sources_hit=hits,
            sources_expected=expected,
            missing=missing,
            reason="required action parameter absent",
        )

    if not evidence:
        if errored:
            return Assessment(
                state=Answerability.TOOL_ERROR,
                sources_hit=hits,
                sources_expected=expected,
                reason=f"lookup failed: {', '.join(s.value for s in errored)}",
            )
        return Assessment(
            state=Answerability.NO_DATA,
            sources_hit=hits,
            sources_expected=expected,
            reason="retrieval succeeded with no matching records",
        )

    if ambiguous_candidates > 1:
        return Assessment(
            state=Answerability.AMBIGUOUS,
            evidence_count=len(evidence),
            sources_hit=hits,
            sources_expected=expected,
            reason=f"{ambiguous_candidates} indistinguishable candidates",
        )

    # Evidence exists, but a source that should have contributed failed. What
    # was found is real; what was not found is unknown rather than absent.
    if errored:
        return Assessment(
            state=Answerability.PARTIALLY_ANSWERABLE,
            evidence_count=len(evidence),
            sources_hit=hits,
            sources_expected=expected,
            missing=[s.value for s in errored],
            reason="some sources failed while others returned evidence",
        )

    # The primary source for this category produced nothing, but a lower-priority
    # one did. That is an answer, with a caveat — not a complete one.
    if expected and hits and expected[0] not in hits:
        return Assessment(
            state=Answerability.PARTIALLY_ANSWERABLE,
            evidence_count=len(evidence),
            sources_hit=hits,
            sources_expected=expected,
            missing=[expected[0].value],
            reason="answered from a secondary source",
        )

    return Assessment(
        state=Answerability.ANSWERABLE,
        evidence_count=len(evidence),
        sources_hit=hits,
        sources_expected=expected,
        reason="primary source produced evidence",
    )


def from_retrieval_result(
    result: RetrievalResult,
    source: MemorySource,
    *,
    memory_type: str = "document",
    provenance: str = "",
    owner_id: str = "",
) -> tuple[List[RetrievedMemory], List[MemorySource]]:
    """
    Adapt a store-level `RetrievalResult` into evidence plus error signal.

    Returns (evidence, errored_sources) so `assess` can tell the difference
    between an empty store and a broken one without every caller re-deriving it.
    """
    if result.status is RetrievalStatus.ERROR:
        return [], [source]

    evidence = [
        RetrievedMemory(
            content=str(item.get("content") or ""),
            source=source,
            memory_type=memory_type,
            provenance=provenance or f"{source.value}:{item.get('section', memory_type)}",
            owner_id=owner_id,
            explicit=False,
            derived=result.status is RetrievalStatus.FALLBACK,
            metadata=item.get("metadata") or {},
        )
        for item in result
        if str(item.get("content") or "").strip()
    ]
    return evidence, []
