"""
Typed results for long-term memory retrieval.

Replaces the `List[dict] | "NO_DATA"` union that `retrieve_skills()` and
`retrieve_projects()` used to return. That union forced every caller to
isinstance-guard, and three of them got it subtly wrong:

  * `search_all()` compared `skills == "NO_DATA"` to decide whether to run the
    resume fallback. Because `retrieve_skills()` swallowed exceptions and
    returned `[]`, a *failed* lookup was reported as `status="OK"` with zero
    results — indistinguishable from a user who genuinely has no skills stored.
    The prompt then omitted both the "no skills data" hint and the refusal
    policy, which is precisely the state in which a model invents skills.

  * `search_all()`'s outer exception handler returned a dict with no status
    keys at all, producing the same failure mode.

`RetrievalStatus.ERROR` makes "we could not find out" a distinct, representable
outcome from "there is nothing to find".

The status *values* are deliberately unchanged (`OK` / `FALLBACK` / `NO_DATA`)
so existing string comparisons and any cached context dictionaries written by a
previous process continue to behave identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List


class RetrievalStatus(str, Enum):
    """Why a retrieval returned the items it did."""

    OK = "OK"
    """Served from the dedicated collection."""

    FALLBACK = "FALLBACK"
    """Dedicated collection was empty; recovered from resume chunks."""

    NO_DATA = "NO_DATA"
    """Nothing is stored for this user. The absence is a fact."""

    ERROR = "ERROR"
    """Lookup failed. The absence is *not* a fact — we simply do not know."""


@dataclass(frozen=True)
class RetrievalResult:
    """
    Items plus the provenance of the lookup that produced them.

    Behaves like a sequence so callers can iterate, len(), and truth-test it
    directly; `status` is consulted only when the *reason* for emptiness
    matters — which, for prompt construction, it always does.
    """

    items: List[Dict[str, Any]] = field(default_factory=list)
    status: RetrievalStatus = RetrievalStatus.OK

    # ── constructors ────────────────────────────────────────────────────
    @classmethod
    def ok(cls, items: List[Dict[str, Any]]) -> "RetrievalResult":
        return cls(items=list(items), status=RetrievalStatus.OK)

    @classmethod
    def fallback(cls, items: List[Dict[str, Any]]) -> "RetrievalResult":
        return cls(items=list(items), status=RetrievalStatus.FALLBACK)

    @classmethod
    def no_data(cls) -> "RetrievalResult":
        return cls(items=[], status=RetrievalStatus.NO_DATA)

    @classmethod
    def error(cls) -> "RetrievalResult":
        return cls(items=[], status=RetrievalStatus.ERROR)

    # ── sequence protocol ───────────────────────────────────────────────
    def __bool__(self) -> bool:
        return bool(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    # ── predicates ──────────────────────────────────────────────────────
    @property
    def is_usable(self) -> bool:
        """True when the lookup succeeded and produced something."""
        return bool(self.items) and self.status is not RetrievalStatus.ERROR

    @property
    def is_trustworthy_absence(self) -> bool:
        """
        True when emptiness is a fact we can state to the model.

        An ERROR is emphatically *not* a trustworthy absence: telling the model
        "no skills data found" after a Qdrant timeout asserts something false.
        """
        return self.status is RetrievalStatus.NO_DATA
