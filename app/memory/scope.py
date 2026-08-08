"""
Whose memory a request may read.

This module is the fix for a structural bug, not a convenience layer.

Memory was partitioned solely by `owner_id`. Guests are correctly issued their
own `guest-<uuid>` identity, and `settings.owner_user_id` was referenced *only*
at login — nowhere in the agent or memory path. `output_mode="recruiter"`
changed nothing but the response tone. So a recruiter session retrieved
`owner_id = "guest-abc123"`: empty profile facts, empty episodes, and vector
searches matching nothing. **The public-facing demo could not discuss the
résumé, skills, or projects it exists to present.**

The mistake was treating ownership as the whole access model. Ownership answers
"whose is this"; it cannot express "the owner's résumé is publicly readable".
`visibility` answers that, and it is orthogonal — which is also what makes real
multi-user sharing possible later without another redesign.

See docs/MEMORY_ARCHITECTURE.md §1.8 and §3.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.auth.models import Principal
from app.config import settings
from app.memory.kinds import Visibility


@dataclass(frozen=True)
class RetrievalScope:
    """Which owner's memory to read, and which visibilities are permitted."""

    owner_id: str
    visibilities: Optional[List[Visibility]]
    """None means every visibility — only ever granted over one's own memory."""

    is_own_memory: bool

    @property
    def public_only(self) -> bool:
        return self.visibilities == [Visibility.PUBLIC]

    def describe(self) -> str:
        """For logs and traces."""
        if self.is_own_memory:
            return f"{self.owner_id} (own, all visibilities)"
        allowed = ", ".join(v.value for v in (self.visibilities or []))
        return f"{self.owner_id} (shared, {allowed})"


def resolve_retrieval_scope(principal: Principal) -> RetrievalScope:
    """
    Decide whose memory this caller reads.

    * **Owner** — their own memory, every visibility.
    * **Guest** — the *owner's* memory, restricted to `public`.

    A guest reading the owner's public memory is the entire point of the
    recruiter view: there is nothing in a guest's own partition and there never
    will be, because guests do not hold the scopes to write memory.

    Private records remain unreachable: the visibility filter is applied in the
    store query itself, not by post-filtering results, so a private record is
    never loaded into a process serving a guest in the first place.
    """
    if principal.is_owner:
        return RetrievalScope(
            owner_id=principal.user_id,
            visibilities=None,
            is_own_memory=True,
        )

    return RetrievalScope(
        owner_id=settings.owner_user_id,
        visibilities=[Visibility.PUBLIC],
        is_own_memory=False,
    )


def can_write_memory(principal: Principal) -> bool:
    """
    Only the owner mutates memory.

    Guests hold `profile:read` and nothing more, so this is belt-and-braces
    alongside the route scopes — mutation authority should be checkable from
    the memory layer without reading the route table.
    """
    return principal.is_owner
