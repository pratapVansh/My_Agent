"""
Identity and authorization primitives.

A Principal is the *verified* answer to "who is making this request". It is
constructed only from a validated JWT — never from request bodies, query
parameters, or path segments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional


class Role(str, Enum):
    OWNER = "owner"
    GUEST = "guest"


class Scope(str, Enum):
    """
    Capability names carried in the token and checked at three layers:
    the route, the identity resolver, and the agent tool registry.
    """

    CHAT = "chat"                       # converse with the assistant

    PROFILE_READ = "profile:read"       # read skills, projects, resume, facts
    PROFILE_WRITE = "profile:write"     # save or forget profile facts

    MEMORY_WRITE = "memory:write"       # upload documents into long-term memory

    ATTENDANCE_READ = "attendance:read"
    ATTENDANCE_WRITE = "attendance:write"

    TIMETABLE_READ = "timetable:read"
    TIMETABLE_WRITE = "timetable:write"

    EMAIL_DRAFT = "email:draft"
    EMAIL_SEND = "email:send"           # actually delivers mail — owner only

    JOBS_SEARCH = "jobs:search"
    JOBS_WRITE = "jobs:write"           # bookmark jobs

    TOOLS_SCRAPE = "tools:scrape"       # drives a headless browser — owner only

    VOICE = "voice"                     # mint a LiveKit room token


# The owner is the account the assistant belongs to: full capability.
OWNER_SCOPES: FrozenSet[str] = frozenset(scope.value for scope in Scope)

# Guests are anonymous visitors (the recruiter-facing demo). They may hold a
# conversation and read the owner's public professional profile. They may not
# send mail, mutate memory, drive the scraper, or read academic records.
#
# This set is the authoritative answer to audit finding C2.
GUEST_SCOPES: FrozenSet[str] = frozenset({
    Scope.CHAT.value,
    Scope.PROFILE_READ.value,
})


def scopes_for_role(role: Role) -> FrozenSet[str]:
    return OWNER_SCOPES if role is Role.OWNER else GUEST_SCOPES


@dataclass(frozen=True)
class Principal:
    """The authenticated caller behind a request."""

    user_id: str
    role: Role
    scopes: FrozenSet[str] = field(default_factory=frozenset)
    token_id: Optional[str] = None       # jti, for revocation
    session_id: Optional[str] = None

    @property
    def is_owner(self) -> bool:
        return self.role is Role.OWNER

    @property
    def is_guest(self) -> bool:
        return self.role is Role.GUEST

    def has_scope(self, scope: "Scope | str") -> bool:
        value = scope.value if isinstance(scope, Scope) else scope
        return value in self.scopes

    def has_all_scopes(self, *required: "Scope | str") -> bool:
        return all(self.has_scope(scope) for scope in required)
