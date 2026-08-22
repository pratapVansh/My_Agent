"""
The tools that can only run with permission — defined once, in one place.

A durable pending action stores a tool *name*, never a callable. Nothing else
would be safe: a pickled function is code, and a database row that can become
executable code is a remote-code-execution primitive wearing an audit trail. So
confirmation across a restart requires the opposite — rebuilding the callable
from a name, through a registry the application controls.

Which raises the question this module answers: *which* registry? The agents
build their tools as closures inside `execute()`, bound to the signed-in user.
A separate lookup table for the resolver would be a second definition of
`send_email`, and second definitions drift — one gains a validation the other
lacks, or a tool is gated in the agent and unguarded in the resolver, and the
divergence is invisible until it matters.

So there is one definition. These factories build the specs, the agents call
them to populate their registries, and the resolver calls the same factories to
reconstruct an approved action. A tool cannot exist in the agent and be missing
from the resolver, because they are the same function — and
`test_every_confirmable_tool_is_resolvable` asserts it stays that way.

Only EXTERNAL_WRITE and DESTRUCTIVE tools live here. Reads and local writes are
never held, never persisted, and never reconstructed; they simply run.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from app.agents.actions import ActionPreview
from app.auth.models import Scope
from app.domain.email import email_repository
from app.memory.memory_manager import memory_manager
from app.services.email_sender_service import email_sender_service
from app.tools.contract import Effect

logger = logging.getLogger(__name__)

# Duplicated from `app.mcp.config` rather than imported, so that this module —
# which the action gateway depends on — does not pull the MCP package (and its
# SDK) into every deployment. The MCP import below is local and guarded.
_MCP_PREFIX = "mcp__"


# ── send_email (EXTERNAL_WRITE) ──────────────────────────────────────────────

def build_send_email_tool(owner_id: str) -> Dict[str, Any]:
    """
    The registry entry for sending mail, bound to one user.

    The binding matters: `owner_id` decides whose drafts a `draft_id` resolves
    against, so a reconstructed action can only ever read the approving user's
    own data. It is taken from the stored action's owner, which was verified
    against the confirming caller before this is ever called.
    """

    async def tool_send_email(tool_input: Mapping[str, Any]) -> Dict[str, Any]:
        to_email = str(tool_input.get("to_email") or "").strip()
        subject = str(tool_input.get("subject") or "").strip()
        body = str(tool_input.get("body") or "").strip()
        draft_id = tool_input.get("draft_id")
        cc_raw = tool_input.get("cc")
        cc = [cc_raw] if isinstance(cc_raw, str) and cc_raw else (cc_raw or None)

        if not to_email:
            return {
                "success": False,
                "error": "to_email is required. Ask the user for the recipient's email address.",
            }

        # If draft_id supplied but body/subject are missing, load from saved draft
        if draft_id and (not subject or not body):
            drafts = await email_repository.get_drafts(
                user_id=owner_id, limit=50, status="draft"
            )
            match = next((d for d in drafts if d["id"] == draft_id), None)
            if match:
                subject = subject or match.get("subject", "")
                body = body or match.get("body", "")

        if not subject or not body:
            return {
                "success": False,
                "error": "subject and body are required. Use email_draft first to compose the email.",
            }

        result = await email_sender_service.send_email(
            to_email=to_email, subject=subject, body=body, cc=cc,
        )

        if result.get("success") and draft_id:
            await email_repository.mark_draft_sent(draft_id, owner_id, to_email)

        return result

    async def preview_send_email(args: Mapping[str, Any]) -> ActionPreview:
        """
        Describe the email that *would* be sent, without sending it.

        Reads only. It resolves a `draft_id` to the actual subject and body for
        two reasons: a preview that cannot show the body is not a preview a
        person can meaningfully approve, and the resolved values become the
        arguments that get hashed, stored and later executed — so what is
        shown, approved and sent are one object rather than three that merely
        ought to agree.

        The precondition checks below are the same ones `tool_send_email`
        performs. They have to live here too: that function no longer runs
        before confirmation, so if this did not reject an address-less send the
        user would be asked to approve an email to nobody.
        """
        to_email = str(args.get("to_email") or "").strip()
        subject = str(args.get("subject") or "").strip()
        body = str(args.get("body") or "").strip()
        draft_id = args.get("draft_id")
        cc_raw = args.get("cc")
        cc = [cc_raw] if isinstance(cc_raw, str) and cc_raw else (cc_raw or None)

        if not to_email:
            return ActionPreview.invalid(
                "to_email is required. Ask the user for the recipient's email address."
            )

        if draft_id and (not subject or not body):
            drafts = await email_repository.get_drafts(
                user_id=owner_id, limit=50, status="draft"
            )
            match = next((d for d in drafts if d["id"] == draft_id), None)
            if match:
                subject = subject or match.get("subject", "")
                body = body or match.get("body", "")

        if not subject or not body:
            return ActionPreview.invalid(
                "subject and body are required. Use email_draft first to compose the email."
            )

        lines = ["You are about to SEND this email:", "", f"To:      {to_email}"]
        if cc:
            lines.append(f"Cc:      {', '.join(cc)}")
        lines += [f"Subject: {subject}", "", body]

        # The resolved arguments, not the ones the model supplied. These are
        # what get hashed, stored and replayed.
        resolved = {
            "to_email": to_email,
            "subject": subject,
            "body": body,
            "cc": cc,
            "draft_id": draft_id,
        }
        return ActionPreview("\n".join(lines), resolved)

    return {
        "description": (
            "Prepare an email for the user's approval. Requires to_email "
            "(full address e.g. name@domain.com), subject, body."
        ),
        "callable": tool_send_email,
        "scope": Scope.EMAIL_SEND.value,
        # Delivers mail to a real person — the one existing tool whose effect
        # leaves this system and cannot be taken back.
        "effect": Effect.EXTERNAL_WRITE,
        "preview": preview_send_email,
    }


# ── forget_preference (DESTRUCTIVE) ──────────────────────────────────────────

def build_forget_preference_tool(owner_id: str) -> Dict[str, Any]:
    """The registry entry for deleting a stored profile fact, bound to one user."""

    async def tool_forget_preference(tool_input: Mapping[str, Any]) -> Dict[str, Any]:
        key = str(tool_input.get("key", "")).strip().lower().replace(" ", "_")
        if not key:
            return {"success": False, "reason": "'key' is required."}
        deleted = await memory_manager.forget_profile_fact(user_id=owner_id, key=key)
        if deleted:
            return {"success": True, "message": f"Forgotten: {key}"}
        return {"success": False, "message": f"No memory found for key '{key}'."}

    async def preview_forget_preference(args: Mapping[str, Any]) -> ActionPreview:
        """
        Show exactly what deleting this key would remove.

        Reads the stored value so the user is approving the removal of a
        specific fact rather than a key name they may not recognise.
        """
        key = str(args.get("key", "")).strip().lower().replace(" ", "_")
        if not key:
            return ActionPreview.invalid("'key' is required.")

        try:
            facts = await memory_manager.get_profile_facts(user_id=owner_id, key=key)
        except Exception:
            facts = []
        current = next((f.get("value") for f in facts if f.get("value")), None)

        if current is None:
            return ActionPreview.invalid(f"No memory found for key '{key}'.")

        return ActionPreview(
            "You are about to permanently DELETE this stored memory:\n\n"
            f"  {key}: {current}\n\n"
            "This cannot be undone.",
            {"key": key},
        )

    return {
        "description": (
            "Delete a specific stored preference by key. "
            "Use when user says 'forget my preference for X'. Args: key (str)."
        ),
        "callable": tool_forget_preference,
        "scope": Scope.PROFILE_WRITE.value,
        # Removes a stored fact with no undo.
        "effect": Effect.DESTRUCTIVE,
        "preview": preview_forget_preference,
    }


# ── The registry ─────────────────────────────────────────────────────────────

ToolBuilder = Callable[[str], Dict[str, Any]]

CONFIRMABLE_TOOLS: Dict[str, ToolBuilder] = {
    "send_email": build_send_email_tool,
    "forget_preference": build_forget_preference_tool,
}
"""Every tool that may be held for confirmation, by name.

A stored action naming anything outside this mapping cannot be reconstructed,
and therefore cannot execute. That is the intended failure: an unknown name is
either a tool that has been removed, or a row nobody should trust."""


# ── The vocabulary of a consequential capability ─────────────────────────────
#
# Two questions have to be answerable *before* any model runs, and both are
# about words rather than about tools — which is why they live here, next to the
# tools whose words they are, rather than in a router:
#
#     is this utterance asking for one of these capabilities?
#     does this sentence claim one of them already happened?
#
# The first decides whether a turn may be answered on the tool-free streaming
# path at all. The second is what makes a fabricated success detectable: no
# reasoning loop and no streamed reply can ever have executed one of these —
# `ActionGateway.confirm_and_execute` is the only caller — so a sentence that
# claims one did is false by construction, not merely suspicious.
#
# The obvious objection is that this is a keyword list, and this codebase is
# rightly sceptical of those: `query_intent` exists precisely because "each new
# phrasing is a new pattern and the list is never finished". The difference is
# what is being enumerated. A list of ways to ask "am I a good fit" is open —
# people invent new ones. This enumerates something *closed*: the two
# consequential capabilities this build actually ships. A phrasing gap here
# costs a turn its escalation, and the gap is bounded by a registry that
# `test_every_confirmable_tool_declares_its_vocabulary` keeps aligned. It is
# also not the only defence — the category classifier escalates action-shaped
# turns on its own, and the completion-claim check catches what both miss.
#
# Nothing here is a security boundary by itself. The boundary is the gateway;
# this decides which door a turn goes through to reach it.


def _normalise(text: str) -> str:
    """Lowercase, apostrophes dropped, punctuation flattened.

    `@` and `.` survive so an address stays one token: "send it to a@b.com"
    should read as a send, not as three unrelated words.
    """
    lowered = (text or "").lower().replace("’", "'").replace("'", "")
    cleaned = re.sub(r"[^a-z0-9@.\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


# Openers that make a sentence a question about stored data rather than an
# instruction to change the world. "show my sent emails" and "did I email her"
# name the capability without requesting it, and gating those would put a
# confirmation in front of a read — the thing requirement 13 forbids.
#
# Deliberately does not list `can`/`could`/`would`: "can you send this email" is
# a request wearing a question mark, and treating it as a read would be exactly
# the miss this exists to prevent.
_READ_ONLY_LEAD = re.compile(
    r"^(?:"
    r"what|which|who|whom|whose|when|where|why"
    r"|how\s+(?:many|much|often|do|does|did|long)"
    r"|show|list|display|read|check|find|search|see|view|tell\s+me\s+(?:what|which|about)"
    r"|look\s+(?:up|at)"
    r"|did|do|does|have|has|had|was|were|is|are|am"
    r")\b"
)


def _any(*patterns: str) -> Tuple[re.Pattern, ...]:
    return tuple(re.compile(p) for p in patterns)


# How a user asks for the capability.
#
# Every pattern requires the utterance to *name* what is being sent — an email
# noun, or an address that names itself. A bare demonstrative does not qualify,
# and that exclusion is deliberate rather than an accident of drafting: "send
# this to him" names no capability, and treating it as one would collide with
# two properties that already hold. It is an AMBIGUOUS_ACTION, whose honest
# answer is a clarifying question rather than a graph invocation with nothing
# more to go on; and "send it" is the affirmative that approves a *pending*
# action, which must not be re-read as a request to prepare another.
#
# What covers the gap is not this list. A turn that slips through reaches a path
# holding no tools, where `claims_consequential_completion` catches it at the
# only point where it could do harm — the moment it claims to have happened.
_SEND_EMAIL_REQUESTS = _any(
    # "send this email", "send the mail", "send that message out"
    r"\bsend\s+(?:(?:this|that|the|an|a|my|out|off|another|again)\s+)*"
    r"(?:e\s?mail|mail|message|msg|note|reply|invite|invitation)\b",
    # "send it to bob@example.com" — the address is the object naming itself.
    r"\bsend\s+.{0,40}?\bto\s+[a-z0-9._%+-]+@",
    # "email it to bob", "email him", "email bob@example.com"
    r"\b(?:e\s?mail|mail)\s+(?:it|this|that|him|her|them|to\b|out\b|[a-z0-9._%+-]+@)",
    # Imperative "email Alice about the meeting" — anchored, so "my email
    # address is ..." and "what is my email id" cannot reach it.
    r"^(?:please\s+|kindly\s+|(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:e\s?mail|mail)\s+(?!address\b|id\b|ids\b|account\b)[a-z]",
    r"\b(?:shoot|fire|ping|drop|dash)\s+(?:\w+\s+)?(?:an?\s+)?(?:e\s?mail|message|note|line)\b",
    r"\bforward\s+(?:this|that|it|the|my)\b",
    r"\breply\s+to\b",
    r"\bresend\b",
)

# How a sentence claims it already happened. Anchored on the *speaker* — "I
# sent", "it has been sent" — never on the user: "you sent three emails last
# week" is a true statement about the store and must survive.
_SEND_EMAIL_CLAIMS = _any(
    r"\b(?:i|ive|i\s+have|we|weve|we\s+have)\s+(?:just\s+|now\s+|already\s+)*"
    r"(?:sent|emailed|e\s?mailed|mailed|forwarded|delivered|fired\s+off)\b",
    r"\b(?:e\s?mail|mail|message|msg|note|reply|invite|it|that|this)\s+"
    r"(?:has|have|is|was|were)\s+(?:now\s+)?been\s+"
    r"(?:sent|delivered|forwarded|emailed|dispatched)\b",
    r"\b(?:e\s?mail|mail|message|msg)\s+sent\b",
    r"\bsent\s+(?:it|this|that|the\s+e\s?mail|your\s+e\s?mail|the\s+message)\b",
    r"\bhas\s+been\s+(?:sent|delivered|dispatched)\b",
    r"\b(?:is|its)\s+on\s+its\s+way\b",
    r"^(?:sent|done\s+sent|all\s+sent)\b",
)

_FORGET_PREFERENCE_REQUESTS = _any(
    r"\bforget\s+(?:my|the|that|this|about|everything|all)\b",
    r"\b(?:delete|remove|erase|wipe|clear|drop|purge)\s+"
    r"(?:my|the|this|that|all|every)\s+(?:\w+\s+){0,3}?"
    r"(?:preference|preferences|memory|memories|fact|facts|note|notes|data"
    r"|record|records|setting|settings|entry|entries)\b",
    r"\b(?:delete|remove|erase|forget)\s+what\s+(?:you|i)\b",
    r"\bstop\s+remembering\b",
    r"\bdont\s+remember\b",
    r"\bunremember\b",
)

_FORGET_PREFERENCE_CLAIMS = _any(
    r"\b(?:i|ive|i\s+have|we|weve|we\s+have)\s+(?:just\s+|now\s+|already\s+)*"
    r"(?:deleted|removed|erased|forgotten|forgot|wiped|cleared|purged)\b",
    r"\b(?:it|that|this|memory|preference|fact|record|note|entry|setting)\s+"
    r"(?:has|have|is|was|were)\s+(?:now\s+)?been\s+"
    r"(?:deleted|removed|erased|forgotten|wiped|cleared|purged)\b",
    r"\bhas\s+been\s+(?:deleted|removed|erased|forgotten|wiped|purged)\b",
    r"\bno\s+longer\s+(?:stored|remembered|saved|in\s+my\s+memory)\b",
    r"^(?:deleted|removed|forgotten|erased|wiped)\b",
)


CONSEQUENTIAL_VOCABULARY: Dict[str, Dict[str, Tuple[re.Pattern, ...]]] = {
    "send_email": {
        "requests": _SEND_EMAIL_REQUESTS,
        "claims": _SEND_EMAIL_CLAIMS,
    },
    "forget_preference": {
        "requests": _FORGET_PREFERENCE_REQUESTS,
        "claims": _FORGET_PREFERENCE_CLAIMS,
    },
}
"""One entry per confirmable tool, keyed identically to `CONFIRMABLE_TOOLS`.

Kept in the same module as the tools themselves so that adding a consequential
capability without saying how it is asked for — or how it would be falsely
claimed — is a test failure rather than a silent hole."""


def names_consequential_capability(text: str) -> Optional[str]:
    """
    The tool this utterance is asking for, or None.

    Returns a name rather than a boolean so a caller can say *which* capability
    a turn is reaching for, which is what a log needs to be worth reading.

    A read-only question that merely mentions a capability — "show my sent
    emails", "did I email her yesterday" — is not a request for it. That
    exclusion is the only reason this can be consulted on every turn without
    putting a confirmation in front of a lookup.
    """
    cleaned = _normalise(text)
    if not cleaned or _READ_ONLY_LEAD.match(cleaned):
        return None
    for tool, vocabulary in CONSEQUENTIAL_VOCABULARY.items():
        if any(pattern.search(cleaned) for pattern in vocabulary["requests"]):
            return tool
    return None


CLAIM_WINDOW_CHARS = 400
"""How much of a growing reply the streaming check needs to look at.

Every claim pattern spans well under a hundred characters, so a window this
size contains any of them whole. It exists because the check runs once per
token: over a full-length reply, re-scanning the whole accumulation each time
is quadratic and cost a quarter-second of CPU on the one path whose entire
purpose is low latency."""


def claims_consequential_completion(
    text: str, *, tail_only: bool = False
) -> Optional[str]:
    """
    The capability this sentence claims to have completed, or None.

    Callers use this only where the answer is already known: inside a reasoning
    loop, and on the tool-free streaming path. Neither can execute a confirmable
    tool — the gateway is the only thing that can — so a match here is not a
    suspicion, it is a false statement caught before it is delivered.

    Unanchored to the reader on purpose: this asks whether the *assistant*
    claimed the act, so "you sent three emails last week" does not match while
    "I've sent it" and "the email has been sent" do.

    `tail_only` drops the patterns anchored to the start of the reply, and is
    for callers passing a *slice* of a longer text: in a slice, `^` means "the
    start of this window", which is an arbitrary point mid-sentence and would
    match things that are not at the start of anything. Those patterns describe
    a bare "Sent." opening a reply, which is decided by its first few tokens and
    is already covered while the reply is still short enough to scan whole.
    """
    cleaned = _normalise(text)
    if not cleaned:
        return None
    for tool, vocabulary in CONSEQUENTIAL_VOCABULARY.items():
        for pattern in vocabulary["claims"]:
            if tail_only and pattern.pattern.startswith("^"):
                continue
            if pattern.search(cleaned):
                return tool
    return None


def claims_completion_in_stream(accumulated: str) -> Optional[str]:
    """
    The same question, asked once per token of a reply still being written.

    Scans the whole reply while it is short and a sliding window once it is
    not. The window slides by one token at a time and is far wider than any
    pattern, so every claim still falls inside one of them whole — this bounds
    the cost without narrowing what is caught.
    """
    if len(accumulated) <= CLAIM_WINDOW_CHARS:
        return claims_consequential_completion(accumulated)
    return claims_consequential_completion(
        accumulated[-CLAIM_WINDOW_CHARS:], tail_only=True
    )


def resolve_confirmable_tool(tool: str, owner_id: str) -> Optional[Dict[str, Any]]:
    """
    Rebuild the registry entry for a stored action.

    Returns None when the name is unknown, which the gateway treats as a
    refusal. Nothing here is derived from the stored row except the name and
    the owner — the callable, the effect and the scope all come from this
    module, so a tampered row cannot change what a tool *is*, only which of the
    known tools it claims to be.
    """
    name = (tool or "").strip()
    builder = CONFIRMABLE_TOOLS.get(name)

    # MCP tools are rebuilt from their own local config rather than being
    # listed here, because which ones exist depends on which servers an
    # operator enabled. The authority is the same either way: the name is all
    # that comes from the stored row, and the callable, effect and scope are
    # reconstructed from configuration this application controls.
    if builder is None and name.startswith(_MCP_PREFIX):
        try:
            from app.mcp.registry import confirmable_builders

            builder = confirmable_builders().get(name)
        except Exception as exc:
            logger.error("Could not resolve MCP tool '%s': %s", name, exc)
            return None

    if builder is None:
        logger.error(
            "Cannot reconstruct action: '%s' is not a known confirmable tool", tool
        )
        return None
    try:
        return builder(owner_id)
    except Exception as exc:
        logger.error("Building '%s' for reconstruction failed: %s", tool, exc)
        return None


__all__ = [
    "CONFIRMABLE_TOOLS",
    "CONSEQUENTIAL_VOCABULARY",
    "ToolBuilder",
    "build_forget_preference_tool",
    "build_send_email_tool",
    "CLAIM_WINDOW_CHARS",
    "claims_completion_in_stream",
    "claims_consequential_completion",
    "names_consequential_capability",
    "resolve_confirmable_tool",
]
