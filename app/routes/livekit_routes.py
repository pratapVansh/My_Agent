"""
LiveKit WebRTC session endpoints.

A LiveKit token grants the holder the right to join a room, publish audio, and
be served by a backend worker that spends Groq/Deepgram/Cartesia credits. It is
therefore issued only to an authenticated caller, for that caller's own room.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import asyncio
import logging

from app.auth.dependencies import require_scope
from app.auth.models import Principal, Scope
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class LiveKitTokenRequest(BaseModel):
    # Both fields are retained for wire compatibility and both are ignored:
    # the identity and the room are derived from the authenticated session.
    user_id: Optional[str] = Field(default=None, description="Deprecated and ignored")
    room_name: Optional[str] = Field(default=None, description="Deprecated and ignored")
    # The browser's conversation id. Voice turns join the thread the user can
    # see rather than a parallel one keyed by room — without this, speaking and
    # typing produce two different conversations and a page reload restores the
    # empty one.
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation thread to attach voice turns to"
    )


class LiveKitTokenResponse(BaseModel):
    token: str
    url: str
    room_name: str


# Room -> worker task. Process-local; see docs/AUDIT_REPORT.md (M15) for the
# shared-state requirement before running more than one instance.
active_workers: Dict[str, asyncio.Task] = {}

# (room_name, identity) -> conversation_id the caller is currently viewing.
#
# Written here at token time and read by the voice worker when it builds a
# participant's state. A registry rather than a worker argument because the
# worker outlives a single connection: the same room is reused when the user
# reconnects, and the conversation they are looking at may have changed (New
# Chat) since the worker started.
voice_conversation_bindings: Dict[tuple, str] = {}


def bind_voice_conversation(room_name: str, identity: str, conversation_id: str) -> None:
    voice_conversation_bindings[(room_name, identity)] = conversation_id


def resolve_voice_conversation(room_name: str, identity: str) -> Optional[str]:
    return voice_conversation_bindings.get((room_name, identity))


def room_name_for(principal: Principal) -> str:
    """
    Derive the room from the verified identity.

    Deriving rather than accepting a client-supplied name is what prevents one
    caller from joining another caller's live voice session.
    """
    return f"voice-{principal.user_id}"


@router.post("/token", response_model=LiveKitTokenResponse)
async def generate_livekit_token(
    request: LiveKitTokenRequest,
    principal: Principal = Depends(require_scope(Scope.VOICE)),
):
    """
    Mint a LiveKit access token for the caller's own voice room.

    Starts the backend voice worker for that room if one is not already running.
    """
    api_key = settings.livekit_api_key
    api_secret = settings.livekit_api_secret
    ws_url = settings.livekit_url

    if not api_key or not api_secret or not ws_url:
        logger.error("LiveKit credentials not configured")
        raise HTTPException(status_code=503, detail="Voice is not configured on this server.")

    room_name = room_name_for(principal)

    # Reject an attempt to steer the session at someone else's room outright
    # rather than silently substituting the correct one.
    if request.room_name and request.room_name != room_name:
        logger.warning(
            "Rejected LiveKit room mismatch: user=%s requested=%s",
            principal.user_id, request.room_name,
        )
        raise HTTPException(
            status_code=403, detail="You cannot join another user's voice session."
        )
    if request.user_id and request.user_id.strip().lower() != principal.user_id.lower():
        raise HTTPException(
            status_code=403, detail="You cannot access another user's data."
        )

    # Bind before the worker starts, so the first spoken turn already lands in
    # the right thread. Ownership is enforced on write by append_turn, which
    # scopes every turn to the authenticated owner — a caller cannot bind their
    # voice to somebody else's conversation.
    if request.conversation_id and request.conversation_id.strip():
        bind_voice_conversation(
            room_name, principal.user_id, request.conversation_id.strip()
        )

    try:
        from livekit import api
        from app.livekit_worker import main as worker_main

        token = api.AccessToken(api_key, api_secret) \
            .with_identity(principal.user_id) \
            .with_name(f"User {principal.user_id}") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )).to_jwt()

        existing = active_workers.get(room_name)
        if existing is None or existing.done():
            def deregister() -> None:
                """Free the room slot the instant the worker starts shutting down.

                Called by the worker before it awaits anything in teardown, so a
                token request that races the shutdown starts a replacement rather
                than handing the caller a room whose worker is on its way out —
                which presents to the user as the assistant never answering.
                """
                active_workers.pop(room_name, None)
                logger.info("Voice worker for room %s deregistered", room_name)

            async def run_worker():
                try:
                    # Scopes travel into the worker so voice turns enforce the
                    # same tool permissions as HTTP requests.
                    await worker_main(
                        room_name,
                        scopes=frozenset(principal.scopes),
                        on_shutdown=deregister,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Voice worker for room %s failed: %s", room_name, e, exc_info=True)
                finally:
                    deregister()

            active_workers[room_name] = asyncio.create_task(
                run_worker(), name=f"voice-worker-{room_name}"
            )
            logger.info("Started voice worker for room %s (user=%s)", room_name, principal.user_id)

        return LiveKitTokenResponse(token=token, url=ws_url, room_name=room_name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error generating LiveKit token for user=%s: %s",
            principal.user_id, e, exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Could not start a voice session.")
