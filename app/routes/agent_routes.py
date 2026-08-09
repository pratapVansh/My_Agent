"""
API routes for multi-agent system.
REST handles normal operations, while WebSocket is reserved for response streaming.
"""
import asyncio
import logging
import re
import uuid
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.config import settings
from app.agents.workflow import run_workflow
from app.agents.streaming_workflow import run_streaming_workflow
from app.tools.job_search_tool import job_search_tool
from app.tools.email_draft_tool import email_draft_tool
from app.tools.attendance_tool import attendance_tool
from app.tools.timetable_tool import timetable_tool, TimetableInput
from app.tools.timetable_pdf_parser import timetable_pdf_parser
from app.memory.memory_manager import memory_manager
from app.domain.academic import academic_repository
from app.services.url_guard import UnsafeURLError
from app.auth.dependencies import (
    authenticate_websocket,
    require_owner,
    require_scope,
    resolve_user_id,
)
from app.auth.models import Principal, Scope
from PyPDF2 import PdfReader
import io


logger = logging.getLogger(__name__)

router = APIRouter()


def internal_error(message: str, exc: Exception, **context) -> HTTPException:
    """
    Log an exception server-side and return a safe 500 for the client.

    Raw exception text can carry connection strings, hostnames, file paths, and
    library internals, so it is logged rather than returned. An error_id ties
    the client-visible response back to the log line.
    """
    error_id = uuid.uuid4().hex[:12]
    logger.error(
        "%s [error_id=%s]%s: %s",
        message,
        error_id,
        "".join(f" {k}={v}" for k, v in context.items()),
        exc,
        exc_info=True,
    )
    detail: Dict[str, Any] = {"message": message, "error_id": error_id}
    if settings.is_development:
        detail["debug"] = str(exc)
    return HTTPException(status_code=500, detail=detail)


async def read_upload_within_limit(file: UploadFile) -> bytes:
    """
    Read an uploaded file, aborting once it exceeds the configured cap.

    Streams in chunks rather than calling file.read() outright: an unbounded
    read pulls the whole body into memory before any size check can reject it.
    """
    max_bytes = settings.max_upload_bytes
    chunks: List[bytes] = []
    total = 0

    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File is too large. Maximum upload size is "
                    f"{max_bytes // (1024 * 1024)} MB."
                ),
            )
        chunks.append(chunk)

    return b"".join(chunks)


# User isolation is now enforced by authentication, not by string validation:
# identity comes from the verified JWT (app/auth/dependencies.resolve_user_id),
# and the only operator-supplied identity — OWNER_USER_ID — is format-checked
# at startup by Settings.validate_auth_config().


class Message(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class AgentRequest(BaseModel):
    """Request model for agent query."""
    query: str = Field(..., description="User's query", min_length=1, max_length=8000)
    # Retained for wire compatibility only. Identity comes from the verified
    # access token; supplying a different value here is rejected with 403.
    user_id: Optional[str] = Field(
        default=None,
        description="Deprecated and ignored — identity is taken from your session.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier (generated if not provided)"
    )
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Optional conversation history"
    )
    output_mode: str = Field(
        default="user",
        description="Speech output mode: user or recruiter"
    )


class AgentResponse(BaseModel):
    """Response model from agent system."""
    display_text: str = Field(..., description="Text formatted for display")
    speech_text: str = Field(..., description="Text formatted for speech")
    metadata: Dict[str, Any] = Field(
        default={},
        description="Additional execution metadata"
    )


class JobSearchRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Deprecated and ignored")
    query: str = Field(..., min_length=1, max_length=2000, description="Job search query")
    location: Optional[str] = Field(default=None, description="Optional location")
    max_results: int = Field(default=10, ge=1, le=25)
    min_score: float = Field(default=0.2, ge=0.0, le=1.0)


class EmailDraftRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Deprecated and ignored")
    query: str = Field(..., min_length=1, max_length=4000, description="Email draft request")
    tone: str = Field(default="professional", description="Email tone")
    recipient_name: Optional[str] = Field(default="", description="Recipient name")


class AttendanceScrapeRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Deprecated and ignored")
    erp_url: str = Field(..., min_length=1, description="ERP login URL")
    username: str = Field(..., min_length=1, description="ERP username")
    password: str = Field(..., min_length=1, description="ERP password")
    selectors: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional CSS selectors for ERP fields and rows"
    )


class TimetableEntryRequest(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6)
    start_time: str = Field(..., description="HH:MM or HH:MM:SS")
    end_time: str = Field(..., description="HH:MM or HH:MM:SS")
    subject: str = Field(..., min_length=1)
    location: Optional[str] = None
    instructor: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TimetableStoreRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Deprecated and ignored")
    entries: List[TimetableEntryRequest] = Field(..., min_length=1, max_length=500)


class TimetableSuggestRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Deprecated and ignored")
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    low_attendance_threshold: float = Field(default=75.0, ge=0.0, le=100.0)


@router.post("/query", response_model=AgentResponse)
async def agent_query(
    request: AgentRequest,
    principal: Principal = Depends(require_scope(Scope.CHAT)),
):
    """
    Process a user query through the multi-agent system with memory.

    The system will:
    1. Retrieve memory context (chat history, preferences, long-term data)
    2. Analyze the query (Planner Agent)
    3. Route to appropriate specialist (Task Agent)
    4. Format the response (Response Agent)
    5. Save interaction to memory

    Returns structured response with display and speech versions.
    """
    try:
        # Identity comes from the verified token; a conflicting body value 403s.
        user_id = resolve_user_id(principal, request.user_id)
        logger.info(
            "Agent query received for user=%s role=%s", user_id, principal.role.value
        )

        # Convert conversation history to dict format
        history = None
        if request.conversation_history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Run workflow with memory integration. Scopes travel with the request
        # so specialist agents can filter their own tools — without this a
        # guest's chat could still reach owner-only tools such as send_email.
        # A guest reads the owner's *public* memory rather than their own empty
        # partition — the fix for the recruiter view being unable to discuss
        # the very work it exists to present.
        from app.memory.scope import resolve_retrieval_scope

        scope = resolve_retrieval_scope(principal)

        result = await run_workflow(
            user_input=request.query,
            # The caller always writes under their *own* identity. Only reads
            # are redirected: a guest must never write into the owner's memory.
            user_id=user_id,
            session_id=request.session_id,
            conversation_history=history,
            output_mode=request.output_mode,
            scopes=principal.scopes,
            memory_owner_id=scope.owner_id,
            memory_visibilities=scope.visibilities,
        )

        # Extract response
        display_text = result.get("display_text", "No response generated")
        speech_text = result.get("speech_text", "No response generated")

        # Build metadata
        metadata = {
            "user_id": result.get("user_id"),
            "session_id": result.get("session_id"),
            "detected_intent": result.get("detected_intent"),
            "selected_agent": result.get("selected_agent"),
            "execution_path": result.get("execution_path", []),
            "memory_used": result.get("memory_prompt") is not None and len(result.get("memory_prompt", "")) > 0,
            "success": result.get("error") is None
        }

        if result.get("error"):
            metadata["error"] = result["error"]

        return AgentResponse(
            display_text=display_text,
            speech_text=speech_text,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Agent execution failed", e, user_id=principal.user_id)


@router.get("/agents")
async def list_agents():
    """
    List available agents in the system.

    Returns information about all agents and their capabilities.
    """
    return {
        "agents": [
            {
                "name": "planner",
                "type": "router",
                "description": "Analyzes queries and routes to appropriate agent"
            },
            {
                "name": "job",
                "type": "task",
                "description": "Job search, applications, and career guidance"
            },
            {
                "name": "email",
                "type": "task",
                "description": "Email composition, management, and organization"
            },
            {
                "name": "academic",
                "type": "task",
                "description": "Academic research, study help, and learning support"
            },
            {
                "name": "profile",
                "type": "task",
                "description": "User profile operations and general assistance"
            },
            {
                "name": "response",
                "type": "formatter",
                "description": "Formats responses for display and speech output"
            }
        ],
        "workflow": "planner -> [task_agent] -> response"
    }


@router.post("/tools/job-search")
async def job_search(
    request: JobSearchRequest,
    principal: Principal = Depends(require_scope(Scope.JOBS_SEARCH)),
):
    """Search jobs using Tavily, then filter/rank with memory personalization."""
    try:
        result = await job_search_tool.search_jobs(
            user_id=resolve_user_id(principal, request.user_id),
            query=request.query,
            location=request.location,
            max_results=request.max_results,
            min_score=request.min_score,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Job search failed", e, user_id=principal.user_id)


@router.post("/tools/email-draft")
async def email_draft(
    request: EmailDraftRequest,
    principal: Principal = Depends(require_scope(Scope.EMAIL_DRAFT)),
):
    """Create a personalized email draft with RAG context. Draft only, never sent."""
    try:
        result = await email_draft_tool.draft_email(
            user_id=resolve_user_id(principal, request.user_id),
            query=request.query,
            tone=request.tone,
            recipient_name=request.recipient_name or "",
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Email draft failed", e, user_id=principal.user_id)


@router.post("/tools/attendance/scrape")
async def scrape_attendance(
    request: AttendanceScrapeRequest,
    principal: Principal = Depends(require_scope(Scope.TOOLS_SCRAPE)),
):
    """
    Scrape attendance from an ERP portal with Playwright and store in PostgreSQL.

    The target URL is validated against the SSRF guard before any request is
    made, so this endpoint cannot be used to reach internal services or cloud
    metadata endpoints on the server's behalf.
    """
    user_id = resolve_user_id(principal, request.user_id)
    try:
        result = await attendance_tool.scrape_and_store(
            user_id=user_id,
            erp_url=request.erp_url,
            username=request.username,
            password=request.password,
            selectors=request.selectors,
        )
        return result
    except UnsafeURLError as e:
        # Client error, not a server fault — and the message is safe to return.
        logger.warning("Blocked attendance scrape for user=%s: %s", user_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        # Never interpolate the exception here: this call site holds ERP
        # credentials, and a driver error can echo request context back.
        raise internal_error("Attendance scrape failed", e, user_id=user_id)


@router.post("/tools/timetable/store")
async def store_timetable(
    request: TimetableStoreRequest,
    principal: Principal = Depends(require_scope(Scope.TIMETABLE_WRITE)),
):
    """Store user-input timetable entries in PostgreSQL."""
    try:
        entries = [
            TimetableInput(
                day_of_week=e.day_of_week,
                start_time=e.start_time,
                end_time=e.end_time,
                subject=e.subject,
                location=e.location,
                instructor=e.instructor,
                metadata=e.metadata,
            )
            for e in request.entries
        ]
        result = await timetable_tool.store_timetable(
            user_id=resolve_user_id(principal, request.user_id), entries=entries
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Timetable store failed", e, user_id=principal.user_id)


@router.post("/tools/timetable/suggest")
async def suggest_classes(
    request: TimetableSuggestRequest,
    principal: Principal = Depends(require_scope(Scope.TIMETABLE_READ)),
):
    """Suggest classes prioritized by low attendance and timetable schedule."""
    try:
        result = await timetable_tool.suggest_classes(
            user_id=resolve_user_id(principal, request.user_id),
            day_of_week=request.day_of_week,
            low_attendance_threshold=request.low_attendance_threshold,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Timetable suggestion failed", e, user_id=principal.user_id)


@router.post("/tools/timetable/upload-pdf")
async def upload_timetable_pdf(
    user_id: Optional[str] = Form(default=None, description="Deprecated and ignored"),
    clear_existing: bool = Form(
        default=True,
        description="If true, deactivates your old timetable before storing the new one"
    ),
    file: UploadFile = File(..., description="PDF file of your semester timetable"),
    principal: Principal = Depends(require_scope(Scope.TIMETABLE_WRITE)),
):
    """
    Upload a PDF timetable for the current semester.

    Workflow:
    1. Extract text from the PDF using PyPDF2
    2. Send the text to Groq LLM to parse into structured entries
       (handles any layout: grid, list, weekly view)
    3. Optionally clear the previous timetable (clear_existing=true by default)
    4. Store all parsed entries in PostgreSQL

    Upload a new PDF any time your timetable changes — the old one is safely
    soft-deleted (not permanently removed) before the new one is stored.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    user_id = resolve_user_id(principal, user_id)
    pdf_bytes = await read_upload_within_limit(file)

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Parse the PDF into structured entries
    parse_result = await timetable_pdf_parser.parse(
        pdf_bytes=pdf_bytes,
        filename=file.filename,
    )

    if not parse_result["success"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not extract any timetable entries from the PDF",
                "parse_notes": parse_result["parse_notes"],
                "raw_text_preview": parse_result["raw_text_preview"],
                "pages": parse_result["pages"],
            },
        )

    # Clear old timetable if requested
    cleared_count = 0
    if clear_existing:
        cleared_count = await academic_repository.clear_timetable(user_id=user_id)

    # Store all parsed entries
    entries = parse_result["entries"]
    store_result = await timetable_tool.store_timetable(user_id=user_id, entries=entries)

    return {
        "success": True,
        "filename": file.filename,
        "pages_in_pdf": parse_result["pages"],
        "entries_parsed": parse_result["entry_count"],
        "entries_stored": store_result["stored_count"],
        "old_entries_cleared": cleared_count,
        "parse_notes": parse_result["parse_notes"],
        "stored_ids": store_result["stored_ids"],
        "message": (
            f"Successfully imported {store_result['stored_count']} classes from "
            f"'{file.filename}'. "
            + (f"Replaced {cleared_count} old entries." if cleared_count else "")
        ),
    }


@router.post("/memory/upload-pdf")
async def upload_pdf_document(
    user_id: Optional[str] = Form(default=None, description="Deprecated and ignored"),
    document_type: str = Form(default="general"),
    file: UploadFile = File(...),
    principal: Principal = Depends(require_scope(Scope.MEMORY_WRITE)),
):
    """
    Upload a PDF document to user's long-term memory.

    This endpoint extracts text from PDFs and stores them in the
    Qdrant vector database for semantic search during conversations.

    Args:
        user_id: User identifier
        document_type: Type of document (resume, projects, skills, notes, research, general)
        file: PDF file to upload

    Returns:
        Document ID and storage confirmation
    """
    try:
        user_id = resolve_user_id(principal, user_id)
        logger.info("Document upload started for user=%s file=%s", user_id, file.filename)

        # Validate file type
        if not (file.filename or "").lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read PDF content, rejecting oversized uploads before buffering them
        content = await read_upload_within_limit(file)

        # Extract text from PDF
        try:
            pdf_reader = PdfReader(io.BytesIO(content))
            text_parts = []

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)

            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

        except HTTPException:
            raise
        except Exception as e:
            logger.warning("PDF text extraction failed for user=%s: %s", user_id, e)
            raise HTTPException(
                status_code=400,
                detail="Could not read this PDF. It may be encrypted, corrupted, or image-only.",
            )

        # Store in long-term memory
        metadata = {
            "source_file": file.filename,
            "document_type": document_type,
            "content_type": "application/pdf",
            "num_pages": len(pdf_reader.pages)
        }

        doc_id = await memory_manager.store_resume(
            user_id=user_id,
            resume_text=full_text,
            metadata=metadata
        )

        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "document_type": document_type,
            "pages_extracted": len(pdf_reader.pages),
            "text_length": len(full_text),
            "message": f"Successfully stored {file.filename} in long-term memory"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("PDF upload failed", e, user_id=user_id)


@router.post("/memory/upload-text")
async def upload_text_document(
    user_id: Optional[str] = Form(default=None, description="Deprecated and ignored"),
    document_type: str = Form(default="general"),
    text_content: str = Form(...),
    document_name: str = Form(default="untitled"),
    principal: Principal = Depends(require_scope(Scope.MEMORY_WRITE)),
):
    """
    Upload plain text to user's long-term memory.

    Use this for copy-pasted content, notes, or any text data.

    Args:
        user_id: User identifier
        document_type: Type of document (resume, projects, skills, notes, research, general)
        text_content: The text content to store
        document_name: Name/title for the document

    Returns:
        Document ID and storage confirmation
    """
    try:
        user_id = resolve_user_id(principal, user_id)
        logger.info("Text upload started for user=%s name=%s", user_id, document_name)

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")

        if len(text_content) > settings.max_text_upload_chars:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Text is too long. Maximum is "
                    f"{settings.max_text_upload_chars:,} characters."
                ),
            )

        metadata = {
            "document_name": document_name,
            "document_type": document_type,
            "content_type": "text/plain"
        }

        doc_id = await memory_manager.store_resume(
            user_id=user_id,
            resume_text=text_content,
            metadata=metadata
        )

        return {
            "success": True,
            "document_id": doc_id,
            "document_name": document_name,
            "document_type": document_type,
            "text_length": len(text_content),
            "message": f"Successfully stored '{document_name}' in long-term memory"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Text upload failed", e, user_id=user_id)


class ProfileFactRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Deprecated and ignored")
    key: str = Field(..., min_length=1, max_length=255, description="Fact key, e.g. 'preferred_tone'")
    value: str = Field(..., min_length=1, max_length=4000)
    source: str = Field(default="explicit", description="explicit | inferred")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    consent_level: str = Field(default="explicit")


@router.get("/memory/profile/{user_id}")
async def list_profile_facts(
    user_id: str,
    key: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_scope(Scope.PROFILE_READ)),
):
    """
    List profile facts for the authenticated user. Paginated.

    The path segment is kept for URL compatibility but is not an identity
    source — requesting someone else's id returns 403.
    """
    try:
        user_id = resolve_user_id(principal, user_id)
        facts = await memory_manager.get_profile_facts(user_id=user_id, key=key)
        page = facts[offset:offset + limit]
        return {
            "success": True,
            "user_id": user_id,
            "count": len(page),
            "total": len(facts),
            "limit": limit,
            "offset": offset,
            "facts": page,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not load profile facts", e, user_id=user_id)


@router.post("/memory/profile")
async def save_profile_fact(
    request: ProfileFactRequest,
    principal: Principal = Depends(require_scope(Scope.PROFILE_WRITE)),
):
    """Save or update a user profile fact (upserts on key)."""
    try:
        user_id = resolve_user_id(principal, request.user_id)
        record_id = await memory_manager.save_profile_fact(
            user_id=user_id,
            key=request.key.strip().lower(),
            value=request.value,
            source=request.source,
            confidence=request.confidence,
            consent_level=request.consent_level,
        )
        if not record_id:
            # save_profile_fact returns "" when the value is rejected as
            # credential-shaped or fails the inferred-confidence policy.
            return {
                "success": False,
                "key": request.key,
                "message": "This value was not stored (it looks like sensitive credential data).",
            }
        return {"success": True, "id": record_id, "key": request.key, "value": request.value}
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not save profile fact", e, user_id=principal.user_id)


@router.delete("/memory/profile/{user_id}/{key}")
async def forget_profile_fact(
    user_id: str,
    key: str,
    principal: Principal = Depends(require_scope(Scope.PROFILE_WRITE)),
):
    """Delete a single profile fact by key."""
    try:
        user_id = resolve_user_id(principal, user_id)
        deleted = await memory_manager.forget_profile_fact(user_id=user_id, key=key)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"No profile fact found for key '{key}'")
        return {"success": True, "deleted_key": key}
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not delete profile fact", e, user_id=user_id, key=key)


@router.delete("/memory/profile/{user_id}")
async def forget_all_profile(
    user_id: str,
    principal: Principal = Depends(require_scope(Scope.PROFILE_WRITE)),
):
    """Delete ALL profile facts for a user."""
    try:
        user_id = resolve_user_id(principal, user_id)
        count = await memory_manager.forget_all_profile(user_id=user_id)
        logger.info("Erased all profile facts for user=%s (count=%d)", user_id, count)
        return {"success": True, "deleted_count": count}
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not delete profile facts", e, user_id=user_id)


@router.get("/memory/episodes/{user_id}")
async def get_episodes(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    principal: Principal = Depends(require_scope(Scope.PROFILE_READ)),
):
    """Return recent episodic memory entries for the authenticated user."""
    try:
        user_id = resolve_user_id(principal, user_id)
        episodes = await memory_manager.get_recent_episodes(user_id=user_id, limit=limit)
        return {
            "success": True,
            "user_id": user_id,
            "count": len(episodes),
            "limit": limit,
            "episodes": episodes,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not load episodes", e, user_id=user_id)


# ── Memory control plane (Phase 6) ──────────────────────────────────────────
#
# Browse, inspect, correct, and erase what the assistant remembers. A system
# that forms memories automatically is only trustworthy if the user can see
# what it concluded and overrule it — extraction is an LLM, and a wrong memory
# is worse than a missing one.
#
# Reads are visibility-scoped through `resolve_retrieval_scope`, so a guest
# reads the *owner's* public records rather than their own empty partition.
# Every mutation is owner-only.


class MemoryRecordPatch(BaseModel):
    """Fields the owner may correct on a stored memory."""

    content: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pinned: Optional[bool] = Field(
        default=None, description="Pinned memories never decay and are never dropped for budget."
    )
    visibility: Optional[str] = Field(
        default=None, pattern="^(private|shared|public)$",
        description="Marking a record public makes it readable in the recruiter view.",
    )


def _record_payload(record, *, include_owner: bool = False) -> Dict[str, Any]:
    payload = {
        "id": str(record.id),
        "kind": record.kind.value,
        "content": record.content,
        "structured": record.structured,
        "importance": record.importance,
        "confidence": record.confidence,
        "pinned": record.pinned,
        "visibility": record.visibility.value,
        "status": record.status.value,
        "version": record.version,
        "source_type": record.source_type.value,
        "source_ref": record.source_ref,
        "derived_from": [str(x) for x in record.derived_from],
        "occurred_at": record.occurred_at.isoformat() if record.occurred_at else None,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "access_count": record.access_count,
    }
    if include_owner:
        payload["owner_id"] = record.owner_id
    return payload


@router.get("/memory/records")
async def list_memory_records(
    kind: Optional[str] = None,
    status: str = Query(default="active", pattern="^(active|superseded|archived)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_scope(Scope.PROFILE_READ)),
):
    """
    Browse remembered records.

    The owner sees everything of theirs; a guest sees the owner's public
    records. The visibility filter is applied in the query, so a private record
    is never loaded in a request serving a guest.
    """
    try:
        from app.memory.kinds import MemoryKind, RecordStatus
        from app.memory.scope import resolve_retrieval_scope
        from app.memory.stores import postgres_record_store

        scope = resolve_retrieval_scope(principal)

        kinds = None
        if kind:
            try:
                kinds = [MemoryKind(kind.strip().lower())]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown memory kind '{kind}'")

        records = await postgres_record_store.list(
            scope.owner_id,
            kinds=kinds,
            statuses=[RecordStatus(status)],
            visibilities=scope.visibilities,
            limit=limit,
            offset=offset,
        )
        return {
            "success": True,
            "scope": scope.describe(),
            "count": len(records),
            "limit": limit,
            "offset": offset,
            "records": [_record_payload(r) for r in records],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not list memories", e, user_id=principal.user_id)


@router.get("/memory/records/{record_id}")
async def get_memory_record(
    record_id: str,
    principal: Principal = Depends(require_scope(Scope.PROFILE_READ)),
):
    """
    Inspect one memory and where it came from — "why do you know this?".

    Provenance is the feature that makes automatic memory formation
    accountable: the user can trace a conclusion back to the material it was
    distilled from, and correct it when the extractor got it wrong.
    """
    try:
        from uuid import UUID as _UUID

        from app.memory.kinds import Visibility
        from app.memory.scope import resolve_retrieval_scope
        from app.memory.stores import postgres_record_store

        scope = resolve_retrieval_scope(principal)
        try:
            parsed = _UUID(record_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid memory id")

        record = await postgres_record_store.get(scope.owner_id, parsed)
        # A record outside the caller's visibility is reported as absent, not
        # forbidden — a 403 would confirm that the id exists.
        if record is None or (
            scope.visibilities is not None and record.visibility not in scope.visibilities
        ):
            raise HTTPException(status_code=404, detail="Memory not found")

        sources = []
        for source_id in record.derived_from:
            source = await postgres_record_store.get(scope.owner_id, source_id)
            if source is not None:
                sources.append(_record_payload(source))

        return {
            "success": True,
            "record": _record_payload(record),
            "derived_from_records": sources,
            "supersedes": str(record.supersedes_id) if record.supersedes_id else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not load memory", e, user_id=principal.user_id)


@router.patch("/memory/records/{record_id}")
async def update_memory_record(
    record_id: str,
    patch: MemoryRecordPatch,
    principal: Principal = Depends(require_owner),
):
    """
    Correct a memory.

    Editing content creates a new version rather than mutating in place, so the
    correction itself is auditable and the original remains recoverable —
    consistent with how conflicts are handled everywhere else.
    """
    try:
        from uuid import UUID as _UUID

        from app.memory.kinds import Visibility
        from app.memory.stores import postgres_record_store

        try:
            parsed = _UUID(record_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid memory id")

        # Validate the payload before touching the database: an empty patch is
        # a client error regardless of whether the record exists, and there is
        # no reason to pay for a lookup to discover that.
        changes: Dict[str, Any] = {}
        if patch.content is not None:
            changes["content"] = patch.content
        if patch.importance is not None:
            changes["importance"] = patch.importance
        if patch.pinned is not None:
            changes["pinned"] = patch.pinned
        if patch.visibility is not None:
            changes["visibility"] = Visibility(patch.visibility)

        if not changes:
            raise HTTPException(status_code=400, detail="No fields to update")

        record = await postgres_record_store.get(principal.user_id, parsed)
        if record is None:
            raise HTTPException(status_code=404, detail="Memory not found")

        updated = record.superseding(**changes)
        stored = await postgres_record_store.supersede(record, updated)

        return {
            "success": True,
            "record": _record_payload(stored),
            "previous_version": record.version,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not update memory", e, user_id=principal.user_id)


@router.delete("/memory/records/{record_id}")
async def forget_memory_record(
    record_id: str,
    principal: Principal = Depends(require_owner),
):
    """
    Erase a memory permanently, along with anything distilled from it.

    Irreversible and cascading by design. A consolidated memory derived from a
    fact the user asked to delete would otherwise outlive that deletion — the
    fact gone from the store yet still reachable through the summary of it.
    """
    try:
        from uuid import UUID as _UUID

        from app.memory.cognition.maintenance import memory_maintenance
        from app.memory.stores import postgres_record_store

        try:
            parsed = _UUID(record_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid memory id")

        if await postgres_record_store.get(principal.user_id, parsed) is None:
            raise HTTPException(status_code=404, detail="Memory not found")

        deleted = await memory_maintenance.forget_record(principal.user_id, parsed)
        logger.info(
            "Erased memory %s for user=%s (%d records including derived)",
            record_id, principal.user_id, deleted,
        )
        return {"success": True, "deleted_count": deleted, "cascaded": deleted > 1}
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not erase memory", e, user_id=principal.user_id)


@router.delete("/memory/all")
async def erase_all_memory(
    confirm: str = "",
    principal: Principal = Depends(require_owner),
):
    """
    Erase everything the assistant remembers about the caller.

    There was no way to do this. `DELETE /memory/records/{id}` removes one
    memory and `DELETE /memory/profile/{user_id}` removes profile facts, but
    nothing cleared the résumé chunks, conversation history, episodes, tool
    memory or the conversational vector store — so "delete my data" could only
    be answered partially, and the internal helper that claimed to do it
    covered one store out of nine.

    Requires `?confirm=erase`: this is irreversible, cascading, and cannot be
    triggered by a stray DELETE.

    Reports per-store results. A partial erasure returns 207 rather than 200,
    because telling the user their data is gone when some of it survives is the
    same false statement as reporting NO_DATA after a failed lookup.
    """
    if confirm != "erase":
        raise HTTPException(
            status_code=400,
            detail="Irreversible operation. Repeat the request with ?confirm=erase",
        )

    try:
        from app.memory.erasure import memory_erasure

        report = await memory_erasure.erase_owner(principal.user_id)
        payload = {"success": report.complete, **report.summary()}

        if not report.complete:
            logger.error(
                "Partial erasure for user=%s; data survives in: %s",
                principal.user_id, report.failed_stores,
            )
            return JSONResponse(status_code=207, content=payload)

        logger.info(
            "Erased all memory for user=%s (%d rows across %d stores)",
            principal.user_id, report.deleted, len(report.results),
        )
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error("Could not erase memory", e, user_id=principal.user_id)


@router.get("/memory/export")
async def export_memory(
    principal: Principal = Depends(require_owner),
):
    """
    Export everything the assistant remembers, as JSON.

    Owner-only and unfiltered by status: an export that silently omitted
    archived or superseded records would misrepresent what is actually held.
    """
    try:
        from app.memory.kinds import RecordStatus
        from app.memory.stores import postgres_record_store

        every_status = [
            RecordStatus.ACTIVE, RecordStatus.SUPERSEDED,
            RecordStatus.ARCHIVED, RecordStatus.DELETED,
        ]
        records: List[Any] = []
        offset = 0
        while True:
            batch = await postgres_record_store.list(
                principal.user_id, statuses=every_status, limit=500, offset=offset
            )
            if not batch:
                break
            records.extend(batch)
            offset += len(batch)
            if len(batch) < 500:
                break

        return {
            "success": True,
            "owner_id": principal.user_id,
            "count": len(records),
            "records": [_record_payload(r, include_owner=True) for r in records],
        }
    except Exception as e:
        raise internal_error("Could not export memory", e, user_id=principal.user_id)


# ── Conversations (Phase 4) ─────────────────────────────────────────────────
#
# A conversation is addressable and resumable. The browser persists its id, so
# a refresh reopens the same thread instead of silently starting a new one —
# which is what used to happen, because the session id was regenerated on every
# page load and chat history was filtered by exactly that id.


@router.get("/conversations")
async def list_conversations(
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str = Query(default="active", pattern="^(active|archived)$"),
    principal: Principal = Depends(require_scope(Scope.CHAT)),
):
    """List the caller's conversation threads, most recently active first."""
    try:
        from app.memory.conversations import conversation_repository

        conversations = await conversation_repository.list_for_owner(
            principal.user_id, limit=limit, offset=offset, status=status
        )
        return {
            "success": True,
            "count": len(conversations),
            "limit": limit,
            "offset": offset,
            "conversations": [c.summary_dict() for c in conversations],
        }
    except Exception as e:
        raise internal_error("Could not list conversations", e, user_id=principal.user_id)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    principal: Principal = Depends(require_scope(Scope.CHAT)),
):
    """
    Fetch a thread and its recent turns, for rehydrating the UI after a reload.

    Scoped to the authenticated owner: an id alone must never read someone
    else's conversation, so an unknown-or-not-yours thread is a flat 404 rather
    than a 403 that would confirm the id exists.
    """
    try:
        from app.memory.conversations import conversation_repository

        conversation = await conversation_repository.get(
            conversation_id, principal.user_id
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        turns = await conversation_repository.recent_turns(
            conversation_id, principal.user_id, limit=limit
        )
        return {
            "success": True,
            "conversation": conversation.summary_dict(),
            "turns": [
                {
                    "sequence": t.sequence,
                    "role": t.role,
                    "content": t.content,
                    "modality": t.modality,
                    "agent": t.agent,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in turns
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(
            "Could not load conversation", e,
            user_id=principal.user_id, conversation_id=conversation_id,
        )


@router.delete("/conversations/{conversation_id}")
async def archive_conversation(
    conversation_id: str,
    principal: Principal = Depends(require_scope(Scope.CHAT)),
):
    """
    Archive a thread.

    Archived, not deleted: the turns remain, and the memories already extracted
    from them stay valid. Hard erasure is a separate, explicit action (Phase 6).
    """
    try:
        from app.memory.conversations import conversation_repository

        archived = await conversation_repository.archive(
            conversation_id, principal.user_id
        )
        if not archived:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"success": True, "conversation_id": conversation_id, "status": "archived"}
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(
            "Could not archive conversation", e,
            user_id=principal.user_id, conversation_id=conversation_id,
        )


@router.websocket("/stream")
async def stream_response(websocket: WebSocket):
    """
    Stream text responses over WebSocket using true LLM token streaming.

    Fix 7: Now wired to run_streaming_workflow which yields tokens as they
    arrive from Groq, giving sub-second first-token latency instead of the
    old approach of blocking on run_workflow and chunking the final result.

    Client sends:
    {
      "type": "query",
      "query": "...",
      "user_id": "...",
      "session_id": "...",          (optional)
      "output_mode": "user|recruiter",
      "conversation_history": [{"role":"user","content":"..."}]
    }

    Server emits (in order):
      {"type": "ack"}
      {"type": "meta",          "selected_agent": ..., "detected_intent": ..., ...}
      {"type": "display_chunk", "index": N, "text": "<token>"}  ← live, many frames
      {"type": "final",         "display_text": "...", "speech_text": "...", "success": bool}
    """
    # CORS middleware does not apply to WebSocket upgrades, so the Origin check
    # has to happen here or any site could open this socket from a browser.
    # It also serves as this endpoint's CSRF defence, since a WebSocket cannot
    # carry a custom header from the browser.
    origin = websocket.headers.get("origin")
    if origin and origin not in settings.allowed_origins:
        logger.warning("Rejected WebSocket connection from disallowed origin: %s", origin)
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    # Authenticate during the handshake so an unauthenticated socket is never
    # accepted in the first place.
    principal = await authenticate_websocket(websocket)
    if principal is None:
        await websocket.close(code=1008, reason="Authentication required")
        return
    if not principal.has_scope(Scope.CHAT):
        await websocket.close(code=1008, reason="Insufficient permissions")
        return

    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type != "query":
                await websocket.send_json({"type": "error", "message": f"Unsupported message type: {msg_type}"})
                continue

            query = (payload.get("query") or "").strip()
            if not query:
                await websocket.send_json({"type": "error", "message": "query is required"})
                continue

            if len(query) > settings.max_query_chars:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Query is too long (maximum {settings.max_query_chars:,} characters).",
                })
                continue

            # Identity is fixed by the handshake token for the whole socket
            # lifetime; anything the client sends in the frame is ignored.
            claimed = payload.get("user_id")
            if claimed and claimed.strip().lower() != principal.user_id.lower():
                logger.warning(
                    "WebSocket user_id mismatch: authenticated=%s claimed=%s",
                    principal.user_id, claimed,
                )
                await websocket.send_json({
                    "type": "error",
                    "message": "You cannot access another user's data.",
                })
                continue
            ws_user_id = principal.user_id

            await websocket.send_json({"type": "ack", "success": True})

            history = payload.get("conversation_history")
            if not isinstance(history, list):
                history = []

            # ── Stream tokens from run_streaming_workflow ─────────────────
            try:
                async for chunk in run_streaming_workflow(
                    user_input=query,
                    user_id=ws_user_id,
                    session_id=payload.get("session_id"),
                    conversation_history=history,
                    output_mode=payload.get("output_mode", "user"),
                    scopes=principal.scopes,
                ):
                    chunk_type = chunk.get("type")

                    if chunk_type == "metadata":
                        await websocket.send_json({
                            "type": "meta",
                            "selected_agent": chunk.get("selected_agent"),
                            "detected_intent": chunk.get("detected_intent"),
                            "execution_path": chunk.get("execution_path", []),
                            "planner_confidence": chunk.get("planner_confidence"),
                        })

                    elif chunk_type == "token":
                        # Each token delivered immediately — true streaming
                        await websocket.send_json({
                            "type": "display_chunk",
                            "index": chunk.get("index", 0),
                            "text": chunk.get("token", ""),
                        })
                        # Yield to event loop so the send isn't batched
                        await asyncio.sleep(0)

                    elif chunk_type == "complete":
                        await websocket.send_json({
                            "type": "final",
                            "display_text": chunk.get("display_text", ""),
                            "speech_text": chunk.get("speech_text", ""),
                            "agent": chunk.get("agent"),
                            "user_id": chunk.get("user_id"),
                            "session_id": chunk.get("session_id"),
                            "success": chunk.get("success", True),
                            "error": chunk.get("error"),
                        })

            except Exception as stream_err:
                logger.error("Streaming workflow failed: %s", stream_err, exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": "Something went wrong while generating a response.",
                })

    except WebSocketDisconnect:
        return
    except Exception as e:
        logger.error("WebSocket session failed: %s", e, exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "The connection encountered an error.",
            })
        except Exception:
            pass
