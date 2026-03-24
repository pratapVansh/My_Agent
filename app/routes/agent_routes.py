"""
API routes for multi-agent system.
REST handles normal operations, while WebSocket is reserved for response streaming.
"""
import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.agents.workflow import run_workflow
from app.tools.job_search_tool import job_search_tool
from app.tools.email_draft_tool import email_draft_tool
from app.tools.attendance_tool import attendance_tool
from app.tools.timetable_tool import timetable_tool, TimetableInput
from app.memory.memory_manager import memory_manager
from PyPDF2 import PdfReader
import io


router = APIRouter()


class Message(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class AgentRequest(BaseModel):
    """Request model for agent query."""
    query: str = Field(..., description="User's query", min_length=1)
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier (generated if not provided)"
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
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., min_length=1, description="Job search query")
    location: Optional[str] = Field(default=None, description="Optional location")
    max_results: int = Field(default=10, ge=1, le=25)
    min_score: float = Field(default=0.2, ge=0.0, le=1.0)


class EmailDraftRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., min_length=1, description="Email draft request")
    tone: str = Field(default="professional", description="Email tone")
    recipient_name: Optional[str] = Field(default="", description="Recipient name")


class AttendanceScrapeRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
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
    user_id: str = Field(..., description="User identifier")
    entries: List[TimetableEntryRequest] = Field(..., min_length=1)


class TimetableSuggestRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    low_attendance_threshold: float = Field(default=75.0, ge=0.0, le=100.0)


def _chunk_text(text: str, chunk_size: int = 140) -> List[str]:
    """Split long responses into chunks for low-latency WebSocket delivery."""
    if not text:
        return []

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []

    for word in words:
        tentative = " ".join(current + [word])
        if len(tentative) > chunk_size and current:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))

    return chunks


@router.post("/query", response_model=AgentResponse)
async def agent_query(request: AgentRequest):
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
        # Convert conversation history to dict format
        history = None
        if request.conversation_history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Run workflow with memory integration
        result = await run_workflow(
            user_input=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_history=history,
            output_mode=request.output_mode
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

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )


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
async def job_search(request: JobSearchRequest):
    """Search jobs using Tavily, then filter/rank with memory personalization."""
    try:
        result = await job_search_tool.search_jobs(
            user_id=request.user_id,
            query=request.query,
            location=request.location,
            max_results=request.max_results,
            min_score=request.min_score,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")


@router.post("/tools/email-draft")
async def email_draft(request: EmailDraftRequest):
    """Create a personalized email draft with RAG context. Draft only, never sent."""
    try:
        result = await email_draft_tool.draft_email(
            user_id=request.user_id,
            query=request.query,
            tone=request.tone,
            recipient_name=request.recipient_name or "",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email draft failed: {str(e)}")


@router.post("/tools/attendance/scrape")
async def scrape_attendance(request: AttendanceScrapeRequest):
    """Scrape attendance from ERP with Playwright and store in PostgreSQL."""
    try:
        result = await attendance_tool.scrape_and_store(
            user_id=request.user_id,
            erp_url=request.erp_url,
            username=request.username,
            password=request.password,
            selectors=request.selectors,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attendance scrape failed: {str(e)}")


@router.post("/tools/timetable/store")
async def store_timetable(request: TimetableStoreRequest):
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
        result = await timetable_tool.store_timetable(user_id=request.user_id, entries=entries)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timetable store failed: {str(e)}")


@router.post("/tools/timetable/suggest")
async def suggest_classes(request: TimetableSuggestRequest):
    """Suggest classes prioritized by low attendance and timetable schedule."""
    try:
        result = await timetable_tool.suggest_classes(
            user_id=request.user_id,
            day_of_week=request.day_of_week,
            low_attendance_threshold=request.low_attendance_threshold,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timetable suggestion failed: {str(e)}")


@router.post("/memory/upload-pdf")
async def upload_pdf_document(
    user_id: str = Form(...),
    document_type: str = Form(default="general"),
    file: UploadFile = File(...)
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
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read PDF content
        content = await file.read()

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

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {str(e)}")


@router.post("/memory/upload-text")
async def upload_text_document(
    user_id: str = Form(...),
    document_type: str = Form(default="general"),
    text_content: str = Form(...),
    document_name: str = Form(default="untitled")
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
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")

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
        raise HTTPException(status_code=500, detail=f"Text upload failed: {str(e)}")


@router.websocket("/stream")
async def stream_response(websocket: WebSocket):
    """
    Stream text responses over WebSocket.
    WebSocket is used only for streaming responses, not normal operations.

    Client message:
    {
      "type": "query",
      "query": "...",
      "user_id": "...",
      "session_id": "...",
      "output_mode": "user|recruiter",
      "conversation_history": [{"role":"user","content":"..."}]
    }
    """
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = payload.get("type")

            if msg_type != "query":
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unsupported message type: {msg_type}",
                    }
                )
                continue

            query = (payload.get("query") or "").strip()
            if not query:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "query is required",
                    }
                )
                continue

            await websocket.send_json({"type": "ack", "success": True})

            history = payload.get("conversation_history")
            if not isinstance(history, list):
                history = []

            result = await run_workflow(
                user_input=query,
                user_id=payload.get("user_id"),
                session_id=payload.get("session_id"),
                conversation_history=history,
                output_mode=payload.get("output_mode", "user"),
            )

            display_text = result.get("display_text", "")
            speech_text = result.get("speech_text", "")

            await websocket.send_json(
                {
                    "type": "meta",
                    "selected_agent": result.get("selected_agent"),
                    "execution_path": result.get("execution_path", []),
                    "user_id": result.get("user_id"),
                    "session_id": result.get("session_id"),
                }
            )

            for idx, chunk in enumerate(_chunk_text(display_text), start=1):
                await websocket.send_json(
                    {
                        "type": "display_chunk",
                        "index": idx,
                        "text": chunk,
                    }
                )
                await asyncio.sleep(0)

            await websocket.send_json(
                {
                    "type": "final",
                    "display_text": display_text,
                    "speech_text": speech_text,
                    "success": result.get("error") is None,
                    "error": result.get("error"),
                }
            )

    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
