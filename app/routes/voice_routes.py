"""
Voice routes:
- WebSocket for low-latency real-time streaming
- Streaming WebSocket with LLM token streaming (NEW - ultra-low latency)
- HTTP endpoint for text + voice response
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import base64
import json
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.agents.workflow import run_workflow
from app.services.voice_service import voice_service
import logging

from app.config import settings
from deepgram import LiveTranscriptionEvents

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """
    Split speech text into sentence-level chunks for pipelined TTS.
    Merges short fragments so Cartesia always gets at least ~15 chars.
    """
    if not text or not text.strip():
        return []
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    result: List[str] = []
    current = ""
    for part in parts:
        current = (current + " " + part).strip() if current else part
        # Emit when long enough or ends with sentence-final punctuation
        if len(current) >= 20 and current[-1] in ".!?":
            result.append(current)
            current = ""
    if current:
        result.append(current)
    return result if result else [text]


router = APIRouter()


class VoiceQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    output_mode: str = Field(default="user", description="user or recruiter")
    voice_id: Optional[str] = None


@router.post("/query")
async def voice_query(request: VoiceQueryRequest):
    """Return both text and voice output for a text query."""
    try:
        user_id = request.user_id.strip().lower()
        print("QUERY user_id:", user_id)

        workflow_result = await run_workflow(
            user_input=request.query,
            user_id=user_id,
            session_id=request.session_id,
            output_mode=request.output_mode,
        )

        display_text = workflow_result.get("display_text", "")
        speech_text = workflow_result.get("speech_text", "")

        tts_result = await voice_service.synthesize_speech(
            text=speech_text,
            voice_id=request.voice_id,
        )

        return {
            "success": True,
            "mode": request.output_mode,
            "display_text": display_text,
            "speech_text": speech_text,
            "voice": tts_result,
            "metadata": {
                "selected_agent": workflow_result.get("selected_agent"),
                "execution_path": workflow_result.get("execution_path", []),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket):
    """
    Real-time voice streaming contract (JSON messages):
    - client -> {"type":"start","user_id":"...","session_id":"...","output_mode":"user|recruiter","voice_id":"..."}
    - client -> {"type":"audio_chunk","audio_base64":"...","mime_type":"audio/wav","language":"en"}
    - client -> {"type":"end_utterance"}

    Server emits:
    - {"type":"partial_transcript","text":"..."}
    - {"type":"final_response","display_text":"...","speech_text":"...","audio_base64":"...","mime_type":"audio/wav"}
    - {"type":"error","message":"..."}
    """
    await websocket.accept()

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    output_mode: str = "user"
    voice_id: Optional[str] = None
    transcript_parts: list[str] = []
    conversation_history: list[Dict[str, str]] = []  # Track conversation in session

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "start":
                user_id = msg.get("user_id")
                session_id = msg.get("session_id")
                output_mode = msg.get("output_mode", "user")
                voice_id = msg.get("voice_id")
                transcript_parts = []

                if not user_id:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "user_id is required in start message",
                        }
                    )
                    continue

                user_id = str(user_id).strip().lower()

                # Don't reset conversation_history on start - maintain context
                await websocket.send_json({"type": "ready", "success": True})
                continue

            if msg_type == "audio_chunk":
                stt_result = await voice_service.transcribe_audio_chunk(
                    audio_b64=msg.get("audio_base64", ""),
                    mime_type=msg.get("mime_type", "audio/wav"),
                    language=msg.get("language", "en"),
                )

                if not stt_result.get("success"):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": stt_result.get("error", "stt failed"),
                        }
                    )
                    continue

                chunk_text = stt_result.get("transcript", "")
                if chunk_text:
                    transcript_parts.append(chunk_text)
                    await websocket.send_json(
                        {
                            "type": "partial_transcript",
                            "text": chunk_text,
                        }
                    )
                continue

            if msg_type == "end_utterance":
                full_text = " ".join(transcript_parts).strip()
                if not full_text:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "No transcript captured",
                        }
                    )
                    continue

                # Add user message to conversation history
                conversation_history.append({"role": "user", "content": full_text})

                workflow_result = await run_workflow(
                    user_input=full_text,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_history=conversation_history,
                    output_mode=output_mode,
                )
                display_text = workflow_result.get("display_text", "")
                speech_text = workflow_result.get("speech_text", "")

                # Add assistant response to conversation history
                conversation_history.append({"role": "assistant", "content": display_text})

                # Keep only last 10 messages to avoid memory overflow
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]

                tts_result = await voice_service.synthesize_speech(
                    text=speech_text,
                    voice_id=voice_id,
                )

                await websocket.send_json(
                    {
                        "type": "final_response",
                        "success": True,
                        "transcript": full_text,
                        "display_text": display_text,
                        "speech_text": speech_text,
                        "audio_base64": tts_result.get("audio_base64", ""),
                        "mime_type": tts_result.get("mime_type", "audio/wav"),
                        "voice_success": tts_result.get("success", False),
                        "voice_error": tts_result.get("error"),
                        "metadata": {
                            "selected_agent": workflow_result.get("selected_agent"),
                            "execution_path": workflow_result.get("execution_path", []),
                        },
                    }
                )
                transcript_parts = []  # Clear only transcript parts, keep conversation history
                continue

            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Unsupported message type: {msg_type}",
                }
            )

    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


@router.websocket("/stream_v2")
async def voice_stream_v2(websocket: WebSocket):
    """
    Ultra-low latency voice streaming with Deepgram WebSocket STT.

    Improvements over /stream:
    - Streaming STT: 50-200ms latency (vs 500-800ms chunk-based)
    - Real-time interim results for immediate feedback
    - Built-in VAD for detecting speech end (vs fixed 2s timer)
    - More responsive user experience

    Protocol:
    - client -> {"type":"start","user_id":"...", "session_id":"...", "output_mode":"user|recruiter", "voice_id":"..."}
    - client -> {"type":"audio_data", "data":"<base64_audio>"}  # Stream audio continuously
    - server <- {"type":"interim_transcript", "text":"..."}     # Real-time partial results
    - server <- {"type":"utterance_end", "transcript":"..."}    # Speech end detected
    - server <- {"type":"final_response", ...}                  # Complete response with audio
    """
    await websocket.accept()

    # Connection state
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    output_mode: str = "user"
    voice_id: Optional[str] = None
    conversation_history: list[Dict[str, str]] = []

    # Deepgram WebSocket connection
    dg_connection = None
    dg_connected = False          # True only when connection is alive and ready
    current_transcript = ""
    transcript_lock = asyncio.Lock()
    # Track the in-progress agent+TTS task so we can cancel it on interrupt
    current_processing_task: Optional[asyncio.Task] = None
    outbound_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    async def outbound_sender():
        """Serialize websocket sends so final responses are flushed immediately."""
        while True:
            payload = await outbound_queue.get()
            if payload is None:
                return
            await websocket.send_json(payload)

    sender_task = asyncio.create_task(outbound_sender())
    background_tasks: set[asyncio.Task] = set()

    async def emit(payload: Dict[str, Any]):
        await outbound_queue.put(payload)

    def emit_threadsafe(payload: Dict[str, Any]):
        loop.call_soon_threadsafe(outbound_queue.put_nowait, payload)

    # Deepgram event handlers (defined outside try block for reuse)
    async def on_message(self, result, **kwargs):
        nonlocal current_transcript

        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return

        if result.is_final:
            current_transcript += sentence + " "
            emit_threadsafe({
                "type": "interim_transcript",
                "text": sentence,
                "is_final": True
            })
        else:
            # Interim result for real-time feedback
            emit_threadsafe({
                "type": "interim_transcript",
                "text": sentence,
                "is_final": False
            })

    async def on_utterance_end(self, utterance_end, **kwargs):
        nonlocal current_transcript, current_processing_task

        # Lock to prevent race with on_message writing current_transcript simultaneously
        async with transcript_lock:
            full_text = current_transcript.strip()
            current_transcript = ""

        if len(full_text) > 0:
            # Cancel any in-flight agent+TTS task and tell frontend to clear audio
            if current_processing_task and not current_processing_task.done():
                current_processing_task.cancel()
                emit_threadsafe({"type": "interrupt"})

            # Notify client that utterance ended
            emit_threadsafe({
                "type": "utterance_end",
                "transcript": full_text
            })

            # Process and send response immediately without blocking receive loop
            current_processing_task = asyncio.create_task(process_transcript(full_text))
            background_tasks.add(current_processing_task)
            current_processing_task.add_done_callback(background_tasks.discard)

    async def on_open(self, open, **kwargs):
        nonlocal dg_connected
        dg_connected = True
        print("[Deepgram] Connection opened")

    async def on_close(self, close, **kwargs):
        nonlocal dg_connected
        # Guard against stale close events from a previous connection firing
        # after a new one has already been established.
        if self is not dg_connection:
            return
        dg_connected = False
        print("[Deepgram] Connection closed")

    async def on_error(self, error, **kwargs):
        nonlocal dg_connected
        if self is not dg_connection:
            return
        dg_connected = False
        logger.warning("[Deepgram] Connection error (will auto-reconnect): %s", error)

        # Schedule a silent reconnect in the background.
        # Only surface the error to the frontend if reconnection fails —
        # the raw Deepgram timeout message is not actionable for the user.
        async def _silent_reconnect():
            await asyncio.sleep(0.5)  # brief pause for SDK cleanup
            ok = await ensure_dg_connected()
            if not ok:
                emit_threadsafe({
                    "type": "error",
                    "message": "Voice connection lost. Please refresh the page.",
                })
                logger.error("[Deepgram] Reconnect failed after error: %s", error)
            else:
                logger.info("[Deepgram] Silently reconnected after error")

        asyncio.create_task(_silent_reconnect())

    try:
        # Initialize Deepgram WebSocket only if API key is present (deferred until after WS accept)
        async def initialize_deepgram():
            """Initialize Deepgram WebSocket connection lazily"""
            nonlocal dg_connection, dg_connected

            if not settings.is_streaming_stt_available:
                return False

            try:
                print("[OK] Initializing Deepgram streaming STT...")

                # Create connection object WITHOUT starting it so we can register
                # handlers first — prevents events from firing before handlers exist.
                new_conn, options = voice_service.create_deepgram_connection()

                new_conn.on(LiveTranscriptionEvents.Open, on_open)
                new_conn.on(LiveTranscriptionEvents.Transcript, on_message)
                new_conn.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
                new_conn.on(LiveTranscriptionEvents.Error, on_error)
                new_conn.on(LiveTranscriptionEvents.Close, on_close)

                if not await new_conn.start(options):
                    raise Exception("Failed to start Deepgram WebSocket connection")

                dg_connection = new_conn
                dg_connected = True
                print("[OK] Deepgram streaming STT initialized successfully")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to initialize Deepgram: {e}")
                dg_connected = False
                return False

        async def ensure_dg_connected() -> bool:
            """Reconnect Deepgram if the connection has dropped."""
            nonlocal dg_connection, dg_connected
            if dg_connected and dg_connection:
                return True
            print("[Deepgram] Reconnecting...")
            # Clean up stale connection
            if dg_connection:
                try:
                    await dg_connection.finish()
                except Exception:
                    pass
                dg_connection = None
            return await initialize_deepgram()

        async def deepgram_keepalive_loop():
            """Send a keepalive to Deepgram every 5s — well within the 10s idle timeout.
            8s was too close: any scheduling jitter caused the window to be missed."""
            nonlocal dg_connected
            while True:
                await asyncio.sleep(5)
                if dg_connection and dg_connected:
                    try:
                        await dg_connection.keep_alive()
                    except Exception as e:
                        logger.warning("[Deepgram] Keepalive failed: %s — marking dead for reconnect", e)
                        dg_connected = False
                        # Attempt a silent reconnect so the next utterance works
                        asyncio.create_task(ensure_dg_connected())

        keepalive_task = asyncio.create_task(deepgram_keepalive_loop())
        background_tasks.add(keepalive_task)

        async def process_transcript(transcript: str):
            """Process complete transcript through agent workflow with streaming TTS."""
            try:
                if not user_id:
                    await emit({
                        "type": "error",
                        "message": "user_id is required before processing transcript",
                    })
                    return

                async with transcript_lock:
                    print("QUERY user_id:", user_id)

                    # Add to conversation history
                    conversation_history.append({"role": "user", "content": transcript})

                    # Run full agent workflow
                    workflow_result = await run_workflow(
                        user_input=transcript,
                        user_id=user_id,
                        session_id=session_id,
                        conversation_history=conversation_history,
                        output_mode=output_mode,
                    )

                    display_text = workflow_result.get("display_text", "")
                    speech_text = workflow_result.get("speech_text", "")

                    # Add response to history
                    conversation_history.append({"role": "assistant", "content": display_text})
                    if len(conversation_history) > 10:
                        conversation_history[:] = conversation_history[-10:]

                    has_tts = bool(
                        speech_text
                        and output_mode == "user"
                        and settings.cartesia_api_key
                        and settings.cartesia_api_key != "your_cartesia_api_key_here"
                    )

                    # Send text response immediately — don't wait for TTS to finish
                    task_result = workflow_result.get("task_result") or {}
                    await emit({
                        "type": "final_response",
                        "success": True,
                        "transcript": transcript,
                        "display_text": display_text,
                        "speech_text": speech_text,
                        "audio_base64": "",       # Audio streams separately via audio_chunk
                        "mime_type": "audio/raw",
                        "voice_success": has_tts,
                        "voice_error": None,
                        "metadata": {
                            "selected_agent": workflow_result.get("selected_agent"),
                            "execution_path": workflow_result.get("execution_path", []),
                        },
                        "job_results": task_result.get("result", {}).get("jobs") if workflow_result.get("selected_agent") == "job" else None,
                    })

                    # Sentence-chunked TTS: split response into sentences and
                    # pipeline each one to Cartesia immediately so first audio
                    # arrives ~sentence_1_duration sooner than full-text TTS.
                    if has_tts:
                        await emit({
                            "type": "audio_stream_start",
                            "sample_rate": settings.cartesia_sample_rate,
                            "encoding": "pcm_s16le",
                            "channels": 1,
                        })
                        try:
                            sentences = _split_sentences(speech_text)
                            for sentence in sentences:
                                if not sentence.strip():
                                    continue
                                async for chunk in voice_service.synthesize_speech_stream(
                                    text=sentence,
                                    voice_id=voice_id,
                                ):
                                    await emit({
                                        "type": "audio_chunk",
                                        "data": base64.b64encode(chunk).decode("ascii"),
                                    })
                        except asyncio.CancelledError:
                            # Task was cancelled by a new utterance — stop cleanly
                            await emit({"type": "audio_stream_end"})
                            raise
                        except Exception as tts_err:
                            print(f"[TTS Stream] Error during streaming: {tts_err}")
                        finally:
                            await emit({"type": "audio_stream_end"})

            except Exception as e:
                await emit({
                    "type": "error",
                    "message": f"Failed to process transcript: {str(e)}",
                })

        # Main message loop
        # Handles both text (JSON) and binary (raw PCM16 audio) frames
        while True:
            data = await websocket.receive()

            # ── Binary frame: raw PCM16 audio from frontend (no base64 overhead) ──
            audio_bytes_raw = data.get("bytes")
            if audio_bytes_raw is not None:
                if not settings.is_streaming_stt_available:
                    await emit({
                        "type": "error",
                        "message": "Deepgram streaming STT not available. Set DEEPGRAM_API_KEY in .env."
                    })
                    continue
                # Reconnect silently if the connection dropped between utterances
                ok = await ensure_dg_connected()
                if ok:
                    try:
                        await dg_connection.send(audio_bytes_raw)
                    except Exception as e:
                        await emit({
                            "type": "error",
                            "message": f"Failed to send audio to Deepgram: {str(e)}"
                        })
                else:
                    await emit({
                        "type": "error",
                        "message": "Deepgram reconnection failed. Please refresh and try again."
                    })
                continue

            # ── Text frame: JSON control messages ──
            text_data = data.get("text")
            if text_data is None:
                continue

            try:
                msg = json.loads(text_data)
            except json.JSONDecodeError:
                await emit({"type": "error", "message": "Invalid JSON in text frame"})
                continue

            msg_type = msg.get("type")

            if msg_type == "start":
                user_id = msg.get("user_id")
                session_id = msg.get("session_id")
                output_mode = msg.get("output_mode", "user")
                voice_id = msg.get("voice_id")
                current_transcript = ""

                if not user_id:
                    await emit({
                        "type": "error",
                        "message": "user_id is required in start message",
                    })
                    continue

                user_id = str(user_id).strip().lower()

                # Initialize Deepgram on first start (lazy loading)
                if not dg_connection and settings.is_streaming_stt_available:
                    streaming_initialized = await initialize_deepgram()
                else:
                    streaming_initialized = bool(dg_connection)

                await emit({
                    "type": "ready",
                    "success": True,
                    "streaming_enabled": streaming_initialized
                })
                continue

            if msg_type == "audio_data":
                # Legacy base64 path — kept for backwards compatibility
                ok = await ensure_dg_connected()
                if ok:
                    try:
                        audio_b64 = msg.get("data", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        await dg_connection.send(audio_bytes)
                    except Exception as e:
                        await emit({
                            "type": "error",
                            "message": f"Failed to send audio to Deepgram: {str(e)}"
                        })
                else:
                    await emit({
                        "type": "error",
                        "message": "Deepgram streaming STT not available. Set DEEPGRAM_API_KEY in .env."
                    })
                continue

            if msg_type == "interrupt":
                # Frontend is interrupting current TTS playback
                if current_processing_task and not current_processing_task.done():
                    current_processing_task.cancel()
                    await emit({"type": "interrupt"})
                continue

            if msg_type == "end_utterance":
                async with transcript_lock:
                    full_text = current_transcript.strip()
                    current_transcript = ""

                if full_text:
                    if current_processing_task and not current_processing_task.done():
                        current_processing_task.cancel()
                        await emit({"type": "interrupt"})
                    current_processing_task = asyncio.create_task(process_transcript(full_text))
                    background_tasks.add(current_processing_task)
                    current_processing_task.add_done_callback(background_tasks.discard)
                else:
                    await emit({
                        "type": "error",
                        "message": "No speech detected. Please try again."
                    })
                continue

            await emit({
                "type": "error",
                "message": f"Unsupported message type: {msg_type}",
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket handler error: %s", e, exc_info=True)
        try:
            await emit({"type": "error", "message": str(e)})
        except Exception as emit_err:
            # WebSocket already closed — cannot send error frame
            logger.debug("Could not send error frame to client: %s", emit_err)
    finally:
        for task in list(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

        await outbound_queue.put(None)
        await asyncio.gather(sender_task, return_exceptions=True)

        # Cleanup Deepgram connection
        if dg_connection:
            try:
                await dg_connection.finish()
            except Exception as dg_err:
                logger.warning("Deepgram connection cleanup failed: %s", dg_err)

