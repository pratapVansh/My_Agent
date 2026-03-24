"""
Voice routes:
- WebSocket for low-latency real-time streaming
- Streaming WebSocket with LLM token streaming (NEW - ultra-low latency)
- HTTP endpoint for text + voice response
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import re
import base64

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.agents.workflow import run_workflow
from app.agents.streaming_workflow import run_streaming_workflow
from app.services.voice_service import voice_service
from app.config import settings
from deepgram import LiveTranscriptionEvents


router = APIRouter()


class VoiceQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    output_mode: str = Field(default="user", description="user or recruiter")
    voice_id: Optional[str] = None


@router.post("/query")
async def voice_query(request: VoiceQueryRequest):
    """Return both text and voice output for a text query."""
    try:
        workflow_result = await run_workflow(
            user_input=request.query,
            user_id=request.user_id,
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
    current_transcript = ""
    is_processing = False

    try:
        # Initialize Deepgram WebSocket if enabled
        if settings.streaming_stt_enabled:
            dg_connection = await voice_service.create_deepgram_websocket()

            # Set up Deepgram event handlers
            async def on_message(self, result, **kwargs):
                nonlocal current_transcript

                sentence = result.channel.alternatives[0].transcript
                if len(sentence) == 0:
                    return

                if result.is_final:
                    current_transcript += sentence + " "
                    await websocket.send_json({
                        "type": "interim_transcript",
                        "text": sentence,
                        "is_final": True
                    })
                else:
                    # Interim result for real-time feedback
                    await websocket.send_json({
                        "type": "interim_transcript",
                        "text": sentence,
                        "is_final": False
                    })

            async def on_utterance_end(self, utterance_end, **kwargs):
                nonlocal is_processing, current_transcript

                if len(current_transcript.strip()) > 0 and not is_processing:
                    is_processing = True
                    full_text = current_transcript.strip()

                    # Notify client that utterance ended
                    await websocket.send_json({
                        "type": "utterance_end",
                        "transcript": full_text
                    })

                    # Process through agent workflow
                    await process_transcript(full_text)

                    # Reset for next utterance
                    current_transcript = ""
                    is_processing = False

            async def on_error(self, error, **kwargs):
                await websocket.send_json({
                    "type": "error",
                    "message": f"Deepgram error: {error}"
                })

            # Register event handlers
            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
            dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        async def process_transcript(transcript: str):
            """Process complete transcript through agent workflow"""
            # Add to conversation history
            conversation_history.append({"role": "user", "content": transcript})

            # Run workflow
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

            # Keep last 10 messages
            if len(conversation_history) > 10:
                conversation_history[:] = conversation_history[-10:]

            # Generate speech
            tts_result = await voice_service.synthesize_speech(
                text=speech_text,
                voice_id=voice_id,
            )

            # Send final response
            await websocket.send_json({
                "type": "final_response",
                "success": True,
                "transcript": transcript,
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
            })

        # Main message loop
        while True:
            # Receive message from client
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "start":
                user_id = msg.get("user_id")
                session_id = msg.get("session_id")
                output_mode = msg.get("output_mode", "user")
                voice_id = msg.get("voice_id")
                current_transcript = ""

                await websocket.send_json({
                    "type": "ready",
                    "success": True,
                    "streaming_enabled": settings.streaming_stt_enabled
                })
                continue

            if msg_type == "audio_data":
                # Forward audio to Deepgram WebSocket
                if dg_connection and settings.streaming_stt_enabled:
                    try:
                        # Decode base64 audio and send to Deepgram
                        audio_b64 = msg.get("data", "")
                        audio_bytes = base64.b64decode(audio_b64)
                        await dg_connection.send(audio_bytes)
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Failed to send audio to Deepgram: {str(e)}"
                        })
                else:
                    # Fallback to chunk-based STT if streaming disabled
                    stt_result = await voice_service.transcribe_audio_chunk(
                        audio_b64=msg.get("data", ""),
                        mime_type=msg.get("mime_type", "audio/wav"),
                        language=msg.get("language", "en"),
                    )

                    if stt_result.get("success"):
                        chunk_text = stt_result.get("transcript", "")
                        if chunk_text:
                            current_transcript += chunk_text + " "
                            await websocket.send_json({
                                "type": "interim_transcript",
                                "text": chunk_text,
                                "is_final": True
                            })
                continue

            if msg_type == "end_utterance":
                # Manual end utterance (backup for when VAD not available)
                full_text = current_transcript.strip()
                if full_text and not is_processing:
                    is_processing = True
                    await process_transcript(full_text)
                    current_transcript = ""
                    is_processing = False
                continue

            # Unknown message type
            await websocket.send_json({
                "type": "error",
                "message": f"Unsupported message type: {msg_type}",
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Cleanup Deepgram connection
        if dg_connection:
            try:
                await dg_connection.finish()
            except:
                pass

