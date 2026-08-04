"""
LiveKit Worker — Phase 5 (Streaming LLM).
Low-latency voice assistant utilizing LiveKit, Deepgram, Groq, and Cartesia.
"""
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import AsyncIterable

from livekit.api import AccessToken, VideoGrants

from app.config import settings
from app.services.deepgram_bridge import DeepgramBridge
from app.agents.streaming_workflow import run_streaming_workflow
from app.agents.workflow import run_workflow
from app.agents.hybrid_router import determine_route
from app.memory.memory_manager import memory_manager
from app.services.langsmith_service import configure_langsmith
from app.services.tts_streamer import tts_streamer
from app.services.text_chunker import text_chunker

# ── Logging configuration ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
_DEEPGRAM_SAMPLE_RATE = 16000
_DEEPGRAM_CHANNELS = 1
_MAX_HISTORY = 20

@dataclass
class ParticipantState:
    identity: str
    session_id: str
    history: list[dict] = field(default_factory=list)
    agent_task: asyncio.Task | None = None
    bridge: DeepgramBridge | None = None
    audio_pipeline_task: asyncio.Task | None = None
    audio_source: 'rtc.AudioSource | None' = None
    audio_track: 'rtc.LocalAudioTrack | None' = None
    cancellation_token: asyncio.Event = field(default_factory=asyncio.Event)

async def main(room_name: str) -> None:
    try:
        from livekit import rtc
    except ImportError:
        logger.error("Missing dependency: pip install livekit>=1.0.0")
        return

    if not settings.is_livekit_configured:
        logger.error("LiveKit not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env")
        return

    if not settings.is_streaming_stt_available:
        logger.error("Deepgram not configured. Set DEEPGRAM_API_KEY in .env")
        return

    configure_langsmith()

    logger.info("Initializing memory system...")
    try:
        await memory_manager.initialize()
        logger.info("✓ Memory system initialized")
    except Exception as exc:
        logger.warning("Memory initialization failed: %s", exc)

    # ── Multi-participant State ──────────────────────────────────────────
    participants: dict[str, ParticipantState] = {}
    room_ref: dict[str, rtc.Room | None] = {"room": None}



    async def run_participant_agent(identity: str, transcript: str):
        state = participants.get(identity)
        if not state:
            return
            
        room = room_ref.get("room")
        if not room:
            return

        logger.info("▶ Routing intent for user=%s | transcript=%s", identity, transcript[:80])
        state.history.append({"role": "user", "content": transcript})

        accumulated = ""


        try:
            route = await determine_route(transcript, list(state.history))
            logger.info("  Intent routed to: %s", route)

            if route == "tool_required":
                # Execute full LangGraph workflow
                workflow_result = await run_workflow(
                    user_input=transcript,
                    user_id=identity,
                    session_id=state.session_id,
                    conversation_history=list(state.history),
                    output_mode="user"
                )
                
                display_text = workflow_result.get("display_text", "")
                speech_text = workflow_result.get("speech_text", "")
                selected_agent = workflow_result.get("selected_agent")
                
                # Publish complete event immediately
                payload = json.dumps({
                    "type": "final_response",
                    "success": True,
                    "transcript": transcript,
                    "display_text": display_text,
                    "metadata": {"selected_agent": selected_agent, "source": "livekit", "route": "tool_required"}
                }).encode("utf-8")
                if room:
                    await room.local_participant.publish_data(payload, topic="agent-response", reliable=True)
                
                state.history.append({"role": "assistant", "content": display_text})
                if len(state.history) > _MAX_HISTORY:
                    state.history[:] = state.history[-_MAX_HISTORY:]
                    
                async def tool_token_consumer() -> AsyncIterable[str]:
                    # Yield the final speech text as a single "token" block for the chunker
                    yield speech_text
                    
                token_stream = tool_token_consumer()
                
            else:
                # Streaming conversational workflow
                workflow = run_streaming_workflow(
                    user_input=transcript,
                    user_id=identity,
                    session_id=state.session_id,
                    conversation_history=list(state.history)
                )

                async def token_consumer() -> AsyncIterable[str]:
                    nonlocal accumulated
                    accumulated = ""
                    selected_agent = None
                    last_flush_time = asyncio.get_running_loop().time()
                    
                    async for event in workflow:
                        if event["type"] == "metadata":
                            selected_agent = event.get("selected_agent")
                            
                        elif event["type"] == "token":
                            token = event["token"]
                            accumulated = event["accumulated"]
                            yield token
                            
                            # UI Batching: Flush every ~100ms
                            now = asyncio.get_running_loop().time()
                            if now - last_flush_time >= 0.1:
                                payload = json.dumps({
                                    "type": "token_batch", 
                                    "text": accumulated
                                }).encode("utf-8")
                                if room:
                                    await room.local_participant.publish_data(payload, topic="agent-response", reliable=True)
                                last_flush_time = now
                                
                        elif event["type"] == "complete":
                            # Final response payload to finalize UI state
                            payload = json.dumps({
                                "type": "final_response",
                                "success": event.get("success", True),
                                "transcript": transcript,
                                "display_text": event.get("display_text", ""),
                                "metadata": {"selected_agent": selected_agent, "source": "livekit", "route": "conversational"}
                            }).encode("utf-8")
                            if room:
                                await room.local_participant.publish_data(payload, topic="agent-response", reliable=True)
                            
                            state.history.append({"role": "assistant", "content": event.get("display_text", "")})
                            if len(state.history) > _MAX_HISTORY:
                                state.history[:] = state.history[-_MAX_HISTORY:]
                                
                token_stream = token_consumer()

            # Wire the pipeline: Tokens -> TextChunker -> TTSStreamer
            chunk_stream = text_chunker.chunk_tokens(token_stream)
            if state.audio_source:
                await tts_streamer.stream_to_track(chunk_stream, state.audio_source)
                
            if room:
                payload = json.dumps({"type": "tts_complete"}).encode("utf-8")
                await room.local_participant.publish_data(payload, topic="agent-response", reliable=True)
                
            logger.info("✓ Workflow complete for user=%s", identity)
        except asyncio.CancelledError:
            logger.info("  Agent task cancelled for %s (Barge-in)", identity)
            # Safe memory commit
            if 'accumulated' in locals() and accumulated:
                state.history.append({"role": "assistant", "content": accumulated + " [Interrupted]"})
            
            # Broadcast interrupted to UI
            if room:
                payload = json.dumps({
                    "type": "interrupted",
                    "partial_text": accumulated if 'accumulated' in locals() else ""
                }).encode("utf-8")
                # Fire and forget safely since we are inside a cancelled task context
                asyncio.create_task(room.local_participant.publish_data(payload, topic="agent-response", reliable=True))
            raise
        except Exception as e:
            logger.error("Error in agent task for %s: %s", identity, e, exc_info=True)
            if room:
                err_payload = json.dumps({
                    "type": "error", "message": f"Processing error: {str(e)[:200]}", "source": "livekit"
                }).encode("utf-8")
                await room.local_participant.publish_data(err_payload, topic="agent-response", reliable=True)

    async def on_utterance(identity: str, transcript: str) -> None:
        state = participants.get(identity)
        if not state:
            return
            
        state.agent_task = asyncio.create_task(
            run_participant_agent(identity, transcript),
            name=f"agent-{identity}"
        )
        
        # Add a done callback to catch silent unhandled exceptions
        def _on_done(t: asyncio.Task):
            try:
                t.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Unhandled exception in agent task for %s: %s", identity, e)
        state.agent_task.add_done_callback(_on_done)

    async def audio_pipeline(stream: rtc.AudioStream, identity: str, bridge: DeepgramBridge) -> None:
        logger.info("[DEBUG] %s: audio_pipeline() started for %s", asyncio.get_running_loop().time(), identity)
        resampler: rtc.AudioResampler | None = None
        frame_count = 0
        try:
            async for event in stream:
                frame = event.frame
                frame_count += 1
                if frame_count == 1:
                    logger.info("[DEBUG] %s: audio_pipeline() received FIRST frame for %s", asyncio.get_running_loop().time(), identity)
                if frame_count % 100 == 0:
                    logger.debug("[DEBUG] %s: audio_pipeline() processed %s frames for %s", asyncio.get_running_loop().time(), frame_count, identity)
                    
                if resampler is None:
                    resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=_DEEPGRAM_SAMPLE_RATE,
                        num_channels=_DEEPGRAM_CHANNELS,
                    )
                resampled_frames = resampler.push(frame)
                for resampled in resampled_frames:
                    if not bridge.connected:
                        if frame_count == 1:
                            logger.info("[DEBUG] %s: bridge not connected, calling ensure_connected()", asyncio.get_running_loop().time())
                        ok = await bridge.ensure_connected()
                        if not ok:
                            logger.error("[DEBUG] %s: ensure_connected() failed!", asyncio.get_running_loop().time())
                            break
                    await bridge.send(bytes(resampled.data))
        except asyncio.CancelledError:
            logger.info("  Audio pipeline cancelled for %s", identity)
        except Exception as exc:
            logger.warning("  Audio pipeline error for %s: %s", identity, exc)
        finally:
            logger.info("[DEBUG] %s: audio_pipeline() closed for %s. Total frames: %s", asyncio.get_running_loop().time(), identity, frame_count)

    # ── Generate a worker token ──────────────────────────────────────────
    token = (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity("voice-worker")
        .with_name("Voice Worker")
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_publish_data=True,
            can_subscribe=True,
        ))
    )
    jwt = token.to_jwt()

    room = rtc.Room()
    room_ref["room"] = room

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        logger.info("✓ Participant connected: %s", participant.identity)
        if participant.identity not in participants:
            participants[participant.identity] = ParticipantState(
                identity=participant.identity,
                session_id=f"lk_{room_name}_{participant.identity}"
            )
        
        # Create dedicated audio source/track for this participant
        audio_source = rtc.AudioSource(24000, 1)
        audio_track = rtc.LocalAudioTrack.create_audio_track(f"agent-mic-{participant.identity}", audio_source)
        participants[participant.identity].audio_source = audio_source
        participants[participant.identity].audio_track = audio_track
        
        # Publish track
        asyncio.create_task(room.local_participant.publish_track(audio_track))

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        logger.info("  Participant disconnected: %s", participant.identity)
        state = participants.pop(participant.identity, None)
        if state:
            if state.agent_task and not state.agent_task.done():
                state.agent_task.cancel()
            if state.audio_pipeline_task and not state.audio_pipeline_task.done():
                state.audio_pipeline_task.cancel()
            if state.bridge:
                asyncio.create_task(state.bridge.close())

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("[DEBUG] %s: on_track_subscribed() start. participant=%s", asyncio.get_running_loop().time(), participant.identity)
            
            state = participants.get(participant.identity)
            if not state:
                logger.info("[DEBUG] %s: Creating new ParticipantState for %s", asyncio.get_running_loop().time(), participant.identity)
                state = ParticipantState(
                    identity=participant.identity,
                    session_id=f"lk_{room_name}_{participant.identity}"
                )
                participants[participant.identity] = state
                
                state.audio_source = rtc.AudioSource(24000, 1)
                state.audio_track = rtc.LocalAudioTrack.create_audio_track(f"agent-mic-{participant.identity}", state.audio_source)
                asyncio.create_task(room.local_participant.publish_track(state.audio_track))
                logger.info("[DEBUG] %s: Published agent audio track to room", asyncio.get_running_loop().time())
                
            if state.audio_pipeline_task and not state.audio_pipeline_task.done():
                state.audio_pipeline_task.cancel()
            if state.bridge:
                asyncio.create_task(state.bridge.close())
                
            # Create a DeepgramBridge specific to this participant
            async def _bridge_callback(transcript: str):
                logger.info("[DEBUG] %s: _bridge_callback() fired. participant=%s, transcript=%s", asyncio.get_running_loop().time(), participant.identity, transcript)
                payload = json.dumps({"type": "utterance_end", "transcript": transcript}).encode("utf-8")
                await room.local_participant.publish_data(payload, topic="agent-response", reliable=True)
                await on_utterance(participant.identity, transcript)
                
            async def _interim_callback(transcript: str):
                payload = json.dumps({"type": "interim_transcript", "text": transcript}).encode("utf-8")
                await room.local_participant.publish_data(payload, topic="agent-response", reliable=False)

            async def _speech_started_callback():
                state = participants.get(participant.identity)
                if state and state.agent_task and not state.agent_task.done():
                    logger.info("  Barge-in detected (SpeechStarted): Cancelling active agent task for %s", participant.identity)
                    state.agent_task.cancel()
                
            logger.info("[DEBUG] %s: Initializing DeepgramBridge for %s", asyncio.get_running_loop().time(), participant.identity)
            bridge = DeepgramBridge(
                on_utterance_end=_bridge_callback,
                on_interim_cb=_interim_callback,
                on_speech_started=_speech_started_callback
            )
            state.bridge = bridge
            
            logger.info("[DEBUG] %s: Spawning audio_pipeline task", asyncio.get_running_loop().time())
            task = asyncio.create_task(
                audio_pipeline(rtc.AudioStream(track), participant.identity, bridge),
                name=f"audio-pipeline-{participant.identity}"
            )
            state.audio_pipeline_task = task
            # The bridge will be lazy-started by audio_pipeline on the first frame

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        logger.info("  Track unsubscribed from %s", participant.identity)
        state = participants.get(participant.identity)
        if state:
            if state.audio_pipeline_task and not state.audio_pipeline_task.done():
                state.audio_pipeline_task.cancel()
            if state.bridge:
                asyncio.create_task(state.bridge.close())
                state.bridge = None

    @room.on("disconnected")
    def on_disconnected() -> None:
        logger.info("  Room disconnected")

    # ── Connect to the room ──────────────────────────────────────────────
    logger.info("Connecting to room '%s' at %s ...", room_name, settings.livekit_url)
    await room.connect(settings.livekit_url, jwt)
    logger.info("✓ Worker joined room: %s", room_name)
    logger.info("  Mode: Phase 7 — Production Barge-in")
    logger.info("  Waiting for participants to publish audio...")

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down worker...")
        for identity, state in participants.items():
            if state.agent_task and not state.agent_task.done():
                state.agent_task.cancel()
            if state.audio_pipeline_task and not state.audio_pipeline_task.done():
                state.audio_pipeline_task.cancel()
            if state.bridge:
                await state.bridge.close()
        
        await room.disconnect()
        
        try:
            await memory_manager.cleanup()
            logger.info("✓ Memory system cleaned up")
        except Exception as exc:
            logger.warning("Memory cleanup error: %s", exc)
        logger.info("✓ Worker shut down cleanly")


if __name__ == "__main__":
    room_name = sys.argv[1] if len(sys.argv) > 1 else "voice-vansh"
    try:
        asyncio.run(main(room_name))
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
