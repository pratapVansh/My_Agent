"""
LiveKit voice worker.

One worker serves one room. Per participant it runs a single pipeline:

    LiveKit audio track
      -> rtc.AudioStream (resamples to 16 kHz mono for us)
      -> Deepgram streaming STT
      -> turn detection (speech_final / UtteranceEnd)
      -> route (heuristic, LLM only when ambiguous)
      -> streaming LLM or tool-calling workflow
      -> TextChunker
      -> Cartesia TTS
      -> LiveKit AudioSource -> browser

Three invariants keep the conversation alive, and all three were once violated:

1. **Exactly one turn per participant at a time.** A new utterance cancels the
   previous turn and waits for it to finish unwinding before starting the next.
   Overlapping turns pushed two TTS streams into one audio source and both
   appended to the same history list.

2. **Every turn publishes a terminal event.** Success, cancellation, stall, and
   crash all end with ``turn_end`` on the data channel. The frontend gates its
   input on that event, so a turn that ends silently leaves the UI stuck on
   "processing" with no way back short of a page reload.

3. **A turn is bounded by progress, never by elapsed time.** Speech is
   real-time: ``capture_frame`` paces itself to playback, so a two-minute answer
   occupies the turn for two minutes by construction. Only the absence of
   progress — no token, no audio frame, no spoken word — means a turn is wedged.

On-screen text is driven by the *audio* clock, not the LLM clock. Groq emits a
full reply in a couple of seconds while speaking it takes far longer, so text
paced by token arrival races ahead of the voice. ``TTSStreamer`` reports when
each chunk will actually be heard and a caption pump reveals words against that
schedule.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import AsyncIterable, AsyncIterator, Awaitable, Callable, FrozenSet, Optional

from livekit.api import AccessToken, VideoGrants

from app.config import settings
from app.services.deepgram_bridge import DeepgramBridge
from app.agents.streaming_workflow import run_streaming_workflow
from app.agents.workflow import run_workflow
from app.agents.hybrid_router import ROUTE_TOOL, determine_route
from app.agents import interruption
from app.memory.memory_manager import memory_manager
from app.services.langsmith_service import configure_langsmith
from app.services.tts_streamer import SpokenChunk, tts_streamer
from app.services.text_chunker import text_chunker
from app.services.voice_service import STT_CHANNELS, STT_SAMPLE_RATE, TTSUnavailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DATA_TOPIC = "agent-response"

# Tolerate a couple of transient reconnects before telling the user STT is down.
_STT_FAILURES_BEFORE_NOTIFY = 3

# How long the room may sit empty before the worker exits. Workers are started
# on demand by /livekit/token and are process-local, so one that never exits is
# a permanent leak of a room connection and a Deepgram socket.
_IDLE_SHUTDOWN_SECONDS = 120.0

_TTS_FAILURE_NOTICE = (
    "I have a reply for you, but my voice output isn't working right now — "
    "it's on screen instead."
)


@dataclass
class TurnProgress:
    """Liveness and results for one spoken turn.

    ``last_activity`` is what the stall watchdog reads. Anything that proves the
    turn is still moving — a token from the LLM, an audio frame handed to
    LiveKit, a word revealed on screen — refreshes it via ``touch()``.
    """

    last_activity: float = field(default_factory=time.monotonic)
    # Text the listener has actually heard, accumulated by the caption pump. Used
    # for the interrupted-turn history, so what we record matches what was said
    # rather than what the model had already generated.
    spoken: str = ""
    # The authoritative full reply, known as soon as the LLM finishes.
    display: str = ""
    # Set by the watchdog before it cancels, so the turn can tell a genuine
    # barge-in apart from its own deadline firing.
    abort_reason: Optional[str] = None

    def touch(self) -> None:
        self.last_activity = time.monotonic()


def _log_task_failure(task: asyncio.Task) -> None:
    """Retrieve a detached task's outcome so failures are logged, not swallowed."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Task '%s' failed: %s", task.get_name(), exc, exc_info=exc)


async def speech_token_stream(
    events: AsyncIterable[dict],
    on_event: Callable[[dict], Awaitable[None]],
) -> AsyncIterator[str]:
    """Convert workflow events into a stream of speakable text.

    Tokens are forwarded as they stream. If a turn finishes without ever
    streaming a token, the completed reply is yielded instead: the planner's
    clarifying questions, unroutable requests, and LLM failures all emit only a
    ``complete`` event, so the chunker saw an empty stream and those answers were
    displayed but never spoken.
    """
    streamed_any = False
    async for event in events:
        await on_event(event)
        kind = event.get("type")
        if kind == "token":
            streamed_any = True
            yield event["token"]
        elif kind == "complete" and not streamed_any:
            text = (event.get("speech_text") or event.get("display_text") or "").strip()
            if text:
                yield text


async def watch_for_stall(
    identity: str, progress: TurnProgress, turn: asyncio.Task
) -> None:
    """Cancel a turn only once it has genuinely stopped making progress.

    Deliberately not a wall-clock deadline. Handing frames to LiveKit paces
    itself to playback, so the elapsed time of a healthy turn equals the length
    of the answer being spoken — a fixed deadline cancels long replies
    mid-sentence, which is exactly what it must not do.

    ``abort_reason`` is set *before* cancelling so the turn can tell this apart
    from a genuine barge-in and report the right thing to the user.
    """
    stall = settings.voice_turn_stall_seconds
    hard_deadline = time.monotonic() + settings.voice_turn_max_seconds

    while not turn.done():
        idle = time.monotonic() - progress.last_activity
        if idle >= stall:
            progress.abort_reason = "stalled"
            logger.error(
                "Turn for %s made no progress for %.1fs — cancelling", identity, idle
            )
            turn.cancel()
            return
        if time.monotonic() >= hard_deadline:
            progress.abort_reason = "too_long"
            logger.error(
                "Turn for %s hit the %.0fs hard ceiling — cancelling",
                identity, settings.voice_turn_max_seconds,
            )
            turn.cancel()
            return
        # Wake when the current idle window would expire, capped so the hard
        # deadline is still noticed promptly.
        await asyncio.sleep(max(0.05, min(stall - idle, 2.0)))


@dataclass
class ParticipantState:
    identity: str
    session_id: str
    history: list[dict] = field(default_factory=list)

    # STT
    bridge: Optional[DeepgramBridge] = None
    audio_pipeline_task: Optional[asyncio.Task] = None

    # TTS output
    audio_source: "Optional[rtc.AudioSource]" = None
    audio_track: "Optional[rtc.LocalAudioTrack]" = None
    track_ready: asyncio.Event = field(default_factory=asyncio.Event)

    # Turn management
    turn_task: Optional[asyncio.Task] = None
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # True only while TTS frames are actually being handed to LiveKit. Ordinary
    # speech barge-in is gated on this: interrupting a turn that has not started
    # speaking yet just loses the reply without the user ever hearing anything
    # to interrupt. An explicit stop command deliberately ignores this gate —
    # see `interruption` — because the wait itself is what is being cut short.
    speaking: bool = False

    # Set immediately before a cancellation the *user* asked for, so the turn's
    # CancelledError handler can tell a deliberate stop apart from a stall or an
    # ordinary barge-in and stay silent instead of apologising for itself.
    stopped_by_user: bool = False


async def main(
    room_name: str,
    scopes: FrozenSet[str] | None = None,
    on_shutdown: Optional[Callable[[], None]] = None,
) -> None:
    """
    Run the voice agent for a room until the room disconnects or goes idle.

    Args:
        room_name: LiveKit room to join.
        scopes: Capabilities of the authenticated participant, forwarded to the
            workflow so voice turns enforce exactly the same tool permissions as
            HTTP requests. None means unrestricted (local CLI use only).
        on_shutdown: called synchronously the moment teardown begins, before any
            await. The caller uses it to deregister this worker so a client
            connecting during teardown gets a fresh worker rather than silently
            joining a room this one is in the middle of leaving.
    """
    try:
        from livekit import rtc
    except ImportError:
        logger.error("Missing dependency: pip install livekit>=1.0.0")
        return

    if not settings.is_livekit_configured:
        logger.error(
            "LiveKit not configured. Set LIVEKIT_URL, LIVEKIT_API_KEY, "
            "LIVEKIT_API_SECRET in .env"
        )
        return

    if not settings.is_streaming_stt_available:
        logger.error("Deepgram not configured. Set DEEPGRAM_API_KEY in .env")
        return

    configure_langsmith()

    try:
        await memory_manager.initialize()
        logger.info("✓ Memory system initialized")
    except Exception as exc:
        logger.warning("Memory initialization failed (continuing): %s", exc)

    participants: dict[str, ParticipantState] = {}
    room = rtc.Room()
    shutdown = asyncio.Event()

    # Strong references to detached tasks. Without a retained reference asyncio
    # may garbage-collect a pending task mid-run, and any exception it raised
    # would surface only as an "unretrieved exception" warning.
    background: set[asyncio.Task] = set()

    def spawn(coro, description: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=description)
        background.add(task)

        def _done(t: asyncio.Task) -> None:
            background.discard(t)
            try:
                t.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Task '%s' failed: %s", description, exc, exc_info=True)

        task.add_done_callback(_done)
        return task

    # ── Data channel ─────────────────────────────────────────────────────

    async def publish(event: dict, reliable: bool = True) -> None:
        """Send one event to the browser, tolerating a closed room."""
        try:
            await room.local_participant.publish_data(
                json.dumps(event).encode("utf-8"),
                topic=_DATA_TOPIC,
                reliable=reliable,
            )
        except Exception as exc:
            # A disconnected room is the normal case during teardown; losing a
            # UI event must never take down the turn that was producing it.
            logger.debug("publish(%s) failed: %s", event.get("type"), exc)

    # ── One spoken turn ──────────────────────────────────────────────────

    async def execute_turn(state: ParticipantState, transcript: str) -> None:
        """Produce and speak one reply, always publishing a terminal event."""
        progress = TurnProgress()
        terminal: dict = {"type": "turn_end", "reason": "complete"}

        state.history.append({"role": "user", "content": transcript})

        # The watchdog cancels this very task rather than a child, so there is no
        # second task that can outlive the turn and keep talking.
        guard = asyncio.create_task(
            watch_for_stall(state.identity, progress, asyncio.current_task()),
            name=f"stall-guard-{state.identity}",
        )
        guard.add_done_callback(_log_task_failure)

        # Each branch below assigns `terminal` *before* awaiting anything, so a
        # second cancellation arriving mid-publish still leaves the finally block
        # with the correct reason to report.
        try:
            await produce_turn(state, transcript, progress)
        except asyncio.CancelledError:
            if progress.abort_reason is None:
                # A real barge-in. Keep what was actually heard so the next
                # turn's context matches the user's experience of it.
                partial = (progress.spoken or progress.display).strip()
                if partial:
                    state.history.append(
                        {"role": "assistant", "content": f"{partial} [interrupted]"}
                    )
                terminal = (
                    {"type": "turn_end", "reason": "stopped_by_user",
                     "partial_text": partial}
                    if state.stopped_by_user
                    else {"type": "interrupted", "partial_text": partial}
                )
                raise

            # The watchdog cancelled us. Absorb the cancellation — this turn is
            # being ended deliberately, not interrupted — so the handler below can
            # still run and report. Re-raising would mark the task cancelled and
            # the terminal event would never reach the browser.
            reason = progress.abort_reason
            terminal = {"type": "turn_end", "reason": reason}
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            await publish({
                "type": "error",
                "message": (
                    "That answer stopped partway through. Please try again."
                    if reason == "stalled"
                    else "That answer was too long to finish. Please ask for a shorter one."
                ),
                "source": reason,
            })
        except TTSUnavailable as exc:
            logger.error("TTS unavailable for %s: %s", state.identity, exc)
            terminal = {"type": "turn_end", "reason": "tts_unavailable"}
            await publish({
                "type": "error", "message": _TTS_FAILURE_NOTICE, "source": "tts",
            })
        except Exception as exc:
            logger.error("Turn failed for %s: %s", state.identity, exc, exc_info=True)
            terminal = {"type": "turn_end", "reason": "error"}
            await publish({
                "type": "error",
                "message": "Sorry, something went wrong handling that. Please try again.",
                "source": "worker",
            })
        finally:
            # Cancel without awaiting: this block runs while a cancellation may
            # still be in flight, and an await here could raise before the
            # terminal event is published — the one thing that must never happen.
            guard.cancel()
            state.speaking = False
            spawn(publish(terminal), f"turn-end-{state.identity}")

    async def produce_turn(
        state: ParticipantState, transcript: str, progress: TurnProgress
    ) -> None:
        """Route, generate, and speak. `progress` carries results and liveness out."""
        route = await determine_route(transcript, list(state.history))
        logger.info("▶ turn user=%s route=%s | %s", state.identity, route, transcript[:80])
        progress.touch()

        if route == ROUTE_TOOL:
            result = await run_workflow(
                user_input=transcript,
                user_id=state.identity,
                session_id=state.session_id,
                conversation_history=list(state.history),
                output_mode="user",
                scopes=scopes,
            )
            progress.touch()
            display_text = result.get("display_text") or ""
            speech_text = result.get("speech_text") or display_text
            progress.display = display_text

            await publish({
                "type": "final_response",
                # run_workflow reports failure through `error`; there is no
                # `success` key, so reading one always yielded True.
                "success": result.get("error") is None,
                "transcript": transcript,
                "display_text": display_text,
                "metadata": {
                    "selected_agent": result.get("selected_agent"),
                    "source": "livekit",
                    "route": route,
                },
            })
            remember(state, display_text)

            async def token_stream() -> AsyncIterable[str]:
                # The tool path has no token stream; hand the finished reply to
                # the chunker as one block so TTS still starts on the first
                # clause rather than after the whole paragraph.
                yield speech_text

        else:
            workflow = run_streaming_workflow(
                user_input=transcript,
                user_id=state.identity,
                session_id=state.session_id,
                conversation_history=list(state.history),
                scopes=scopes,
            )

            selected_agent: Optional[str] = None

            async def on_event(event: dict) -> None:
                # Deliberately publishes no per-token text. On-screen words come
                # from the caption pump on the audio clock: the model finishes a
                # reply in seconds that takes a minute to speak, so text paced by
                # token arrival raced far ahead of the voice.
                nonlocal selected_agent
                progress.touch()
                if event.get("type") == "metadata":
                    selected_agent = event.get("selected_agent")
                elif event.get("type") == "complete":
                    display_text = event.get("display_text") or ""
                    progress.display = display_text
                    await publish({
                        "type": "final_response",
                        "success": bool(event.get("success", True)),
                        "transcript": transcript,
                        "display_text": display_text,
                        "metadata": {
                            "selected_agent": selected_agent,
                            "source": "livekit",
                            "route": route,
                        },
                    })
                    remember(state, display_text)

            def token_stream() -> AsyncIterable[str]:
                return speech_token_stream(workflow, on_event)

        await speak(state, token_stream(), progress)

    def remember(state: ParticipantState, text: str) -> None:
        """Append the assistant's reply to in-process turn history."""
        if not text:
            return
        state.history.append({"role": "assistant", "content": text})
        limit = settings.voice_history_turns * 2
        if len(state.history) > limit:
            del state.history[:-limit]

    async def caption_pump(
        state: ParticipantState,
        queue: "asyncio.Queue[Optional[SpokenChunk]]",
        progress: TurnProgress,
    ) -> None:
        """Reveal text word by word, timed to when each word is actually heard.

        Each chunk carries the wall-clock instant its audio starts playing, so
        words are paced against the audio clock and re-anchored on every chunk —
        drift cannot accumulate over a long reply. Deltas go out reliably (LiveKit
        orders them) and `final_response` remains the authoritative full text.
        """
        offset = settings.voice_caption_offset_seconds
        while True:
            chunk = await queue.get()
            if chunk is None:
                return

            words = chunk.text.split()
            if not words:
                continue
            per_word = chunk.duration / len(words)

            for index, word in enumerate(words):
                target = chunk.starts_at + offset + index * per_word
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                progress.spoken = f"{progress.spoken} {word}".strip()
                progress.touch()
                await publish({"type": "caption", "delta": word})

    async def speak(
        state: ParticipantState, tokens: AsyncIterable[str], progress: TurnProgress
    ) -> None:
        """Chunk tokens, stream them to the audio track, and caption them."""
        # The track is published asynchronously on join; frames handed over
        # before publishing completes are discarded by LiveKit, which silently
        # ate the beginning of the first reply.
        try:
            await asyncio.wait_for(state.track_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            raise TTSUnavailable(
                f"agent audio track never became ready for {state.identity}"
            ) from None
        if state.audio_source is None:
            raise TTSUnavailable("agent audio source is gone")

        audio_source = state.audio_source
        queue: "asyncio.Queue[Optional[SpokenChunk]]" = asyncio.Queue()
        pump = asyncio.create_task(
            caption_pump(state, queue, progress), name=f"captions-{state.identity}"
        )
        pump.add_done_callback(_log_task_failure)

        state.speaking = True
        try:
            frames = await tts_streamer.stream_to_track(
                text_chunker.chunk_tokens(tokens),
                audio_source,
                on_chunk=queue.put_nowait,
                on_progress=progress.touch,
            )
            if frames == 0:
                logger.warning("Turn for %s produced no audio", state.identity)
                return

            # Let the captions finish, then wait for the tail of the audio still
            # sitting in LiveKit's playout buffer. Only now has the listener
            # actually heard the whole reply, which is what makes `turn_end` the
            # right moment for the UI to commit the message and stop the
            # speaking indicator.
            queue.put_nowait(None)
            await pump
            progress.touch()
            await audio_source.wait_for_playout()
        finally:
            state.speaking = False
            pump.cancel()

    async def start_turn(state: ParticipantState, transcript: str) -> None:
        """Cancel any in-flight turn, then run this one. Serialised per participant."""
        async with state.turn_lock:
            await cancel_turn(state)
            state.turn_task = asyncio.create_task(
                execute_turn(state, transcript), name=f"turn-{state.identity}"
            )

            def _done(t: asyncio.Task) -> None:
                try:
                    t.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.error("Turn task crashed for %s: %s", state.identity, exc)

            state.turn_task.add_done_callback(_done)

    async def cancel_turn(state: ParticipantState) -> None:
        """Stop the current turn and wait for it to unwind.

        Awaiting matters: returning while the old turn is still inside
        capture_frame leaves its buffered speech playing over the new one.
        """
        task, state.turn_task = state.turn_task, None
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if state.audio_source is not None:
            await tts_streamer.clear_pending_audio(state.audio_source)
        state.speaking = False

    # ── Audio track (agent -> browser) ───────────────────────────────────

    async def ensure_agent_track(state: ParticipantState) -> None:
        """Create and publish this participant's agent audio track exactly once."""
        if state.audio_source is not None:
            return
        # The streamer owns the format and the playout depth; caption timing is
        # derived from that same queue, so both must come from one place.
        state.audio_source = tts_streamer.create_audio_source()
        state.audio_track = rtc.LocalAudioTrack.create_audio_track(
            f"agent-voice-{state.identity}", state.audio_source
        )
        try:
            await room.local_participant.publish_track(
                state.audio_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            state.track_ready.set()
            logger.info(
                "Published agent audio track for %s (%d Hz)",
                state.identity, tts_streamer.sample_rate,
            )
        except Exception as exc:
            logger.error("Could not publish agent track for %s: %s", state.identity, exc)
            source, state.audio_source = state.audio_source, None
            state.audio_track = None
            if source is not None:
                try:
                    await source.aclose()
                except Exception:
                    pass

    # ── Audio pipeline (browser -> Deepgram) ─────────────────────────────

    async def audio_pipeline(
        stream: "rtc.AudioStream", identity: str, bridge: DeepgramBridge
    ) -> None:
        """Forward microphone audio to Deepgram.

        AudioStream is constructed with Deepgram's exact input format, so it does
        the resampling internally. Doing it again with a second
        rtc.AudioResampler here was redundant work on every frame, and its
        channel count was hardcoded independently of the stream's.
        """
        frames = 0
        notified = False
        try:
            async for event in stream:
                frames += 1
                if not bridge.connected:
                    # ensure_connected() de-duplicates concurrent attempts and
                    # applies its own capped backoff, so calling it from the
                    # frame path does not mean one dial-out per frame.
                    if not await bridge.ensure_connected():
                        if (
                            not notified
                            and bridge.consecutive_failures >= _STT_FAILURES_BEFORE_NOTIFY
                        ):
                            logger.error(
                                "STT unavailable for %s after %d failed reconnects",
                                identity, bridge.consecutive_failures,
                            )
                            await publish({
                                "type": "error",
                                "message": (
                                    "Speech recognition is temporarily unavailable. "
                                    "Please try again shortly."
                                ),
                                "source": "stt",
                            })
                            notified = True
                        # Drop this frame but keep consuming, so the pipeline
                        # recovers by itself once Deepgram returns.
                        continue
                    notified = False
                await bridge.send(bytes(event.frame.data))
        except asyncio.CancelledError:
            logger.info("Audio pipeline cancelled for %s", identity)
        except Exception as exc:
            logger.warning("Audio pipeline error for %s: %s", identity, exc, exc_info=True)
        finally:
            logger.debug("Audio pipeline closed for %s (frames=%d)", identity, frames)

    def get_or_create_state(identity: str) -> ParticipantState:
        state = participants.get(identity)
        if state is None:
            # Attach to the conversation the browser is showing, so speaking and
            # typing continue one thread. Falling back to a room-derived id keeps
            # voice working when the client is older than this field or storage
            # was unavailable — a separate thread is worse than none.
            from app.routes.livekit_routes import resolve_voice_conversation

            conversation_id = (
                resolve_voice_conversation(room_name, identity)
                or f"lk_{room_name}_{identity}"
            )
            logger.info(
                "Voice participant %s attached to conversation %s", identity, conversation_id
            )
            state = ParticipantState(identity=identity, session_id=conversation_id)
            participants[identity] = state
        return state

    async def attach_stt(track: "rtc.Track", identity: str) -> None:
        """Wire a newly subscribed microphone track to a fresh Deepgram bridge."""
        logger.info("Audio track subscribed for %s", identity)
        state = get_or_create_state(identity)
        await ensure_agent_track(state)

        # Tear the previous pipeline down completely first. Overlapping bridges
        # stream the same audio to two Deepgram sockets and deliver every
        # transcript — and therefore every reply — twice.
        if state.audio_pipeline_task and not state.audio_pipeline_task.done():
            state.audio_pipeline_task.cancel()
            await asyncio.gather(state.audio_pipeline_task, return_exceptions=True)
        if state.bridge:
            old, state.bridge = state.bridge, None
            try:
                await old.close()
            except Exception as exc:
                logger.error("Error closing previous bridge for %s: %s", identity, exc)

        async def on_turn(transcript: str) -> None:
            await publish({"type": "utterance_end", "transcript": transcript})

            # A bare stop command is not a question. Cancelling the turn and
            # then handing "stop" to the language model produces a reply *to
            # the word stop* — the old task ends and a new one immediately
            # takes its place, which from the listener's side is the previous
            # task carrying on. So it is cancelled here and nothing is asked.
            if interruption.is_pure_stop(transcript):
                logger.info("Stop command from %s: %r", identity, transcript[:40])
                state.stopped_by_user = True
                await cancel_turn(state)
                state.stopped_by_user = False
                # Record what was said so a follow-up ("carry on") still has a
                # thread, without inviting an answer to it.
                state.history.append({"role": "user", "content": transcript})
                await publish({"type": "turn_end", "reason": "stopped_by_user"})
                return

            # "No, tell me about my projects" cancels *and* asks. Cancel first
            # so the old turn stops immediately, then run the remaining request
            # rather than dropping a question the user did ask.
            if interruption.carries_new_request(transcript):
                request = interruption.strip_stop_prefix(transcript)
                logger.info(
                    "Stop-with-request from %s: %r → %r",
                    identity, transcript[:40], request[:40],
                )
                state.stopped_by_user = True
                await cancel_turn(state)
                state.stopped_by_user = False
                spawn(start_turn(state, request), f"start-turn-{identity}")
                return

            spawn(start_turn(state, transcript), f"start-turn-{identity}")

        async def on_interim(transcript: str) -> None:
            await publish({"type": "interim_transcript", "text": transcript}, reliable=False)
            # Transcript-confirmed barge-in. Requiring real words (rather than
            # acting on raw VAD) is what stops a cough, a keypress, or the
            # assistant's own voice echoing back from killing the reply.
            task = state.turn_task
            if not task or task.done():
                return

            stop = interruption.is_stop_command(transcript)

            # An explicit stop command cancels whatever the turn is doing —
            # including an LLM call or a tool round-trip that has not produced
            # a syllable yet. Gating on `speaking` made "stop" a no-op during
            # exactly the wait people most want to cut short, and the word-count
            # floor rejected every one-word way of saying it.
            if not stop:
                if not state.speaking:
                    return
                if len(transcript.split()) < settings.voice_bargein_min_words:
                    return

            logger.info(
                "Barge-in by %s (%s): %s",
                identity, "stop command" if stop else "speech", transcript[:40],
            )
            if stop:
                # Mark it before cancelling so the turn's CancelledError handler
                # reports a user stop rather than a stall, and so the audio
                # already queued is dropped rather than played out.
                state.stopped_by_user = True
            task.cancel()
            if stop and state.audio_source is not None:
                # Drop buffered speech immediately. Without this the sentence
                # in LiveKit's playout buffer keeps playing after the cancel,
                # which is precisely what "stop" was asking not to happen.
                try:
                    await tts_streamer.clear_pending_audio(state.audio_source)
                except Exception as exc:
                    logger.debug("clear_pending_audio after stop failed: %s", exc)
                state.speaking = False

        async def on_speech_started() -> None:
            # VAD onset is a hint only — it fires on any sound. Acting on it
            # cancelled replies before a single word was audible.
            logger.debug("VAD onset for %s (speaking=%s)", identity, state.speaking)

        bridge = DeepgramBridge(
            on_utterance_end=on_turn,
            on_interim=on_interim,
            on_speech_started=on_speech_started,
        )
        state.bridge = bridge

        # Ask AudioStream for exactly what Deepgram was configured with.
        state.audio_pipeline_task = asyncio.create_task(
            audio_pipeline(
                rtc.AudioStream(
                    track, sample_rate=STT_SAMPLE_RATE, num_channels=STT_CHANNELS
                ),
                identity,
                bridge,
            ),
            name=f"audio-pipeline-{identity}",
        )

    async def teardown_participant(identity: str) -> None:
        state = participants.pop(identity, None)
        if not state:
            return
        await cancel_turn(state)
        if state.audio_pipeline_task and not state.audio_pipeline_task.done():
            state.audio_pipeline_task.cancel()
            await asyncio.gather(state.audio_pipeline_task, return_exceptions=True)
        if state.bridge:
            await state.bridge.close()
            state.bridge = None
        if state.audio_track:
            try:
                await room.local_participant.unpublish_track(state.audio_track.sid)
            except Exception as exc:
                logger.debug("unpublish_track for %s failed: %s", identity, exc)
        if state.audio_source:
            try:
                await state.audio_source.aclose()
            except Exception as exc:
                logger.debug("audio_source.aclose for %s failed: %s", identity, exc)
        state.audio_source = None
        state.audio_track = None
        logger.info("Cleaned up participant %s", identity)

    # ── Room events ──────────────────────────────────────────────────────

    @room.on("participant_connected")
    def _on_participant_connected(participant: "rtc.RemoteParticipant") -> None:
        logger.info("✓ Participant connected: %s", participant.identity)
        spawn(
            ensure_agent_track(get_or_create_state(participant.identity)),
            f"track-{participant.identity}",
        )

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: "rtc.RemoteParticipant") -> None:
        logger.info("Participant disconnected: %s", participant.identity)
        spawn(teardown_participant(participant.identity), f"teardown-{participant.identity}")

    @room.on("track_subscribed")
    def _on_track_subscribed(
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            spawn(attach_stt(track, participant.identity), f"stt-{participant.identity}")

    @room.on("track_unsubscribed")
    def _on_track_unsubscribed(
        track: "rtc.Track",
        publication: "rtc.RemoteTrackPublication",
        participant: "rtc.RemoteParticipant",
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        logger.info("Track unsubscribed from %s", participant.identity)
        state = participants.get(participant.identity)
        if not state:
            return

        async def stop_stt() -> None:
            if state.audio_pipeline_task and not state.audio_pipeline_task.done():
                state.audio_pipeline_task.cancel()
                await asyncio.gather(state.audio_pipeline_task, return_exceptions=True)
            if state.bridge:
                bridge, state.bridge = state.bridge, None
                await bridge.close()

        spawn(stop_stt(), f"stop-stt-{participant.identity}")

    @room.on("disconnected")
    def _on_disconnected(*args) -> None:
        # Exiting is essential, not cosmetic. The worker is registered in
        # active_workers by room name; if it lingers on a dead room the entry
        # never clears, so the next /livekit/token call for that user reuses a
        # worker that will never hear them and nothing ever answers again.
        logger.warning("Room disconnected — shutting worker down so it can be restarted")
        shutdown.set()

    # ── Connect ──────────────────────────────────────────────────────────

    jwt = (
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
        .to_jwt()
    )

    logger.info("Connecting to room '%s' at %s ...", room_name, settings.livekit_url)
    await room.connect(settings.livekit_url, jwt)
    logger.info("✓ Worker joined room: %s", room_name)

    # participant_connected does not fire for participants already in the room,
    # and the browser normally joins first because it is what starts this
    # worker. Without this sweep the very first caller can be ignored entirely.
    for participant in list(room.remote_participants.values()):
        logger.info("Adopting participant already in room: %s", participant.identity)
        spawn(
            ensure_agent_track(get_or_create_state(participant.identity)),
            f"track-{participant.identity}",
        )
        for publication in list(participant.track_publications.values()):
            track = getattr(publication, "track", None)
            if track is not None and track.kind == rtc.TrackKind.KIND_AUDIO:
                spawn(attach_stt(track, participant.identity), f"stt-{participant.identity}")

    async def idle_watchdog() -> None:
        """Exit once the room has been empty for a while, so workers don't leak."""
        while not shutdown.is_set():
            await asyncio.sleep(15.0)
            if room.remote_participants:
                continue
            logger.info(
                "Room %s empty; shutting down in %.0fs unless someone joins",
                room_name, _IDLE_SHUTDOWN_SECONDS,
            )
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=_IDLE_SHUTDOWN_SECONDS)
            except asyncio.TimeoutError:
                if not room.remote_participants:
                    logger.info("Room %s idle — worker exiting", room_name)
                    shutdown.set()

    watchdog = asyncio.create_task(idle_watchdog(), name="idle-watchdog")

    try:
        await shutdown.wait()
    except asyncio.CancelledError:
        pass
    finally:
        # Deregister first, synchronously. Anyone requesting a token from here
        # on gets a brand-new worker instead of joining a room this one is
        # already leaving.
        if on_shutdown is not None:
            try:
                on_shutdown()
            except Exception as exc:
                logger.error("on_shutdown callback failed: %s", exc)

        logger.info("Shutting down worker for room %s...", room_name)
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)

        for identity in list(participants):
            try:
                await teardown_participant(identity)
            except Exception as exc:
                logger.error("Teardown failed for %s: %s", identity, exc)

        for task in list(background):
            task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)

        try:
            await room.disconnect()
        except Exception as exc:
            logger.debug("room.disconnect() failed: %s", exc)

        # Deliberately does NOT call memory_manager.cleanup(). This worker runs
        # as a task inside the FastAPI process and shares the process-wide
        # Postgres engine with every REST request and every other room's worker;
        # cleanup() disposes that engine. Engine lifecycle belongs solely to the
        # application lifespan handler in app/main.py.
        logger.info("✓ Worker shut down cleanly (shared resources left intact)")


if __name__ == "__main__":
    try:
        asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "voice-vansh"))
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
