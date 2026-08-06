"""
Deepgram streaming STT bridge.

Owns the lifecycle of one Deepgram WebSocket: connect, reconnect with capped
backoff, transcript accumulation, and clean teardown.

The bridge reports *facts* about what the microphone heard and takes no policy
decisions. In particular it does not decide what constitutes an interruption —
that belongs to the caller, which is the only layer that knows whether the
assistant is currently speaking. An earlier version cancelled the in-flight
reply from inside this class on every interim transcript and on every raw VAD
event, which meant a cough, a keyboard tap, or the assistant's own voice
leaking back through the speakers silently killed the reply before a single
word was audible.

Turn detection uses two Deepgram signals:
  * ``speech_final`` on a final result — driven by ``endpointing``, typically
    ~300 ms after the speaker stops. This is the fast path.
  * ``UtteranceEnd`` — driven by ``utterance_end_ms``, which the API floors at
    1000 ms. This is the backstop for when endpointing does not fire (noisy
    audio, no clear silence boundary).

Whichever arrives first harvests and clears the transcript buffer, so the
second one finds nothing and cannot double-fire the same turn.

All Deepgram SDK configuration (model, sample rate, VAD, endpointing) lives in
``voice_service.create_deepgram_connection()`` so there is exactly one place
where Deepgram options are defined.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from deepgram import LiveTranscriptionEvents

from app.services.voice_service import voice_service

logger = logging.getLogger(__name__)

# Ceiling for reconnect backoff. Long enough to stop hammering a downed
# provider, short enough that recovery still feels immediate to the speaker.
_MAX_RECONNECT_BACKOFF = 8.0


class DeepgramBridge:
    """Manages a single Deepgram streaming WebSocket connection.

    Args:
        on_utterance_end: awaited with a complete user turn.
        on_interim: awaited with each partial transcript. The caller uses this
            both to show live text and to decide about interruption.
        on_speech_started: awaited on Deepgram's VAD onset. A *hint* only —
            VAD fires on any sound, so it must not be treated as "the user said
            something".
    """

    def __init__(
        self,
        on_utterance_end: Callable[[str], Awaitable[None]],
        on_interim: Optional[Callable[[str], Awaitable[None]]] = None,
        on_speech_started: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._on_utterance_end_cb = on_utterance_end
        self._on_interim_cb = on_interim
        self._on_speech_started_cb = on_speech_started

        # ── Connection state ─────────────────────────────────────────────
        self._connection = None
        self._connected: bool = False
        self._closed: bool = False          # set by close(), prevents reconnect

        # Serialises reconnects. Error events and per-frame ensure_connected()
        # calls can fire together during a network blip; without this lock each
        # would build its own socket, leaking the loser and delivering every
        # transcript twice.
        self._reconnect_lock = asyncio.Lock()
        self._consecutive_failures: int = 0

        # ── Transcript accumulation ──────────────────────────────────────
        self._current_transcript: str = ""
        self._transcript_lock = asyncio.Lock()

        # ── Background tasks owned by this bridge ────────────────────────
        self._background_tasks: set[asyncio.Task] = set()

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """True when the Deepgram WebSocket is alive and ready for audio."""
        return self._connected and not self._closed

    @property
    def consecutive_failures(self) -> int:
        """Number of back-to-back reconnect failures (0 once reconnected)."""
        return self._consecutive_failures

    async def start(self) -> bool:
        """Open the Deepgram WebSocket. Returns True on success."""
        if self._closed:
            logger.warning("[DeepgramBridge] Cannot start — bridge is closed")
            return False

        try:
            conn, options = voice_service.create_deepgram_connection()

            # Register handlers BEFORE start() so no open/error event is missed.
            conn.on(LiveTranscriptionEvents.Open, self._on_open)
            conn.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started_event)
            conn.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            conn.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end_event)
            conn.on(LiveTranscriptionEvents.Error, self._on_error)
            conn.on(LiveTranscriptionEvents.Close, self._on_close)

            if not await conn.start(options):
                raise RuntimeError("Deepgram WebSocket start() returned False")

            self._connection = conn
            self._connected = True
            # No keepalive task here: the SDK already runs one because
            # voice_service passes options={"keepalive": "true"}. Sending a
            # second stream of KeepAlive frames from this class was pure
            # duplicate work on every connection.
            logger.info("[DeepgramBridge] Streaming STT connected")
            return True

        except Exception as exc:
            logger.error("[DeepgramBridge] Failed to initialize: %s", exc)
            self._connected = False
            return False

    async def ensure_connected(self) -> bool:
        """
        Reconnect if the connection has dropped. No-op if already connected.

        Serialised by a lock so concurrent callers await one shared reconnect
        instead of racing to build competing sockets. Applies capped exponential
        backoff: the audio pipeline can reach this per frame (~50/s), which
        would otherwise hammer Deepgram throughout an outage.
        """
        if self._closed:
            return False
        if self._connected and self._connection:
            return True

        async with self._reconnect_lock:
            # Re-check under the lock: a concurrent caller may have reconnected
            # while this one waited, making another attempt unnecessary.
            if self._closed:
                return False
            if self._connected and self._connection:
                return True

            if self._consecutive_failures:
                backoff = min(
                    _MAX_RECONNECT_BACKOFF,
                    0.5 * (2 ** (self._consecutive_failures - 1)),
                )
                logger.info(
                    "[DeepgramBridge] Backing off %.1fs before reconnect attempt %d",
                    backoff, self._consecutive_failures + 1,
                )
                await asyncio.sleep(backoff)
                if self._closed:
                    return False

            logger.info("[DeepgramBridge] Reconnecting...")

            if self._connection:
                stale, self._connection = self._connection, None
                try:
                    await stale.finish()
                except Exception:
                    pass

            ok = await self.start()
            self._consecutive_failures = 0 if ok else self._consecutive_failures + 1
            return ok

    async def send(self, audio_bytes: bytes) -> bool:
        """Send raw linear16 / 16 kHz / mono audio. Returns False if dropped."""
        if not (self._connection and self._connected):
            return False
        try:
            await self._connection.send(audio_bytes)
            return True
        except Exception as exc:
            logger.warning("[DeepgramBridge] Send failed: %s", exc)
            self._connected = False
            return False

    async def close(self) -> None:
        """Cleanly shut down the connection and all background tasks."""
        self._closed = True
        self._connected = False

        # Cancel background reconnects first, so none of them resurrects a
        # connection after the bridge is meant to be gone.
        pending = [t for t in self._background_tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._background_tasks.clear()

        if self._connection:
            conn, self._connection = self._connection, None
            try:
                await conn.finish()
            except Exception as exc:
                logger.warning("[DeepgramBridge] Connection cleanup error: %s", exc)

        logger.info("[DeepgramBridge] Bridge closed")

    # ── Deepgram event handlers ──────────────────────────────────────────
    #
    # The SDK invokes handlers as ``handler(connection, payload=...)``. Because
    # these are bound methods, ``self`` is the bridge and ``args[0]`` is the SDK
    # connection — which is what makes the stale-connection guards below work.
    # ─────────────────────────────────────────────────────────────────────

    def _is_stale(self, args) -> bool:
        """True when an event belongs to a connection we have already replaced."""
        conn = args[0] if args else None
        return conn is not None and self._connection is not None and conn is not self._connection

    async def _on_open(self, *args, **kwargs):
        self._connected = True
        # A successful open clears the failure streak, so the next blip starts
        # its backoff from zero instead of inheriting an old escalation.
        self._consecutive_failures = 0
        logger.info("[DeepgramBridge] Connection opened")

    async def _on_close(self, *args, **kwargs):
        if self._is_stale(args):
            return
        self._connected = False
        logger.info("[DeepgramBridge] Connection closed")

    async def _on_error(self, *args, **kwargs):
        if self._is_stale(args):
            return
        error = kwargs.get("error") or (args[1] if len(args) > 1 else None)
        self._connected = False
        logger.warning("[DeepgramBridge] Connection error (will auto-reconnect): %s", error)

        if self._closed:
            return

        async def _silent_reconnect():
            await asyncio.sleep(0.5)  # brief pause for SDK cleanup
            if await self.ensure_connected():
                logger.info("[DeepgramBridge] Silently reconnected after error")
            else:
                logger.error("[DeepgramBridge] Reconnect failed after error: %s", error)

        self._spawn(_silent_reconnect(), "dg-reconnect")

    async def _on_speech_started_event(self, *args, **kwargs):
        """Deepgram VAD onset. A hint that *something* was heard, not speech."""
        if self._is_stale(args):
            return
        if self._on_speech_started_cb:
            try:
                await self._on_speech_started_cb()
            except Exception as exc:
                logger.error("[DeepgramBridge] on_speech_started error: %s", exc)

    async def _on_transcript(self, *args, **kwargs):
        """Handle interim and final transcript events."""
        if self._is_stale(args):
            return
        result = kwargs.get("result") or (args[1] if len(args) > 1 else None)
        if result is None:
            return

        try:
            sentence = result.channel.alternatives[0].transcript
        except (AttributeError, IndexError):
            return
        if not sentence or not sentence.strip():
            return

        if not result.is_final:
            logger.debug("[STT interim] %s", sentence)
            if self._on_interim_cb:
                try:
                    await self._on_interim_cb(sentence)
                except Exception as exc:
                    logger.error("[DeepgramBridge] on_interim error: %s", exc)
            return

        async with self._transcript_lock:
            self._current_transcript += sentence + " "
        logger.debug("[STT final] %s (speech_final=%s)", sentence, result.speech_final)

        # Endpointing says the speaker has stopped: start the reply now rather
        # than waiting the full utterance_end_ms floor of one second.
        if getattr(result, "speech_final", False):
            await self._flush_utterance("speech_final")

    async def _on_utterance_end_event(self, *args, **kwargs):
        """Backstop turn boundary for when endpointing did not fire."""
        if self._is_stale(args):
            return
        await self._flush_utterance("utterance_end")

    async def _flush_utterance(self, trigger: str) -> None:
        """Hand the buffered transcript to the caller as one complete turn.

        Harvest-and-clear under the lock is what makes speech_final and
        UtteranceEnd safe to both be wired up: whichever fires first takes the
        text, and the other finds an empty buffer and returns.
        """
        async with self._transcript_lock:
            full_text = self._current_transcript.strip()
            self._current_transcript = ""

        if not full_text:
            return

        logger.info("[STT turn:%s] %s", trigger, full_text)
        try:
            await self._on_utterance_end_cb(full_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[DeepgramBridge] Utterance callback error: %s", exc, exc_info=True
            )

    # ── Internals ────────────────────────────────────────────────────────

    def _spawn(self, coro, name: str) -> None:
        """Run a detached coroutine while retaining a strong reference.

        Without the retained reference asyncio may garbage-collect the task
        mid-run and any exception surfaces only as an "unretrieved exception"
        warning instead of through this logger.
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
