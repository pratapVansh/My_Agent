"""
Deepgram Streaming Bridge — Phase 2.

Manages the lifecycle of a Deepgram WebSocket connection for streaming
speech-to-text.  Handles connection, reconnection, keepalive, transcript
accumulation (with proper locking), and cleanup.

Designed as a callback-driven service:
  - Phase 2: consumer passes ``print()`` as the callback.
  - Phase 3: consumer passes ``run_workflow()`` instead — zero bridge changes.

All Deepgram SDK configuration (model, sample_rate, encoding, VAD, etc.)
is delegated to ``voice_service.create_deepgram_connection()`` so there is
exactly ONE place where Deepgram options are defined.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from deepgram import LiveTranscriptionEvents

from app.services.voice_service import voice_service

logger = logging.getLogger(__name__)

# Type alias: async callback that receives the completed transcript string.
OnUtteranceEnd = Callable[[str], Awaitable[None]]


class DeepgramBridge:
    """
    Manages a single Deepgram streaming WebSocket connection.

    Usage::

        bridge = DeepgramBridge(on_utterance_end=my_callback)
        await bridge.start()
        await bridge.send(audio_bytes)   # linear16, 16 kHz, mono
        ...
        await bridge.close()
    """

    def __init__(
        self,
        on_utterance_end: Callable[[str], Awaitable[None]],
        on_interim_cb: Optional[Callable[[str], Awaitable[None]]] = None,
        on_speech_started: Optional[Callable[[], Awaitable[None]]] = None,
        keepalive_interval: float = 3.0,
    ) -> None:
        self._on_utterance_end_cb = on_utterance_end
        self._on_interim_cb = on_interim_cb
        self._on_speech_started = on_speech_started
        self._keepalive_interval = keepalive_interval

        # ── Connection state ─────────────────────────────────────────────
        self._connection = None
        self._connected: bool = False
        self._closed: bool = False          # set True by close(), prevents reconnect

        # ── Transcript accumulation ──────────────────────────────────────
        self._current_transcript: str = ""
        self._transcript_lock = asyncio.Lock()

        # ── Background tasks owned by this bridge ────────────────────────
        self._keepalive_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """True when the Deepgram WebSocket is alive and ready for audio."""
        return self._connected and not self._closed

    async def start(self) -> bool:
        """Initialize and start the Deepgram WebSocket connection.

        Returns True on success, False on failure.
        """
        if self._closed:
            logger.warning("[DeepgramBridge] Cannot start — bridge is closed")
            return False

        try:
            logger.info("[DeepgramBridge] Initializing streaming STT...")

            # Reuse the project-wide Deepgram configuration from voice_service.
            conn, options = voice_service.create_deepgram_connection()

            # Register handlers BEFORE starting to prevent missed events.
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

            # Start keepalive loop (well within Deepgram's 10 s idle timeout)
            self._keepalive_task = asyncio.create_task(
                self._keepalive_loop(), name="dg-keepalive"
            )

            logger.info("[DeepgramBridge] Streaming STT initialized successfully")
            return True

        except Exception as exc:
            logger.error("[DeepgramBridge] Failed to initialize: %s", exc)
            self._connected = False
            return False

    async def ensure_connected(self) -> bool:
        """Reconnect if the connection has dropped.  No-op if already connected."""
        if self._closed:
            return False
        if self._connected and self._connection:
            return True

        logger.info("[DeepgramBridge] Reconnecting...")

        # Tear down stale connection
        if self._connection:
            try:
                await self._connection.finish()
            except Exception:
                pass
            self._connection = None

        # Cancel stale keepalive task
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):
                pass

        return await self.start()

    async def send(self, audio_bytes: bytes) -> None:
        """Send raw audio bytes to Deepgram.

        The caller is responsible for providing audio in the format that
        Deepgram expects (linear16, 16 kHz, mono) — the same format
        configured by ``voice_service.create_deepgram_connection()``.
        """
        if self._connection and self._connected:
            try:
                await self._connection.send(audio_bytes)
            except Exception as exc:
                logger.warning("[DeepgramBridge] Send failed: %s", exc)
                self._connected = False

    async def close(self) -> None:
        """Cleanly shut down the connection and all background tasks."""
        self._closed = True

        # Cancel keepalive first
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._keepalive_task = None

        # Finish Deepgram connection
        if self._connection:
            try:
                await self._connection.finish()
            except Exception as exc:
                logger.warning("[DeepgramBridge] Connection cleanup error: %s", exc)
            self._connection = None

        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        self._background_tasks.clear()

        self._connected = False
        logger.info("[DeepgramBridge] Bridge closed")

    # ── Deepgram event handlers ──────────────────────────────────────────
    #
    # The Deepgram SDK passes the connection object as the first positional
    # argument to all event handlers.  Because these are *bound methods*,
    # ``self`` is already the DeepgramBridge instance and ``_conn`` receives
    # the SDK connection.
    # ─────────────────────────────────────────────────────────────────────

    async def _on_open(self, *args, **kwargs):
        self._connected = True
        logger.info("[DeepgramBridge] Connection opened")

    async def _on_close(self, *args, **kwargs):
        # Guard against stale close events from a previous connection
        _conn = args[0] if args else None
        if _conn is not None and _conn is not self._connection:
            return
        self._connected = False
        logger.info("[DeepgramBridge] Connection closed")

    async def _on_error(self, *args, **kwargs):
        _conn = args[0] if args else None
        error = args[1] if len(args) > 1 else kwargs.get('error')
        if _conn is not None and _conn is not self._connection:
            return
        self._connected = False
        logger.warning(
            "[DeepgramBridge] Connection error (will auto-reconnect): %s", error
        )

        # Schedule a silent reconnect in the background.
        async def _silent_reconnect():
            await asyncio.sleep(0.5)  # brief pause for SDK cleanup
            ok = await self.ensure_connected()
            if ok:
                logger.info("[DeepgramBridge] Silently reconnected after error")
            else:
                logger.error("[DeepgramBridge] Reconnect failed after error: %s", error)

        task = asyncio.create_task(_silent_reconnect(), name="dg-reconnect")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _on_speech_started_event(self, *args, **kwargs):
        """Handle Deepgram VAD SpeechStarted: trigger early barge-in."""
        logger.info("[STT SpeechStarted] VAD detected speech")
        if self._on_speech_started:
            try:
                await self._on_speech_started()
            except Exception as exc:
                logger.error("[DeepgramBridge] on_speech_started error: %s", exc)

    async def _on_transcript(self, *args, **kwargs):
        """Handle interim and final transcript events from Deepgram."""
        result = args[1] if len(args) > 1 else kwargs.get('result')
        if not result:
            return
        sentence = result.channel.alternatives[0].transcript
        if not sentence:
            return

        if result.is_final:
            # Acquire lock to prevent race with _on_utterance_end_event
            async with self._transcript_lock:
                self._current_transcript += sentence + " "
            logger.info("[STT final]   %s", sentence)
        else:
            logger.info("[STT interim] %s", sentence)
            
            # Fallback VAD: if we get interim transcript, user is definitely speaking
            if self._on_speech_started:
                try:
                    await self._on_speech_started()
                except Exception as exc:
                    pass

            if self._on_interim_cb:
                try:
                    await self._on_interim_cb(sentence)
                except Exception as exc:
                    logger.error("[DeepgramBridge] on_interim_cb error: %s", exc)

    async def _on_utterance_end_event(self, *args, **kwargs):
        """Handle Deepgram VAD utterance-end: harvest transcript and fire callback."""
        async with self._transcript_lock:
            full_text = self._current_transcript.strip()
            self._current_transcript = ""

        if full_text:
            logger.info("[STT utterance_end] %s", full_text)
            try:
                await self._on_utterance_end_cb(full_text)
            except Exception as exc:
                logger.error(
                    "[DeepgramBridge] Utterance callback error: %s", exc, exc_info=True
                )

    # ── Internal background tasks ────────────────────────────────────────

    async def _keepalive_loop(self) -> None:
        """Send keepalive every N seconds to prevent Deepgram's 10 s idle timeout."""
        try:
            while True:
                await asyncio.sleep(self._keepalive_interval)
                if self._connection and self._connected:
                    try:
                        await self._connection.keep_alive()
                    except Exception as exc:
                        logger.warning(
                            "[DeepgramBridge] Keepalive failed: %s — marking for reconnect",
                            exc,
                        )
                        self._connected = False
                        task = asyncio.create_task(
                            self.ensure_connected(), name="dg-reconnect-keepalive"
                        )
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
        except asyncio.CancelledError:
            pass  # normal shutdown
