"""
LiveKit TTS streamer.

Consumes an async stream of text chunks, synthesises each one with Cartesia, and
pushes exactly-20 ms PCM frames into a LiveKit ``AudioSource``.

``AudioSource.capture_frame`` withholds its acknowledgement once the native
playout queue is full, so pushing frames self-paces to real time and this module
must not add sleeps of its own. That pacing is also what makes on-screen
captions possible: ``queued_duration`` tells us exactly how far ahead of the
listener's ear we are, so each chunk can be reported with the wall-clock time it
will actually be *heard* rather than the time it was generated.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterable, Callable, Optional

from livekit import rtc

from app.config import settings
from app.services.voice_service import TTSUnavailable, voice_service

logger = logging.getLogger(__name__)

_FRAME_DURATION_MS = 20
_FRAME_SECONDS = _FRAME_DURATION_MS / 1000
_BYTES_PER_SAMPLE = 2  # pcm_s16le

# Playout buffer held inside LiveKit. Deep enough that the gap between two
# Cartesia requests never drains it into an audible seam, shallow enough that
# capture_frame still tracks real time closely (which keeps caption timing
# honest). Barge-in discards it explicitly, so depth costs no interrupt latency.
_QUEUE_SIZE_MS = 2000


@dataclass(frozen=True)
class SpokenChunk:
    """A chunk of speech together with when the listener will hear it.

    ``starts_at`` is on the ``time.monotonic()`` timeline, the same clock
    ``AudioSource.queued_duration`` is derived from.
    """

    text: str
    starts_at: float
    duration: float


class TTSStreamer:
    """Streams text -> Cartesia -> LiveKit AudioSource."""

    @property
    def sample_rate(self) -> int:
        """Rate the audio track must be created with.

        Read from settings rather than hardcoded: the AudioSource, the frame
        headers, and the Cartesia request all have to agree. When they drifted
        the audio still played, just at the wrong speed and pitch — a failure
        with no error message anywhere.
        """
        return settings.cartesia_sample_rate

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def queue_size_ms(self) -> int:
        return _QUEUE_SIZE_MS

    def create_audio_source(self) -> rtc.AudioSource:
        """Build an AudioSource matching this streamer's output format."""
        return rtc.AudioSource(
            self.sample_rate, self.num_channels, queue_size_ms=_QUEUE_SIZE_MS
        )

    def _frame_geometry(self) -> tuple[int, int]:
        samples = int(self.sample_rate * _FRAME_SECONDS)
        return samples, samples * self.num_channels * _BYTES_PER_SAMPLE

    async def clear_pending_audio(self, audio_source: rtc.AudioSource) -> None:
        """
        Drop audio already queued inside LiveKit but not yet played.

        Cancelling this task stops us handing over *new* frames, but frames
        already accepted by the AudioSource sit in the playout buffer and keep
        playing — so without this the assistant talks over the user for as long
        as the buffer is deep after a barge-in.
        """
        try:
            audio_source.clear_queue()
        except Exception as exc:
            logger.warning("[TTSStreamer] clear_queue() failed: %s", exc)

    async def stream_to_track(
        self,
        text_stream: AsyncIterable[str],
        audio_source: rtc.AudioSource,
        voice_id: str | None = None,
        on_chunk: Optional[Callable[[SpokenChunk], None]] = None,
        on_progress: Optional[Callable[[], None]] = None,
    ) -> int:
        """
        Synthesise a stream of text chunks into a LiveKit audio track.

        Args:
            on_chunk: called synchronously with each chunk's playback schedule,
                *before* its frames are handed over, so a caption can be shown
                in step with the voice.
            on_progress: called as synthesis and frame pushing advance. Lets the
                caller distinguish a long reply from a stalled one — a spoken
                answer legitimately takes as long as it takes to say.

        Returns:
            Number of frames pushed. Zero means the listener heard nothing,
            which the caller needs to distinguish from a normal reply.

        Raises:
            TTSUnavailable: synthesis failed; the caller should tell the user.
        """
        samples_per_channel, bytes_per_frame = self._frame_geometry()
        sample_rate, num_channels = self.sample_rate, self.num_channels
        frames_pushed = 0

        try:
            async for text_chunk in text_stream:
                text_chunk = text_chunk.strip()
                if not text_chunk:
                    continue

                logger.debug("[TTSStreamer] Synthesizing: %s", text_chunk[:60])

                # Collect the chunk's audio before pushing any of it. The exact
                # duration is what lets on_chunk report a truthful schedule, and
                # the playout buffer covers the extra few hundred milliseconds.
                audio = bytearray()
                async for audio_bytes in voice_service.synthesize_speech_stream(
                    text_chunk, voice_id=voice_id
                ):
                    audio.extend(audio_bytes)
                    if on_progress:
                        on_progress()

                if not audio:
                    continue

                # Pad to a whole frame. Cartesia's final packet rarely lands on
                # a 20 ms boundary, and dropping the remainder clipped the last
                # few milliseconds off every chunk.
                remainder = len(audio) % bytes_per_frame
                if remainder:
                    audio.extend(b"\x00" * (bytes_per_frame - remainder))

                frame_count = len(audio) // bytes_per_frame
                if on_chunk:
                    # Everything already queued plays before this chunk's first
                    # sample, so that is precisely when it will be heard.
                    on_chunk(
                        SpokenChunk(
                            text=text_chunk,
                            starts_at=time.monotonic() + audio_source.queued_duration,
                            duration=frame_count * _FRAME_SECONDS,
                        )
                    )

                for index in range(frame_count):
                    offset = index * bytes_per_frame
                    await audio_source.capture_frame(
                        rtc.AudioFrame(
                            data=bytes(audio[offset : offset + bytes_per_frame]),
                            sample_rate=sample_rate,
                            num_channels=num_channels,
                            samples_per_channel=samples_per_channel,
                        )
                    )
                    frames_pushed += 1
                    if on_progress:
                        on_progress()

            return frames_pushed

        except asyncio.CancelledError:
            logger.info("[TTSStreamer] Cancelled (barge-in) after %d frames", frames_pushed)
            await self.clear_pending_audio(audio_source)
            raise
        except TTSUnavailable:
            await self.clear_pending_audio(audio_source)
            raise
        except Exception as exc:
            logger.error("[TTSStreamer] Stream error: %s", exc, exc_info=True)
            await self.clear_pending_audio(audio_source)
            raise


tts_streamer = TTSStreamer()
