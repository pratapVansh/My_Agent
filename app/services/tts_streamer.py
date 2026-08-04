"""
LiveKit TTS Streamer.
Reads an async stream of text chunks (sentences/tokens), requests PCM audio from Cartesia,
and pushes exactly 20ms chunks into a LiveKit AudioSource, leveraging LiveKit's internal backpressure for pacing.
"""
import asyncio
import logging
from typing import AsyncIterable

from livekit import rtc
from app.services.voice_service import voice_service

logger = logging.getLogger(__name__)

class TTSStreamer:
    """Streams text -> Cartesia TTS -> LiveKit AudioSource."""
    
    def __init__(self):
        # 24kHz, 16-bit mono = 24000 samples/sec * 2 bytes/sample = 48000 bytes/sec
        # 20ms chunk = 48000 * 0.02 = 960 bytes
        self._sample_rate = 24000
        self._num_channels = 1
        self._bytes_per_sample = 2
        self._chunk_duration_ms = 20
        self._bytes_per_chunk = int(self._sample_rate * self._num_channels * self._bytes_per_sample * (self._chunk_duration_ms / 1000))
        self._samples_per_channel = int(self._sample_rate * (self._chunk_duration_ms / 1000))

    async def stream_to_track(
        self,
        text_stream: AsyncIterable[str],
        audio_source: rtc.AudioSource,
        voice_id: str | None = None
    ) -> None:
        """
        Consume a stream of text chunks, synthesize audio, and push to LiveKit.
        Cleanly exits on cancellation (barge-in).
        """
        buffer = bytearray()
        
        try:
            async for text_chunk in text_stream:
                text_chunk = text_chunk.strip()
                if not text_chunk:
                    continue
                    
                logger.info("[TTSStreamer] Synthesizing chunk: %s", text_chunk[:50])
                
                # Fetch audio stream from Cartesia
                async for audio_bytes in voice_service.synthesize_speech_stream(text_chunk, voice_id=voice_id):
                    if audio_bytes:
                        buffer.extend(audio_bytes)
                        
                        # Process all complete 20ms chunks in the buffer
                        while len(buffer) >= self._bytes_per_chunk:
                            chunk_data = buffer[:self._bytes_per_chunk]
                            del buffer[:self._bytes_per_chunk]
                            
                            frame = rtc.AudioFrame(
                                data=bytes(chunk_data),
                                sample_rate=self._sample_rate,
                                num_channels=self._num_channels,
                                samples_per_channel=self._samples_per_channel,
                            )
                            # LiveKit handles backpressure natively
                            await audio_source.capture_frame(frame)
                            
        except asyncio.CancelledError:
            logger.info("[TTSStreamer] Stream cancelled (barge-in)")
            raise
        except Exception as exc:
            logger.error("[TTSStreamer] Stream error: %s", exc, exc_info=True)
            raise
        finally:
            buffer.clear()
            logger.debug("[TTSStreamer] Stream cleanup complete")

tts_streamer = TTSStreamer()
