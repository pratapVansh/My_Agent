"""
Low-latency voice service.
- Speech-to-text via Deepgram (HTTP and WebSocket streaming)
- Text-to-speech via Cartesia
"""
from __future__ import annotations

import base64
import json
from typing import Any, AsyncGenerator, Dict, Optional
import asyncio

import httpx
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions

from app.config import settings


class VoiceService:
    """Unified voice service for STT and TTS."""

    def __init__(self):
        # Optimized httpx client with connection pooling, keep-alive, and HTTP/2
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, read=15.0),  # Reduced timeouts for faster failures
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            ),
            http2=True  # Enable HTTP/2 for better multiplexing
        )

    async def transcribe_audio_chunk(
        self,
        audio_b64: str,
        mime_type: str = "audio/wav",
        language: str = "en",
    ) -> Dict[str, Any]:
        """Transcribe a single audio chunk with Deepgram."""
        if not settings.deepgram_api_key:
            return {
                "success": False,
                "transcript": "",
                "error": "Missing DEEPGRAM_API_KEY",
            }

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            return {
                "success": False,
                "transcript": "",
                "error": "Invalid audio_base64 payload",
            }

        params = {
            "model": settings.deepgram_model,
            "smart_format": "true",
            "language": language,
            "interim_results": "false",
            "punctuate": "true",
        }

        try:
            resp = await self._client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers={
                    "Authorization": f"Token {settings.deepgram_api_key}",
                    "Content-Type": mime_type,
                },
                content=audio_bytes,
            )
            resp.raise_for_status()
            payload = resp.json()

            transcript = ""
            channels = payload.get("results", {}).get("channels", [])
            if channels:
                alternatives = channels[0].get("alternatives", [])
                if alternatives:
                    transcript = alternatives[0].get("transcript", "") or ""

            return {
                "success": True,
                "transcript": transcript.strip(),
                "provider": "deepgram",
            }
        except Exception as e:
            return {
                "success": False,
                "transcript": "",
                "error": f"Deepgram transcription failed: {str(e)}",
            }

    def create_deepgram_connection(self):
        """
        Create a Deepgram WebSocket connection object and options WITHOUT starting it.

        The caller must register event handlers and then call `await conn.start(options)`.
        This ordering ensures no events (open/close/error) are missed due to a race
        between connection start and handler registration.

        Returns:
            (dg_connection, LiveOptions) — connection is NOT yet started
        """
        if not settings.deepgram_api_key:
            raise ValueError("Missing DEEPGRAM_API_KEY for streaming STT")

        deepgram_config = DeepgramClientOptions(
            options={"keepalive": "true"}
        )
        deepgram_client = DeepgramClient(
            settings.deepgram_api_key,
            deepgram_config
        )

        dg_connection = deepgram_client.listen.asyncwebsocket.v("1")

        options = LiveOptions(
            model=settings.deepgram_model,
            language="en",
            smart_format=True,
            punctuate=True,
            interim_results=settings.deepgram_interim_results,
            utterance_end_ms=str(settings.deepgram_utterance_end_ms),
            vad_events=settings.deepgram_vad_events,
            encoding="linear16",
            sample_rate=16000,
            channels=1
        )

        return dg_connection, options

    async def create_deepgram_websocket(self):
        """
        Create and start a Deepgram WebSocket connection (no event handlers registered).

        Prefer `create_deepgram_connection()` when you need to attach event handlers
        before the connection opens.
        """
        dg_connection, options = self.create_deepgram_connection()
        if await dg_connection.start(options):
            return dg_connection
        raise Exception("Failed to start Deepgram WebSocket connection")

    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate speech audio from text with Cartesia."""
        if not settings.cartesia_api_key or settings.cartesia_api_key == "your_cartesia_api_key_here":
            # Return success but without audio if API key is not configured
            return {
                "success": True,
                "audio_base64": "",
                "mime_type": "audio/wav",
                "error": "TTS not configured - Set CARTESIA_API_KEY in .env",
            }

        payload = {
            "model_id": settings.cartesia_model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": voice_id or settings.cartesia_voice_id,
            },
            "output_format": {
                "container": "wav",
                "sample_rate": settings.cartesia_sample_rate,
                "encoding": "pcm_f32le",
            },
        }

        try:
            resp = await self._client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "X-API-Key": settings.cartesia_api_key,
                    "Cartesia-Version": settings.cartesia_version,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            audio_b64 = base64.b64encode(resp.content).decode("ascii")
            return {
                "success": True,
                "audio_base64": audio_b64,
                "provider": "cartesia",
                "mime_type": "audio/wav",
            }
        except Exception as e:
            return {
                "success": False,
                "audio_base64": "",
                "error": f"Cartesia synthesis failed: {str(e)}",
            }

    async def synthesize_speech_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream raw PCM S16LE audio chunks from Cartesia (24kHz mono).

        Yields bytes chunks as they arrive from Cartesia instead of waiting for
        the full response — reduces time-to-first-audio from ~1-2s to ~200-400ms.
        """
        if not settings.cartesia_api_key or settings.cartesia_api_key == "your_cartesia_api_key_here":
            return

        if not text or not text.strip():
            return

        payload = {
            "model_id": settings.cartesia_model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": voice_id or settings.cartesia_voice_id,
            },
            "output_format": {
                "container": "raw",
                "sample_rate": settings.cartesia_sample_rate,
                "encoding": "pcm_s16le",
            },
        }

        try:
            async with self._client.stream(
                "POST",
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "X-API-Key": settings.cartesia_api_key,
                    "Cartesia-Version": settings.cartesia_version,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=httpx.Timeout(5.0, read=30.0),
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    if chunk:
                        yield chunk
        except Exception as e:
            print(f"[TTS Stream] Cartesia streaming failed: {e}")

    async def close(self):
        await self._client.aclose()


voice_service = VoiceService()
