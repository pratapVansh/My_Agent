"""
Low-latency voice service.
- Speech-to-text via Deepgram (HTTP and WebSocket streaming)
- Text-to-speech via Cartesia
"""
from __future__ import annotations

import base64
from typing import Any, Dict, Optional
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

    async def create_deepgram_websocket(self):
        """
        Create a streaming WebSocket connection to Deepgram for real-time STT.

        This provides significantly lower latency than chunk-based HTTP API:
        - HTTP chunk: 500-800ms latency
        - WebSocket streaming: 50-200ms latency

        Returns:
            Deepgram WebSocket connection configured for low-latency streaming
        """
        if not settings.deepgram_api_key:
            raise ValueError("Missing DEEPGRAM_API_KEY for streaming STT")

        try:
            # Configure Deepgram client
            deepgram_config = DeepgramClientOptions(
                options={"keepalive": "true"}
            )
            deepgram_client = DeepgramClient(
                settings.deepgram_api_key,
                deepgram_config
            )

            # Get WebSocket connection
            dg_connection = deepgram_client.listen.asyncwebsocket.v("1")

            # Configure streaming options for ultra-low latency
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

            # Start the connection
            if await dg_connection.start(options):
                return dg_connection
            else:
                raise Exception("Failed to start Deepgram WebSocket connection")

        except Exception as e:
            raise Exception(f"Failed to create Deepgram WebSocket: {str(e)}")

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

    async def close(self):
        await self._client.aclose()


voice_service = VoiceService()
