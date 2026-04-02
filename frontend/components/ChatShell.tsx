"use client";

import { useEffect, useRef, useState } from "react";
import {
  askWithVoice,
  ChatMessage,
  ChatMode
} from "@/lib/api";
import { ListeningIndicator, SpeakingIndicator, ThinkingIndicator, TypingIndicator } from "./StatusIndicator";
import { CloseChatModal, ClearChatModal } from "./ChatModals";

type ChatShellProps = {
  mode: ChatMode;
  title: string;
  subtitle: string;
};

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

const USER_ID_STORAGE_KEY = "user_id";
const STABLE_USER_ID = "vansh";

const AGENT_LABELS: Record<string, string> = {
  job: "Job Agent",
  email: "Email Agent",
  academic: "Research Agent",
  profile: "Profile Agent",
  planner: "Planner",
  response: "Formatter",
  clarification: "Clarifying",
};

function generateStableId(prefix: "user" | "session") {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${uid()}_${Date.now()}`;
}

function getOrCreateUserId() {
  if (typeof window === "undefined") {
    return STABLE_USER_ID;
  }

  const existing = window.localStorage.getItem(USER_ID_STORAGE_KEY);
  if (existing !== STABLE_USER_ID) {
    window.localStorage.setItem(USER_ID_STORAGE_KEY, STABLE_USER_ID);
  }

  return STABLE_USER_ID;
}

function createConversationSessionId() {
  return generateStableId("session");
}

function getVoiceWebSocketUrl() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
  const wsBase = apiBase.replace(/^http/i, "ws");
  return `${wsBase}/api/v1/voice/stream_v2`;
}

function playAudio(base64?: string, mimeType?: string) {
  if (!base64) return;
  const audio = new Audio(`data:${mimeType ?? "audio/wav"};base64,${base64}`);
  return audio;
}

function downsampleTo16kHz(input: Float32Array, inputSampleRate: number): Float32Array {
  if (inputSampleRate === 16000) {
    return input;
  }

  const sampleRateRatio = inputSampleRate / 16000;
  const outputLength = Math.round(input.length / sampleRateRatio);
  const output = new Float32Array(outputLength);

  let outputOffset = 0;
  let inputOffset = 0;

  while (outputOffset < output.length) {
    const nextInputOffset = Math.round((outputOffset + 1) * sampleRateRatio);
    let accumulator = 0;
    let count = 0;

    for (let i = inputOffset; i < nextInputOffset && i < input.length; i += 1) {
      accumulator += input[i];
      count += 1;
    }

    output[outputOffset] = count > 0 ? accumulator / count : 0;
    outputOffset += 1;
    inputOffset = nextInputOffset;
  }

  return output;
}

function float32ToPcm16(input: Float32Array): Int16Array {
  const pcm16 = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const s = Math.max(-1, Math.min(1, input[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16;
}

// Playback sample rate must match Cartesia output (24kHz)
const PLAYBACK_SAMPLE_RATE = 24000;

/**
 * Decode base64 → PCM S16LE bytes → Float32 AudioBuffer and schedule
 * playback on the provided AudioContext, chaining buffers end-to-end.
 *
 * KEY FIX: if we've fallen behind (first chunk or slow network), schedule
 * at ctx.currentTime + 5ms instead of ctx.currentTime + 10ms + nextStartTime,
 * which previously caused audible gaps when chunks arrived slower than their
 * own duration.
 *
 * Returns the updated nextStartTime.
 */
function scheduleAudioChunk(
  ctx: AudioContext,
  base64Pcm: string,
  nextStartTime: number,
): number {
  const binary = atob(base64Pcm);
  const byteLen = binary.length;
  if (byteLen < 2) return nextStartTime;

  const numSamples = Math.floor(byteLen / 2);
  const buffer = ctx.createBuffer(1, numSamples, PLAYBACK_SAMPLE_RATE);
  const channelData = buffer.getChannelData(0);

  for (let i = 0; i < numSamples; i += 1) {
    const lo = binary.charCodeAt(i * 2);
    const hi = binary.charCodeAt(i * 2 + 1);
    let sample = (hi << 8) | lo;
    if (sample >= 0x8000) sample -= 0x10000; // sign extend
    channelData[i] = sample / 32768.0;
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);

  // Chain to previous buffer if we're ahead of schedule,
  // otherwise start immediately (5ms safety margin) to avoid gaps.
  const startAt = nextStartTime > ctx.currentTime
    ? nextStartTime
    : ctx.currentTime + 0.005;
  source.start(startAt);

  return startAt + buffer.duration;
}

export default function ChatShell({ mode, title, subtitle }: ChatShellProps) {
  const [userId, setUserId] = useState("");
  const [sessionId, setSessionId] = useState("");

  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [showCloseModal, setShowCloseModal] = useState(false);
  const [showClearModal, setShowClearModal] = useState(false);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [showMemoryPanel, setShowMemoryPanel] = useState(false);
  const [profileFacts, setProfileFacts] = useState<Array<{key: string; value: string; source?: string}>>([]);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [voiceMode, setVoiceMode] = useState<"push-to-talk" | "always-on">("push-to-talk");
  const [timetableUploading, setTimetableUploading] = useState(false);
  const [timetableUploadResult, setTimetableUploadResult] = useState<{success: boolean; message: string} | null>(null);
  const timetableInputRef = useRef<HTMLInputElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<AudioWorkletNode | null>(null);
  const userMessageAddedRef = useRef<boolean>(false);
  // Accumulates FINAL Deepgram results — used as the authoritative transcript on stop
  const interimTranscriptRef = useRef<string>("");
  // Tracks the latest DISPLAYED text (final or interim) — fallback when user
  // stops speaking before Deepgram fires a final result
  const latestInterimRef = useRef<string>("");
  // Audio frames buffered while WS is still connecting
  const audioBufferRef = useRef<ArrayBuffer[]>([]);
  // Keep voiceMode accessible inside WS closures without stale state
  const voiceModeRef = useRef<"push-to-talk" | "always-on">("push-to-talk");

  // Streaming TTS playback — separate AudioContext from the recording one
  const playbackContextRef = useRef<AudioContext | null>(null);
  const nextAudioStartTimeRef = useRef<number>(0);
  const audioStreamEndTimerRef = useRef<ReturnType<typeof window.setTimeout> | null>(null);

  const colors = mode === "user"
    ? {
        bg: "bg-user-bg",
        surface: "bg-user-surface",
        gradient: "from-user-primary to-user-secondary",
        accentGradient: "from-user-secondary to-user-accent",
        primary: "bg-user-primary",
        text: "text-user-primary"
      }
    : {
        bg: "bg-recruiter-bg",
        surface: "bg-recruiter-surface",
        gradient: "from-recruiter-primary to-recruiter-secondary",
        accentGradient: "from-recruiter-secondary to-recruiter-accent",
        primary: "bg-recruiter-primary",
        text: "text-recruiter-primary"
      };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    const stableUserId = getOrCreateUserId();
    setUserId(stableUserId);
    setSessionId(createConversationSessionId());
  }, []);

  // Keep ref in sync so stopStreamingCapture always reads the latest transcript
  useEffect(() => {
    interimTranscriptRef.current = interimTranscript;
  }, [interimTranscript]);

  // Keep voiceModeRef in sync for use inside WS closures
  useEffect(() => {
    voiceModeRef.current = voiceMode;
  }, [voiceMode]);

  const fetchProfileFacts = async () => {
    if (!userId) return;
    setMemoryLoading(true);
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
      const res = await fetch(`${apiBase}/api/v1/agents/memory/profile/${userId}`);
      const data = await res.json();
      setProfileFacts(data.facts || []);
    } catch (e) {
      console.error("Failed to fetch profile facts", e);
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleTimetableUpload = async (file: File) => {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      setTimetableUploadResult({ success: false, message: "Please select a PDF file." });
      return;
    }
    const { resolvedUserId } = ensureIdentity();
    setTimetableUploading(true);
    setTimetableUploadResult(null);
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
      const formData = new FormData();
      formData.append("user_id", resolvedUserId);
      formData.append("clear_existing", "true");
      formData.append("file", file);
      const res = await fetch(`${apiBase}/api/v1/agents/tools/timetable/upload-pdf`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data?.detail?.message ?? data?.detail ?? "Upload failed";
        setTimetableUploadResult({ success: false, message: String(detail) });
        return;
      }
      setTimetableUploadResult({
        success: true,
        message: `Imported ${data.entries_stored} classes from "${data.filename}".${data.old_entries_cleared ? ` Replaced ${data.old_entries_cleared} old entries.` : ""}`,
      });
      addMessage({
        id: uid(),
        role: "assistant",
        text: `Timetable uploaded! I found ${data.entries_stored} classes in "${data.filename}". You can now ask me things like "What classes do I have today?" or "Show me my full schedule".`,
        createdAt: Date.now(),
      });
    } catch (e) {
      setTimetableUploadResult({ success: false, message: "Network error. Is the server running?" });
    } finally {
      setTimetableUploading(false);
      if (timetableInputRef.current) timetableInputRef.current.value = "";
    }
  };

  const forgetFact = async (key: string) => {
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
      await fetch(`${apiBase}/api/v1/agents/memory/profile/${userId}/${encodeURIComponent(key)}`, {
        method: "DELETE"
      });
      setProfileFacts(prev => prev.filter(f => f.key !== key));
    } catch (e) {
      console.error("Failed to forget fact", e);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSending]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      const processor = processorNodeRef.current;
      if (processor) {
        processor.disconnect();
      }

      const source = sourceNodeRef.current;
      if (source) {
        source.disconnect();
      }

      const context = audioContextRef.current;
      if (context && context.state !== "closed") {
        void context.close();
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }

      if (websocketRef.current) {
        websocketRef.current.close();
      }

      const playbackCtx = playbackContextRef.current;
      if (playbackCtx && playbackCtx.state !== "closed") {
        void playbackCtx.close();
      }

      if (audioStreamEndTimerRef.current !== null) {
        window.clearTimeout(audioStreamEndTimerRef.current);
      }
    };
  }, []);

  const addMessage = (msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  };

  const ensureIdentity = () => {
    let resolvedUserId = userId;
    if (!resolvedUserId) {
      resolvedUserId = getOrCreateUserId();
      setUserId(resolvedUserId);
    }

    let resolvedSessionId = sessionId;
    if (!resolvedSessionId) {
      resolvedSessionId = createConversationSessionId();
      setSessionId(resolvedSessionId);
    }

    return { resolvedUserId, resolvedSessionId };
  };

  const ensureVoiceSocket = async (resolvedUserId: string, resolvedSessionId: string) => {
    const current = websocketRef.current;
    if (current && current.readyState === WebSocket.OPEN) {
      current.send(JSON.stringify({
        type: "start",
        user_id: resolvedUserId,
        session_id: resolvedSessionId,
        output_mode: mode
      }));
      return current;
    }

    if (current && current.readyState === WebSocket.CONNECTING) {
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error("WebSocket connect timeout")), 5000);
        const onOpen = () => {
          window.clearTimeout(timeout);
          resolve();
        };
        const onError = () => {
          window.clearTimeout(timeout);
          reject(new Error("WebSocket connection failed"));
        };
        current.addEventListener("open", onOpen, { once: true });
        current.addEventListener("error", onError, { once: true });
      });

      current.send(JSON.stringify({
        type: "start",
        user_id: resolvedUserId,
        session_id: resolvedSessionId,
        output_mode: mode
      }));
      return current;
    }

    const ws = new WebSocket(getVoiceWebSocketUrl());

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const msgType = payload.type;

        if (msgType === "interim_transcript") {
          const text = String(payload.text ?? "");
          if (payload.is_final) {
            interimTranscriptRef.current = (interimTranscriptRef.current + " " + text).trim();
          }
          // Always track the latest shown text so stopStreamingCapture can fall
          // back to it if the user stops before Deepgram fires a final result.
          latestInterimRef.current = interimTranscriptRef.current || text;
          setInterimTranscript(latestInterimRef.current);
          setInput(latestInterimRef.current);
          return;
        }

        if (msgType === "utterance_end") {
          // Backend VAD detected speech end — stop capture and show processing immediately
          audioBufferRef.current = [];
          const processor = processorNodeRef.current;
          if (processor) { processor.disconnect(); processorNodeRef.current = null; }
          const source = sourceNodeRef.current;
          if (source) { source.disconnect(); sourceNodeRef.current = null; }
          const context = audioContextRef.current;
          if (context && context.state !== "closed") { void context.close(); audioContextRef.current = null; }
          if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach((t) => t.stop());
            mediaStreamRef.current = null;
          }

          // Use ref for fresh transcript (state may be stale in this closure)
          const userText = (payload.transcript as string | undefined)?.trim()
            || interimTranscriptRef.current.trim();

          setIsListening(false);
          setInterimTranscript("");
          interimTranscriptRef.current = "";
          latestInterimRef.current = "";
          setInput("");
          setCurrentAgent(null);

          if (userText && !userMessageAddedRef.current) {
            addMessage({ id: uid(), role: "user", text: userText, createdAt: Date.now() });
            userMessageAddedRef.current = true;
          }
          setIsSending(true);
          return;
        }

        if (msgType === "interrupt") {
          // Backend cancelled the in-flight TTS — stop playback immediately
          if (playbackContextRef.current && playbackContextRef.current.state !== "closed") {
            void playbackContextRef.current.close();
            playbackContextRef.current = null;
          }
          nextAudioStartTimeRef.current = 0;
          if (audioStreamEndTimerRef.current !== null) {
            window.clearTimeout(audioStreamEndTimerRef.current);
            audioStreamEndTimerRef.current = null;
          }
          setIsSpeaking(false);
          return;
        }

        if (msgType === "audio_stream_start") {
          // Prepare playback AudioContext for incoming PCM chunks
          if (mode === "user") {
            // Close any existing context to avoid scheduling on a stale timeline
            if (playbackContextRef.current && playbackContextRef.current.state !== "closed") {
              void playbackContextRef.current.close();
            }
            playbackContextRef.current = new AudioContext({ sampleRate: PLAYBACK_SAMPLE_RATE });
            // Browsers auto-suspend AudioContext — resume or audio is silent
            if (playbackContextRef.current.state === "suspended") {
              void playbackContextRef.current.resume();
            }
            nextAudioStartTimeRef.current = 0;
            if (audioStreamEndTimerRef.current !== null) {
              window.clearTimeout(audioStreamEndTimerRef.current);
              audioStreamEndTimerRef.current = null;
            }
            setIsSpeaking(true);
          }
          return;
        }

        if (msgType === "audio_chunk") {
          // Decode PCM S16LE chunk and schedule for seamless playback
          if (mode === "user" && payload.data && playbackContextRef.current) {
            // Resume in case context was suspended between chunks
            if (playbackContextRef.current.state === "suspended") {
              void playbackContextRef.current.resume();
            }
            nextAudioStartTimeRef.current = scheduleAudioChunk(
              playbackContextRef.current,
              String(payload.data),
              nextAudioStartTimeRef.current,
            );
          }
          return;
        }

        if (msgType === "audio_stream_end") {
          // Schedule isSpeaking → false after the last queued chunk finishes
          if (mode === "user" && playbackContextRef.current) {
            const ctx = playbackContextRef.current;
            const msUntilEnd = Math.max(
              0,
              (nextAudioStartTimeRef.current - ctx.currentTime) * 1000
            );
            if (audioStreamEndTimerRef.current !== null) {
              window.clearTimeout(audioStreamEndTimerRef.current);
            }
            audioStreamEndTimerRef.current = window.setTimeout(() => {
              setIsSpeaking(false);
              setCurrentAudio(null);
              audioStreamEndTimerRef.current = null;
              // Always-on: auto-restart listening after speaking ends
              if (voiceModeRef.current === "always-on") {
                startStreamingCapture().catch(console.error);
              }
            }, msUntilEnd + 150); // +150ms buffer for scheduler jitter
          } else {
            setIsSpeaking(false);
            if (voiceModeRef.current === "always-on") {
              window.setTimeout(() => startStreamingCapture().catch(console.error), 400);
            }
          }
          return;
        }

        if (msgType === "final_response") {
          const transcript = String(payload.transcript ?? "").trim();
          const displayText = String(payload.display_text ?? "");

          // Clear processing timeout if it exists
          const ws = websocketRef.current;
          if (ws && (ws as any).__processingTimeout) {
            window.clearTimeout((ws as any).__processingTimeout);
            (ws as any).__processingTimeout = null;
          }

          // Transition: PROCESSING → IDLE (audio_stream_start will set SPEAKING)
          setIsSending(false);
          setInterimTranscript("");
          setInput("");

          const selectedAgent = payload.metadata?.selected_agent as string | undefined;
          setCurrentAgent(selectedAgent || null);

          // Only add user message if not already added
          if (transcript && !userMessageAddedRef.current) {
            addMessage({ id: uid(), role: "user", text: transcript, createdAt: Date.now() });
          }

          // Reset flag for next utterance
          userMessageAddedRef.current = false;

          addMessage({
            id: uid(),
            role: "assistant",
            text: displayText,
            audioBase64: payload.audio_base64,
            mimeType: payload.mime_type,
            agentName: selectedAgent,
            jobResults: payload.job_results || undefined,
            createdAt: Date.now()
          });

          // Legacy fallback: play blob audio only if no streaming audio is coming
          // (audio_base64 non-empty means old /stream endpoint or TTS not available)
          if (payload.audio_base64 && mode === "user") {
            handlePlayVoice(payload.audio_base64, payload.mime_type);
          }
          return;
        }

        if (msgType === "error") {
          // Clear processing timeout if it exists
          const ws = websocketRef.current;
          if (ws && (ws as any).__processingTimeout) {
            window.clearTimeout((ws as any).__processingTimeout);
            (ws as any).__processingTimeout = null;
          }

          console.log("Received error from backend:", payload.message);

          // Reset all states on error
          setIsListening(false);
          setIsSending(false);
          setIsSpeaking(false);
          setInterimTranscript("");
          setInput("");
          userMessageAddedRef.current = false;

          const message = String(payload.message ?? "Voice stream error");

          // Only show error message if it's not about "no speech detected"
          if (!message.toLowerCase().includes("no speech detected")) {
            addMessage({ id: uid(), role: "assistant", text: `Error: ${message}`, createdAt: Date.now() });
          }
          return;
        }
      } catch (err) {
        console.error("Failed to parse websocket message", err);
      }
    };

    ws.onerror = () => {
      setIsListening(false);
      setIsSending(false);
      addMessage({ id: uid(), role: "assistant", text: "Error: Voice socket connection failed", createdAt: Date.now() });
    };

    ws.onclose = () => {
      websocketRef.current = null;
    };

    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error("WebSocket connect timeout")), 7000);

      const onOpen = () => {
        window.clearTimeout(timeout);
        ws.removeEventListener("error", onError);
        resolve();
      };

      const onError = () => {
        window.clearTimeout(timeout);
        ws.removeEventListener("open", onOpen);
        reject(new Error("WebSocket connection failed"));
      };

      ws.addEventListener("open", onOpen, { once: true });
      ws.addEventListener("error", onError, { once: true });
    });

    websocketRef.current = ws;
    ws.send(JSON.stringify({
      type: "start",
      user_id: resolvedUserId,
      session_id: resolvedSessionId,
      output_mode: mode
    }));
    return ws;
  };

  const stopStreamingCapture = () => {
    // ── INSTANT: stop listening state immediately ───────────────────────────
    setIsListening(false);
    audioBufferRef.current = [];

    // ── Tear down audio pipeline ────────────────────────────────────────────
    const processor = processorNodeRef.current;
    if (processor) { processor.disconnect(); processorNodeRef.current = null; }
    const source = sourceNodeRef.current;
    if (source) { source.disconnect(); sourceNodeRef.current = null; }
    const context = audioContextRef.current;
    if (context && context.state !== "closed") { void context.close(); audioContextRef.current = null; }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }

    // Read from ref — avoids stale closure when transcript updated after render
    // Prefer accumulated finals; fall back to the latest shown interim so we
    // don't lose speech when the user stops before Deepgram fires a final result.
    const captured = (interimTranscriptRef.current || latestInterimRef.current).trim();
    const hasTranscript = captured.length > 0;

    setInput("");
    setInterimTranscript("");
    interimTranscriptRef.current = "";
    latestInterimRef.current = "";

    const ws = websocketRef.current;

    if (!hasTranscript) {
      // Nothing was spoken — reset cleanly
      setIsSending(false);
      userMessageAddedRef.current = false;
      return;
    }

    // ── Show user message + Processing indicator IMMEDIATELY ───────────────
    if (!userMessageAddedRef.current) {
      addMessage({ id: uid(), role: "user", text: captured, createdAt: Date.now() });
      userMessageAddedRef.current = true;
    }
    setIsSending(true);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "end_utterance" }));

      // Safety timeout: reset if backend never responds
      const timeoutId = window.setTimeout(() => {
        setIsSending(false);
        setIsSpeaking(false);
        userMessageAddedRef.current = false;
        addMessage({ id: uid(), role: "assistant", text: "Request timeout. Please try again.", createdAt: Date.now() });
      }, 30000);
      (ws as any).__processingTimeout = timeoutId;
    } else {
      // WS not connected — still show message, but reset processing
      setIsSending(false);
      userMessageAddedRef.current = false;
    }
  };

  const startStreamingCapture = async () => {
    const { resolvedUserId, resolvedSessionId } = ensureIdentity();

    // ── INSTANT feedback — set state before ANY async work ─────────────────
    setIsListening(true);
    setInterimTranscript("");
    interimTranscriptRef.current = "";
    latestInterimRef.current = "";
    setInput("");
    setIsSending(false);
    userMessageAddedRef.current = false;
    audioBufferRef.current = [];

    // ── Request mic — fast after first permission grant ─────────────────────
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
    } catch (err) {
      setIsListening(false);
      throw err;
    }

    mediaStreamRef.current = stream;

    // ── Audio pipeline setup (AudioWorkletNode — replaces deprecated ScriptProcessorNode) ──
    const audioContext = new AudioContext();
    await audioContext.resume();
    await audioContext.audioWorklet.addModule("/pcm-processor.js");
    const sourceNode = audioContext.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");

    audioContextRef.current = audioContext;
    sourceNodeRef.current = sourceNode;
    processorNodeRef.current = workletNode;

    workletNode.port.onmessage = (event) => {
      const inputData: Float32Array = event.data;
      const ws = websocketRef.current;
      try {
        const downsampled = downsampleTo16kHz(inputData, audioContext.sampleRate);
        const pcm16 = float32ToPcm16(downsampled);
        const frame = pcm16.buffer.slice(0);

        if (ws && ws.readyState === WebSocket.OPEN) {
          const buf = audioBufferRef.current;
          if (buf.length > 0) {
            for (const buffered of buf.splice(0)) ws.send(buffered);
          }
          ws.send(frame);
        } else {
          // Buffer while WS is still connecting — keep at most ~2 s of audio
          audioBufferRef.current.push(frame);
          if (audioBufferRef.current.length > 50) audioBufferRef.current.shift();
        }
      } catch (err) {
        console.error("Failed to send PCM audio chunk", err);
      }
    };

    sourceNode.connect(workletNode);
    workletNode.connect(audioContext.destination);

    // ── Connect WS in background — audio starts immediately above ──────────
    ensureVoiceSocket(resolvedUserId, resolvedSessionId).catch((err) => {
      console.error("Failed to connect voice socket:", err);
      setIsListening(false);
      addMessage({ id: uid(), role: "assistant", text: "Error: Unable to start voice streaming", createdAt: Date.now() });
    });
  };


  const stopTtsPlayback = () => {
    // Immediately stop any in-flight TTS audio playback
    if (playbackContextRef.current && playbackContextRef.current.state !== "closed") {
      void playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
    nextAudioStartTimeRef.current = 0;
    if (audioStreamEndTimerRef.current !== null) {
      window.clearTimeout(audioStreamEndTimerRef.current);
      audioStreamEndTimerRef.current = null;
    }
    setIsSpeaking(false);
    // Tell backend to cancel its TTS stream
    const ws = websocketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "interrupt" }));
    }
  };

  const handleMic = async () => {
    console.log("Mic clicked - current state:", {isListening, isSending, isSpeaking});

    // If agent is speaking, interrupt it and start listening
    if (isSpeaking && !isListening) {
      console.log("Interrupting TTS to start listening...");
      stopTtsPlayback();
      try {
        await startStreamingCapture();
      } catch (error) {
        console.error("Failed to start voice stream after interrupt", error);
        setIsListening(false);
      }
      return;
    }

    // Block starting a new recording while agent is still processing (not speaking)
    if (!isListening && isSending) {
      console.log("Mic click blocked - currently processing");
      return;
    }

    // Toggle: LISTENING → IDLE (stop and process)
    if (isListening) {
      console.log("Stopping listening...");
      stopStreamingCapture();
      return;
    }

    // Transition: IDLE → LISTENING (start recording)
    console.log("Starting listening...");
    try {
      await startStreamingCapture();
    } catch (error) {
      console.error("Failed to start voice stream", error);
      setIsListening(false);
      setIsSending(false);
      setIsSpeaking(false);
      addMessage({ id: uid(), role: "assistant", text: "Error: Unable to start voice streaming", createdAt: Date.now() });
    }
  };

  const handleSend = async (textOverride?: string) => {
    const text = (textOverride ?? input).trim();
    if (!text || isSending) return;

    const { resolvedUserId, resolvedSessionId } = ensureIdentity();

    setIsSending(true);
    setInput("");
    setInterimTranscript("");

    addMessage({ id: uid(), role: "user", text, createdAt: Date.now() });

    try {
      const response = await askWithVoice({
        query: text,
        userId: resolvedUserId,
        sessionId: resolvedSessionId,
        mode
      });

      const msgId = uid();
      addMessage({
        id: msgId,
        role: "assistant",
        text: response.displayText,
        audioBase64: response.audioBase64,
        mimeType: response.mimeType,
        createdAt: Date.now()
      });

      // Auto-play voice response (USER mode only)
      if (response.audioBase64 && mode === "user") {
        handlePlayVoice(response.audioBase64, response.mimeType);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error occurred";
      addMessage({ id: uid(), role: "assistant", text: `Error: ${message}`, createdAt: Date.now() });
    } finally {
      setIsSending(false);
    }
  };

  const handlePlayVoice = (base64?: string, mimeType?: string) => {
    // Stop any existing audio playback
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.onended = null;
      currentAudio.onerror = null;
      setIsSpeaking(false);
      setCurrentAudio(null);
    }

    const audio = playAudio(base64, mimeType);
    if (audio) {
      setCurrentAudio(audio);
      setIsSpeaking(true);

      // Transition: SPEAKING → IDLE when audio ends
      audio.onended = () => {
        console.log("Audio playback ended - resetting to IDLE state");
        setIsSpeaking(false);
        setCurrentAudio(null);
      };

      // Also handle errors and interruptions
      audio.onerror = (e) => {
        console.error("Audio playback error:", e);
        setIsSpeaking(false);
        setCurrentAudio(null);
      };

      audio.play().catch((err) => {
        console.error("Audio playback failed:", err);
        setIsSpeaking(false);
        setCurrentAudio(null);
      });
    }
  };

  const handleEndChat = () => {
    window.location.href = "/";
  };

  const handleClearChat = () => {
    setMessages([]);
    setShowClearModal(false);
    setInput("");
    setInterimTranscript("");
    setSessionId(createConversationSessionId());

    // Stop audio and reset all states
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.onended = null;
      currentAudio.onerror = null;
      setCurrentAudio(null);
    }

    // Stop streaming playback
    if (audioStreamEndTimerRef.current !== null) {
      window.clearTimeout(audioStreamEndTimerRef.current);
      audioStreamEndTimerRef.current = null;
    }
    if (playbackContextRef.current && playbackContextRef.current.state !== "closed") {
      void playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
    nextAudioStartTimeRef.current = 0;

    setIsSpeaking(false);
    setIsSending(false);
    setIsListening(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <main className={`min-h-screen ${colors.bg} relative overflow-hidden`}>
      {/* Animated background gradient */}
      <div className="absolute inset-0 opacity-30">
        <div className={`absolute top-0 right-0 w-96 h-96 bg-gradient-to-br ${colors.gradient} rounded-full blur-3xl animate-pulse-slow`} />
        <div className={`absolute bottom-0 left-0 w-96 h-96 bg-gradient-to-br ${colors.accentGradient} rounded-full blur-3xl animate-pulse-slow`} style={{ animationDelay: "1s" }} />
      </div>

      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-7xl gap-6 p-6">
        {/* Main chat section */}
        <section className="flex min-h-[90vh] flex-1 flex-col rounded-3xl glass-strong shadow-2xl overflow-hidden">
          {/* Header */}
          <header className={`${colors.surface} px-6 py-4 border-b border-white/10`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${colors.gradient} flex items-center justify-center shadow-lg`}>
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">{title}</h1>
                  <p className="text-sm text-white/60">{subtitle}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {/* Hidden file input for timetable upload */}
                <input
                  ref={timetableInputRef}
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void handleTimetableUpload(file);
                  }}
                />
                {/* Timetable upload button */}
                <button
                  onClick={() => timetableInputRef.current?.click()}
                  disabled={timetableUploading}
                  className={`p-2.5 rounded-xl glass hover:glass-strong transition-all duration-200 group ${
                    timetableUploading ? "opacity-50 cursor-not-allowed" : "text-white/80 hover:text-white"
                  }`}
                  title="Upload timetable PDF"
                >
                  {timetableUploading ? (
                    <svg className="w-5 h-5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  )}
                </button>
                <button
                  onClick={() => { setShowMemoryPanel(true); fetchProfileFacts(); }}
                  className="p-2.5 rounded-xl glass hover:glass-strong transition-all duration-200 text-white/80 hover:text-white group"
                  title="Memory panel"
                >
                  <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </button>
                <button
                  onClick={() => setShowClearModal(true)}
                  className="p-2.5 rounded-xl glass hover:glass-strong transition-all duration-200 text-white/80 hover:text-white group"
                  title="Clear chat"
                >
                  <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
                <button
                  onClick={() => setShowCloseModal(true)}
                  className="p-2.5 rounded-xl glass hover:glass-strong transition-all duration-200 text-white/80 hover:text-white group"
                  title="End chat"
                >
                  <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          </header>

          {/* Timetable upload status toast */}
          {timetableUploadResult && (
            <div className={`px-6 py-2 flex items-center justify-between border-b border-white/10 text-sm ${
              timetableUploadResult.success ? "text-green-300" : "text-red-300"
            }`}>
              <span>{timetableUploadResult.success ? "✓" : "✗"} {timetableUploadResult.message}</span>
              <button
                onClick={() => setTimetableUploadResult(null)}
                className="ml-3 text-white/40 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
          )}

          {/* Status indicators */}
          {(isListening || isSending || isSpeaking) && (
            <div className="px-6 py-3 border-b border-white/10">
              {isListening && (
                <div>
                  <ListeningIndicator />
                  {interimTranscript && (
                    <p className="text-xs text-white/50 italic mt-1">
                      Transcribing: {interimTranscript}...
                    </p>
                  )}
                </div>
              )}
              {isSending && !isListening && (
                <div className="flex items-center gap-3">
                  <ThinkingIndicator />
                  {currentAgent && (
                    <span className={`px-2.5 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${colors.gradient} text-white/90 animate-pulse`}>
                      {AGENT_LABELS[currentAgent] ?? currentAgent}
                    </span>
                  )}
                </div>
              )}
              {isSpeaking && !isListening && !isSending && <SpeakingIndicator />}
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                <div className={`w-20 h-20 rounded-3xl bg-gradient-to-br ${colors.gradient} flex items-center justify-center shadow-xl`}>
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">Start a Conversation</h2>
                  <p className="text-white/60 max-w-md">
                    {mode === "user"
                      ? "Tap the microphone button below and speak. Your voice assistant is ready to help!"
                      : "Type a message or use voice input to begin chatting."}
                  </p>
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <article
                key={msg.id}
                className={`flex items-start gap-3 animate-slide-up ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {msg.role === "assistant" && (
                  <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${colors.gradient} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                )}

                <div className={`max-w-[75%] ${msg.role === "user" ? "order-first" : ""}`}>
                  {/* Agent label badge */}
                  {msg.role === "assistant" && msg.agentName && (
                    <div className="mb-1 flex items-center gap-1.5">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-gradient-to-r ${colors.gradient} text-white/80`}>
                        <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                        {AGENT_LABELS[msg.agentName] ?? msg.agentName}
                      </span>
                    </div>
                  )}

                  <div
                    className={`rounded-2xl px-5 py-3 ${
                      msg.role === "user"
                        ? `bg-gradient-to-br ${colors.gradient} text-white shadow-lg`
                        : "glass-strong text-white"
                    }`}
                  >
                    <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.text}</p>
                  </div>

                  {/* Rich job cards */}
                  {msg.role === "assistant" && msg.jobResults && msg.jobResults.length > 0 && (
                    <div className="mt-2 space-y-2">
                      {msg.jobResults.map((job, i) => (
                        <div key={i} className="glass rounded-xl p-3 border border-white/10">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <h4 className="text-sm font-semibold text-white truncate">{job.title}</h4>
                              {job.snippet && (
                                <p className="text-xs text-white/55 mt-1 line-clamp-2">{job.snippet}</p>
                              )}
                            </div>
                            {job.rank_score !== undefined && (
                              <span className="text-[10px] text-white/40 flex-shrink-0 mt-0.5">
                                {Math.round(job.rank_score * 100)}% match
                              </span>
                            )}
                          </div>
                          {job.url && (
                            <a
                              href={job.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className={`mt-2 inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-gradient-to-r ${colors.gradient} text-white text-xs font-medium hover:opacity-90 transition-opacity`}
                            >
                              Apply →
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {msg.role === "assistant" && msg.audioBase64 && mode === "recruiter" && (
                    <button
                      onClick={() => handlePlayVoice(msg.audioBase64, msg.mimeType)}
                      className={`mt-2 px-4 py-2 rounded-xl glass hover:glass-strong transition-all duration-200 text-white text-xs font-medium flex items-center gap-2 group`}
                    >
                      <svg className="w-4 h-4 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                      </svg>
                      Play Voice
                    </button>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${colors.accentGradient} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                )}
              </article>
            ))}

            <TypingIndicator show={isSending && !isListening} />
            <div ref={messagesEndRef} />
          </div>

          {/* Input footer */}
          <footer className={`${colors.surface} px-6 py-4 border-t border-white/10`}>
            <div className="flex gap-3 items-end">
              <button
                onClick={handleMic}
                disabled={isSending || isSpeaking}
                className={`p-3 rounded-xl ${
                  isListening
                    ? `bg-gradient-to-br ${colors.gradient} shadow-lg scale-110`
                    : (isSending || isSpeaking)
                    ? "glass opacity-50 cursor-not-allowed"
                    : "glass hover:glass-strong"
                } transition-all duration-200 text-white ${mode === "user" ? "flex-1" : "flex-shrink-0"} group btn-ripple`}
                type="button"
                title={
                  isSending
                    ? "Processing your request..."
                    : isSpeaking
                    ? "Speaking response..."
                    : isListening
                    ? "Click to stop and process"
                    : "Click to start speaking"
                }
              >
                {isSending ? (
                  <svg className="w-6 h-6 animate-spin mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                ) : isSpeaking ? (
                  <svg className="w-6 h-6 animate-pulse mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  </svg>
                ) : (
                  <svg className={`w-6 h-6 ${isListening ? "animate-pulse" : "group-hover:scale-110"} transition-transform ${mode === "user" ? "mx-auto" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                )}
                {mode === "user" && (
                  <span className="ml-3 text-sm font-medium">
                    {isListening
                      ? "Click to stop & process"
                      : isSending
                      ? "Processing..."
                      : isSpeaking
                      ? "Speaking..."
                      : "Tap to speak"}
                  </span>
                )}
              </button>

              {mode === "recruiter" && (
                <>
                  <div className="flex-1 glass-strong rounded-xl overflow-hidden">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={isListening ? "Listening... Speak now!" : "Type your message here or click the mic..."}
                      className={`w-full px-4 py-3 bg-transparent text-white placeholder-white/40 outline-none resize-none text-sm ${interimTranscript ? "italic" : ""}`}
                      rows={1}
                      style={{ minHeight: "48px", maxHeight: "120px" }}
                      disabled={isListening}
                    />
                  </div>

                  <button
                    onClick={() => void handleSend()}
                    disabled={isSending || isSpeaking || isListening || !input.trim()}
                    className={`p-3 rounded-xl bg-gradient-to-br ${colors.gradient} hover:shadow-lg hover:scale-105 disabled:opacity-50 disabled:scale-100 disabled:cursor-not-allowed transition-all duration-200 text-white flex-shrink-0 btn-ripple`}
                    type="button"
                    title="Send message"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </button>
                </>
              )}
            </div>

            {/* Voice mode toggle (user mode only) */}
            {mode === "user" && (
              <div className="mt-2 flex justify-center">
                <button
                  onClick={() => setVoiceMode(v => v === "push-to-talk" ? "always-on" : "push-to-talk")}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs transition-all duration-200 ${
                    voiceMode === "always-on"
                      ? `bg-gradient-to-r ${colors.gradient} text-white font-medium shadow`
                      : "glass text-white/50 hover:text-white/80"
                  }`}
                  title={voiceMode === "always-on" ? "Switch to push-to-talk" : "Switch to always-on (auto-restart after response)"}
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                  {voiceMode === "always-on" ? "Always-on" : "Push-to-talk"}
                </button>
              </div>
            )}
          </footer>
        </section>

      </div>

      {/* Memory Panel */}
      {showMemoryPanel && (
        <div className="fixed inset-0 z-50 flex justify-end" onClick={() => setShowMemoryPanel(false)}>
          <div
            className="relative h-full w-80 glass-strong shadow-2xl border-l border-white/10 flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            {/* Panel header */}
            <div className={`px-5 py-4 border-b border-white/10 flex items-center justify-between ${colors.surface}`}>
              <div className="flex items-center gap-2">
                <svg className={`w-5 h-5 ${colors.text}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <span className="text-white font-semibold text-sm">Memory</span>
              </div>
              <button
                onClick={() => setShowMemoryPanel(false)}
                className="p-1.5 rounded-lg glass hover:glass-strong transition-all text-white/60 hover:text-white"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Facts list */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {memoryLoading && (
                <div className="flex items-center justify-center py-8">
                  <svg className="w-5 h-5 animate-spin text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </div>
              )}

              {!memoryLoading && profileFacts.length === 0 && (
                <div className="text-center py-8">
                  <p className="text-white/40 text-sm">No profile facts saved yet.</p>
                  <p className="text-white/30 text-xs mt-1">The AI learns from your conversations.</p>
                </div>
              )}

              {!memoryLoading && profileFacts.map((fact) => (
                <div key={fact.key} className="glass rounded-xl p-3 flex items-start justify-between gap-2 group">
                  <div className="flex-1 min-w-0">
                    <p className="text-[10px] text-white/40 uppercase tracking-wide font-medium">{fact.key.replace(/_/g, " ")}</p>
                    <p className="text-sm text-white mt-0.5 break-words">{fact.value}</p>
                    {fact.source && (
                      <p className="text-[10px] text-white/30 mt-1">{fact.source}</p>
                    )}
                  </div>
                  <button
                    onClick={() => forgetFact(fact.key)}
                    className="flex-shrink-0 p-1.5 rounded-lg text-white/20 hover:text-red-400 hover:bg-red-400/10 transition-all opacity-0 group-hover:opacity-100"
                    title="Forget this"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>

            {/* Refresh button */}
            <div className={`px-4 py-3 border-t border-white/10 ${colors.surface}`}>
              <button
                onClick={fetchProfileFacts}
                disabled={memoryLoading}
                className="w-full py-2 rounded-xl glass hover:glass-strong transition-all text-white/60 hover:text-white text-xs disabled:opacity-50"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modals */}
      <CloseChatModal
        isOpen={showCloseModal}
        onClose={() => setShowCloseModal(false)}
        onConfirm={handleEndChat}
        mode={mode}
      />

      <ClearChatModal
        isOpen={showClearModal}
        onClose={() => setShowClearModal(false)}
        onConfirm={handleClearChat}
        mode={mode}
      />
    </main>
  );
}
