"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  archiveConversation,
  askWithVoice,
  ChatMessage,
  ChatMode,
  fetchConversation,
  fetchProfileFacts,
  forgetProfileFact,
  getLiveKitToken,
  uploadTimetablePdf
} from "@/lib/api";
import { Identity, ensureSession, hasScope, logout } from "@/lib/auth";
import { RemoteTrack, Room, RoomEvent, Track } from "livekit-client";
import { ListeningIndicator, SpeakingIndicator, ThinkingIndicator } from "./StatusIndicator";
import { CloseChatModal, ClearChatModal } from "./ChatModals";

type ChatShellProps = {
  mode: ChatMode;
  title: string;
  subtitle: string;
};

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

const AGENT_LABELS: Record<string, string> = {
  job: "Job Agent",
  email: "Email Agent",
  academic: "Research Agent",
  profile: "Profile Agent",
  planner: "Planner",
  response: "Formatter",
  clarification: "Clarifying",
};

function createConversationSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `session_${crypto.randomUUID()}`;
  }
  return `session_${uid()}_${Date.now()}`;
}

// The conversation id is persisted so a refresh, a browser restart, or a
// navigation away and back resumes the same thread. Previously it was minted
// fresh on every mount and never stored, so the server — which scopes chat
// history by exactly this id — saw each page load as a brand-new conversation
// and the thread was silently lost.
//
// Scoped per mode so the owner and recruiter views keep separate threads.
function conversationStorageKey(mode: ChatMode) {
  return `my_agent.conversation_id.${mode}`;
}

function loadStoredConversationId(mode: ChatMode): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(conversationStorageKey(mode));
  } catch {
    // Private browsing and some embedded webviews deny storage access. A
    // non-resumable session is a degraded experience, not a broken one.
    return null;
  }
}

function storeConversationId(mode: ChatMode, conversationId: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(conversationStorageKey(mode), conversationId);
  } catch {
    /* storage unavailable — continue without persistence */
  }
}

function clearStoredConversationId(mode: ChatMode) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(conversationStorageKey(mode));
  } catch {
    /* storage unavailable */
  }
}

export default function ChatShell({ mode, title, subtitle }: ChatShellProps) {
  const router = useRouter();
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [authState, setAuthState] = useState<"loading" | "ready" | "unauthenticated">("loading");
  const [sessionId, setSessionId] = useState("");

  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [showCloseModal, setShowCloseModal] = useState(false);
  const [showClearModal, setShowClearModal] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState("");
  const [partialAssistantMessage, setPartialAssistantMessage] = useState("");
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [showMemoryPanel, setShowMemoryPanel] = useState(false);
  const [profileFacts, setProfileFacts] = useState<Array<{key: string; value: string; source?: string}>>([]);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [timetableUploading, setTimetableUploading] = useState(false);
  const [timetableUploadResult, setTimetableUploadResult] = useState<{success: boolean; message: string} | null>(null);
  const timetableInputRef = useRef<HTMLInputElement>(null);

  const [livekitStatus, setLivekitStatus] = useState<"idle" | "connecting" | "connected" | "error">("idle");
  // Set when the browser blocks autoplay, so the user can be offered a tap to
  // enable sound. Without this the assistant speaks into a muted element and
  // appears to never reply.
  const [audioBlocked, setAudioBlocked] = useState(false);
  const livekitRoomRef = useRef<Room | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const userMessageAddedRef = useRef<boolean>(false);
  // Guards against a second connect starting while the first is still in
  // flight (React Strict Mode double-invoke, or a fast double-click on the
  // mic). Without it, two LiveKit rooms are opened for one participant.
  const livekitConnectingRef = useRef<boolean>(false);
  // Audio elements created by track.attach(), so unmount can remove them.
  // They were previously appended to document.body and leaked on every
  // reconnect, leaving several elements playing the same track.
  const audioElementsRef = useRef<HTMLMediaElement[]>([]);
  // The authoritative reply, held from `final_response` until the audio finishes.
  // The bubble is committed at turn_end so the permanent message appears when the
  // assistant has actually finished saying it, rather than the moment the model
  // stopped generating — which for a long answer is a minute earlier.
  const pendingReplyRef = useRef<{ text: string; agentName?: string; jobResults?: ChatMessage["jobResults"] } | null>(null);
  // Mirror of the live caption. endTurn needs the current value synchronously and
  // is called from a handler closure registered once at connect time, so reading
  // the state variable would see a stale value — and reading it through a
  // setState updater would mean calling addMessage from inside an updater, which
  // React may invoke twice and would duplicate the message.
  const spokenTextRef = useRef<string>("");
  // Fails the turn open again if the worker never sends a terminal event, so a
  // dropped data message can't leave the mic permanently disabled.
  const turnWatchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  // Establish a session before anything can be sent. Recruiter mode
  // self-provisions a read-only guest session; user mode requires a real
  // sign-in and redirects to /login when there isn't one.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const session = await ensureSession(mode);
        if (cancelled) return;

        if (!session) {
          setAuthState("unauthenticated");
          router.replace(`/login?next=${encodeURIComponent(mode === "user" ? "/user" : "/recruiter")}`);
          return;
        }

        setIdentity(session);

        // Resume the stored thread. A refresh must never start a new
        // conversation, so the stored id is kept whatever the server says:
        //
        //  - turns returned      → rehydrate the transcript;
        //  - 404 (no turns yet)  → keep the id; the thread is created by the
        //                          first turn, and a brand-new id here would
        //                          orphan a conversation the user is mid-way
        //                          through;
        //  - network/API failure → keep the id and leave the view empty rather
        //                          than silently forking the thread. Minting a
        //                          fresh id on a transient error is how a
        //                          refresh came to look like Clear Chat.
        //
        // Only the genuine absence of a stored id creates one.
        const stored = loadStoredConversationId(mode);

        if (stored) {
          setSessionId(stored);
          try {
            const existing = await fetchConversation(stored);
            if (cancelled) return;
            if (existing) {
              setMessages(
                existing.turns.map((turn) => ({
                  id: `${stored}_${turn.sequence}`,
                  role: turn.role,
                  text: turn.content,
                  createdAt: turn.created_at ? Date.parse(turn.created_at) : Date.now(),
                  agentName: turn.agent ?? undefined,
                }))
              );
            }
          } catch (e) {
            console.warn("Could not resume the previous conversation", e);
          }
        } else {
          const fresh = createConversationSessionId();
          storeConversationId(mode, fresh);
          setSessionId(fresh);
        }

        setAuthState("ready");
      } catch (e) {
        if (cancelled) return;
        console.error("Could not establish a session", e);
        setAuthState("unauthenticated");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [mode, router]);

  const canEditMemory = hasScope(identity, "profile:write");
  const canManageTimetable = hasScope(identity, "timetable:write");

  const loadProfileFacts = async () => {
    if (!identity) return;
    setMemoryLoading(true);
    try {
      setProfileFacts(await fetchProfileFacts(identity.user_id));
    } catch (e) {
      console.error("Failed to fetch profile facts", e);
      setProfileFacts([]);
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleTimetableUpload = async (file: File) => {
    if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
      setTimetableUploadResult({ success: false, message: "Please select a PDF file." });
      return;
    }
    setTimetableUploading(true);
    setTimetableUploadResult(null);
    try {
      const data = await uploadTimetablePdf(file);
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
      setTimetableUploadResult({
        success: false,
        message: e instanceof Error ? e.message : "Upload failed.",
      });
    } finally {
      setTimetableUploading(false);
      if (timetableInputRef.current) timetableInputRef.current.value = "";
    }
  };

  const forgetFact = async (key: string) => {
    if (!identity) return;
    try {
      await forgetProfileFact(identity.user_id, key);
      setProfileFacts(prev => prev.filter(f => f.key !== key));
    } catch (e) {
      console.error("Failed to forget fact", e);
    }
  };

  const handleSignOut = async () => {
    await logout();
    router.replace("/login");
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSending]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (turnWatchdogRef.current) clearTimeout(turnWatchdogRef.current);
      audioElementsRef.current.forEach((el) => el.remove());
      audioElementsRef.current = [];
      if (livekitRoomRef.current) {
        void livekitRoomRef.current.disconnect();
        livekitRoomRef.current = null;
      }
    };
  }, []);

  const addMessage = (msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  };

  const ensureConversationId = () => {
    if (sessionId) return sessionId;
    const fresh = createConversationSessionId();
    storeConversationId(mode, fresh);
    setSessionId(fresh);
    return fresh;
  };

  // Start a new thread. Three operations are deliberately distinct here:
  //
  //   Refresh   — same conversation, rehydrated (handled at mount; never calls this)
  //   New Chat  — new conversation, previous one left active and listable
  //   Clear     — new conversation, previous one archived
  //
  // Only an explicit user action reaches this function. A reload must never
  // create a conversation, which is the bug this separation exists to prevent.
  const startNewConversation = async ({ archivePrevious }: { archivePrevious: boolean }) => {
    const previous = sessionId;
    const fresh = createConversationSessionId();
    storeConversationId(mode, fresh);
    setSessionId(fresh);
    setMessages([]);
    if (previous && archivePrevious) {
      try {
        await archiveConversation(previous);
      } catch (e) {
        console.warn("Could not archive the previous conversation", e);
      }
    }
  };

  
  
  



  
  // The mic is a toggle for the whole voice session, not for a single utterance:
  // once it is on it stays on across turns, which is what makes barge-in
  // possible. It is never disabled while the assistant is thinking or speaking —
  // doing that removed the only way to interrupt, and any turn whose terminal
  // event went missing left the button dead permanently.
  const handleMic = async () => {
    if (livekitConnectingRef.current) return;

    if (isListening) {
      setIsListening(false);
      endTurn();
      const room = livekitRoomRef.current;
      if (room) await room.localParticipant.setMicrophoneEnabled(false);
      return;
    }

    setIsListening(true);
    userMessageAddedRef.current = false;
    setInterimTranscript("");
    setInput("");
    try {
      if (!livekitRoomRef.current) {
        await connectLiveKit();
      } else {
        await livekitRoomRef.current.localParticipant.setMicrophoneEnabled(true);
        await enableAudioPlayback();
      }
    } catch (e) {
      console.error("Failed to start LiveKit mic", e);
      setIsListening(false);
    }
  };

  const handleSend = async (textOverride?: string) => {
    const text = (textOverride ?? input).trim();
    if (!text || isSending || authState !== "ready") return;

    const conversationId = ensureConversationId();

    setIsSending(true);
    setInput("");
    setInterimTranscript("");

    addMessage({ id: uid(), role: "user", text, createdAt: Date.now() });

    try {
      const response = await askWithVoice({
        query: text,
        sessionId: conversationId,
        mode
      });

      const msgId = uid();
      addMessage({
        id: msgId,
        role: "assistant",
        text: response.displayText,
        createdAt: Date.now()
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error occurred";
      addMessage({ id: uid(), role: "assistant", text: `Error: ${message}`, createdAt: Date.now() });
    } finally {
      setIsSending(false);
    }
  };

  // Every path that ends a turn funnels through here, so the input controls can
  // never be left disabled by a missing or out-of-order event. The live caption
  // is promoted to a permanent message on the way out.
  const endTurn = ({ interrupted = false }: { interrupted?: boolean } = {}) => {
    if (turnWatchdogRef.current) {
      clearTimeout(turnWatchdogRef.current);
      turnWatchdogRef.current = null;
    }
    const pending = pendingReplyRef.current;
    const spoken = spokenTextRef.current.trim();
    pendingReplyRef.current = null;
    spokenTextRef.current = "";

    // A completed turn commits the authoritative text. An interrupted one commits
    // only what was actually spoken — the full reply was generated but the user
    // never heard the rest of it, so showing it would misrepresent the exchange.
    const text = interrupted ? spoken : (pending?.text || spoken);
    if (text) {
      addMessage({
        id: uid(),
        role: "assistant",
        text,
        agentName: pending?.agentName,
        jobResults: interrupted ? undefined : pending?.jobResults,
        createdAt: Date.now(),
      });
    }

    setPartialAssistantMessage("");
    setIsSending(false);
    setIsSpeaking(false);
    userMessageAddedRef.current = false;
  };

  const beginTurn = () => {
    pendingReplyRef.current = null;
    spokenTextRef.current = "";
    setPartialAssistantMessage("");
    setIsSending(true);
    if (turnWatchdogRef.current) clearTimeout(turnWatchdogRef.current);
    // Backstop only — the worker cancels a genuinely stalled turn itself. This
    // catches the case where its terminal event never reaches us at all. It has
    // to outlast a long spoken reply, so it is refreshed by every caption.
    turnWatchdogRef.current = setTimeout(() => {
      console.warn("Turn watchdog fired — no terminal event from the worker");
      endTurn();
    }, 60000);
  };

  // Any sign the turn is still alive pushes the backstop out. Without this a
  // reply that takes longer than the watchdog to speak would be torn down by
  // the client even though the worker was streaming it perfectly well.
  const keepTurnAlive = () => {
    if (!turnWatchdogRef.current) return;
    clearTimeout(turnWatchdogRef.current);
    turnWatchdogRef.current = setTimeout(() => {
      console.warn("Turn watchdog fired — no terminal event from the worker");
      endTurn();
    }, 60000);
  };

  const connectLiveKit = async () => {
    // Idempotency guard — see livekitConnectingRef.
    if (livekitConnectingRef.current || livekitRoomRef.current) {
      console.log("LiveKit: connect already in progress or established; skipping");
      return;
    }
    livekitConnectingRef.current = true;

    setLivekitStatus("connecting");
    try {
      // The room is derived server-side from the authenticated identity, so
      // there is nothing for the client to choose (or spoof) here.
      const { token, url, room_name: roomName } = await getLiveKitToken(ensureConversationId());

      const room = new Room({
        // Echo cancellation is what stops the assistant's own voice being
        // captured, transcribed, and answered as if the user had said it.
        audioCaptureDefaults: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      room.on(RoomEvent.Disconnected, () => {
        setLivekitStatus("idle");
        setIsListening(false);
        endTurn();
        audioElementsRef.current.forEach((el) => el.remove());
        audioElementsRef.current = [];
        livekitRoomRef.current = null;
        livekitConnectingRef.current = false;
        console.log("LiveKit: room disconnected");
      });

      room.on(RoomEvent.Reconnecting, () => console.log("LiveKit: reconnecting…"));
      room.on(RoomEvent.Reconnected, () => {
        console.log("LiveKit: reconnected");
        // A reconnect can drop the mic publication; re-assert the intent.
        if (isListening) void room.localParticipant.setMicrophoneEnabled(true);
      });

      // Browsers refuse to play audio without a user gesture. Surface it rather
      // than letting the assistant speak inaudibly into a blocked element.
      room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
        setAudioBlocked(!room.canPlaybackAudio);
      });

      room.on(RoomEvent.ParticipantConnected, (p) => {
        console.log("LiveKit: participant joined:", p.identity);
      });

      room.on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
        try {
          const strData = new TextDecoder().decode(payload);
          const parsed = JSON.parse(strData);
          
          switch (parsed.type) {
            case "utterance_end": {
              // The user's turn is complete. The mic deliberately stays open so
              // they can interrupt the reply.
              const transcript = String(parsed.transcript ?? "").trim();
              if (transcript && !userMessageAddedRef.current) {
                addMessage({ id: uid(), role: "user", text: transcript, createdAt: Date.now() });
                userMessageAddedRef.current = true;
              }
              beginTurn();
              setInterimTranscript("");
              setInput("");
              break;
            }

            case "interim_transcript": {
              const text = String(parsed.text ?? "");
              setInterimTranscript(text);
              setInput(text);
              break;
            }

            case "caption": {
              // One word, timed by the worker to the instant it is heard. This
              // is the only thing that advances the on-screen reply, which is
              // what keeps text and voice in step.
              const delta = String(parsed.delta ?? "");
              if (!delta) break;
              spokenTextRef.current = spokenTextRef.current
                ? `${spokenTextRef.current} ${delta}`
                : delta;
              setPartialAssistantMessage(spokenTextRef.current);
              setIsSending(false);
              setIsSpeaking(true);
              keepTurnAlive();
              break;
            }

            case "final_response": {
              const transcript = String(parsed.transcript ?? "").trim();
              const displayText = String(parsed.display_text ?? "");
              const selectedAgent = parsed.metadata?.selected_agent as string | undefined;

              setCurrentAgent(selectedAgent || null);

              if (transcript && !userMessageAddedRef.current) {
                addMessage({ id: uid(), role: "user", text: transcript, createdAt: Date.now() });
              }
              userMessageAddedRef.current = false;

              // Held, not rendered. The model has finished generating but the
              // assistant is still speaking; rendering the whole reply here is
              // exactly what put the text a minute ahead of the audio.
              pendingReplyRef.current = {
                text: displayText,
                agentName: selectedAgent,
                jobResults: parsed.job_results || undefined,
              };
              keepTurnAlive();
              break;
            }

            case "turn_end": {
              // The single terminal event for a turn — success, timeout, or
              // failure alike. Everything the UI disabled is re-enabled here.
              endTurn();
              break;
            }

            case "interrupted": {
              // Barge-in. isListening is untouched because the mic stays hot.
              endTurn({ interrupted: true });
              break;
            }

            case "error": {
              console.error("Voice worker error:", parsed.message);
              // Commit anything already said first, so the reply and the notice
              // read in the order they happened. A TTS failure in particular
              // carries a perfectly good reply that belongs on screen.
              endTurn();
              addMessage({
                id: uid(),
                role: "assistant",
                text: `Error: ${parsed.message}`,
                createdAt: Date.now(),
              });
              break;
            }

            default:
              break;
          }
        } catch (e) {
          console.error("Failed to parse LiveKit data channel message:", e);
        }
      });

      room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind !== Track.Kind.Audio) return;
        const el = track.attach();
        el.autoplay = true;
        // Hidden rather than absent: an element detached from the document does
        // not reliably play in every browser.
        el.style.display = "none";
        document.body.appendChild(el);
        audioElementsRef.current.push(el);
        console.log("LiveKit: agent audio attached");
      });

      room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        if (track.kind !== Track.Kind.Audio) return;
        track.detach().forEach((el) => {
          el.remove();
          audioElementsRef.current = audioElementsRef.current.filter((x) => x !== el);
        });
        console.log("LiveKit: agent audio detached");
      });

      await room.connect(url, token);
      // Publish the mic only after the room is up, so the worker (which this
      // same request started) has a chance to join and subscribe.
      await room.localParticipant.setMicrophoneEnabled(true);

      // connect() runs inside the mic click, so this normally succeeds outright;
      // it is the fallback for browsers that still withhold playback.
      try {
        await room.startAudio();
        setAudioBlocked(false);
      } catch {
        setAudioBlocked(!room.canPlaybackAudio);
      }

      livekitRoomRef.current = room;
      setLivekitStatus("connected");
      console.log("✓ LiveKit: connected to room", roomName);
    } catch (err) {
      console.error("LiveKit connection failed:", err);
      setLivekitStatus("error");
      setIsListening(false);
      addMessage({
        id: uid(),
        role: "assistant",
        text: "I couldn't start the voice session. Check your microphone permission and try again.",
        createdAt: Date.now(),
      });
      setTimeout(() => setLivekitStatus("idle"), 3000);
    } finally {
      livekitConnectingRef.current = false;
    }
  };

  const enableAudioPlayback = async () => {
    const room = livekitRoomRef.current;
    if (!room) return;
    try {
      await room.startAudio();
      setAudioBlocked(false);
    } catch (e) {
      console.error("Could not start audio playback", e);
    }
  };

  const handleEndChat = () => {
    window.location.href = "/";
  };

  const handleClearChat = () => {
    setShowClearModal(false);
    setInput("");
    setInterimTranscript("");
    endTurn();
    setIsListening(false);
    void livekitRoomRef.current?.localParticipant.setMicrophoneEnabled(false);
    // Starts a genuinely new persisted thread and archives the old one, rather
    // than only emptying the view — otherwise the next reload would resume the
    // conversation the user just asked to clear.
    void startNewConversation({ archivePrevious: true });
  };

  // New Chat keeps the previous thread active and listable — it is a way to
  // start a second conversation, not a way to discard the first.
  const handleNewChat = () => {
    setInput("");
    setInterimTranscript("");
    endTurn();
    setIsListening(false);
    void livekitRoomRef.current?.localParticipant.setMicrophoneEnabled(false);
    void startNewConversation({ archivePrevious: false });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Hold the UI until a session exists, so nothing can be typed into a chat
  // that would be rejected the moment it is sent.
  if (authState !== "ready") {
    return (
      <main className={`min-h-screen ${colors.bg} flex items-center justify-center`}>
        <div className="flex flex-col items-center gap-4 text-white/60">
          <svg className="w-8 h-8 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <p className="text-sm">
            {authState === "loading" ? "Starting your session…" : "Redirecting to sign in…"}
          </p>
        </div>
      </main>
    );
  }

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
                  <p className="text-sm text-white/60">
                    {subtitle}
                    {identity?.role === "guest" && (
                      <span className="ml-2 px-2 py-0.5 rounded-full bg-white/10 text-[10px] uppercase tracking-wide align-middle">
                        Guest · read-only
                      </span>
                    )}
                  </p>
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
                {/* Timetable upload — owner only; a guest has no write scope */}
                {canManageTimetable && (
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
                )}
                {/* Memory panel — owner only; guests have no stored facts */}
                {canEditMemory && (
                <button
                  onClick={() => { setShowMemoryPanel(true); void loadProfileFacts(); }}
                  className="p-2.5 rounded-xl glass hover:glass-strong transition-all duration-200 text-white/80 hover:text-white group"
                  title="Memory panel"
                >
                  <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </button>
                )}
                {identity?.role === "owner" && (
                <button
                  onClick={() => void handleSignOut()}
                  className="p-2.5 rounded-xl glass hover:glass-strong transition-all duration-200 text-white/80 hover:text-white group"
                  title="Sign out"
                >
                  <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                </button>
                )}
                <button
                  onClick={handleNewChat}
                  className="p-2.5 rounded-xl glass hover:glass-strong transition-all duration-200 text-white/80 hover:text-white group"
                  title="New chat"
                >
                  <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
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

          {/* Playback blocked by the browser's autoplay policy */}
          {audioBlocked && (
            <button
              onClick={() => void enableAudioPlayback()}
              className="px-6 py-2 w-full text-left border-b border-white/10 text-sm text-amber-300 hover:text-amber-200 transition-colors"
            >
              🔇 Your browser blocked audio playback — click here to enable sound.
            </button>
          )}

          {/* Status indicators. The mic stays open across a turn, so these are
              driven by what the assistant is doing rather than by isListening. */}
          {(isListening || isSending || isSpeaking) && (
            <div className="px-6 py-3 border-b border-white/10">
              {isSpeaking ? (
                <SpeakingIndicator />
              ) : isSending ? (
                <div className="flex items-center gap-3">
                  <ThinkingIndicator />
                  {currentAgent && (
                    <span className={`px-2.5 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${colors.gradient} text-white/90 animate-pulse`}>
                      {AGENT_LABELS[currentAgent] ?? currentAgent}
                    </span>
                  )}
                </div>
              ) : (
                <div>
                  <ListeningIndicator />
                  {interimTranscript && (
                    <p className="text-xs text-white/50 italic mt-1">
                      Transcribing: {interimTranscript}...
                    </p>
                  )}
                </div>
              )}
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

            {/* Thinking state lives in the status bar above: the mic now stays
                open across a turn, so a duplicate indicator keyed on
                !isListening could never render anyway. */}
            {partialAssistantMessage && (
              <article className={`flex items-start gap-4 flex-row`}>
                <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${colors.gradient} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div className={`flex-1 glass-strong rounded-2xl p-4 max-w-[85%] border border-white/20 shadow-xl prose prose-invert`}>
                  <p className="text-white text-[15px] leading-relaxed m-0">{partialAssistantMessage}</p>
                </div>
              </article>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input footer */}
          <footer className={`${colors.surface} px-6 py-4 border-t border-white/10`}>
            <div className="flex gap-3 items-end">
              <button
                onClick={handleMic}
                disabled={livekitStatus === "connecting"}
                className={`p-3 rounded-xl ${
                  isListening
                    ? `bg-gradient-to-br ${colors.gradient} shadow-lg scale-110`
                    : livekitStatus === "connecting"
                    ? "glass opacity-50 cursor-not-allowed"
                    : "glass hover:glass-strong"
                } transition-all duration-200 text-white ${mode === "user" ? "flex-1" : "flex-shrink-0"} group btn-ripple`}
                type="button"
                title={
                  isListening
                    ? "Voice is live — just speak, or click to end the session"
                    : livekitStatus === "connecting"
                    ? "Connecting…"
                    : "Click to start a voice session"
                }
              >
                {isSending && !isListening ? (
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
                    {livekitStatus === "connecting"
                      ? "Connecting…"
                      : isListening && isSpeaking
                      ? "Speaking — talk to interrupt"
                      : isListening && isSending
                      ? "Thinking…"
                      : isListening
                      ? "Listening — click to end"
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
                onClick={() => void loadProfileFacts()}
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
