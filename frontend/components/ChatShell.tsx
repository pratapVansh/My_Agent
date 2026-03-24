"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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

declare global {
  interface Window {
    webkitSpeechRecognition?: new () => SpeechRecognition;
  }

  interface SpeechRecognition extends EventTarget {
    lang: string;
    interimResults: boolean;
    continuous: boolean;
    maxAlternatives: number;
    onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
    onerror: ((this: SpeechRecognition, ev: Event) => void) | null;
    onstart: (() => void) | null;
    onend: (() => void) | null;
    start(): void;
    stop(): void;
  }

  interface SpeechRecognitionEvent extends Event {
    readonly results: SpeechRecognitionResultList;
  }
}

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function playAudio(base64?: string, mimeType?: string) {
  if (!base64) return;
  const audio = new Audio(`data:${mimeType ?? "audio/wav"};base64,${base64}`);
  return audio;
}

export default function ChatShell({ mode, title, subtitle }: ChatShellProps) {
  const userId = useMemo(() => `${mode}_demo_user`, [mode]);
  const sessionId = useMemo(() => `${mode}_session_${uid()}`, [mode]);

  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [showCloseModal, setShowCloseModal] = useState(false);
  const [showClearModal, setShowClearModal] = useState(false);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);
  const [interimTranscript, setInterimTranscript] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);

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
    scrollToBottom();
  }, [messages, isSending]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const addMessage = (msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  };

  const handleMic = () => {
    const SpeechRec =
      typeof window !== "undefined"
        ? ((globalThis as any).SpeechRecognition || (globalThis as any).webkitSpeechRecognition)
        : undefined;

    if (!SpeechRec) {
      alert("Speech recognition not supported in this browser");
      return;
    }

    if (isListening && recognitionRef.current) {
      // Stop listening
      recognitionRef.current.stop();
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      return;
    }

    // Start new recognition session
    const rec = new SpeechRec();
    recognitionRef.current = rec;
    rec.lang = "en-US";
    rec.interimResults = true; // Enable real-time results
    rec.continuous = true; // Keep listening
    rec.maxAlternatives = 1;

    let finalTranscript = "";

    rec.onstart = () => {
      setIsListening(true);
      setInterimTranscript("");
      finalTranscript = "";
    };

    rec.onend = () => {
      setIsListening(false);
      setInterimTranscript("");

      // Auto-send if we have a final transcript
      if (finalTranscript.trim()) {
        setInput(finalTranscript.trim());
        // Auto-submit after a short delay
        setTimeout(() => {
          const text = finalTranscript.trim();
          if (text && !isSending) {
            handleSend();
          }
        }, 300);
      }
    };

    rec.onresult = (ev: any) => {
      let interim = "";
      let final = "";

      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const transcript = ev.results[i][0].transcript;
        if (ev.results[i].isFinal) {
          final += transcript + " ";
        } else {
          interim += transcript;
        }
      }

      if (final) {
        finalTranscript += final;
        setInput(finalTranscript.trim() + " " + interim);
      } else {
        setInput(finalTranscript.trim() + " " + interim);
      }

      setInterimTranscript(interim);

      // Reset silence timer
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }

      // Auto-stop after 2 seconds of silence if we have content
      if (finalTranscript.trim()) {
        silenceTimerRef.current = setTimeout(() => {
          if (recognitionRef.current) {
            recognitionRef.current.stop();
          }
        }, 2000);
      }
    };

    rec.onerror = (event: any) => {
      console.error("Speech recognition error:", event.error);
      setIsListening(false);
      setInterimTranscript("");
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
    };

    try {
      rec.start();
    } catch (error) {
      console.error("Failed to start recognition:", error);
      setIsListening(false);
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isSending) return;

    setIsSending(true);
    setInput("");
    setInterimTranscript("");

    addMessage({ id: uid(), role: "user", text, createdAt: Date.now() });

    try {
      const response = await askWithVoice({
        query: text,
        userId,
        sessionId,
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
        setTimeout(() => {
          handlePlayVoice(response.audioBase64, response.mimeType);
        }, 500);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error occurred";
      addMessage({ id: uid(), role: "assistant", text: `Error: ${message}`, createdAt: Date.now() });
    } finally {
      setIsSending(false);
    }
  };

  const handlePlayVoice = (base64?: string, mimeType?: string) => {
    if (currentAudio) {
      currentAudio.pause();
      setIsSpeaking(false);
    }

    const audio = playAudio(base64, mimeType);
    if (audio) {
      setCurrentAudio(audio);
      setIsSpeaking(true);
      audio.onended = () => setIsSpeaking(false);
      audio.play();
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
    if (currentAudio) {
      currentAudio.pause();
      setIsSpeaking(false);
    }
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
              {isSending && !isListening && <ThinkingIndicator />}
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

                <div className={`max-w-[70%] ${msg.role === "user" ? "order-first" : ""}`}>
                  <div
                    className={`rounded-2xl px-5 py-3 ${
                      msg.role === "user"
                        ? `bg-gradient-to-br ${colors.gradient} text-white shadow-lg`
                        : "glass-strong text-white"
                    }`}
                  >
                    <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.text}</p>
                  </div>

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
                disabled={isSending}
                className={`p-3 rounded-xl ${isListening ? `bg-gradient-to-br ${colors.gradient} shadow-lg scale-110` : "glass hover:glass-strong"} transition-all duration-200 text-white ${mode === "user" ? "flex-1" : "flex-shrink-0"} group btn-ripple`}
                type="button"
                title="Voice input"
              >
                <svg className={`w-6 h-6 ${isListening ? "animate-pulse" : "group-hover:scale-110"} transition-transform ${mode === "user" ? "mx-auto" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                {mode === "user" && (
                  <span className="ml-3 text-sm font-medium">
                    {isListening ? "Listening... Speak now!" : "Tap to speak"}
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
                    onClick={handleSend}
                    disabled={isSending || !input.trim()}
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
          </footer>
        </section>

      </div>

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
