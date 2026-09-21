"use client";

import { RefObject } from "react";
import { ChatMessage, ChatMode } from "@/lib/api";
import { AGENT_LABELS } from "@/lib/conversation";

type MessageListProps = {
  messages: ChatMessage[];
  partialAssistantMessage: string;
  messagesEndRef: RefObject<HTMLDivElement>;
  mode: ChatMode;
  colors: { gradient: string; accentGradient: string };
};

/**
 * The transcript. Presentational only — it renders what it is given and owns
 * no state, so the scroll anchor is passed down rather than created here.
 */
export function MessageList({
  messages,
  partialAssistantMessage,
  messagesEndRef,
  mode,
  colors,
}: MessageListProps) {
  return (
    <>
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
    </>
  );
}
