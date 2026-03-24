"use client";

import { useState } from "react";

type CloseChatModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  mode: "user" | "recruiter";
};

export function CloseChatModal({ isOpen, onClose, onConfirm, mode }: CloseChatModalProps) {
  if (!isOpen) return null;

  const colors = mode === "user"
    ? "from-user-primary to-user-secondary"
    : "from-recruiter-primary to-recruiter-secondary";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center animate-fade-in">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-md mx-4 animate-scale-in">
        <div className="glass-strong rounded-2xl p-6 shadow-2xl">
          {/* Header */}
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${colors} flex items-center justify-center`}>
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <h3 className="text-xl font-bold text-white">End Chat Session?</h3>
              <p className="text-sm text-white/60">This action cannot be undone</p>
            </div>
          </div>

          {/* Body */}
          <p className="text-white/80 mb-6">
            Are you sure you want to end this chat session? All conversation history will be cleared.
          </p>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2.5 rounded-lg glass hover:glass-strong transition-all duration-200 text-white font-medium"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              className={`flex-1 px-4 py-2.5 rounded-lg bg-gradient-to-r ${colors} hover:shadow-lg hover:scale-105 transition-all duration-200 text-white font-medium`}
            >
              End Chat
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

type ClearChatModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  mode: "user" | "recruiter";
};

export function ClearChatModal({ isOpen, onClose, onConfirm, mode }: ClearChatModalProps) {
  if (!isOpen) return null;

  const colors = mode === "user"
    ? "from-user-primary to-user-secondary"
    : "from-recruiter-primary to-recruiter-secondary";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center animate-fade-in">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      <div className="relative z-10 w-full max-w-md mx-4 animate-scale-in">
        <div className="glass-strong rounded-2xl p-6 shadow-2xl">
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${colors} flex items-center justify-center`}>
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </div>
            <div>
              <h3 className="text-xl font-bold text-white">Clear All Messages?</h3>
              <p className="text-sm text-white/60">Start a fresh conversation</p>
            </div>
          </div>
          <p className="text-white/80 mb-6">
            This will remove all messages from the current chat. You can continue chatting after clearing.
          </p>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2.5 rounded-lg glass hover:glass-strong transition-all duration-200 text-white font-medium"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              className={`flex-1 px-4 py-2.5 rounded-lg bg-gradient-to-r ${colors} hover:shadow-lg hover:scale-105 transition-all duration-200 text-white font-medium`}
            >
              Clear Chat
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
