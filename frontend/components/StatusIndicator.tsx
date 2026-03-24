export function ListeningIndicator() {
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-full glass-strong animate-pulse-slow">
      <div className="relative">
        <div className="w-4 h-4 rounded-full bg-red-500 animate-pulse" />
        <div className="absolute top-0 left-0 w-4 h-4 rounded-full bg-red-500 pulse-ring" />
      </div>
      <span className="text-sm font-medium text-white">Listening...</span>
      <div className="voice-wave text-white opacity-75">
        <div className="wave-bar" style={{ height: "6px" }} />
        <div className="wave-bar" style={{ height: "12px" }} />
        <div className="wave-bar" style={{ height: "18px" }} />
        <div className="wave-bar" style={{ height: "12px" }} />
        <div className="wave-bar" style={{ height: "6px" }} />
      </div>
    </div>
  );
}

export function SpeakingIndicator() {
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-full glass-strong">
      <div className="relative">
        <div className="w-4 h-4 rounded-full bg-green-500 animate-pulse" />
        <div className="absolute top-0 left-0 w-4 h-4 rounded-full bg-green-500 pulse-ring" />
      </div>
      <span className="text-sm font-medium text-white">Agent speaking...</span>
      <div className="voice-wave text-green-400">
        <div className="wave-bar" style={{ height: "8px" }} />
        <div className="wave-bar" style={{ height: "16px" }} />
        <div className="wave-bar" style={{ height: "12px" }} />
        <div className="wave-bar" style={{ height: "20px" }} />
        <div className="wave-bar" style={{ height: "12px" }} />
        <div className="wave-bar" style={{ height: "8px" }} />
      </div>
    </div>
  );
}

export function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-full glass-strong">
      <div className="relative">
        <div className="w-4 h-4 rounded-full bg-blue-500 animate-spin border-2 border-blue-300 border-t-transparent" />
      </div>
      <span className="text-sm font-medium text-white">Processing...</span>
      <div className="typing-indicator text-blue-400">
        <div className="typing-dot" />
        <div className="typing-dot" />
        <div className="typing-dot" />
      </div>
    </div>
  );
}

export function TypingIndicator({ show }: { show: boolean }) {
  if (!show) return null;

  return (
    <div className="flex items-start gap-3 max-w-[85%] animate-slide-in">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      </div>
      <div className="flex items-center gap-2 px-5 py-3 rounded-2xl glass-strong">
        <div className="typing-indicator text-white">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
