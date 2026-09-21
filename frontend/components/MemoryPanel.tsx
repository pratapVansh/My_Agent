"use client";

type ProfileFact = { key: string; value: string; source?: string };

type MemoryPanelProps = {
  showMemoryPanel: boolean;
  setShowMemoryPanel: (open: boolean) => void;
  profileFacts: ProfileFact[];
  memoryLoading: boolean;
  forgetFact: (key: string) => void;
  loadProfileFacts: () => void | Promise<void>;
  colors: { surface: string; text: string };
};

/**
 * The slide-over listing stored profile facts, owner-only. Presentational:
 * loading and forgetting are owned by ChatShell, which holds the session.
 */
export function MemoryPanel({
  showMemoryPanel,
  setShowMemoryPanel,
  profileFacts,
  memoryLoading,
  forgetFact,
  loadProfileFacts,
  colors,
}: MemoryPanelProps) {
  return (
    <>
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
    </>
  );
}
