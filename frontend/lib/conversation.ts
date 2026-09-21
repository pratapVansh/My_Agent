import { ChatMode } from "@/lib/api";

export function uid() {
  return Math.random().toString(36).slice(2, 10);
}

export const AGENT_LABELS: Record<string, string> = {
  job: "Job Agent",
  email: "Email Agent",
  academic: "Research Agent",
  profile: "Profile Agent",
  planner: "Planner",
  response: "Formatter",
  clarification: "Clarifying",
};

export function createConversationSessionId() {
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
export function conversationStorageKey(mode: ChatMode) {
  return `my_agent.conversation_id.${mode}`;
}

export function loadStoredConversationId(mode: ChatMode): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(conversationStorageKey(mode));
  } catch {
    // Private browsing and some embedded webviews deny storage access. A
    // non-resumable session is a degraded experience, not a broken one.
    return null;
  }
}

export function storeConversationId(mode: ChatMode, conversationId: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(conversationStorageKey(mode), conversationId);
  } catch {
    /* storage unavailable — continue without persistence */
  }
}
