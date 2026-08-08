export type ChatMode = "user" | "recruiter";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  audioBase64?: string;
  mimeType?: string;
  createdAt: number;
  agentName?: string;
  jobResults?: JobResult[];
};

export type JobResult = {
  title: string;
  url: string;
  snippet: string;
  rank_score: number;
};

export type EmailDraft = {
  subject: string;
  greeting: string;
  body: string;
  closing: string;
  signature: string;
};

export type AttendanceSuggestion = {
  subject: string;
  day_of_week: number;
  start_time: string;
  end_time: string;
  location?: string;
  attendance_percentage: number;
  risk: string;
  suggestion_reason: string;
};

/**
 * Backend API client.
 *
 * No function here takes a userId: the server derives identity from the
 * session cookie, so sending one from the browser would be meaningless at best
 * and rejected at worst. Every call goes through authFetch, which attaches
 * credentials, adds the CSRF header, and transparently refreshes an expired
 * access token once before giving up.
 */
import { authFetch } from "./auth";

async function readError(resp: Response, fallback: string): Promise<string> {
  try {
    const body = await resp.json();
    if (typeof body.detail === "string") return body.detail;
    if (body.detail?.message) return body.detail.message;
  } catch {
    /* response was not JSON */
  }
  return fallback;
}

export async function askWithVoice(params: {
  query: string;
  sessionId: string;
  mode: ChatMode;
}): Promise<{
  displayText: string;
  speechText: string;
}> {
  const resp = await authFetch(`/api/v1/agents/query`, {
    method: "POST",
    body: JSON.stringify({
      query: params.query,
      session_id: params.sessionId,
      output_mode: params.mode
    })
  });

  if (!resp.ok) {
    throw new Error(await readError(resp, `Request failed (${resp.status})`));
  }

  const data = await resp.json();
  return {
    displayText: data.display_text ?? "",
    speechText: data.speech_text ?? "",
  };
}

export async function fetchJobs(query: string): Promise<JobResult[]> {
  const resp = await authFetch(`/api/v1/agents/tools/job-search`, {
    method: "POST",
    body: JSON.stringify({ query, max_results: 6, min_score: 0.2 })
  });

  if (!resp.ok) {
    throw new Error(await readError(resp, `Job search failed (${resp.status})`));
  }

  const data = await resp.json();
  return (data.results ?? []) as JobResult[];
}

export async function draftEmail(query: string): Promise<EmailDraft | null> {
  const resp = await authFetch(`/api/v1/agents/tools/email-draft`, {
    method: "POST",
    body: JSON.stringify({ query, tone: "professional" })
  });

  if (!resp.ok) {
    throw new Error(await readError(resp, `Email draft failed (${resp.status})`));
  }

  const data = await resp.json();
  return data.draft ?? null;
}

export async function fetchAttendanceSuggestions(): Promise<AttendanceSuggestion[]> {
  const resp = await authFetch(`/api/v1/agents/tools/timetable/suggest`, {
    method: "POST",
    body: JSON.stringify({ low_attendance_threshold: 75 })
  });

  if (!resp.ok) {
    throw new Error(await readError(resp, `Attendance suggestions failed (${resp.status})`));
  }

  const data = await resp.json();
  return (data.suggestions ?? []) as AttendanceSuggestion[];
}

export async function fetchProfileFacts(
  userId: string
): Promise<Array<{ key: string; value: string; source?: string }>> {
  const resp = await authFetch(
    `/api/v1/agents/memory/profile/${encodeURIComponent(userId)}`,
    { method: "GET" }
  );
  if (!resp.ok) {
    throw new Error(await readError(resp, `Could not load memory (${resp.status})`));
  }
  const data = await resp.json();
  return data.facts ?? [];
}

export async function forgetProfileFact(userId: string, key: string): Promise<void> {
  const resp = await authFetch(
    `/api/v1/agents/memory/profile/${encodeURIComponent(userId)}/${encodeURIComponent(key)}`,
    { method: "DELETE" }
  );
  if (!resp.ok) {
    throw new Error(await readError(resp, `Could not forget fact (${resp.status})`));
  }
}

export async function uploadTimetablePdf(file: File): Promise<{
  entries_stored: number;
  filename: string;
  old_entries_cleared: number;
}> {
  const formData = new FormData();
  formData.append("clear_existing", "true");
  formData.append("file", file);

  const resp = await authFetch(`/api/v1/agents/tools/timetable/upload-pdf`, {
    method: "POST",
    body: formData,
  });

  if (!resp.ok) {
    throw new Error(await readError(resp, "Upload failed"));
  }
  return resp.json();
}

export type ConversationSummary = {
  id: string;
  title: string;
  status: string;
  modality: string;
  turn_count: number;
  started_at: string | null;
  last_active_at: string | null;
};

export type ConversationTurn = {
  sequence: number;
  role: "user" | "assistant";
  content: string;
  modality: string;
  agent?: string | null;
  created_at: string | null;
};

/** List the signed-in user's conversation threads, most recent first. */
export async function fetchConversations(): Promise<ConversationSummary[]> {
  const resp = await authFetch(`/api/v1/agents/conversations`, { method: "GET" });
  if (!resp.ok) {
    throw new Error(await readError(resp, `Could not load conversations (${resp.status})`));
  }
  const data = await resp.json();
  return (data.conversations ?? []) as ConversationSummary[];
}

/**
 * Fetch one thread and its turns.
 *
 * Returns null for 404 rather than throwing: a stored conversation id that no
 * longer exists (cleared server-side, or belonging to a previous account on
 * this browser) is an expected condition, and the caller simply starts fresh.
 */
export async function fetchConversation(
  conversationId: string
): Promise<{ conversation: ConversationSummary; turns: ConversationTurn[] } | null> {
  const resp = await authFetch(
    `/api/v1/agents/conversations/${encodeURIComponent(conversationId)}`,
    { method: "GET" }
  );
  if (resp.status === 404) return null;
  if (!resp.ok) {
    throw new Error(await readError(resp, `Could not load conversation (${resp.status})`));
  }
  const data = await resp.json();
  return { conversation: data.conversation, turns: data.turns ?? [] };
}

/** Archive a thread. Turns are retained; extracted memories stay valid. */
export async function archiveConversation(conversationId: string): Promise<void> {
  const resp = await authFetch(
    `/api/v1/agents/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" }
  );
  if (!resp.ok && resp.status !== 404) {
    throw new Error(await readError(resp, `Could not archive conversation (${resp.status})`));
  }
}

/**
 * Request a LiveKit token for this session's own voice room.
 * The room is derived server-side from the authenticated identity.
 */
export async function getLiveKitToken(conversationId?: string): Promise<{
  token: string;
  url: string;
  room_name: string;
}> {
  const resp = await authFetch(`/api/v1/voice/token`, {
    method: "POST",
    // Voice turns join the conversation on screen. Without this the worker
    // derives its own thread from the room name, and everything said out loud
    // is persisted somewhere the UI never reads.
    body: JSON.stringify(conversationId ? { conversation_id: conversationId } : {}),
  });

  if (!resp.ok) {
    throw new Error(await readError(resp, `Could not start voice session (${resp.status})`));
  }

  return resp.json();
}
