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

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function askWithVoice(params: {
  query: string;
  userId: string;
  sessionId: string;
  mode: ChatMode;
}): Promise<{
  displayText: string;
  speechText: string;
  audioBase64?: string;
  mimeType?: string;
}> {
  const resp = await fetch(`${API_BASE}/api/v1/voice/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: params.query,
      user_id: params.userId,
      session_id: params.sessionId,
      output_mode: params.mode
    })
  });

  if (!resp.ok) {
    throw new Error(`Voice query failed: ${resp.status}`);
  }

  const data = await resp.json();
  return {
    displayText: data.display_text ?? "",
    speechText: data.speech_text ?? "",
    audioBase64: data.voice?.audio_base64,
    mimeType: data.voice?.mime_type
  };
}

export async function fetchJobs(userId: string, query: string): Promise<JobResult[]> {
  const resp = await fetch(`${API_BASE}/api/v1/agents/tools/job-search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, query, max_results: 6, min_score: 0.2 })
  });

  if (!resp.ok) {
    throw new Error(`Job search failed: ${resp.status}`);
  }

  const data = await resp.json();
  return (data.results ?? []) as JobResult[];
}

export async function draftEmail(userId: string, query: string): Promise<EmailDraft | null> {
  const resp = await fetch(`${API_BASE}/api/v1/agents/tools/email-draft`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, query, tone: "professional" })
  });

  if (!resp.ok) {
    throw new Error(`Email draft failed: ${resp.status}`);
  }

  const data = await resp.json();
  return data.draft ?? null;
}

export async function fetchAttendanceSuggestions(userId: string): Promise<AttendanceSuggestion[]> {
  const resp = await fetch(`${API_BASE}/api/v1/agents/tools/timetable/suggest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, low_attendance_threshold: 75 })
  });

  if (!resp.ok) {
    throw new Error(`Attendance suggestions failed: ${resp.status}`);
  }

  const data = await resp.json();
  return (data.suggestions ?? []) as AttendanceSuggestion[];
}
