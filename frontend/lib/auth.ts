/**
 * Client-side session handling.
 *
 * Tokens are never touched by this code. They live in HttpOnly cookies the
 * browser attaches automatically, so JavaScript cannot read — and therefore
 * cannot leak — them. What this module does hold is the CSRF token, which is
 * deliberately readable and carries no authority on its own: it only proves
 * the caller can read this origin's cookie jar.
 */

export type Role = "owner" | "guest";

export type Identity = {
  user_id: string;
  role: Role;
  scopes: string[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const CSRF_COOKIE = "csrf_token";
const CSRF_HEADER = "X-CSRF-Token";

export class AuthError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "AuthError";
    this.status = status;
  }
}

function readCookie(name: string): string | null {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(
    new RegExp(`(?:^|;\\s*)${name}=([^;]*)`)
  );
  return match ? decodeURIComponent(match[1]) : null;
}

function buildHeaders(init?: HeadersInit, isJson = true): Headers {
  const headers = new Headers(init);
  if (isJson && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const csrf = readCookie(CSRF_COOKIE);
  if (csrf) headers.set(CSRF_HEADER, csrf);
  return headers;
}

/** Serialises refreshes so a burst of 401s triggers exactly one refresh call. */
let refreshInFlight: Promise<boolean> | null = null;

async function refreshSession(): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
          method: "POST",
          credentials: "include",
          headers: buildHeaders(),
        });
        return resp.ok;
      } catch {
        return false;
      } finally {
        // Cleared on the next tick so concurrent callers all observe this run.
        setTimeout(() => {
          refreshInFlight = null;
        }, 0);
      }
    })();
  }
  return refreshInFlight;
}

/**
 * fetch() with credentials, CSRF, and one automatic retry after a token refresh.
 *
 * Access tokens are short-lived by design; transparently refreshing on the
 * first 401 keeps that invisible to the user instead of bouncing them to login
 * every fifteen minutes.
 */
export async function authFetch(
  path: string,
  init: RequestInit = {},
  retryOnUnauthorized = true
): Promise<Response> {
  const isFormData = init.body instanceof FormData;

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: buildHeaders(init.headers, !isFormData),
  });

  if (response.status === 401 && retryOnUnauthorized) {
    const refreshed = await refreshSession();
    if (refreshed) {
      return authFetch(path, init, false);
    }
  }

  return response;
}

/** Current identity, or null when there is no valid session. */
export async function fetchIdentity(): Promise<Identity | null> {
  try {
    const resp = await authFetch("/api/v1/auth/me", { method: "GET" });
    if (!resp.ok) return null;
    return (await resp.json()) as Identity;
  } catch {
    return null;
  }
}

export async function login(username: string, password: string): Promise<Identity> {
  const resp = await fetch(`${API_BASE}/api/v1/auth/login`, {
    method: "POST",
    credentials: "include",
    headers: buildHeaders(),
    body: JSON.stringify({ username, password }),
  });

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new AuthError(
      typeof body.detail === "string" ? body.detail : "Sign in failed.",
      resp.status
    );
  }

  const data = await resp.json();
  return { user_id: data.user_id, role: data.role, scopes: data.scopes };
}

/** Start an anonymous, read-only session (the public recruiter view). */
export async function startGuestSession(): Promise<Identity> {
  const resp = await fetch(`${API_BASE}/api/v1/auth/guest`, {
    method: "POST",
    credentials: "include",
    headers: buildHeaders(),
  });

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new AuthError(
      typeof body.detail === "string" ? body.detail : "Could not start a session.",
      resp.status
    );
  }

  const data = await resp.json();
  return { user_id: data.user_id, role: data.role, scopes: data.scopes };
}

export async function logout(): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/v1/auth/logout`, {
      method: "POST",
      credentials: "include",
      headers: buildHeaders(),
    });
  } catch {
    // Cookies are cleared server-side; a network failure here is not fatal.
  }
}

/**
 * Resolve a usable session for a page.
 *
 * Recruiter mode self-provisions a guest session so a visitor never sees a
 * login wall. User mode requires a real sign-in and returns null when absent,
 * letting the caller redirect.
 */
export async function ensureSession(mode: "user" | "recruiter"): Promise<Identity | null> {
  const existing = await fetchIdentity();
  if (existing) {
    // A guest token cannot drive the owner's view — force a real sign-in.
    if (mode === "user" && existing.role !== "owner") return null;
    return existing;
  }

  if (mode === "recruiter") {
    return startGuestSession();
  }
  return null;
}

export function hasScope(identity: Identity | null, scope: string): boolean {
  return !!identity?.scopes.includes(scope);
}
