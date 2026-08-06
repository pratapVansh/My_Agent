"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { AuthError, login } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const nextPath = searchParams.get("next") || "/user";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    setError(null);
    setIsSubmitting(true);
    try {
      await login(username, password);
      // Full navigation so the new session cookie is picked up everywhere.
      router.replace(nextPath);
      router.refresh();
    } catch (err) {
      setError(
        err instanceof AuthError
          ? err.message
          : "Could not reach the server. Is the backend running?"
      );
      setPassword("");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-20 left-20 w-72 h-72 bg-blue-500 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: "1s" }} />
      </div>

      <div className="relative z-10 flex min-h-screen items-center justify-center p-6">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <div className="inline-flex w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 items-center justify-center shadow-2xl mb-5">
              <svg className="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-white">Sign in</h1>
            <p className="text-white/60 mt-2 text-sm">
              Your personal assistant, your data.
            </p>
          </div>

          <form
            onSubmit={handleSubmit}
            className="glass-strong rounded-3xl p-8 shadow-2xl space-y-5"
          >
            {error && (
              <div
                role="alert"
                className="rounded-xl bg-red-500/15 border border-red-400/30 px-4 py-3 text-sm text-red-200"
              >
                {error}
              </div>
            )}

            <div>
              <label htmlFor="username" className="block text-xs font-medium uppercase tracking-wide text-white/50 mb-2">
                Username
              </label>
              <input
                id="username"
                name="username"
                type="text"
                autoComplete="username"
                required
                autoFocus
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={isSubmitting}
                className="w-full rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-white placeholder-white/30 outline-none focus:border-purple-400/60 focus:ring-2 focus:ring-purple-400/30 transition disabled:opacity-50"
                placeholder="your username"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-xs font-medium uppercase tracking-wide text-white/50 mb-2">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isSubmitting}
                className="w-full rounded-xl bg-white/5 border border-white/10 px-4 py-3 text-white placeholder-white/30 outline-none focus:border-purple-400/60 focus:ring-2 focus:ring-purple-400/30 transition disabled:opacity-50"
                placeholder="••••••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting || !username || !password}
              className="w-full rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 px-4 py-3 font-semibold text-white shadow-lg transition hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isSubmitting ? "Signing in…" : "Sign in"}
            </button>
          </form>

          <p className="text-center text-white/40 text-sm mt-6">
            Just looking around?{" "}
            <Link href="/recruiter" className="text-purple-300 hover:text-purple-200 underline underline-offset-4">
              Continue as a guest
            </Link>
          </p>
        </div>
      </div>
    </main>
  );
}
