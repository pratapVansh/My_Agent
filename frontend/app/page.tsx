"use client";

import Link from "next/link";
import { useState } from "react";

export default function Home() {
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-20 left-20 w-72 h-72 bg-blue-500 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: "1s" }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-pink-500 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: "2s" }} />
      </div>

      {/* Grid pattern overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.05)_1px,transparent_1px)] bg-[size:50px_50px]" />

      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center p-6">
        {/* Header */}
        <div className="text-center mb-16 space-y-6 animate-slide-up">
          {/* Logo */}
          <div className="flex justify-center mb-8">
            <div className="relative">
              <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-2xl rotate-6 hover:rotate-0 transition-transform duration-500">
                <svg className="w-14 h-14 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div className="absolute -inset-2 bg-gradient-to-r from-blue-500 to-pink-500 rounded-3xl blur-2xl opacity-50 animate-pulse-slow" />
            </div>
          </div>

          <h1 className="text-6xl md:text-7xl font-black text-white mb-4">
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-gradient">
              AI Assistant
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-white/80 max-w-2xl mx-auto font-light">
            Your intelligent companion for productivity, career growth, and communication
          </p>

          {/* Feature badges */}
          <div className="flex flex-wrap justify-center gap-3 mt-8">
            {["Voice Enabled", "Real-time", "Smart Tools"].map((badge, idx) => (
              <span
                key={badge}
                className="px-4 py-2 rounded-full glass-strong text-white/90 text-sm font-medium animate-slide-up"
                style={{ animationDelay: `${idx * 100}ms` }}
              >
                {badge}
              </span>
            ))}
          </div>
        </div>

        {/* Cards */}
        <div className="grid md:grid-cols-2 gap-6 max-w-5xl w-full mb-12">
          {/* User Card */}
          <Link href="/user" onMouseEnter={() => setHoveredCard("user")} onMouseLeave={() => setHoveredCard(null)}>
            <div className={`group relative h-80 rounded-3xl glass-strong shadow-2xl overflow-hidden transition-all duration-500 hover:scale-105 hover:shadow-blue-500/50 ${hoveredCard === "user" ? "ring-2 ring-blue-400" : ""}`}>
              {/* Gradient overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 opacity-80 group-hover:opacity-90 transition-opacity" />

              {/* Animated shapes */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl transform translate-x-32 -translate-y-32 group-hover:scale-150 transition-transform duration-700" />
              <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/10 rounded-full blur-2xl transform -translate-x-24 translate-y-24 group-hover:scale-150 transition-transform duration-700" />

              {/* Content */}
              <div className="relative h-full flex flex-col justify-between p-8">
                <div>
                  <div className="w-16 h-16 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                  <h2 className="text-3xl font-bold text-white mb-3">User Mode</h2>
                  <p className="text-white/90 text-lg mb-6">
                    Personal AI assistant for job search, email drafting, and academic support with voice interaction.
                  </p>
                </div>

                <div className="space-y-2">
                  {["Job Search", "Email Assistant", "Attendance Tracking"].map((feature, idx) => (
                    <div key={feature} className="flex items-center gap-2 text-white/90 animate-slide-in" style={{ animationDelay: `${idx * 50}ms` }}>
                      <svg className="w-5 h-5 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span className="text-sm font-medium">{feature}</span>
                    </div>
                  ))}
                </div>

                {/* Arrow */}
                <div className="absolute bottom-8 right-8 w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center group-hover:bg-white/30 group-hover:translate-x-2 transition-all duration-300">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </div>
              </div>
            </div>
          </Link>

          {/* Recruiter Card */}
          <Link href="/recruiter" onMouseEnter={() => setHoveredCard("recruiter")} onMouseLeave={() => setHoveredCard(null)}>
            <div className={`group relative h-80 rounded-3xl glass-strong shadow-2xl overflow-hidden transition-all duration-500 hover:scale-105 hover:shadow-cyan-500/50 ${hoveredCard === "recruiter" ? "ring-2 ring-cyan-400" : ""}`}>
              {/* Gradient overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-600 via-blue-600 to-teal-600 opacity-80 group-hover:opacity-90 transition-opacity" />

              {/* Animated shapes */}
              <div className="absolute top-0 left-0 w-64 h-64 bg-white/10 rounded-full blur-3xl transform -translate-x-32 -translate-y-32 group-hover:scale-150 transition-transform duration-700" />
              <div className="absolute bottom-0 right-0 w-48 h-48 bg-white/10 rounded-full blur-2xl transform translate-x-24 translate-y-24 group-hover:scale-150 transition-transform duration-700" />

              {/* Content */}
              <div className="relative h-full flex flex-col justify-between p-8">
                <div>
                  <div className="w-16 h-16 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                    <svg className="w-9 h-9 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h2 className="text-3xl font-bold text-white mb-3">Recruiter Mode</h2>
                  <p className="text-white/90 text-lg mb-6">
                    Professional AI assistant designed for recruitment professionals and candidate engagement.
                  </p>
                </div>

                <div className="space-y-2">
                  {["Smart Responses", "Professional Tone", "Voice Interaction"].map((feature, idx) => (
                    <div key={feature} className="flex items-center gap-2 text-white/90 animate-slide-in" style={{ animationDelay: `${idx * 50}ms` }}>
                      <svg className="w-5 h-5 text-cyan-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span className="text-sm font-medium">{feature}</span>
                    </div>
                  ))}
                </div>

                {/* Arrow */}
                <div className="absolute bottom-8 right-8 w-12 h-12 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center group-hover:bg-white/30 group-hover:translate-x-2 transition-all duration-300">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </div>
              </div>
            </div>
          </Link>
        </div>

        {/* Footer info */}
        <div className="text-center space-y-4 animate-fade-in" style={{ animationDelay: "400ms" }}>
          <p className="text-white/60 text-sm">
            Powered by advanced AI with voice recognition and natural language processing
          </p>
          <div className="flex items-center justify-center gap-6 text-white/40 text-xs">
            <span className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              All systems operational
            </span>
          </div>
        </div>
      </div>
    </main>
  );
}
