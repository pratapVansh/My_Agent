import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Assistant - Smart Chat Platform",
  description: "Intelligent AI assistant with voice interaction, job search, email drafting, and academic support"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
