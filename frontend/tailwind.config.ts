import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        // Modern color palette
        user: {
          primary: "#6366f1", // Indigo
          secondary: "#8b5cf6", // Purple
          accent: "#ec4899", // Pink
          bg: "#0f172a", // Dark slate
          surface: "#1e293b",
          muted: "#64748b"
        },
        recruiter: {
          primary: "#0891b2", // Cyan
          secondary: "#06b6d4", // Light cyan
          accent: "#14b8a6", // Teal
          bg: "#082f49", // Dark cyan
          surface: "#0c4a6e",
          muted: "#6b7280"
        },
        neutral: {
          50: "#fafafa",
          100: "#f4f4f5",
          200: "#e4e4e7",
          300: "#d4d4d8",
          800: "#27272a",
          900: "#18181b"
        }
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "bounce-slow": "bounce 2s infinite",
        "slide-up": "slideUp 0.3s ease-out",
        "slide-in": "slideIn 0.2s ease-out",
        "fade-in": "fadeIn 0.2s ease-out",
        "scale-in": "scaleIn 0.2s ease-out",
        "wave": "wave 1.5s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite"
      },
      keyframes: {
        slideUp: {
          "0%": { transform: "translateY(10px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" }
        },
        slideIn: {
          "0%": { transform: "translateX(-10px)", opacity: "0" },
          "100%": { transform: "translateX(0)", opacity: "1" }
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" }
        },
        scaleIn: {
          "0%": { transform: "scale(0.95)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" }
        },
        wave: {
          "0%, 100%": { transform: "scaleY(0.5)" },
          "50%": { transform: "scaleY(1.5)" }
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" }
        }
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))"
      },
      backdropBlur: {
        xs: "2px"
      }
    }
  },
  plugins: []
};

export default config;
