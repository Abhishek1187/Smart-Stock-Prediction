/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
      colors: {
        terminal: {
          bg: "#08131f",
          panel: "#0f1e2f",
          panelSoft: "#15273a",
          border: "#26415a",
          accent: "#2dd4bf",
          text: "#dce8f5",
          muted: "#8fa7bf",
          danger: "#fb7185",
          success: "#22c55e",
          warning: "#f59e0b",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(45, 212, 191, 0.35), 0 12px 28px rgba(0, 0, 0, 0.45)",
      },
    },
  },
  plugins: [],
};
