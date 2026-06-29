import type { Config } from "tailwindcss";

// Palette lifted straight from the old CSS variables so the dark "electric"
// theme is unchanged — just expressed as Tailwind tokens now.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0e1a",
        panel: "#121829",
        "panel-2": "#1a2138",
        border: "#232c44",
        text: "#e6ebf5",
        muted: "#8b95ad",
        accent: "#ffd23f", // electric yellow
        "accent-2": "#3fa9ff", // electric blue
      },
    },
  },
  plugins: [],
};

export default config;
