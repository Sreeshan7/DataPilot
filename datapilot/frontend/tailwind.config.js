/** Colours are CSS variables (see src/styles/globals.css) so light mode is a token swap, not a rewrite. */
const token = (name) => `hsl(var(--${name}) / <alpha-value>)`;

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: token("background"),
        surface: token("surface"),
        raised: token("raised"),
        foreground: token("foreground"),
        muted: { DEFAULT: token("muted"), foreground: token("muted-foreground") },
        subtle: token("subtle"),
        border: token("border"),
        ring: token("ring"),
        accent: { DEFAULT: token("accent"), foreground: token("accent-foreground"), soft: token("accent-soft") },
        success: token("success"),
        warning: token("warning"),
        danger: token("danger"),
        info: token("info"),
      },
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ['"JetBrains Mono Variable"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: { lg: "0.75rem", xl: "1rem", "2xl": "1.25rem" },
      keyframes: {
        "fade-up": { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "none" } },
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        shimmer: { from: { backgroundPosition: "200% 0" }, to: { backgroundPosition: "-200% 0" } },
        "phrase-in": { from: { opacity: "0", transform: "translateY(10px)", filter: "blur(2px)" }, to: { opacity: "1", transform: "none", filter: "none" } },
        "phrase-out": { from: { opacity: "1", transform: "none" }, to: { opacity: "0", transform: "translateY(-10px)", filter: "blur(2px)" } },
        "pulse-dot": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
      },
      animation: {
        "fade-up": "fade-up 220ms cubic-bezier(0.2,0.7,0.2,1) both",
        "fade-in": "fade-in 180ms ease-out both",
        shimmer: "shimmer 2.2s linear infinite",
        "phrase-in": "phrase-in 420ms cubic-bezier(0.2,0.7,0.2,1) both",
        "phrase-out": "phrase-out 320ms ease-in both",
        "pulse-dot": "pulse-dot 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
