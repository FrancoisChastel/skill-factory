export const theme = {
  bg: "#080b09",
  bgPanel: "#0f1511",
  bgPanel2: "#141b16",
  border: "#233026",
  text: "#e7f2ea",
  muted: "#7d907f",
  accent: "#4ade80", // neon green
  accentDim: "#1c3b28",
  accentGlow: "rgba(74, 222, 128, 0.35)",
  good: "#4ade80",
  warn: "#f5a524",
  bad: "#f26d6d",
  mono: '"SF Mono", "JetBrains Mono", "Fira Code", ui-monospace, Menlo, monospace',
  sans: 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif',
} as const;

// Shared easing curve (ease-out-expo) for confident, snappy motion.
export const EXPO: [number, number, number, number] = [0.16, 1, 0.3, 1];
