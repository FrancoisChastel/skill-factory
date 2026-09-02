import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";
import { theme, EXPO } from "./theme";

/** Full-frame dark backdrop: subtle grid, vignette, and a drifting accent glow. */
export const Backdrop: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const drift = interpolate(frame, [0, durationInFrames], [0, 1]);
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg }}>
      {/* grid */}
      <AbsoluteFill
        style={{
          backgroundImage: `linear-gradient(${theme.border} 1px, transparent 1px), linear-gradient(90deg, ${theme.border} 1px, transparent 1px)`,
          backgroundSize: "64px 64px",
          opacity: 0.25,
          maskImage: "radial-gradient(circle at 50% 45%, black, transparent 78%)",
          WebkitMaskImage: "radial-gradient(circle at 50% 45%, black, transparent 78%)",
        }}
      />
      {/* drifting glow */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(600px 600px at ${20 + drift * 60}% ${30 + drift * 20}%, ${theme.accentGlow}, transparent 60%)`,
          filter: "blur(30px)",
          opacity: 0.5,
        }}
      />
      {/* vignette */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.55) 100%)",
        }}
      />
      {children}
    </AbsoluteFill>
  );
};

/** Fade + rise in, driven by a spring so motion feels physical. */
export const FadeUp: React.FC<{
  delay?: number;
  y?: number;
  children: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ delay = 0, y = 28, children, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delay, fps, config: { damping: 200 } });
  const opacity = interpolate(frame - delay, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <div style={{ opacity, transform: `translateY(${(1 - s) * y}px)`, ...style }}>
      {children}
    </div>
  );
};

/** Blinking terminal cursor. */
export const Cursor: React.FC<{ color?: string }> = ({ color = theme.accent }) => {
  const frame = useCurrentFrame();
  const on = Math.floor(frame / 15) % 2 === 0;
  return (
    <span
      style={{
        display: "inline-block",
        width: "0.55em",
        height: "1.05em",
        marginLeft: "0.08em",
        background: color,
        opacity: on ? 1 : 0,
        transform: "translateY(0.14em)",
        boxShadow: `0 0 14px ${theme.accentGlow}`,
      }}
    />
  );
};

/** A labeled score bar that fills from `from` to `to`. */
export const ScoreBar: React.FC<{
  label: string;
  from: number;
  to: number;
  delay?: number;
  width?: number;
}> = ({ label, from, to, delay = 0, width = 720 }) => {
  const frame = useCurrentFrame();
  const t = interpolate(frame - delay, [0, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.bezier(...EXPO),
  });
  const value = from + (to - from) * t;
  const delta = Math.round((to - from) * 100);
  return (
    <div style={{ width, fontFamily: theme.mono }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          color: theme.muted,
          fontSize: 26,
          marginBottom: 14,
        }}
      >
        <span style={{ color: theme.text }}>{label}</span>
        <span>
          <span style={{ color: theme.muted }}>{from.toFixed(3)}</span>
          <span style={{ color: theme.muted }}> → </span>
          <span style={{ color: theme.accent }}>{value.toFixed(3)}</span>
        </span>
      </div>
      <div
        style={{
          height: 22,
          borderRadius: 12,
          background: theme.bgPanel2,
          border: `1px solid ${theme.border}`,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${value * 100}%`,
            background: `linear-gradient(90deg, ${theme.accentDim}, ${theme.accent})`,
            boxShadow: `0 0 24px ${theme.accentGlow}`,
          }}
        />
      </div>
      <div
        style={{
          marginTop: 12,
          textAlign: "right",
          color: theme.accent,
          fontSize: 24,
          opacity: t,
        }}
      >
        +{Math.round(delta * t)} pts
      </div>
    </div>
  );
};

/** A small pill/chip for provider & harness names. */
export const Chip: React.FC<{ label: string; delay?: number; active?: boolean }> = ({
  label,
  delay = 0,
  active = false,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delay, fps, config: { damping: 200 } });
  return (
    <div
      style={{
        transform: `scale(${0.8 + s * 0.2})`,
        opacity: s,
        padding: "14px 26px",
        borderRadius: 999,
        fontFamily: theme.mono,
        fontSize: 28,
        color: active ? theme.bg : theme.text,
        background: active ? theme.accent : theme.bgPanel2,
        border: `1px solid ${active ? theme.accent : theme.border}`,
        boxShadow: active ? `0 0 30px ${theme.accentGlow}` : "none",
      }}
    >
      {label}
    </div>
  );
};

/** A titled panel/card. */
export const Panel: React.FC<{
  title: string;
  children?: React.ReactNode;
  delay?: number;
  width?: number;
  accent?: boolean;
}> = ({ title, children, delay = 0, width = 360, accent = false }) => {
  return (
    <FadeUp delay={delay}>
      <div
        style={{
          width,
          padding: 28,
          borderRadius: 18,
          background: theme.bgPanel,
          border: `1px solid ${accent ? theme.accent : theme.border}`,
          boxShadow: accent ? `0 0 40px ${theme.accentGlow}` : "0 20px 60px rgba(0,0,0,0.4)",
        }}
      >
        <div
          style={{
            fontFamily: theme.mono,
            fontSize: 22,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: accent ? theme.accent : theme.muted,
            marginBottom: 16,
          }}
        >
          {title}
        </div>
        <div style={{ fontFamily: theme.sans, fontSize: 26, color: theme.text, lineHeight: 1.4 }}>
          {children}
        </div>
      </div>
    </FadeUp>
  );
};
