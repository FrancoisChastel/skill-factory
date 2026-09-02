import React from "react";
import { interpolate, useCurrentFrame, spring, useVideoConfig } from "remotion";
import { theme } from "./theme";

const NODES = [
  { label: "Skill", sub: "SKILL.md" },
  { label: "Harness", sub: "run(skill, task)" },
  { label: "Rollout", sub: "output" },
  { label: "Metric", sub: "score + feedback" },
  { label: "Optimizer", sub: "edit" },
];

/** Animated pentagon loop of the five primitives, with a pulse traveling the ring. */
export const LoopDiagram: React.FC<{ size?: number; perNode?: number; startDelay?: number }> = ({
  size = 620,
  perNode = 32,
  startDelay = 20,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const cx = size / 2;
  const cy = size / 2;
  const R = size * 0.36;

  const pts = NODES.map((_, i) => {
    const a = (-90 + i * 72) * (Math.PI / 180);
    return { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) };
  });

  // Pulse progress around the ring (0..5), one node per `perNode` frames.
  const prog = Math.max(0, (frame - startDelay) / perNode);
  const activeIndex = Math.floor(prog) % NODES.length;

  const ringDash = 2 * Math.PI * R;

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ position: "absolute", inset: 0 }}>
        {/* dim base ring */}
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={theme.accentDim} strokeWidth={3} />
        {/* animated flow ring */}
        <circle
          cx={cx}
          cy={cy}
          r={R}
          fill="none"
          stroke={theme.accent}
          strokeWidth={3}
          strokeLinecap="round"
          strokeDasharray={`${ringDash * 0.12} ${ringDash}`}
          strokeDashoffset={-((frame * 6) % ringDash)}
          style={{ filter: `drop-shadow(0 0 8px ${theme.accentGlow})` }}
          transform={`rotate(-90 ${cx} ${cy})`}
        />
        {/* traveling pulse dot */}
        <PulseDot cx={cx} cy={cy} R={R} prog={prog} />
      </svg>

      {NODES.map((n, i) => {
        const active = i === activeIndex && frame > startDelay;
        const s = spring({ frame: frame - i * 6, fps, config: { damping: 200 } });
        const scale = (0.7 + s * 0.3) * (active ? 1.12 : 1);
        return (
          <div
            key={n.label}
            style={{
              position: "absolute",
              left: pts[i].x,
              top: pts[i].y,
              transform: `translate(-50%, -50%) scale(${scale})`,
              opacity: s,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 6,
              width: 200,
            }}
          >
            <div
              style={{
                padding: "16px 22px",
                borderRadius: 14,
                background: active ? theme.accent : theme.bgPanel,
                color: active ? theme.bg : theme.text,
                border: `1px solid ${active ? theme.accent : theme.border}`,
                fontFamily: theme.mono,
                fontSize: 30,
                fontWeight: 600,
                boxShadow: active ? `0 0 40px ${theme.accentGlow}` : "0 12px 30px rgba(0,0,0,0.5)",
                whiteSpace: "nowrap",
              }}
            >
              {n.label}
            </div>
            <div style={{ fontFamily: theme.mono, fontSize: 18, color: theme.muted }}>{n.sub}</div>
          </div>
        );
      })}
    </div>
  );
};

const PulseDot: React.FC<{ cx: number; cy: number; R: number; prog: number }> = ({
  cx,
  cy,
  R,
  prog,
}) => {
  const angleDeg = -90 + prog * 72;
  const a = angleDeg * (Math.PI / 180);
  const x = cx + R * Math.cos(a);
  const y = cy + R * Math.sin(a);
  const pulse = interpolate(prog % 1, [0, 0.5, 1], [1, 1.6, 1]);
  return (
    <circle cx={x} cy={y} r={9 * pulse} fill={theme.accent} style={{ filter: `drop-shadow(0 0 10px ${theme.accent})` }} />
  );
};
