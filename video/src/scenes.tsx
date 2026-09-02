import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { theme } from "./theme";
import { Backdrop, FadeUp, Cursor, ScoreBar, Chip, Panel } from "./components";
import { LoopDiagram } from "./LoopDiagram";

const Center: React.FC<{ children: React.ReactNode; gap?: number }> = ({ children, gap = 28 }) => (
  <AbsoluteFill
    style={{ justifyContent: "center", alignItems: "center", flexDirection: "column", gap }}
  >
    {children}
  </AbsoluteFill>
);

const Kicker: React.FC<{ children: React.ReactNode; delay?: number }> = ({ children, delay = 0 }) => (
  <FadeUp delay={delay}>
    <div
      style={{
        fontFamily: theme.mono,
        fontSize: 26,
        letterSpacing: "0.32em",
        textTransform: "uppercase",
        color: theme.accent,
      }}
    >
      {children}
    </div>
  </FadeUp>
);

const H1: React.FC<{ children: React.ReactNode; delay?: number; size?: number }> = ({
  children,
  delay = 0,
  size = 88,
}) => (
  <FadeUp delay={delay}>
    <div
      style={{
        fontFamily: theme.sans,
        fontSize: size,
        fontWeight: 800,
        color: theme.text,
        letterSpacing: "-0.03em",
        textAlign: "center",
        lineHeight: 1.05,
      }}
    >
      {children}
    </div>
  </FadeUp>
);

const Sub: React.FC<{ children: React.ReactNode; delay?: number }> = ({ children, delay = 0 }) => (
  <FadeUp delay={delay}>
    <div style={{ fontFamily: theme.sans, fontSize: 34, color: theme.muted, textAlign: "center" }}>
      {children}
    </div>
  </FadeUp>
);

// 1 — Title
export const TitleScene: React.FC = () => (
  <Backdrop>
    <Center>
      <Kicker delay={0}>scientific skill optimization</Kicker>
      <FadeUp delay={8}>
        <div
          style={{
            fontFamily: theme.mono,
            fontSize: 118,
            fontWeight: 700,
            color: theme.text,
            letterSpacing: "-0.02em",
          }}
        >
          Skill Factory
          <Cursor />
        </div>
      </FadeUp>
      <Sub delay={20}>Train agent skills like you train weights.</Sub>
    </Center>
  </Backdrop>
);

// 2 — Problem
export const ProblemScene: React.FC = () => {
  const frame = useCurrentFrame();
  const wobble = Math.sin(frame / 8) * 1.2;
  return (
    <Backdrop>
      <Center>
        <Kicker>the old way</Kicker>
        <FadeUp delay={10}>
          <div
            style={{
              transform: `rotate(${wobble}deg)`,
              width: 640,
              padding: 32,
              borderRadius: 16,
              background: theme.bgPanel,
              border: `1px solid ${theme.border}`,
              fontFamily: theme.mono,
              fontSize: 26,
              color: theme.muted,
              boxShadow: "0 24px 70px rgba(0,0,0,0.5)",
            }}
          >
            <div style={{ color: theme.accent, marginBottom: 14 }}># SKILL.md</div>
            Read the invoice and return the fields as JSON.
            <br />
            Try to find the vendor, date, and total. <span style={{ color: theme.warn }}>?</span>
          </div>
        </FadeUp>
        <Sub delay={26}>Edit the prompt. Hope. Repeat.</Sub>
        <FadeUp delay={40}>
          <div style={{ fontFamily: theme.sans, fontSize: 40, color: theme.text, fontWeight: 700 }}>
            How do you <span style={{ color: theme.accent }}>know</span> it got better?
          </div>
        </FadeUp>
      </Center>
    </Backdrop>
  );
};

// 3 — Idea
export const IdeaScene: React.FC = () => (
  <Backdrop>
    <Center>
      <Kicker>the idea</Kicker>
      <H1 delay={8} size={72}>
        A skill is a <span style={{ color: theme.accent }}>trainable parameter.</span>
      </H1>
      <FadeUp delay={22}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 30,
            fontFamily: theme.mono,
            fontSize: 44,
            color: theme.text,
          }}
        >
          <span>SKILL.md</span>
          <span style={{ color: theme.accent }}>→</span>
          <span style={{ color: theme.accent }}>θ</span>
        </div>
      </FadeUp>
      <Sub delay={34}>Define a metric. Measure. Optimize.</Sub>
    </Center>
  </Backdrop>
);

// 4 — Loop
export const LoopScene: React.FC = () => (
  <Backdrop>
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", flexDirection: "column" }}>
      <div style={{ position: "absolute", top: 70 }}>
        <Kicker>the loop</Kicker>
      </div>
      <LoopDiagram size={680} />
      <FadeUp delay={30} style={{ position: "absolute", bottom: 80 }}>
        <div style={{ fontFamily: theme.mono, fontSize: 30, color: theme.muted }}>
          rollout <span style={{ color: theme.accent }}>→</span> reflect{" "}
          <span style={{ color: theme.accent }}>→</span> edit{" "}
          <span style={{ color: theme.accent }}>→</span> validation-gate
        </div>
      </FadeUp>
    </AbsoluteFill>
  </Backdrop>
);

// 5 — Metrics
export const MetricsScene: React.FC = () => (
  <Backdrop>
    <Center gap={44}>
      <Kicker>define "good"</Kicker>
      <div style={{ display: "flex", gap: 28 }}>
        <Panel title="Golden set" delay={8} accent>
          Match a labeled expected answer.
        </Panel>
        <Panel title="Programmatic" delay={16}>
          Deterministic checks: valid JSON, required keys, regex.
        </Panel>
        <Panel title="LLM judge" delay={24}>
          Rubric-scored by a model, with feedback.
        </Panel>
      </div>
      <FadeUp delay={40}>
        <div style={{ fontFamily: theme.mono, fontSize: 30, color: theme.text }}>
          combine by weight <span style={{ color: theme.accent }}>→</span> one score{" "}
          <span style={{ color: theme.muted }}>+ feedback</span>
        </div>
      </FadeUp>
    </Center>
  </Backdrop>
);

// 6 — Results
export const ResultsScene: React.FC = () => (
  <Backdrop>
    <Center gap={54}>
      <Kicker>real, validation-gated gains</Kicker>
      <ScoreBar label="invoice-extractor · JSON" from={0.559} to={1.0} delay={12} />
      <ScoreBar label="ticket-classifier · classification" from={0.151} to={1.0} delay={30} />
      <Sub delay={70}>Verified live on a local model — one reflective edit.</Sub>
    </Center>
  </Backdrop>
);

// 7 — Providers
export const ProvidersScene: React.FC = () => (
  <Backdrop>
    <Center gap={40}>
      <Kicker>any model, any harness</Kicker>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", justifyContent: "center", maxWidth: 1100 }}>
        {["Anthropic", "OpenAI", "Azure", "Ollama", "vLLM", "OpenRouter", "Groq", "LM Studio"].map(
          (p, i) => (
            <Chip key={p} label={p} delay={8 + i * 4} active={i === 0} />
          )
        )}
      </div>
      <FadeUp delay={50}>
        <div style={{ fontFamily: theme.mono, fontSize: 30, color: theme.muted }}>
          Claude Code · Codex · Cursor · Gemini CLI
        </div>
      </FadeUp>
    </Center>
  </Backdrop>
);

// 8 — Export
export const ExportScene: React.FC = () => {
  const frame = useCurrentFrame();
  const typed = "npx skills add dist/invoice-extractor";
  const chars = Math.floor(interpolate(frame, [15, 70], [0, typed.length], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }));
  return (
    <Backdrop>
      <Center gap={40}>
        <Kicker>ship it anywhere</Kicker>
        <FadeUp delay={8}>
          <div
            style={{
              width: 820,
              borderRadius: 16,
              background: theme.bgPanel,
              border: `1px solid ${theme.border}`,
              boxShadow: "0 24px 70px rgba(0,0,0,0.5)",
              overflow: "hidden",
            }}
          >
            <div style={{ display: "flex", gap: 8, padding: "14px 18px", background: theme.bgPanel2 }}>
              {[theme.bad, theme.warn, theme.good].map((c) => (
                <div key={c} style={{ width: 14, height: 14, borderRadius: 99, background: c }} />
              ))}
            </div>
            <div style={{ padding: 30, fontFamily: theme.mono, fontSize: 30, color: theme.text }}>
              <span style={{ color: theme.accent }}>$ </span>
              {typed.slice(0, chars)}
              <Cursor />
            </div>
          </div>
        </FadeUp>
        <Sub delay={20}>Portable SKILL.md — the open npx skills format.</Sub>
      </Center>
    </Backdrop>
  );
};

// 9 — CTA
export const CTAScene: React.FC = () => (
  <Backdrop>
    <Center gap={30}>
      <FadeUp>
        <div style={{ fontFamily: theme.mono, fontSize: 96, fontWeight: 700, color: theme.text }}>
          Skill Factory
        </div>
      </FadeUp>
      <FadeUp delay={12}>
        <div
          style={{
            fontFamily: theme.mono,
            fontSize: 36,
            color: theme.accent,
            padding: "16px 30px",
            borderRadius: 12,
            border: `1px solid ${theme.accent}`,
            boxShadow: `0 0 40px ${theme.accentGlow}`,
          }}
        >
          pip install skill-factory
        </div>
      </FadeUp>
      <Sub delay={24}>github.com/FrancoisChastel/skill-factory · MIT</Sub>
    </Center>
  </Backdrop>
);
