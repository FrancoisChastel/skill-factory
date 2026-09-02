import React from "react";
import { AbsoluteFill, Series } from "remotion";
import {
  TitleScene,
  ProblemScene,
  IdeaScene,
  LoopScene,
  MetricsScene,
  ResultsScene,
  ProvidersScene,
  ExportScene,
  CTAScene,
} from "./scenes";
import { theme } from "./theme";

export const SCENES = [
  { C: TitleScene, d: 120 },
  { C: ProblemScene, d: 140 },
  { C: IdeaScene, d: 110 },
  { C: LoopScene, d: 230 },
  { C: MetricsScene, d: 150 },
  { C: ResultsScene, d: 190 },
  { C: ProvidersScene, d: 130 },
  { C: ExportScene, d: 130 },
  { C: CTAScene, d: 150 },
];

export const TOTAL_FRAMES = SCENES.reduce((a, s) => a + s.d, 0);

export const Explainer: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg }}>
      <Series>
        {SCENES.map(({ C, d }, i) => (
          <Series.Sequence key={i} durationInFrames={d}>
            <C />
          </Series.Sequence>
        ))}
      </Series>
    </AbsoluteFill>
  );
};
