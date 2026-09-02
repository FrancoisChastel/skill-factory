import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
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

const FADE_FRAMES = 12;
const VO_START = 10; // frames into each scene before narration begins

// Scene durations are fitted to the narration audio (vo frames + breathing room),
// so the narrated and silent cuts share one timeline. vo: public/vo/<n>.mp3.
export const SCENES: { C: React.FC; d: number; vo: string }[] = [
  { C: TitleScene, d: 162, vo: "vo/1.mp3" },
  { C: ProblemScene, d: 192, vo: "vo/2.mp3" },
  { C: IdeaScene, d: 218, vo: "vo/3.mp3" },
  { C: LoopScene, d: 342, vo: "vo/4.mp3" },
  { C: MetricsScene, d: 254, vo: "vo/5.mp3" },
  { C: ResultsScene, d: 257, vo: "vo/6.mp3" },
  { C: ProvidersScene, d: 163, vo: "vo/7.mp3" },
  { C: ExportScene, d: 172, vo: "vo/8.mp3" },
  { C: CTAScene, d: 166, vo: "vo/9.mp3" },
];

export const TOTAL_FRAMES =
  SCENES.reduce((a, s) => a + s.d, 0) - FADE_FRAMES * (SCENES.length - 1);

export const Explainer: React.FC<{ narration?: boolean }> = ({ narration = false }) => {
  // TransitionSeries requires its Sequence/Transition children directly (no
  // fragments), so the timeline is assembled as a flat array.
  const children: React.ReactNode[] = [];
  SCENES.forEach(({ C, d, vo }, i) => {
    children.push(
      <TransitionSeries.Sequence key={`s${i}`} durationInFrames={d}>
        <C />
        {narration ? (
          <Sequence from={VO_START}>
            <Audio src={staticFile(vo)} />
          </Sequence>
        ) : null}
      </TransitionSeries.Sequence>
    );
    if (i < SCENES.length - 1) {
      children.push(
        <TransitionSeries.Transition
          key={`t${i}`}
          presentation={fade()}
          timing={linearTiming({ durationInFrames: FADE_FRAMES })}
        />
      );
    }
  });

  return (
    <AbsoluteFill style={{ backgroundColor: theme.bg }}>
      <TransitionSeries>{children}</TransitionSeries>
    </AbsoluteFill>
  );
};
