import React from "react";
import { Composition } from "remotion";
import { Explainer, TOTAL_FRAMES } from "./Explainer";

const common = {
  component: Explainer,
  durationInFrames: TOTAL_FRAMES,
  fps: 30,
  width: 1920,
  height: 1080,
} as const;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition id="Explainer" {...common} defaultProps={{ narration: false }} />
      <Composition id="ExplainerNarrated" {...common} defaultProps={{ narration: true }} />
    </>
  );
};
