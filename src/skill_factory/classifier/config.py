"""Classifier-mode run configuration (YAML, ``kind: classifier``).

    kind: classifier
    name: pr-mismatch
    dataset: dataset.jsonl                  # id, state, label, group, source
    labels: {positive: mismatch, negative: consistent}
    probeset: seed_probeset.json            # the seed artifact
    pool: candidates.json                   # optional: more questions to select from
    workspace: ../../runs/pr-mismatch       # cache, reports, seal
    splits:
      fractions: {train: 0.5, val: 0.25}    # test gets the rest
      held_out_sources: [synthetic/repo-d]
      read_before_split: []
    systemone: {provider: typesafe, model: jev-latest, max_usd: 2.0}
    objective:
      fpr_budget: 0.05
      constraints: [{split: val, max_fpr: 0.1}]
    policy: policy.py:verdict               # optional, default: flag at threshold
    optimizer: {rounds: 3, patience: 2, max_questions: 6, max_new_questions: 6, goal: "..."}
    optimizer_provider: {name: anthropic, model: claude-opus-5-5}   # omit: selection only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from skill_factory.config import ProviderConfig
from skill_factory.classifier.objective import Objective
from skill_factory.classifier.splits import SplitPlan

_KNOWN = {
    "kind", "name", "dataset", "labels", "probeset", "pool", "workspace", "splits", "systemone",
    "objective", "policy", "optimizer", "optimizer_provider", "keep_questions",
}


@dataclass(frozen=True)
class ClassifierConfig:
    name: str
    dataset_path: Path
    probeset_path: Path
    workspace: Path
    positive: str = "positive"
    negative: str = "negative"
    pool_path: Path | None = None
    splits: SplitPlan = field(default_factory=SplitPlan)
    systemone: Mapping[str, Any] = field(default_factory=dict)
    objective: Objective = field(default_factory=Objective)
    policy: str | None = None
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    proposer: ProviderConfig | None = None
    keep_questions: tuple[str, ...] | None = None
    base_dir: Path = Path(".")


def is_classifier_config(path: str | Path) -> bool:
    """True when the YAML at ``path`` declares ``kind: classifier``."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(raw, dict) and raw.get("kind") == "classifier"


def load_classifier_config(path: str | Path) -> ClassifierConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a YAML mapping")
    unknown = sorted(set(raw) - _KNOWN)
    if unknown:
        raise ValueError(f"unknown classifier config keys: {unknown}")
    for key in ("dataset", "probeset"):
        if key not in raw:
            raise ValueError(f"classifier config is missing {key!r}")
    base = p.parent
    labels = raw.get("labels") or {}
    name = str(raw.get("name") or p.parent.name)
    proposer = raw.get("optimizer_provider")
    keep = raw.get("keep_questions")
    return ClassifierConfig(
        name=name,
        dataset_path=base / str(raw["dataset"]),
        probeset_path=base / str(raw["probeset"]),
        workspace=base / str(raw.get("workspace") or f"runs/{name}"),
        positive=str(labels.get("positive", "positive")),
        negative=str(labels.get("negative", "negative")),
        pool_path=base / str(raw["pool"]) if raw.get("pool") else None,
        splits=SplitPlan.from_dict(raw.get("splits")),
        systemone=dict(raw.get("systemone") or {}),
        objective=Objective.from_dict(raw.get("objective")),
        policy=str(raw["policy"]) if raw.get("policy") else None,
        optimizer=dict(raw.get("optimizer") or {}),
        proposer=ProviderConfig.from_dict(proposer, default_model="claude-opus-5-5") if proposer else None,
        keep_questions=tuple(map(str, keep)) if keep is not None else None,
        base_dir=base,
    )
