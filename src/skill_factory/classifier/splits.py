"""Honest splits: train, validation, test, and whole groups held out.

Tuning reads training failures, keeps an edit only if it beats validation, and
is reported on examples neither step saw: a test slice of the same sources, and
sources held out whole, which shows how tuning carries to families it never read.

Three rules keep the numbers honest:

* **Group-aware.** The split is a hash of the example's group, so near-duplicates
  (one template, one repository) never straddle train and test.
* **Read before the split.** Examples someone read while choosing the split count
  as training data, so nothing in a held-out score was seen beforehand.
* **Sealed.** Test and held-out results refuse to print until the configuration
  is frozen (``lab freeze``), and every opening is logged with the hash it was
  opened under. A test set looked at under one configuration and reported under
  another says so in the report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from skill_factory.classifier.dataset import Example

SPLITS = ("train", "val", "test", "held-out")
SEALED = frozenset({"test", "held-out"})
LABELS = {"train": "training", "val": "validation", "test": "test", "held-out": "held-out"}


@dataclass(frozen=True)
class SplitPlan:
    """Where each example falls. Fractions apply to examples not held out; test gets the rest."""

    train: float = 0.5
    val: float = 0.25
    held_out_sources: frozenset[str] = frozenset()
    held_out_groups: frozenset[str] = frozenset()
    read_before_split: frozenset[str] = frozenset()
    salt: str = ""

    def __post_init__(self) -> None:
        if not (0 < self.train < 1 and 0 < self.val < 1 and self.train + self.val <= 1):
            raise ValueError("split fractions need 0 < train, 0 < val, train + val <= 1")

    def split_of(self, ex: Example) -> str:
        if ex.id in self.read_before_split or ex.group in self.read_before_split:
            return "train"
        if ex.split is not None:
            if ex.split not in SPLITS:
                raise ValueError(f"example {ex.id!r} has unknown split {ex.split!r}")
            return ex.split
        if ex.source in self.held_out_sources or ex.group in self.held_out_groups:
            return "held-out"
        digest = hashlib.sha256(f"{self.salt}{ex.group}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        if u < self.train:
            return "train"
        return "val" if u < self.train + self.val else "test"

    def assign(self, examples: Iterable[Example]) -> dict[str, list[Example]]:
        out: dict[str, list[Example]] = {s: [] for s in SPLITS}
        for ex in examples:
            out[self.split_of(ex)].append(ex)
        _check_groups(out)
        return out

    @classmethod
    def from_dict(cls, data: Mapping | None) -> "SplitPlan":
        data = data or {}
        fractions = data.get("fractions") or {}
        return cls(
            train=float(fractions.get("train", 0.5)),
            val=float(fractions.get("val", 0.25)),
            held_out_sources=frozenset(map(str, data.get("held_out_sources") or [])),
            held_out_groups=frozenset(map(str, data.get("held_out_groups") or [])),
            read_before_split=frozenset(map(str, data.get("read_before_split") or [])),
            salt=str(data.get("salt", "")),
        )


def _check_groups(by_split: Mapping[str, Sequence[Example]]) -> None:
    """A group in two splits leaks; it can only happen through explicit splits or read-before entries."""
    where: dict[str, set[str]] = {}
    for split, exs in by_split.items():
        for ex in exs:
            where.setdefault(ex.group, set()).add(split)
    leaking = sorted(g for g, s in where.items() if len(s) > 1)
    if leaking:
        shown = ", ".join(leaking[:5]) + (" ..." if len(leaking) > 5 else "")
        raise ValueError(
            f"{len(leaking)} group(s) span several splits: {shown}. Name the whole group in "
            "read_before_split, or give its examples the same explicit split."
        )


class SealedError(RuntimeError):
    """Raised when sealed splits are opened before the configuration is frozen."""


@dataclass
class Seal:
    """The freeze record and the log of every opening of the sealed splits (``seal.json``)."""

    path: Path
    frozen: dict | None = None
    views: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, workspace: str | Path) -> "Seal":
        path = Path(workspace) / "seal.json"
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, data.get("frozen"), list(data.get("views") or []))

    def freeze(self, content_hash: str, *, baseline: str | None = None, note: str = "", force: bool = False) -> None:
        """Seal ``content_hash`` as the configuration to report, and ``baseline`` as the one it is compared with."""
        changed = self.frozen and (self.frozen["hash"], self.frozen.get("baseline")) != (content_hash, baseline)
        if changed and self.views and not force:
            raise SealedError(
                "the sealed splits were already opened under another configuration; refreezing "
                "means the test numbers are no longer unseen. Pass --force to refreeze and have "
                "the report say so."
            )
        self.frozen = {"hash": content_hash, "baseline": baseline, "at": _now(), "note": note}
        self._save()

    def open(self, content_hash: str, splits: Iterable[str], *, baseline: str | None = None) -> int:
        """Record an opening of sealed splits; returns how many times they have been opened.

        ``baseline`` is the hash of a second configuration read in the same opening (the
        "before" column); it must be the one recorded at freeze, so a report cannot be used
        to score any number of other candidates on the sealed splits.
        """
        wanted = sorted(set(splits) & SEALED)
        if not wanted:
            return len(self.views)
        if not self.frozen:
            raise SealedError(
                f"{', '.join(wanted)} sealed: freeze the configuration first (skill-factory lab freeze)"
            )
        if self.frozen["hash"] != content_hash:
            raise SealedError(
                f"{', '.join(wanted)} sealed: this probe set ({content_hash[:12]}) is not the frozen "
                f"one ({self.frozen['hash'][:12]}); freeze it to report on unseen data"
            )
        if baseline is not None and baseline not in (self.frozen["hash"], self.frozen.get("baseline")):
            raise SealedError(
                f"{', '.join(wanted)} sealed: the baseline ({baseline[:12]}) is not the one recorded at "
                "freeze; refreeze with --baseline to compare against it"
            )
        self.views.append({"hash": content_hash, "baseline": baseline, "at": _now(), "splits": wanted})
        self._save()
        return len(self.views)

    @property
    def contaminated(self) -> bool:
        """True when sealed splits were opened under a configuration other than the frozen one."""
        if not self.frozen:
            return False
        frozen = (self.frozen["hash"], self.frozen.get("baseline"))
        return any(
            v["hash"] != frozen[0] or (v.get("baseline") is not None and v["baseline"] not in frozen)
            for v in self.views
        )

    def summary(self) -> str:
        if not self.frozen:
            return "not frozen: test and held-out splits are sealed"
        lines = [f"frozen {self.frozen['at']} as {self.frozen['hash'][:12]}", f"sealed splits opened {len(self.views)} time(s)"]
        if self.contaminated:
            lines.append("WARNING: sealed splits were opened under an earlier configuration")
        return "; ".join(lines)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"frozen": self.frozen, "views": self.views}
        self.path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
