"""The answer cache: one line per answer, keyed by (state hash, question hash, model).

    {"s": "<state hash>", "q": "<question hash>", "m": "jev-latest", "p": 0.93, "v": "jev-1.13.0"}

``m`` is the model asked for, ``v`` the version that answered, ``r`` the
replicate (present only for re-asks, see drift.py). The format is skill-scanner's
``answers.jsonl``, so a lab cached there can be rescored here without a call.

Answers are appended as they arrive, so an interrupted run loses nothing, and a
new idea only pays for its new questions.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Mapping

Key = tuple[str, str, str, int]


class AnswerCache:
    """An append-only JSONL file of answers with an in-memory index. Thread-safe."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path is not None else None
        self._index: dict[Key, float] = {}
        self._versions: dict[Key, str] = {}
        self._lock = threading.Lock()
        self._needs_newline = False
        if self.path is not None and self.path.exists():
            self._load()

    def __len__(self) -> int:
        return len(self._index)

    def get(self, state_hash: str, question_hash: str, model: str, replicate: int = 0) -> float | None:
        return self._index.get((state_hash, question_hash, model, replicate))

    def has(self, state_hash: str, question_hash: str, model: str, replicate: int = 0) -> bool:
        return (state_hash, question_hash, model, replicate) in self._index

    def version(self, state_hash: str, question_hash: str, model: str, replicate: int = 0) -> str | None:
        return self._versions.get((state_hash, question_hash, model, replicate))

    def put_many(self, rows: Iterable[Mapping]) -> None:
        """Add answers: dicts with s, q, m, p and optional v, r."""
        lines = []
        with self._lock:
            for row in rows:
                key = (str(row["s"]), str(row["q"]), str(row["m"]), int(row.get("r", 0)))
                self._index[key] = float(row["p"])
                if row.get("v"):
                    self._versions[key] = str(row["v"])
                out = {"s": key[0], "q": key[1], "m": key[2], "p": float(row["p"])}
                if row.get("v"):
                    out["v"] = str(row["v"])
                if key[3]:
                    out["r"] = key[3]
                lines.append(json.dumps(out))
            if self.path is not None and lines:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    # Start on a fresh line after a cut-off last line, so it stays the only casualty.
                    fh.write(("\n" if self._needs_newline else "") + "\n".join(lines) + "\n")
                self._needs_newline = False

    def models(self) -> set[str]:
        return {k[2] for k in self._index}

    def _load(self) -> None:
        assert self.path is not None
        text = self.path.read_text(encoding="utf-8")
        self._needs_newline = bool(text) and not text.endswith("\n")
        lines = text.split("\n")
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                a = json.loads(line)
                key = (str(a["s"]), str(a["q"]), str(a["m"]), int(a.get("r", 0)))
                p = float(a["p"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                # A crash mid-append can only cut the last line: drop it (it will be re-asked).
                # Anything else is corruption.
                if lineno == len(lines) and not text.endswith("\n"):
                    with self.path.open("r+", encoding="utf-8") as fh:
                        fh.truncate(len(text[: text.rfind("\n") + 1].encode("utf-8")))
                    self._needs_newline = False
                    break
                raise ValueError(f"{self.path}:{lineno}: corrupt answer line ({exc})") from exc
            self._index[key] = p
            if a.get("v"):
                self._versions[key] = str(a["v"])
