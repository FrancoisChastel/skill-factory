"""Reports: the before-and-after table per split, so overfitting is visible.

The jev work saw it in this table: two new static rules caught 11 more training
skills and 2 more validation skills, then one more test skill and no held-out
one, while the tuned questions held on every split. Every run prints the same
table, and the figure next to it.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from skill_factory.classifier.evaluation import QuestionStat, SplitReport
from skill_factory.classifier.metrics import pct
from skill_factory.classifier.optimizer import ClassifierResult
from skill_factory.classifier.probeset import describe_score
from skill_factory.classifier.select import Candidate
from skill_factory.classifier.splits import LABELS, SEALED, SPLITS


def before_after_table(
    before: Mapping[str, SplitReport],
    after: Mapping[str, SplitReport],
    *,
    sealed: Iterable[str] = (),
) -> str:
    """One row per split: detection and false flags per verdict level, before -> after."""
    sample = next(iter(after.values()), None) or next(iter(before.values()), None)
    levels = sample.levels[1:] if sample else ("flag",)
    head = ["Split"]
    for lvl in levels:
        head += [f"Positives {lvl}", f"Negatives {lvl}"]
    head.append("AUC")
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    sealed = set(sealed)
    for split in SPLITS:
        if split in sealed:
            lines.append(f"| {LABELS[split]} | sealed: freeze the probe set (`lab freeze`) to report it |"
                         + " |" * (len(head) - 2))
            continue
        b, a = before.get(split), after.get(split)
        if a is None or (a.positives + a.negatives) == 0:
            continue
        cells = [f"{LABELS[split]} ({a.positives} / {a.negatives})"]
        for lvl in levels:
            cells.append(_arrow(b.rates(lvl).recall if b else None, a.rates(lvl).recall) if a.positives else "n/a (no positives)")
            cells.append(_arrow(b.rates(lvl).fpr if b else None, a.rates(lvl).fpr) if a.negatives else "n/a (no negatives)")
        cells.append(f"{b.auc:.3f} -> **{a.auc:.3f}**" if b else f"{a.auc:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def coverage_table(statuses: Mapping[str, Mapping[str, int]]) -> str:
    """How many examples of each split the model actually read."""
    lines = ["| Split | Judged | Over budget | Skipped | No state |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        counts = statuses.get(split)
        if not counts:
            continue
        lines.append(
            f"| {LABELS[split]} | {counts.get('judged', 0)} | {counts.get('over budget', 0)} | "
            f"{counts.get('skipped', 0)} | {counts.get('no state', 0)} |"
        )
    return "\n".join(lines)


def questions_table(stats: Sequence[QuestionStat], in_set: Iterable[str], fpr_budget: float, limit: int = 40) -> str:
    in_set = set(in_set)
    lines = [
        f"| Question | In set | Train AUC | Val AUC | Val recall at {pct(fpr_budget)} (t from train) |",
        "|---|:---:|---:|---:|---:|",
    ]
    for s in list(stats)[:limit]:
        lines.append(
            f"| `{s.qid}` | {'yes' if s.qid in in_set else ''} | {s.auc.get('train', 0.5):.3f} | "
            f"{s.auc.get('val', 0.5):.3f} | {s.recall.get('val', 0.0):.0%} (t={s.threshold:.3f}) |"
        )
    if len(stats) > limit:
        lines.append(f"| ... {len(stats) - limit} more | | | | |")
    return "\n".join(lines)


def pareto_table(front: Sequence[Candidate]) -> str:
    lines = ["| Candidate | Questions | Train recall | Train FPR | Threshold |", "|---|---:|---:|---:|---:|"]
    for c in front:
        lines.append(f"| `{c.name}` | {len(c.questions)} | {pct(c.train.recall)} | {c.train.fpr:.2%} | {c.threshold:.4f} |")
    return "\n".join(lines)


def classifier_report(
    result: ClassifierResult,
    *,
    seal_summary: str = "",
    cost_per_pass: float | None = None,
    statuses: Mapping[str, Mapping[str, int]] | None = None,
    sealed_after: Mapping[str, SplitReport] | None = None,
    sealed_before: Mapping[str, SplitReport] | None = None,
    sealed_splits: Iterable[str] = SEALED,
) -> str:
    best, obj = result.best, result.objective
    after = {**result.after, **(sealed_after or {})}
    before = {**result.before, **(sealed_before or {})}
    sealed = [s for s in SPLITS if s in set(sealed_splits) and s not in after]
    level = obj.level or "detection"
    lines = [
        f"# Classifier report: {best.name}",
        "",
        f"- **Objective:** {obj.split} recall ({level}) at {pct(obj.fpr_budget)} false positives, "
        "thresholds fitted on train",
        f"- **Constraints:** {'; '.join(c.describe() for c in obj.constraints) or 'none'}",
        f"- **Policy:** `{result.policy_name}`",
        f"- **Before -> after ({obj.split}):** {pct(result.baseline_score)} -> **{pct(result.best_score)}**",
        f"- **Probe set:** `{best.content_hash()[:12]}`, {len(best.bank)} questions, model `{best.model}`"
        + (f" (answered as `{best.pinned_version}`)" if best.pinned_version else "")
        + f", state budget {best.state_budget:,} characters",
        f"- **Score:** `{describe_score(best.score)}` >= {best.threshold:.4f}" if best.threshold is not None
        else f"- **Score:** `{describe_score(best.score)}`",
        f"- **Spent:** ${result.usd:.4f} ({result.input_tokens:,} input tokens)"
        + (f"; one pass of this probe set over train+val costs about ${cost_per_pass:.4f}" if cost_per_pass else ""),
    ]
    if seal_summary:
        lines.append(f"- **Seal:** {seal_summary}")
    if result.failed_requests:
        lines.append(
            f"- **WARNING: {result.failed_requests} requests failed.** Their answers are missing and the "
            "examples count as unanswered (misses), so recall below is a lower bound. Rerun to fill the "
            f"cache. First error: {_md(result.errors[0]) if result.errors else 'unknown'}"
        )
    lines += [
        "",
        "## Before and after, per split",
        "",
        "![Before and after, per split](splits.svg)",
        "",
        before_after_table(before, after, sealed=sealed),
        "",
        "## Rounds",
        "",
        "| Round | Change | Why | Candidate | Train | Val | Decision |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for h in result.history:
        tr, va = h.reports["train"].rates(obj.level), h.reports["val"].rates(obj.level)
        decision = "kept" if h.accepted else "refused: " + "; ".join(h.reasons)
        lines.append(
            f"| {h.round} | {_md(h.change)} | {_md(h.rationale)[:200]} | `{_md(h.candidate)}` | "
            f"{pct(tr.recall)} at {tr.fpr:.2%} | {pct(va.recall)} at {va.fpr:.2%} | {_md(decision)} |"
        )
    lines += ["", "## Questions", "", questions_table(result.stats, best.bank.ids, obj.fpr_budget)]
    if result.front:
        lines += ["", "## Pareto front (train: recall, false positives, questions)", "", pareto_table(result.front)]
    if statuses:
        lines += ["", "## Coverage", "", coverage_table(statuses)]
    return "\n".join(lines) + "\n"


def splits_svg(before: Mapping[str, SplitReport], after: Mapping[str, SplitReport], level: str | None = None) -> str:
    """Two panels, recall and false-positive rate, one row per split, before (grey) and after (blue)."""
    rows = [s for s in SPLITS if s in after and (after[s].positives + after[s].negatives) > 0]
    if not rows:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'
    fprs = [r.rates(level).fpr for r in [*after.values(), *before.values()]]
    fpr_max = max(0.01, max(fprs) * 1.25)
    w, label_w, panel_w, gap, row_h, top = 760, 150, 260, 50, 46, 56
    height = top + row_h * len(rows) + 34
    grey, blue, ink, rule = "#9aa3ad", "#2f6fdf", "#4a5560", "#c9d0d6"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{height}" viewBox="0 0 {w} {height}" '
        'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="12">',
        f'<text x="{label_w}" y="22" fill="{ink}" font-weight="600">Positives detected</text>',
        f'<text x="{label_w + panel_w + gap}" y="22" fill="{ink}" font-weight="600">Negatives flagged</text>',
        f'<rect x="{w - 170}" y="12" width="10" height="10" fill="{grey}"/>'
        f'<text x="{w - 155}" y="21" fill="{ink}">before</text>',
        f'<rect x="{w - 100}" y="12" width="10" height="10" fill="{blue}"/>'
        f'<text x="{w - 85}" y="21" fill="{ink}">after</text>',
    ]
    for panel, (x0, scale, fmt) in enumerate(
        [(label_w, 1.0, lambda v: f"{v:.0%}"), (label_w + panel_w + gap, fpr_max, lambda v: f"{v:.1%}")]
    ):
        y_axis = top + row_h * len(rows)
        out.append(f'<line x1="{x0}" y1="{top - 8}" x2="{x0}" y2="{y_axis}" stroke="{rule}"/>')
        for tick in (0.0, 0.5, 1.0):
            tx = x0 + panel_w * tick
            out.append(f'<text x="{tx:.1f}" y="{y_axis + 16}" fill="{ink}" text-anchor="middle">{fmt(scale * tick)}</text>')
        for i, split in enumerate(rows):
            y = top + i * row_h
            if panel == 0:
                out.append(f'<text x="{label_w - 10}" y="{y + 20}" fill="{ink}" text-anchor="end">{LABELS[split]}</text>')
            for j, (src, color) in enumerate(((before, grey), (after, blue))):
                rep = src.get(split)
                if rep is None:
                    continue
                v = rep.rates(level).recall if panel == 0 else rep.rates(level).fpr
                bw = max(1.0, panel_w * min(1.0, v / scale))
                by = y + 4 + j * 15
                out.append(f'<rect x="{x0}" y="{by}" width="{bw:.1f}" height="12" rx="2" fill="{color}"/>')
                label = f"{v:.1%}" if panel == 0 else f"{v:.2%}"
                out.append(f'<text x="{x0 + bw + 5:.1f}" y="{by + 10}" fill="{ink}" font-size="11">{label}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _arrow(before: float | None, after: float) -> str:
    return f"**{pct(after)}**" if before is None else f"{pct(before)} -> **{pct(after)}**"


def _md(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")
