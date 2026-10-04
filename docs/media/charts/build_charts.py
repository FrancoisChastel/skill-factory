"""Render the README charts as light and dark SVGs (stdlib only).

    python docs/media/charts/build_charts.py

Every number below is copied from a measured run documented in docs/classifier.md
or the README results table. Change a number there first, then here, then rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif'
WIDTH = 880
PAD = 40


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    border: str
    tile: str
    ink: str
    ink2: str
    grid: str
    axis: str
    accent: str
    context: str


# Accent and context grays pass the dataviz palette validator on both surfaces
# (CVD dE 15.9, normal-vision dE >= 17, contrast >= 3:1); ink follows GitHub's own.
LIGHT = Theme("light", "#ffffff", "#d1d9e0", "#f6f8fa", "#1f2328", "#59636e",
              "#eef1f4", "#d1d9e0", "#2a78d6", "#898781")
DARK = Theme("dark", "#0d1117", "#3d444d", "#151b23", "#f0f6fc", "#9198a1",
             "#1f242c", "#3d444d", "#3987e5", "#898781")


@dataclass(frozen=True)
class Point:
    label: str
    detail: str
    fpr: float      # validation false-flag rate, percent
    recall: float   # validation recall, percent


# jev in skill-scanner, rescored from 70k cached answers (docs/classifier.md, case study).
JEV_SEED = Point("Released judge", "29.0% at 1.28%", 1.28, 29.0)
JEV_HAND = Point("Shipped by hand", "79.6% at 0.64%", 0.64, 79.6)
JEV_UNION = Point("Greedy union: refused", "87.1%, but 1.50% flagged", 1.50, 87.1)
JEV_BEST = Point("skill-factory optimize", "83.9% at 0.43%", 0.43, 83.9)
JEV_BUDGET = 0.5

# Skill mode, local gemma-4-31b-it-mlx (README results table).
SKILLS = [
    ("invoice-extractor", "JSON extraction", 0.559, 1.000),
    ("ticket-classifier", "classification", 0.151, 1.000),
]


def text(x: float, y: float, s: str, size: int, fill: str, *, weight: int = 400,
         anchor: str = "start", cls: str = "", spacing: float = 0) -> str:
    extra = f' class="{cls}"' if cls else ""
    if spacing:
        extra += f' letter-spacing="{spacing}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')


def document(height: int, theme: Theme, title: str, desc: str, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="t d" '
        f'font-family=\'{FONT}\'>',
        f'<title id="t">{escape(title)}</title>',
        f'<desc id="d">{escape(desc)}</desc>',
        "<style>.num{font-variant-numeric:tabular-nums}</style>",
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="16" '
        f'fill="{theme.bg}" stroke="{theme.border}"/>',
        *body,
        "</svg>",
    ])


def header(theme: Theme, eyebrow: str, title: str, subtitle: str) -> list[str]:
    return [
        text(PAD, 50, eyebrow.upper(), 11, theme.ink2, weight=600, spacing=1.2),
        text(PAD, 84, title, 24, theme.ink, weight=650),
        text(PAD, 110, subtitle, 14, theme.ink2),
    ]


def tile(theme: Theme, x: float, y: float, w: float, value: str, label: str,
         detail: str) -> list[str]:
    return [
        f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="98" rx="12" fill="{theme.tile}"/>',
        f'<rect x="{x + 20:.1f}" y="{y + 20}" width="16" height="3" rx="1.5" fill="{theme.accent}"/>',
        text(x + 20, y + 56, value, 30, theme.ink, weight=650),
        text(x + 20, y + 76, label, 13, theme.ink),
        text(x + 20, y + 91, detail, 12, theme.ink2, cls="num"),
    ]


def tiles(theme: Theme, y: float, items: list[tuple[str, str, str]]) -> list[str]:
    gap = 14
    w = (WIDTH - 2 * PAD - gap * (len(items) - 1)) / len(items)
    out: list[str] = []
    for i, (value, label, detail) in enumerate(items):
        out += tile(theme, PAD + i * (w + gap), y, w, value, label, detail)
    return out


def dot(x: float, y: float, r: float, fill: str, ring: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{ring}" stroke-width="2"/>'


def two_line_label(theme: Theme, x: float, y: float, p: Point, anchor: str,
                   strong: bool = False) -> list[str]:
    return [
        text(x, y, p.label, 13, theme.ink, weight=650 if strong else 600, anchor=anchor),
        text(x, y + 17, p.detail, 12, theme.ink2, anchor=anchor, cls="num"),
    ]


def jev_plot(theme: Theme, left: float, top: float, right: float, bottom: float) -> list[str]:
    def sx(fpr: float) -> float:
        return left + fpr / 1.6 * (right - left)

    def sy(recall: float) -> float:
        return bottom - recall / 100 * (bottom - top)

    out = [f'<rect x="{left}" y="{top}" width="{sx(JEV_BUDGET) - left:.1f}" '
           f'height="{bottom - top}" fill="{theme.accent}" fill-opacity="0.07"/>']
    for r in (25, 50, 75, 100):
        out.append(f'<line x1="{left}" x2="{right}" y1="{sy(r):.1f}" y2="{sy(r):.1f}" '
                   f'stroke="{theme.grid}"/>')
        out.append(text(left - 10, sy(r) + 4, f"{r}%", 11, theme.ink2, anchor="end", cls="num"))
    out.append(f'<line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="{theme.axis}"/>')
    for f in (0, 0.4, 0.8, 1.2, 1.6):
        out.append(text(sx(f), bottom + 20, f"{f:g}%", 11, theme.ink2, anchor="middle", cls="num"))
    out.append(f'<line x1="{sx(JEV_BUDGET):.1f}" x2="{sx(JEV_BUDGET):.1f}" y1="{top}" '
               f'y2="{bottom}" stroke="{theme.accent}" stroke-opacity="0.45"/>')
    out.append(text(sx(JEV_BUDGET) + 8, top + 16, "0.5% false-flag target", 11, theme.ink2))
    out.append(text(left + 8, top + 16, "↖ better", 11, theme.ink2, weight=600))
    out.append(text(left, top - 14, "Malicious skills caught", 12, theme.ink2, weight=600))
    out.append(text(right, bottom + 46, "Benign skills falsely flagged →", 12,
                    theme.ink2, weight=600, anchor="end"))

    seed, best = (sx(JEV_SEED.fpr), sy(JEV_SEED.recall)), (sx(JEV_BEST.fpr), sy(JEV_BEST.recall))
    ctrl = (seed[0] - 230, seed[1] + 6)  # never below the seed: recall did not dip
    out.append(
        f'<defs><marker id="head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" '
        f'stroke="{theme.accent}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round"/></marker></defs>')
    out.append(
        f'<path d="M{seed[0] - 10:.1f},{seed[1] + 4:.1f} Q{ctrl[0]:.1f},{ctrl[1]:.1f} '
        f'{best[0] + 9:.1f},{best[1] + 13:.1f}" fill="none" stroke="{theme.accent}" '
        f'stroke-width="2" stroke-linecap="round" marker-end="url(#head)"/>')
    out.append(text(ctrl[0] + 105, bottom - 38, "optimize: 3 s of selection, no new calls", 12,
                    theme.ink2, anchor="middle"))

    hand, union = (sx(JEV_HAND.fpr), sy(JEV_HAND.recall)), (sx(JEV_UNION.fpr), sy(JEV_UNION.recall))
    out.append(dot(*hand, 5, theme.context, theme.bg))
    out += two_line_label(theme, hand[0] + 12, hand[1] + 22, JEV_HAND, "start")
    out.append(f'<circle cx="{union[0]:.1f}" cy="{union[1]:.1f}" r="5" fill="{theme.bg}" '
               f'stroke="{theme.context}" stroke-width="2"/>')
    out += two_line_label(theme, union[0] - 12, union[1] - 3, JEV_UNION, "end")
    out.append(dot(*seed, 5, theme.context, theme.bg))
    out += two_line_label(theme, seed[0] + 12, seed[1] - 3, JEV_SEED, "start")
    out.append(f'<circle cx="{best[0]:.1f}" cy="{best[1]:.1f}" r="13" fill="{theme.accent}" '
               f'fill-opacity="0.16"/>')
    out.append(dot(*best, 6.5, theme.accent, theme.bg))
    out += two_line_label(theme, best[0] - 18, best[1] - 3, JEV_BEST, "end", strong=True)
    return out


def jev_chart(theme: Theme) -> str:
    body = header(theme, "Case study · jev in skill-scanner",
                  "Catching malicious agent skills: 29% → 84%",
                  "Validation split of 1,666 real skills, rescored offline from 70,000 "
                  "cached jev answers.")
    body += tiles(theme, 136, [
        ("2.9×", "more malicious skills caught", "29.0% → 83.9% recall"),
        ("3×", "fewer benign skills flagged", "1.28% → 0.43% false flags"),
        ("3 s", "to find it, with no new calls", "selection over the answer cache"),
    ])
    body += jev_plot(theme, left=96, top=290, right=WIDTH - PAD - 8, bottom=520)
    desc = ("Recall against false-flag rate on validation. The released judge catches 29.0% "
            "at 1.28% false flags; the hand-shipped probe set 79.6% at 0.64%; a greedy union "
            "87.1% at 1.50%, refused by the gate; skill-factory optimize 83.9% at 0.43%.")
    return document(590, theme, "jev: 29% to 84% of malicious skills caught", desc, body)


def skill_chart(theme: Theme) -> str:
    left, right = 250, WIDTH - PAD - 96
    body = header(theme, "Skill mode · local gemma-4-31b, zero API cost",
                  "One validation-gated edit, two skills fixed",
                  "Validation score of the seed SKILL.md and of the optimized one.")
    body += [dot(right - 150, 50 - 4, 5, theme.context, theme.bg),
             text(right - 140, 50, "seed", 12, theme.ink2),
             dot(right - 88, 50 - 4, 5, theme.accent, theme.bg),
             text(right - 78, 50, "optimized", 12, theme.ink2)]

    def sx(v: float) -> float:
        return left + v * (right - left)

    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="146" y2="264" stroke="{theme.grid}"/>')
        body.append(text(sx(v), 284, f"{v:g}", 11, theme.ink2, anchor="middle", cls="num"))
    for i, (name, task, seed, best) in enumerate(SKILLS):
        y = 182 + i * 54
        body += [
            text(PAD, y - 2, name, 14, theme.ink, weight=600),
            text(PAD, y + 15, task, 12, theme.ink2),
            f'<line x1="{sx(seed):.1f}" x2="{sx(best):.1f}" y1="{y}" y2="{y}" '
            f'stroke="{theme.accent}" stroke-opacity="0.35" stroke-width="4" stroke-linecap="round"/>',
            dot(sx(seed), y, 6, theme.context, theme.bg),
            dot(sx(best), y, 6.5, theme.accent, theme.bg),
            text(sx(seed), y - 14, f"{seed:.3f}", 12, theme.ink2, anchor="middle", cls="num"),
            text(sx(best), y - 14, f"{best:.3f}", 12, theme.ink, weight=600, anchor="middle", cls="num"),
            text(WIDTH - PAD, y + 5, f"+{round((best - seed) * 100)} pts", 15, theme.ink,
                 weight=650, anchor="end", cls="num"),
        ]
    desc = ("Validation score before and after one reflective edit: invoice-extractor 0.559 "
            "to 1.000, ticket-classifier 0.151 to 1.000.")
    return document(306, theme, "Skill mode results", desc, body)


def pr_strip(theme: Theme) -> str:
    body = header(theme, "Live · examples/pr-mismatch on jev-1.13.0",
                  "Catching pull requests that hide what they change",
                  "One question selected from the pool; sealed test split opened once, "
                  "after freeze.")
    body += tiles(theme, 136, [
        ("3 / 3", "hidden changes caught", "sealed test split"),
        ("0 / 9", "honest PRs flagged", "sealed test split"),
        ("$0.0025", "System One spend", "32 requests, 60,700 tokens"),
        ("3 s", "end-to-end optimize", "0% → 80% validation recall"),
    ])
    desc = ("pr-mismatch live run: 3 of 3 hidden changes caught and 0 of 9 honest PRs flagged "
            "on the sealed test split, for $0.0025 in 3 seconds.")
    return document(266, theme, "pr-mismatch live results", desc, body)


CHARTS: dict[str, Callable[[Theme], str]] = {
    "jev-results": jev_chart,
    "skill-results": skill_chart,
    "pr-mismatch": pr_strip,
}


def main() -> None:
    for name, render in CHARTS.items():
        for theme in (LIGHT, DARK):
            path = OUT / f"{name}-{theme.name}.svg"
            path.write_text(render(theme) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(OUT.parents[2])}")


if __name__ == "__main__":
    main()
