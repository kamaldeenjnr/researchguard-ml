"""Builds the README figures from the committed benchmark results.

Run:  python docs/img/make_figures.py
Writes optimism-{light,dark}.svg and split-{light,dark}.svg next to this file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RES = HERE.parents[1] / "benchmarks" / "results"

THEMES = {
    "light": dict(bg="#ffffff", border="#d0d7de", text="#1f2328", muted="#59636e",
                  grid="#e6e9ec", zero="#8c959f", tile="#eef1f4", tile_text="#1f2328",
                  hot="#C4412B", hot_text="#ffffff", band="#f6f8fa",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(bg="#0d1117", border="#30363d", text="#e6edf3", muted="#9198a1",
                 grid="#21262d", zero="#6e7681", tile="#21262d", tile_text="#e6edf3",
                 hot="#C4412B", hot_text="#ffffff", band="#161b22",
                 series=["#3987e5", "#d95926", "#199e70"]),
}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"
MODELS = [("logistic_regression", "Logistic regression"),
          ("random_forest", "Random forest"),
          ("gradient_boosting", "Gradient boosting")]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_rows():
    syn = pd.read_csv(RES / "benchmark_results.csv")
    real = pd.read_csv(RES / "real_diabetes130.csv")
    rows = [
        ("grouped_leaky", "Same patients, recognisable", "synthetic"),
        ("grouped_control", "Same patients, no hazard (control)", "synthetic"),
        ("temporal_drift", "Pattern changes over time", "synthetic"),
        ("temporal_control", "Stable over time (control)", "synthetic"),
    ]
    out = []
    for key, label, kind in rows:
        d = syn[syn.scenario == key]
        out.append((label, kind, {m: d[d.model == m].optimism for m, _ in MODELS}))
    out.append(("99,343 hospital visits, by patient", "real",
                {m: real[real.model == m].optimism for m, _ in MODELS}))
    return out


def optimism_svg(t: dict) -> str:
    rows = load_rows()
    W, left, right, top, row_h = 920, 330, 870, 150, 66
    x0, x1 = -0.05, 0.30
    X = lambda v: left + (v - x0) / (x1 - x0) * (right - left)
    H = top + row_h * len(rows) + 22 + 56
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-label="Optimism of random cross-validation by scenario and model. Large on '
         f'synthetic data with recognisable patients (up to 0.18 AUC) and changing patterns (up to 0.13); '
         f'near zero on controls and on real hospital data.">',
         f'<style>text{{font-family:{FONT}}}</style>',
         f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="10" fill="{t["bg"]}" stroke="{t["border"]}"/>',
         f'<text x="32" y="44" font-size="21" font-weight="700" fill="{t["text"]}">How much does ordinary random CV flatter the score?</text>',
         f'<text x="32" y="70" font-size="14.5" fill="{t["muted"]}">Optimism = random-CV AUC − design-aware AUC. Dot = mean, line = range across seeds. 0 means no inflation.</text>']
    # legend
    lx = 32
    for i, (_, name) in enumerate(MODELS):
        s.append(f'<circle cx="{lx+6}" cy="100" r="5" fill="{t["series"][i]}"/>'
                 f'<text x="{lx+18}" y="105" font-size="14" fill="{t["text"]}">{name}</text>')
        lx += 30 + len(name) * 7.6
    # real-data band
    ry = top + row_h * 4 + 22
    s.append(f'<rect x="16" y="{ry-6}" width="{W-32}" height="{row_h+8}" rx="8" fill="{t["band"]}"/>')
    # grid + ticks
    gy0, gy1 = top - 8, ry + row_h + 2
    for v in (0.0, 0.1, 0.2, 0.3):
        c = t["zero"] if v == 0 else t["grid"]
        s.append(f'<line x1="{X(v):.1f}" y1="{gy0}" x2="{X(v):.1f}" y2="{gy1}" stroke="{c}" stroke-width="1"/>')
        lab = "0" if v == 0 else f"+{v:.1f}"
        s.append(f'<text x="{X(v):.1f}" y="{gy1+20}" font-size="13" text-anchor="middle" fill="{t["muted"]}">{lab}</text>')
    s.append(f'<text x="{right}" y="{gy1+40}" font-size="12.5" text-anchor="end" fill="{t["muted"]}">AUC points of inflation</text>')
    s.append(f'<text x="32" y="{top-14}" font-size="11.5" letter-spacing="1.2" fill="{t["muted"]}">SYNTHETIC (BUILT TO CONTAIN OR LACK A HAZARD) · 10 SEEDS</text>')
    s.append(f'<text x="32" y="{ry+8}" font-size="11.5" letter-spacing="1.2" fill="{t["muted"]}">REAL DATA · DIABETES 130-US HOSPITALS · 5 SEEDS</text>')
    for r, (label, kind, data) in enumerate(rows):
        cy = (top + row_h * r + row_h / 2 + 4) if kind == "synthetic" else (ry + row_h / 2 + 8)
        s.append(f'<text x="32" y="{cy+5}" font-size="15" fill="{t["text"]}">{esc(label)}</text>')
        for i, (m, _) in enumerate(MODELS):
            v = data[m]
            y = cy + (i - 1) * 14
            col = t["series"][i]
            s.append(f'<line x1="{X(v.min()):.1f}" y1="{y}" x2="{X(v.max()):.1f}" y2="{y}" stroke="{col}" '
                     f'stroke-width="2" stroke-linecap="round"/>')
            s.append(f'<circle cx="{X(v.mean()):.1f}" cy="{y}" r="5" fill="{col}" stroke="{t["bg"]}" stroke-width="2"/>')
    # selective direct labels
    leaky_rf = rows[0][2]["random_forest"].mean()
    s.append(f'<text x="{X(rows[0][2]["random_forest"].max())+10:.1f}" y="{top + row_h/2 + 9}" font-size="13.5" '
             f'font-weight="600" fill="{t["text"]}">+{leaky_rf:.2f}</text>')
    real_max = max(rows[4][2][m].max() for m, _ in MODELS)
    s.append(f'<text x="{X(real_max)+12:.1f}" y="{ry + row_h/2 + 13}" font-size="13.5" font-weight="600" '
             f'fill="{t["text"]}">≈ 0 — structure found, no inflation</text>')
    s.append("</svg>")
    return "\n".join(s)


def split_svg(t: dict) -> str:
    W, H = 920, 360
    tw, th, gap = 38, 30, 5
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'role="img" aria-label="Random split versus patient-grouped split. In the random split, '
         f'visits from five of eight patients land in both training and test data. In the grouped '
         f'split, each patient is entirely in training or entirely in test.">',
         f'<style>text{{font-family:{FONT}}}</style>',
         f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="10" fill="{t["bg"]}" stroke="{t["border"]}"/>',
         f'<text x="32" y="44" font-size="21" font-weight="700" fill="{t["text"]}">The same patient on both sides of the split</text>',
         f'<text x="32" y="70" font-size="14.5" fill="{t["muted"]}">8 patients × 3 visits. Each tile is one visit, labelled with its patient.</text>']
    visits = [(p, v) for p in range(1, 9) for v in range(1, 4)]
    random_test = {(1, 2), (3, 1), (3, 3), (5, 2), (6, 1), (8, 3)}
    grouped_test = {(p, v) for p, v in visits if p in (3, 7)}
    panels = [("Ordinary random split", random_test, 32),
              ("ResearchGuard: patient-grouped split", grouped_test, 478)]
    for title, test, px in panels:
        train = [x for x in visits if x not in test]
        tst = [x for x in visits if x in test]
        shared = {p for p, _ in tst} & {p for p, _ in train}
        s.append(f'<text x="{px}" y="112" font-size="16" font-weight="700" fill="{t["text"]}">{esc(title)}</text>')
        for box, items, bx, cols, name in ((0, train, px, 6, "Train"), (1, tst, px + 6 * (tw + gap) + gap + 18, 2, "Test")):
            bw = cols * (tw + gap) + gap
            s.append(f'<text x="{bx}" y="140" font-size="12" letter-spacing="1.2" fill="{t["muted"]}">{name.upper()}</text>')
            s.append(f'<rect x="{bx-0.5}" y="148.5" width="{bw}" height="{3*(th+gap)+gap}" rx="7" fill="none" stroke="{t["border"]}"/>')
            for k, (p, v) in enumerate(items):
                x = bx + gap + (k % cols) * (tw + gap)
                y = 148 + gap + (k // cols) * (th + gap)
                hot = p in shared
                s.append(f'<rect x="{x}" y="{y}" width="{tw}" height="{th}" rx="4" fill="{t["hot"] if hot else t["tile"]}"/>'
                         f'<text x="{x+tw/2}" y="{y+20}" font-size="12.5" text-anchor="middle" font-weight="{600 if hot else 400}" '
                         f'fill="{t["hot_text"] if hot else t["tile_text"]}">P{p}</text>')
        if shared:
            msg = [f"{len(shared)} of 8 patients appear in both train and test.",
                   "The model can recognise them, so the test is too easy."]
        else:
            msg = ["Every patient is entirely in train or in test.",
                   "The score reflects performance on new patients."]
        for j, line in enumerate(msg):
            s.append(f'<text x="{px}" y="{290 + j*22}" font-size="14.5" fill="{t["text"] if j == 0 else t["muted"]}">{line}</text>')
    s.append(f'<rect x="32" y="{H-30}" width="12" height="12" rx="3" fill="{t["hot"]}"/>'
             f'<text x="52" y="{H-20}" font-size="12.5" fill="{t["muted"]}">Patient with visits on both sides</text>')
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    for name, t in THEMES.items():
        (HERE / f"optimism-{name}.svg").write_text(optimism_svg(t), encoding="utf-8")
        (HERE / f"split-{name}.svg").write_text(split_svg(t), encoding="utf-8")
    print("written")
