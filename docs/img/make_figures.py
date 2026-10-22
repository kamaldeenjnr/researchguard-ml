"""Builds the two README figures from the committed benchmark results.

Run:  pip install -e ".[figures]" && python docs/img/make_figures.py
Writes docs/img/fig1_splits.svg and docs/img/fig2_optimism.svg.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

HERE = Path(__file__).parent
RES = HERE.parents[1] / "benchmarks" / "results"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "svg.fonttype": "path", "figure.facecolor": "white", "axes.facecolor": "white",
})
MODELS = [("logistic_regression", "Logistic regression", "o", "#2a78d6"),
          ("random_forest", "Random forest", "s", "#eb6834"),
          ("gradient_boosting", "Gradient boosting", "^", "#1baf7a")]


def fig1_splits():
    """Fold-by-fold view of which visits are tested, for 8 patients x 3 visits."""
    patients = np.repeat(np.arange(8), 3)
    X = np.zeros((24, 1))
    designs = [("(a) Random 4-fold CV", KFold(4, shuffle=True, random_state=1).split(X)),
               ("(b) Patient-grouped 4-fold CV", GroupKFold(4).split(X, groups=patients))]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    for ax, (title, splits) in zip(axes, designs):
        shared_total = set()
        for k, (tr, te) in enumerate(splits):
            row = np.zeros(24)
            row[te] = 1
            ax.scatter(np.arange(24), np.full(24, k), marker="s", s=62,
                       c=np.where(row == 1, "#1f2328", "#d0d7de"), linewidths=0)
            shared = set(patients[te]) & set(patients[tr])
            shared_total |= shared
            for p in shared:  # mark patients split across train and test in this fold
                idx = np.flatnonzero(patients == p)
                ax.plot([idx[0] - 0.4, idx[-1] + 0.4], [k + 0.36] * 2, color="#C4412B", lw=1.6)
        for b in range(1, 8):
            ax.axvline(b * 3 - 0.5, color="white", lw=3)
        ax.set_xticks(np.arange(1, 24, 3), [f"P{i}" for i in range(1, 9)])
        ax.tick_params(axis="x", length=0)
        ax.set_yticks(range(4), [f"Fold {k+1}" for k in range(4)])
        ax.set_ylim(3.6, -0.6)
        ax.set_xlim(-0.8, 23.8)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(f"{len(shared_total)} of 8 patients split between train and test",
                      fontsize=9.5, color="#57606a")
    fig.tight_layout(w_pad=3)
    fig.savefig(HERE / "fig1_splits.svg")
    plt.close(fig)


def fig2_optimism():
    syn = pd.read_csv(RES / "benchmark_results.csv")
    real = pd.read_csv(RES / "real_diabetes130.csv")
    rows = [("Grouped, patients recognisable", syn[syn.scenario == "grouped_leaky"]),
            ("Grouped, no hazard (control)", syn[syn.scenario == "grouped_control"]),
            ("Temporal, relationship drifts", syn[syn.scenario == "temporal_drift"]),
            ("Temporal, stable (control)", syn[syn.scenario == "temporal_control"]),
            ("Diabetes 130-US, by patient", real)]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ys = [0, 1, 2, 3, 4.6]
    for y, (label, d) in zip(ys, rows):
        for j, (m, name, marker, col) in enumerate(MODELS):
            v = d[d.model == m].optimism
            yy = y + (j - 1) * 0.22
            ax.plot([v.min(), v.max()], [yy, yy], color=col, lw=1.4, solid_capstyle="butt")
            ax.plot(v.mean(), yy, marker=marker, color=col, ms=6, mec="white", mew=0.8,
                    label=name if y == 0 else None, ls="none")
    ax.axvline(0, color="#57606a", lw=0.9)
    ax.axhline(3.8, color="#d0d7de", lw=0.8)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(5.1, -0.6)
    ax.set_xlim(-0.06, 0.31)
    ax.set_xlabel("Optimism: AUC under random CV minus AUC under design-aware CV")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.text(0.305, -0.45, "Synthetic, 10 seeds", ha="right", fontsize=9, color="#57606a")
    ax.text(0.305, 4.05, "Real data, 5 seeds", ha="right", fontsize=9, color="#57606a")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.84), fontsize=9,
              handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_optimism.svg")
    plt.close(fig)


if __name__ == "__main__":
    fig1_splits()
    fig2_optimism()
    print("written")
