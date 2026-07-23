#!/usr/bin/env python3
"""Regenerate the figures for the ProspectMap prospectivity report from the COMMITTED artifacts. Two figures:

  fig-inflation.pdf - the two ways a prospectivity map lies. (a) Random- vs spatial-cross-validation AUC per case
                      and on the real US-MVT belt: random CV inflates the AUC (largest on the real data), and the
                      honest number is the spatial-CV one. (b) On the real data, the learned classifier beats
                      classical Weights of Evidence under honest spatial CV.
  fig-woe.pdf       - the Weights-of-Evidence contrasts and their studentized significance for the six evidence
                      layers of the real US-MVT belt: large raw contrasts but low studentized significance, the
                      honest state of real geoscience evidence.

Run:  python make_figs.py     (from repo root)
Deps: matplotlib, numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

INK = "#1a1a2e"
GRID = "#d8d8e0"
RAND = "#b23a48"
SPAT = "#1b6ca8"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.4, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
})


def _load():
    return json.loads((DATA / "pm.json").read_text(encoding="utf-8"))


def fig_inflation():
    d = _load()
    cases = [c for c in d["cases"] if c["randomAuc"] is not None]
    cases.sort(key=lambda c: -(c["gap"] or 0))
    ids = [c["id"] for c in cases]
    rnd = [c["randomAuc"] for c in cases]
    spt = [c["spatialAuc"] for c in cases]
    y = np.arange(len(cases))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})

    for yi, r, s in zip(y, rnd, spt):
        a1.plot([s, r], [yi, yi], color="#c9c9d2", linewidth=1.3, zorder=1)
    a1.scatter(spt, y, s=30, color=SPAT, edgecolor=INK, linewidth=0.4, zorder=3, label="spatial CV (honest)")
    a1.scatter(rnd, y, s=30, color=RAND, edgecolor=INK, linewidth=0.4, zorder=3, label="random CV (inflated)")
    a1.set_yticks(y)
    a1.set_yticklabels([f"{i}{'  *' if i=='REAL-USMVT' else ''}" for i in ids], fontsize=7.4)
    a1.set_xlabel("ROC-AUC (Weights of Evidence)")
    a1.set_title("(a) random CV inflates the AUC;\nspatial CV is the honest number", fontsize=8.2)
    a1.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    a1.set_axisbelow(True)
    a1.axvline(0.5, color="#999", linewidth=0.8, linestyle=":")
    a1.legend(fontsize=7.0, frameon=True, facecolor="white", edgecolor=GRID, loc="lower right")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    # (b) WoE vs MLP on real, spatial CV
    rc = d["real"]["spatial_cv"]
    names = ["Weights of\nEvidence", "learned MLP"]
    vals = [rc["wofe_roc_auc"], rc["mlp_roc_auc"]]
    cols = ["#e07a3f", "#1b6ca8"]
    bars = a2.bar(names, vals, color=cols, edgecolor=INK, linewidth=0.6, width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.6, fontweight="bold")
    a2.axhline(0.5, color="#999", linewidth=0.8, linestyle=":")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("spatial-CV ROC-AUC (real US-MVT)")
    a2.set_title("(b) learned beats WoE\n(honest spatial CV)", fontsize=8.2)
    a2.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    a2.set_axisbelow(True)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(HERE / "fig-inflation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_woe():
    d = _load()
    layers = d["real_woe"]
    ids = [L["id"] for L in layers]
    contrast = [L["contrast"] for L in layers]
    tstar = [L["tStar"] for L in layers]
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    bars = ax.bar(x, contrast, color="#1b6ca8", edgecolor=INK, linewidth=0.5, width=0.6, zorder=3,
                  label="WoE contrast $C=W^+-W^-$")
    ax.set_ylabel("WoE contrast", color="#1b6ca8")
    ax.tick_params(axis="y", labelcolor="#1b6ca8")
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=25, ha="right", fontsize=8)
    ax.set_title("Real US-MVT: large WoE contrasts, but low studentized\nsignificance, the honest state of real evidence",
                 fontsize=8.6)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top",):
        ax.spines[s].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, tstar, "s--", color="#e07a3f", linewidth=1.4, markersize=6, zorder=4, label="studentized $C/s(C)$")
    ax2.axhline(1.645, color="#b23a48", linewidth=1.1, linestyle=":", label="95% significance ($C/s(C)=1.645$)")
    ax2.set_ylabel("studentized contrast", color="#e07a3f")
    ax2.tick_params(axis="y", labelcolor="#e07a3f")
    ax2.set_ylim(0, 2.0)
    ax2.spines["top"].set_visible(False)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7.0, frameon=True, facecolor="white",
              edgecolor=GRID, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "fig-woe.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_inflation()
    fig_woe()
    print("wrote fig-inflation.pdf, fig-woe.pdf")


if __name__ == "__main__":
    main()
