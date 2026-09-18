#!/usr/bin/env python3
"""Regenerate the figures for the ProspectMap prospectivity report from the COMMITTED artifacts. Two figures:

  fig-inflation.pdf - the two ways a prospectivity map lies. (a) Random- vs spatial-cross-validation AUC per case
                      and on the real US-MVT belt: random CV inflates the AUC (largest on the real data), and the
                      spatial-CV number is the realistic estimate. (b) On the real data, the AUC of Weights of
                      Evidence (scored over all map cells) and of the learned MLP (scored on its labelled sample of
                      deposit and sampled non-deposit cells) under each evaluation protocol.
  fig-woe.pdf       - the Weights-of-Evidence contrasts and their studentized significance for the six evidence
                      layers of the real US-MVT belt: large raw contrasts but low studentized significance.

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
    a1.scatter(spt, y, s=30, color=SPAT, edgecolor=INK, linewidth=0.4, zorder=3, label="spatial CV")
    a1.scatter(rnd, y, s=30, color=RAND, edgecolor=INK, linewidth=0.4, zorder=3, label="random CV (inflated)")
    a1.set_yticks(y)
    a1.set_yticklabels([f"{i}{'  *' if i=='REAL-USMVT' else ''}" for i in ids], fontsize=7.4)
    a1.set_xlabel("ROC-AUC (Weights of Evidence)")
    a1.set_title("(a) random CV inflates the AUC;\nspatial CV is the realistic estimate", fontsize=8.2)
    a1.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    a1.set_axisbelow(True)
    a1.axvline(0.5, color="#999", linewidth=0.8, linestyle=":")
    a1.legend(fontsize=7.0, frameon=True, facecolor="white", edgecolor=GRID, loc="upper left")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    # (b) the real belt: AUC by evaluation protocol. Weights of Evidence is scored over all active map cells
    # (the case's cv block: random and spatial-block folds); the learned MLP over its labelled sample (deposit
    # cells + sampled, distance-buffered non-deposit cells; pm-learned-real.json). pm-learned-real.json stores the
    # WofE AUC WITHOUT cross-validation (the trace's roc_auc) under spatial_cv.wofe_roc_auc (real_learned.py), so it
    # is plotted as "no CV". The two models are scored on different cell sets.
    real = next(c for c in d["cases"] if c["id"] == "REAL-USMVT")
    rl = d["real"]
    protocols = ["no CV", "random CV", "spatial CV"]
    woe = [rl["spatial_cv"]["wofe_roc_auc"], real["randomAuc"], real["spatialAuc"]]
    mlp = [np.nan, rl["random_cv"]["mlp_roc_auc"], rl["spatial_cv"]["mlp_roc_auc"]]
    x = np.arange(len(protocols))
    bw = 0.36
    for off, vals, col, lab in [(-bw / 2, woe, "#e07a3f", "Weights of Evidence (all cells)"),
                                (bw / 2, mlp, "#1b6ca8", f"learned MLP (labelled sample, n={rl['nEval']})")]:
        a2.bar(x + off, vals, bw, color=col, edgecolor=INK, linewidth=0.6, zorder=3, label=lab)
        for xi, v in zip(x + off, vals):
            if np.isfinite(v):
                a2.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels(protocols, fontsize=8.0)
    a2.axhline(0.5, color="#999", linewidth=0.8, linestyle=":")
    a2.set_ylim(0, 1.3)
    a2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.set_ylabel("ROC-AUC (real US-MVT)")
    a2.set_title("(b) real belt: AUC by protocol", fontsize=8.2)
    a2.legend(fontsize=6.6, frameon=True, facecolor="white", edgecolor=GRID, loc="upper left")
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
    fig, ax = plt.subplots(figsize=(6.2, 3.55))
    bars = ax.bar(x, contrast, color="#1b6ca8", edgecolor=INK, linewidth=0.5, width=0.6, zorder=3,
                  label="WoE contrast $C=W^+-W^-$")
    ax.set_ylabel("WoE contrast", color="#1b6ca8")
    ax.tick_params(axis="y", labelcolor="#1b6ca8")
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=25, ha="right", fontsize=8)
    ax.set_title("Real US-MVT: large WoE contrasts,\nbut low studentized significance",
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
              edgecolor=GRID, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / "fig-woe.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_inflation()
    fig_woe()
    print("wrote fig-inflation.pdf, fig-woe.pdf")


if __name__ == "__main__":
    main()
