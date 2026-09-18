#!/usr/bin/env python3
"""Regenerate the figures of the ProspectMap technical report from data/pm.json (built from the committed artifacts
by data/build_pm.py). Every AUC in a figure is labelled with its protocol and cell set.

  fig-inflation.pdf - (a) per case, the Weights-of-Evidence held-out ROC-AUC under random versus spatial-block
                      cross-validation (5 folds; each fold fitted on its training folds only; every map cell scored
                      once; pooled). (b) the real US-MVT belt, AUC by protocol: WofE without CV (fitting AUC); WofE,
                      the learned MLP and a distance-to-known-deposit baseline under the same random and
                      spatial-block CV over all map cells; and, separately, the MLP-only CV on its labelled sample.
  fig-woe.pdf       - the six real layers' WoE contrasts C (bars) and studentized contrasts C/s(C) (line), with the
                      one-sided (1.645) and two-sided (1.96) 5% significance levels. Layers whose 2x2 table has a zero
                      count are marked: their W-, C and s(C) rest on the 0.5 correction.

Run:  python manuscripts/prospectivity/data/build_pm.py && python manuscripts/prospectivity/figures/make_figs.py
Deps: matplotlib, numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

INK = "#1a1a2e"
GRID = "#d8d8e0"
RAND = "#b23a48"
SPAT = "#1b6ca8"
WOFE = "#e07a3f"
MLP = "#1b6ca8"
BASE = "#8a8a9a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.4, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
    "pdf.fonttype": 42,
})


def _load() -> dict:
    d = json.loads((DATA / "pm.json").read_text(encoding="utf-8"))
    if d.get("schema") != "prospectmap.manuscript-data/v2":
        raise SystemExit("data/pm.json is not prospectmap.manuscript-data/v2: run data/build_pm.py first")
    return d


def _bar_labels(ax, xs, vals, dy=0.015, size=6.2):
    for xi, v in zip(xs, vals):
        if v is not None and np.isfinite(v):
            ax.text(xi, v + dy, f"{v:.3f}", ha="center", va="bottom", fontsize=size, rotation=90)


def fig_inflation() -> None:
    d = _load()
    cases = sorted(d["cases"], key=lambda c: -c["wofe_cv_gap_random_minus_spatial"])
    ids = [c["id"] for c in cases]
    rnd = [c["wofe_random_cv_auc"] for c in cases]
    spt = [c["wofe_spatial_cv_auc"] for c in cases]
    y = np.arange(len(cases))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1.0, 1.35]})

    # (a) every case: the same WofE protocol under the two fold schemes
    for yi, r, s in zip(y, rnd, spt):
        a1.plot([s, r], [yi, yi], color="#c9c9d2", linewidth=1.3, zorder=1)
    # random under spatial, larger: where the two AUCs coincide (K-OROGENIC) both markers stay visible
    a1.scatter(rnd, y, s=46, color=RAND, edgecolor=INK, linewidth=0.4, zorder=3, label="random CV")
    a1.scatter(spt, y, s=20, color=SPAT, edgecolor=INK, linewidth=0.4, zorder=4, label="spatial-block CV")
    a1.set_yticks(y)
    a1.set_yticklabels([f"{i}{'  *' if i == 'REAL-USMVT' else ''}" for i in ids], fontsize=7.2)
    a1.set_xlabel("WofE held-out ROC-AUC\n(5 folds, fitted on the training folds,\nevery map cell scored once, pooled)",
                  fontsize=7.6)
    a1.set_title("(a) Weights of Evidence, per case:\nrandom vs spatial-block CV", fontsize=8.2)
    a1.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    a1.set_axisbelow(True)
    a1.axvline(0.5, color="#999", linewidth=0.8, linestyle=":")
    a1.legend(fontsize=6.6, frameon=True, facecolor="white", edgecolor=GRID, loc="center left")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)

    # (b) the real belt: every AUC with its protocol; the like-for-like groups share cells, folds and aggregation
    r = d["real"]
    cv = r["cv_all_cells_pooled"]
    ls = r["mlp_labelled_sample_cv"]
    groups = [
        ("no CV\n(fit)", r["wofe_fit_no_cv_auc"], None, None),
        ("random\nCV", cv["random"]["wofe_auc"], cv["random"]["mlp_auc"], cv["random"]["distance_baseline_auc"]),
        ("spatial-\nblock CV", cv["spatial"]["wofe_auc"], cv["spatial"]["mlp_auc"],
         cv["spatial"]["distance_baseline_auc"]),
    ]
    bw = 0.26
    x = np.arange(len(groups))
    series = [
        (-bw, [g[1] for g in groups], WOFE, "Weights of Evidence", None),
        (0.0, [g[2] for g in groups], MLP, "learned MLP", None),
        (bw, [g[3] for g in groups], BASE, "distance-to-deposit baseline (no geology)", "///"),
    ]
    for off, vals, col, lab, hatch in series:
        xs = [xi + off for xi, v in zip(x, vals) if v is not None]
        vs = [v for v in vals if v is not None]
        a2.bar(xs, vs, bw, color=col, edgecolor=INK, linewidth=0.6, zorder=3, label=lab, hatch=hatch)
        _bar_labels(a2, xs, vs)

    # the MLP-only CV on its labelled sample: a different cell set and aggregation, set apart and not compared
    xl = len(groups) + 0.45 + np.array([0.0, 0.8])
    lvals = [ls["random_auc_mean_of_folds"], ls["spatial_auc_mean_of_folds"]]
    a2.bar(xl, lvals, bw, color="#bcd3e6", edgecolor=INK, linewidth=0.6, zorder=3, hatch="..",
           label=f"MLP only, labelled sample ({ls['n_deposit_cells']} + {ls['n_sampled_non_deposit_cells']} cells, "
                 "mean of per-fold AUCs)")
    _bar_labels(a2, xl, lvals)
    a2.axvline(len(groups) - 0.05, color="#999", linewidth=0.8, linestyle="--")

    ticks = list(x) + list(xl)
    labels = [g[0] for g in groups] + ["random", "spatial"]
    a2.set_xticks(ticks)
    a2.set_xticklabels(labels, fontsize=6.6)
    a2.text(xl.mean(), -0.2, "labelled sample\n(MLP only)", ha="center", va="top", fontsize=6.4,
            transform=a2.get_xaxis_transform())
    a2.text(1.0, -0.2, "all map cells", ha="center", va="top", fontsize=6.4,
            transform=a2.get_xaxis_transform())
    a2.axhline(0.5, color="#999", linewidth=0.8, linestyle=":")
    a2.set_ylim(0, 1.16)
    a2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.set_ylabel("ROC-AUC (real US-MVT)")
    a2.set_title("(b) real belt: AUC by protocol", fontsize=8.2)
    a2.legend(fontsize=5.9, frameon=True, facecolor="white", edgecolor=GRID, loc="upper center",
              bbox_to_anchor=(0.5, -0.33), ncol=2)
    a2.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    a2.set_axisbelow(True)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(HERE / "fig-inflation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_woe() -> None:
    d = _load()
    layers = d["real"]["woe_layers"]
    ids = [L["id"] for L in layers]
    contrast = [L["contrast_C"] for L in layers]
    stud = [L["studentized_C"] for L in layers]
    zero = [L["haldane_correction"] for L in layers]
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.bar(x, contrast, color="#1b6ca8", edgecolor=INK, linewidth=0.5, width=0.6, zorder=3,
           label="WoE contrast $C=W^+-W^-$")
    ax.set_ylabel("WoE contrast $C$", color="#1b6ca8")
    ax.tick_params(axis="y", labelcolor="#1b6ca8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}{' *' if z else ''}" for i, z in zip(ids, zero)], rotation=25, ha="right", fontsize=8)
    ax.set_title("Real US-MVT: WoE contrasts and studentized contrasts", fontsize=8.6)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, stud, "s--", color="#e07a3f", linewidth=1.4, markersize=6, zorder=4,
             label="studentized contrast $C/s(C)$")
    ax2.axhline(1.645, color="#b23a48", linewidth=1.0, linestyle=":", label="one-sided 5% level (1.645)")
    ax2.axhline(1.96, color="#b23a48", linewidth=1.0, linestyle="--", label="two-sided 5% level (1.96)")
    ax2.set_ylabel("studentized contrast $C/s(C)$", color="#e07a3f")
    ax2.tick_params(axis="y", labelcolor="#e07a3f")
    ax2.set_ylim(0, max(stud) * 1.18)
    ax2.spines["top"].set_visible(False)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=6.8, frameon=True, facecolor="white",
              edgecolor=GRID, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    if any(zero):
        fig.text(0.5, -0.03, "* a zero count in the layer's 2x2 table: $W^-$, $C$ and $s(C)$ rest on the 0.5 correction",
                 ha="center", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(HERE / "fig-woe.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_inflation()
    fig_woe()
    print("wrote fig-inflation.pdf, fig-woe.pdf")


if __name__ == "__main__":
    main()
