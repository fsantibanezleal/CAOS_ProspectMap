#!/usr/bin/env python3
"""Build data/pm.json, the numbers the ProspectMap technical report quotes and plots, from the committed artifacts
under data/derived/:

  case-results.json     the TypeScript engine's bake: per case, the Weights-of-Evidence fit without cross-validation
                        and the random / spatial-block cross-validation; for REAL-USMVT, the per-layer weights and
                        their 2x2 counts, and the conditional-independence tests of the full-data fit;
  pm-learned-real.json  the real-lane head-to-head: the learned MLP and WofE under one cross-validation protocol, the
                        out-of-fold logistic regression and the distance-to-known-deposit baseline under the same
                        folds, and the MLP-only labelled-sample CV.

Every key names its protocol and its cell set, so no value can be read under the wrong protocol. s(C) is recomputed
from the 2x2 counts with the engine's formula (Haldane 0.5 on every cell of a table with a zero count) and checked
against the engine's studentized contrast. Run from the repository root (stdlib only):

    python manuscripts/prospectivity/data/build_pm.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DERIVED = ROOT / "data" / "derived"
REAL = "REAL-USMVT"
SCHEMA = "prospectmap.manuscript-data/v2"

PROTOCOLS = {
    "wofe_fit_no_cv": (
        "Weights of Evidence fitted on every deposit cell (each layer's maximizing-contrast threshold, its weights and "
        "the prior) and scored on the same map cells; nothing is held out, so it is a fitting AUC."
    ),
    "cv_all_cells_pooled": (
        "5-fold cross-validation over every map cell with the engine's folds: spatial-block (20x20-cell blocks, "
        "fold = blockId % 5, so every held-out block borders training blocks) or random (seed 17). Each fold's model "
        "is fitted on the training folds' cells only (for WofE: thresholds, weights and prior); every cell is scored "
        "once while its fold is held out; the held-out scores are pooled into one Mann-Whitney ROC AUC."
    ),
    "logistic_regression": (
        "The engine's logistic regression (IRLS, ridge 1e-3 on the slopes) on the layers' binary patterns. Without "
        "CV: fitted on every cell at the full-data thresholds (the betas are those slopes). Under CV: per fold, the "
        "patterns at the training folds' thresholds and the fit on the training folds' cells only; same folds, cells "
        "and aggregation as cv_all_cells_pooled."
    ),
    "distance_baseline": (
        "The engine's distance-to-known-deposit score exp(-d / 4), d the cell distance to the nearest training-fold "
        "deposit, under the same folds and aggregation; it learns no geology."
    ),
    "mlp_labelled_sample_cv": (
        "The learned MLP only, on its labelled sample (the deposit cells and non-deposit cells sampled more than 2 "
        "cells from any deposit); 20x20-cell blocks assigned to 5 folds in shuffled order (spatial) or a uniform "
        "5-way split (random); the AUC is the mean of the 5 per-fold AUCs. No WofE value exists on this basis."
    ),
    "woe_layers": (
        "Per real evidence layer, fitted on every deposit cell: the maximizing-contrast threshold tStar, W+ and W-, "
        "the contrast C = W+ - W-, its standard error s(C) = sqrt(1/n_B_D + 1/n_B_notD + 1/n_notB_D + 1/n_notB_notD) "
        "(Haldane 0.5 added to every count of a table with a zero count), and the studentized contrast C / s(C); "
        "one-sided significance at 5% is C / s(C) > 1.645, two-sided at 5% is > 1.96. pattern_area_fraction is the "
        "share of counted map cells inside the pattern; deposit_fraction_in_pattern the share of deposit cells "
        "inside it. When no deposit lies outside the pattern, W-, C and s(C) rest on the 0.5 correction, and C/s(C) "
        "depends on it (studentized_C_if_correction_0_1 repeats it with 0.1). Exact inference needs no correction: "
        "fisher_exact_p_one_sided is the one-sided Fisher exact p-value that deposits are over-represented in the "
        "pattern, and contrast_C_exact_95_interval the exact conditional (Cornfield) 95% interval for C = ln(odds "
        "ratio), its upper bound null (+infinity) when no deposit lies outside the pattern."
    ),
    "conditional_independence_fit_no_cv": (
        "Computed on the full-data fit (every deposit cell, no CV), so the cross-validation protocol does not affect "
        "it. Omnibus test (Agterberg and Cheng 2002): T is the sum of the posterior over the map cells, which equals "
        "N(D) under conditional independence; s(T) = sqrt(sum P(1 - P)); z = (T - N(D)) / s(T); the CI ratio is "
        "N(D) / T, which the tool reads as a violation below 0.85. Pairwise: for each layer pair, the Yates-corrected "
        "2x2 chi-square of the two binary patterns within the deposit cells plus the same within the non-deposit "
        "cells (df = 2), and Cramer's V = sqrt(chi2 / n)."
    ),
}


def _read(name: str) -> dict:
    return json.loads((DERIVED / name).read_text(encoding="utf-8"))


def contrast_se(n_bd: int, n_bdbar: int, n_bbard: int, n_bbardbar: int) -> tuple[float, bool]:
    """s(C) from the 2x2 counts, as frontend/src/mpm/wofe.ts::weightsFromCounts computes it."""
    haldane = 0 in (n_bd, n_bdbar, n_bbard, n_bbardbar)
    k = 0.5 if haldane else 0.0
    return math.sqrt(sum(1.0 / (n + k) for n in (n_bd, n_bdbar, n_bbard, n_bbardbar))), haldane


def studentized_contrast(n_bd: int, n_bdbar: int, n_bbard: int, n_bbardbar: int, k: float) -> float:
    """C / s(C) with `k` added to every count of a table that has a zero count (the engine uses k = 0.5)."""
    kk = k if 0 in (n_bd, n_bdbar, n_bbard, n_bbardbar) else 0.0
    a, b, c, d = n_bd + kk, n_bdbar + kk, n_bbard + kk, n_bbardbar + kk
    return math.log((a * d) / (b * c)) / math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _cond_log_weights(n_bd: int, n_bdbar: int, n_bbard: int, n_bbardbar: int) -> tuple[list[int], list[float]]:
    """Support and log weights (at odds ratio 1) of n_B_D given the margins: the central hypergeometric."""
    n_b, n_d = n_bd + n_bdbar, n_bd + n_bbard
    n = n_b + n_bbard + n_bbardbar
    xs = list(range(max(0, n_b + n_d - n), min(n_b, n_d) + 1))
    return xs, [_log_choose(n_d, x) + _log_choose(n - n_d, n_b - x) for x in xs]


def _tail(xs: list[int], lw: list[float], obs: int, log_psi: float, upper: bool) -> float:
    lt = [w + x * log_psi for x, w in zip(xs, lw)]
    sel = [v for x, v in zip(xs, lt) if (x >= obs if upper else x <= obs)]
    return math.exp(_logsumexp(sel) - _logsumexp(lt))


def fisher_exact_p_greater(n_bd: int, n_bdbar: int, n_bbard: int, n_bbardbar: int) -> float:
    """One-sided Fisher exact p-value that deposits are over-represented in the pattern: P(n_B_D >= observed) under
    independence given the margins. Exact, so it needs no correction for zero counts."""
    xs, lw = _cond_log_weights(n_bd, n_bdbar, n_bbard, n_bbardbar)
    return _tail(xs, lw, n_bd, 0.0, upper=True)


def exact_contrast_interval(n_bd: int, n_bdbar: int, n_bbard: int, n_bbardbar: int,
                            level: float = 0.95) -> tuple[float, float | None]:
    """The exact conditional (Cornfield) interval for C = ln(odds ratio): each bound puts (1 - level) / 2 in the
    corresponding tail of the noncentral hypergeometric distribution of n_B_D. The upper bound is None (+infinity)
    when no deposit lies outside the pattern."""
    xs, lw = _cond_log_weights(n_bd, n_bdbar, n_bbard, n_bbardbar)
    alpha = (1.0 - level) / 2.0

    def solve(upper_tail: bool) -> float:
        lo, hi = -60.0, 60.0  # on ln(psi); the tail probability is monotone in psi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            p = _tail(xs, lw, n_bd, mid, upper=upper_tail)
            if (p < alpha) == upper_tail:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lower = solve(upper_tail=True)
    upper = None if n_bd == xs[-1] else solve(upper_tail=False)
    return lower, upper


def build() -> dict:
    cr = _read("case-results.json")
    learned = _read("pm-learned-real.json")
    clf = learned["classifier"]
    if clf.get("protocol", {}).get("cell_set") != "all_map_cells":
        raise SystemExit("pm-learned-real.json does not declare the all-cells head-to-head protocol (schema v2 or later)")

    cases = []
    for cid, c in cr["cases"].items():
        cv = c["cv"]
        cases.append({
            "id": cid,
            "real_or_synthetic": c["realOrSynthetic"],
            "n_cells": c["nCells"],
            "n_deposit_cells": c["nDeposits"],
            "wofe_fit_no_cv_auc": c["rocAuc"],
            "wofe_random_cv_auc": cv["randomAuc"],
            "wofe_spatial_cv_auc": cv["spatialAuc"],
            "wofe_cv_gap_random_minus_spatial": cv["inflationGap"],
            "wofe_capture_at_10pct_area_spatial_cv": c["capture"]["prediction"]["captureAt10"],
        })

    real = cr["cases"][REAL]
    if abs(real["cv"]["spatialAuc"] - clf["spatial_cv"]["wofe_roc_auc"]) > 5e-7:
        raise SystemExit("pm-learned-real.json and case-results.json disagree on the WofE spatial-CV AUC: "
                         "re-run the real lane (science/real_wofe_oof.mjs, pipeline.real_learned) after the bake")
    layers = []
    for L in real["layers"]:
        s_c, haldane = contrast_se(L["nBD"], L["nBDbar"], L["nBbarD"], L["nBbarDbar"])
        if abs(L["contrast"] / s_c - L["studC"]) > 1e-9:
            raise SystemExit(f"s(C) recomputed for {L['id']} disagrees with the engine's studentized contrast")
        n_counted = L["nBD"] + L["nBDbar"] + L["nBbarD"] + L["nBbarDbar"]
        counts = (L["nBD"], L["nBDbar"], L["nBbarD"], L["nBbarDbar"])
        c_lo, c_hi = exact_contrast_interval(*counts)
        layers.append({
            "id": L["id"],
            "tStar_threshold": L["tStar"],
            "W_plus": L["wPlus"],
            "W_minus": L["wMinus"],
            "contrast_C": L["contrast"],
            "s_C": s_c,
            "studentized_C": L["studC"],
            # a zero count (typically no deposit outside the pattern) makes W-, C and s(C) rest on the 0.5 correction
            "haldane_correction": haldane,
            "studentized_C_if_correction_0_1": studentized_contrast(*counts, k=0.1),
            "fisher_exact_p_one_sided": fisher_exact_p_greater(*counts),
            "contrast_C_exact_95_interval": [c_lo, c_hi],
            "pattern_area_fraction": (L["nBD"] + L["nBDbar"]) / n_counted,
            "deposit_fraction_in_pattern": L["nBD"] / (L["nBD"] + L["nBbarD"]),
            "counts": {"n_B_D": L["nBD"], "n_B_notD": L["nBDbar"], "n_notB_D": L["nBbarD"],
                       "n_notB_notD": L["nBbarDbar"]},
        })

    sp, rd = clf["spatial_cv"], clf["random_cv"]
    ls = clf["labelled_sample_cv"]
    return {
        "schema": SCHEMA,
        "source": {
            "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
            "case_results": "data/derived/case-results.json",
            "learned_real": "data/derived/pm-learned-real.json",
            "builder": "manuscripts/prospectivity/data/build_pm.py",
        },
        "protocols": PROTOCOLS,
        "cases": cases,
        "real": {
            "id": REAL,
            "name": real["name"],
            "n_cells": real["nCells"],
            "n_deposit_cells": real["nDeposits"],
            "wofe_fit_no_cv_auc": real["rocAuc"],
            "logistic_regression_fit_no_cv": {
                "auc": real["lr"]["rocAuc"],
                "betas": {b["id"]: b["beta"] for b in real["lr"]["betas"]},
            },
            "cv_all_cells_pooled": {
                "spatial": {"wofe_auc": real["cv"]["spatialAuc"], "mlp_auc": sp["mlp_roc_auc"],
                            "logistic_regression_auc": sp["lr_roc_auc"],
                            "distance_baseline_auc": sp["distance_null_roc_auc"], "verdict_mlp_vs_wofe": sp["winner"]},
                "random": {"wofe_auc": real["cv"]["randomAuc"], "mlp_auc": rd["mlp_roc_auc"],
                           "logistic_regression_auc": rd["lr_roc_auc"],
                           "distance_baseline_auc": rd["distance_null_roc_auc"]},
                "wofe_gap_random_minus_spatial": real["cv"]["inflationGap"],
                "mlp_gap_random_minus_spatial": clf["inflation_gap"],
                "verdict_rule": "mlp or wofe when one AUC exceeds the other by more than 0.005, else tie",
            },
            "mlp_labelled_sample_cv": {
                "n_deposit_cells": ls["n_pos"],
                "n_sampled_non_deposit_cells": ls["n_neg"],
                "spatial_auc_mean_of_folds": ls["spatial_mlp_roc_auc"],
                "random_auc_mean_of_folds": ls["random_mlp_roc_auc"],
                "gap_random_minus_spatial": ls["inflation_gap"],
            },
            "woe_layers": layers,
            "conditional_independence_fit_no_cv": {
                "omnibus": {"T_sum_of_posterior": real["ci"]["T"], "N_D": real["ci"]["nD"], "s_T": real["ci"]["sT"],
                            "z": real["ci"]["z"], "ci_ratio_N_D_over_T": real["ci"]["ciRatio"]},
                "pairwise": [{"layers": [p["a"], p["b"]], "chi2_stratified_yates_df2": p["chi2"],
                              "cramers_v": p["cramersV"]} for p in real["ci"]["pairwise"]],
            },
        },
    }


def main() -> None:
    out = build()
    path = HERE / "pm.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    r = out["real"]
    print(f"wrote {path.relative_to(ROOT)}: {len(out['cases'])} cases; {r['id']} WofE fit {r['wofe_fit_no_cv_auc']:.3f}, "
          f"spatial CV WofE {r['cv_all_cells_pooled']['spatial']['wofe_auc']:.3f} / "
          f"MLP {r['cv_all_cells_pooled']['spatial']['mlp_auc']:.3f}; "
          f"studC {min(L['studentized_C'] for L in r['woe_layers']):.2f}-{max(L['studentized_C'] for L in r['woe_layers']):.2f}")


if __name__ == "__main__":
    main()
