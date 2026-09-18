"""The learned classifier vs Weights of Evidence on a real cube: the protocol and the artifact block, torch-free so CI
can test them (pipeline/real_learned.py does the training).

Two AUCs are only comparable when both come from one protocol: the same folds, the same held-out cells, the same
labels and the same aggregation. The protocol here is the one behind the WofE cross-validation AUCs of the committed
bake (case-results.json, produced by the TypeScript engine's analyzeCube):

- folds: the engine's spatialBlockFolds (20x20-cell blocks, fold = blockId % k) and randomFolds (seed 17);
- refit: each model is refitted on the training folds only (WofE: weights from the training-fold deposits; the MLP:
  the training-fold deposits plus distance-buffered negatives sampled from the training folds);
- scoring: every map cell is scored exactly once, while its fold is held out;
- aggregation: the held-out scores are pooled over the folds into one rank (Mann-Whitney) ROC AUC over all cells.

The engine's held-out WofE posterior and its folds come from science/real_wofe_oof.mjs. `check_oof_matches_bake`
re-derives the pooled WofE AUCs from that export with `rank_auc`, the estimator the MLP also gets, and refuses to go
on unless they equal the bake, so the two values in `spatial_cv` are like for like by construction.

The same export carries the engine's distance-to-known-deposit baseline under the same folds (nearestDepositScore,
exp(-d / 4) with d the cell distance to the nearest training-fold deposit). It learns no geology, so its AUC is the
ranking skill that proximity to known deposits alone reaches under the same folds. It is reported beside the pair as a
reference, never as a contestant: the engine's folds interleave 20x20-cell blocks, so every held-out block borders
training blocks.

The WofE AUC without cross-validation (weights fitted on every deposit, scored on the same cells) goes under `nocv`,
never under a cross-validation key: GitHub issue #41 found it stored as `spatial_cv.wofe_roc_auc`, beside a `winner`
flag that compared it with an MLP value computed on a different cell set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEARNED_REAL_SCHEMA = "prospectmap.learned/v2"
WINNER_MARGIN = 0.005  # the tie band of the synthetic lane's head-to-head (science/train_mpm.py)
CELL_SET = "all_map_cells"
# AUCs are written at 6 decimals: the App shows 3, and a 4-decimal value such as 0.6365 would display as 0.636
# beside the live engine value 0.6365238 shown as 0.637 (double rounding).
AUC_DIGITS = 6


def rank_auc(scores, labels) -> float:
    """ROC AUC as the Mann-Whitney U statistic with average ranks for tied scores: the estimator of the engine's
    rocAuc (frontend/src/mpm/validate.ts). Returns 0.5 when a class is empty, as the engine does."""
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(labels, dtype=np.float64) > 0.5
    if s.shape != pos.shape:
        raise ValueError(f"scores {s.shape} and labels {pos.shape} differ in shape")
    if np.isnan(s).any():
        raise ValueError("rank_auc got NaN scores; every evaluated cell needs a held-out score")
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts          # 0-based position of each distinct score in ascending order
    avg_rank = first + (counts + 1) / 2.0       # 1-based average rank shared by the tied cells
    rank_sum = float(avg_rank[inverse.ravel()][pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def winner(mlp_auc: float, wofe_auc: float, margin: float = WINNER_MARGIN) -> str:
    """'mlp' or 'wofe' when one AUC exceeds the other by more than `margin`, else 'tie'. Only ever call it on two AUCs
    measured under one protocol."""
    if mlp_auc > wofe_auc + margin:
        return "mlp"
    if wofe_auc > mlp_auc + margin:
        return "wofe"
    return "tie"


def sample_negatives_in_pool(y: np.ndarray, rows: np.ndarray, cols: np.ndarray, pool: np.ndarray,
                             rng: np.random.Generator, *, nx: int, ny: int,
                             ratio: int = 3, buffer_cells: int = 2) -> np.ndarray:
    """Presence-only negatives for one training split: non-deposit cells drawn from `pool` only, each more than
    `buffer_cells` (Chebyshev) from every deposit IN the pool, `ratio` per pool deposit. The held-out fold's deposits
    never shape the training set (the per-fold sampling of pipeline/pu_conformal.py's sample_negatives)."""
    pool = np.asarray(pool, dtype=np.int64)
    pos = pool[y[pool] > 0.5]
    near = np.zeros((ny, nx), dtype=bool)
    grid = np.zeros((ny, nx), dtype=bool)
    grid[rows[pos], cols[pos]] = True
    for dr in range(-buffer_cells, buffer_cells + 1):
        for dc in range(-buffer_cells, buffer_cells + 1):
            src_r = slice(max(0, -dr), ny - max(0, dr))
            dst_r = slice(max(0, dr), ny - max(0, -dr))
            src_c = slice(max(0, -dc), nx - max(0, dc))
            dst_c = slice(max(0, dc), nx - max(0, -dc))
            near[dst_r, dst_c] |= grid[src_r, src_c]
    cand = pool[y[pool] < 0.5]
    keep = cand[~near[rows[cand], cols[cand]]]
    if len(keep) == 0:
        keep = cand
    n_neg = min(len(keep), ratio * max(1, len(pos)))
    return rng.choice(keep, size=n_neg, replace=False)


def load_wofe_oof(path: str | Path) -> dict:
    """Read the engine export written by science/real_wofe_oof.mjs into numpy arrays."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != "prospectmap.wofe-oof/v1":
        raise ValueError(f"{path}: unexpected schema {d.get('schema')!r}")
    out = {key: d[key] for key in ("case_id", "nx", "ny", "k", "block_cells", "random_seed", "layer_ids",
                                   "null_scale_cells")}
    out["cells"] = np.asarray(d["cells"], dtype=np.int64)
    out["deposit_cells"] = np.asarray(d["deposit_cells"], dtype=np.int64)

    def floats(values: list) -> np.ndarray:
        return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)

    for scheme in ("spatial", "random"):
        s = d[scheme]
        out[scheme] = {
            "folds": np.asarray(s["folds"], dtype=np.int64),
            "wofe": floats(s["wofe"]),
            "engine_auc": float(s["engine_auc"]),
            "distance_null": floats(s["distance_null"]),
            "distance_null_engine_auc": float(s["distance_null_engine_auc"]),
        }
    return out


def check_oof_matches_bake(oof: dict, case_result: dict, tol: float = 1e-12) -> dict:
    """Re-derive the pooled WofE AUCs from the exported held-out scores and require them to equal both the engine's
    own value in the export and the committed bake's cv block. Returns {'spatial': auc, 'random': auc}.

    A mismatch means the export is not the bake's cross-validation (another cube, layer set or fold scheme), and the
    head-to-head would silently compare the MLP against a different WofE, so it raises instead."""
    cv = case_result["cv"]
    if int(cv["k"]) != int(oof["k"]) or int(cv["blockCells"]) != int(oof["block_cells"]):
        raise ValueError(f"fold scheme differs: bake k={cv['k']} blockCells={cv['blockCells']}, "
                         f"export k={oof['k']} block_cells={oof['block_cells']}")
    labels = np.isin(oof["cells"], oof["deposit_cells"])
    if len(oof["cells"]) != int(case_result["nCells"]) or int(labels.sum()) != int(case_result["nDeposits"]):
        raise ValueError(f"cell set differs: bake {case_result['nCells']} cells / {case_result['nDeposits']} deposits, "
                         f"export {len(oof['cells'])} / {int(labels.sum())}")
    aucs = {}
    for scheme, key in (("spatial", "spatialAuc"), ("random", "randomAuc")):
        auc = rank_auc(oof[scheme]["wofe"], labels)
        for name, ref in (("the engine export", oof[scheme]["engine_auc"]), ("the committed bake", float(cv[key]))):
            if abs(auc - ref) > tol:
                raise ValueError(f"WofE {scheme}-CV AUC {auc!r} re-derived from the export != {name} ({ref!r})")
        aucs[scheme] = auc
    return aucs


def distance_null_aucs(oof: dict, tol: float = 1e-12) -> dict:
    """Pooled AUCs of the engine's distance-to-known-deposit baseline under the export's folds, re-derived with
    rank_auc and required to equal the engine's own values. Returns {'spatial': auc, 'random': auc}."""
    labels = np.isin(oof["cells"], oof["deposit_cells"])
    aucs = {}
    for scheme in ("spatial", "random"):
        auc = rank_auc(oof[scheme]["distance_null"], labels)
        ref = oof[scheme]["distance_null_engine_auc"]
        if abs(auc - ref) > tol:
            raise ValueError(f"distance-null {scheme}-CV AUC {auc!r} re-derived from the export != the engine ({ref!r})")
        aucs[scheme] = auc
    return aucs


def build_classifier_block(*, case_result: dict, wofe_cv: dict, mlp_cv: dict, labelled: dict,
                           k: int, block_cells: int, random_seed: int,
                           null_cv: dict | None = None, null_scale_cells: float = 4.0) -> dict:
    """Assemble the `classifier` block of pm-learned-real.json.

    wofe_cv / mlp_cv: pooled held-out AUCs under the shared protocol, {'spatial': float, 'random': float}.
    null_cv: the distance-to-known-deposit baseline under the same protocol (same keys), a reference beside the pair.
    labelled: the MLP-only labelled-sample CV, {'spatial', 'random', 'n_pos', 'n_neg'}; it has no WofE counterpart,
    so it carries no WofE value and no winner.
    """
    n_cells = int(case_result["nCells"])
    n_dep = int(case_result["nDeposits"])

    def rd(v: float) -> float:
        return round(float(v), AUC_DIGITS)

    mlp_spatial, wofe_spatial = rd(mlp_cv["spatial"]), rd(wofe_cv["spatial"])
    spatial_cv = {  # the one comparison in the file: both values under the shared protocol
        "mlp_roc_auc": mlp_spatial,
        "wofe_roc_auc": wofe_spatial,
        "winner": winner(mlp_spatial, wofe_spatial),  # from the written values, so the file can be re-checked
    }
    random_cv = {"mlp_roc_auc": rd(mlp_cv["random"]), "wofe_roc_auc": rd(wofe_cv["random"])}
    baseline = {}
    if null_cv is not None:  # a reference under the same protocol, never a contestant for `winner`
        spatial_cv["distance_null_roc_auc"] = rd(null_cv["spatial"])
        random_cv["distance_null_roc_auc"] = rd(null_cv["random"])
        baseline = {"baseline": (
            f"distance_null: the engine's distance-to-known-deposit score exp(-d / {null_scale_cells:g}), d the cell "
            "distance to the nearest training-fold deposit, scored under the same folds; it learns no geology, so "
            "its AUC is the ranking skill that proximity to known deposits alone reaches, and a model must beat it "
            "to claim it learned geology. The spatial folds interleave blocks, so every held-out block borders "
            "training blocks")}
    return {
        "protocol": {
            "cell_set": CELL_SET,
            "n_cells": n_cells,
            "n_deposit_cells": n_dep,
            "k": int(k),
            "block_cells": int(block_cells),
            "random_seed": int(random_seed),
            "folds": (f"the TS engine's folds: spatial = spatialBlockFolds ({block_cells}x{block_cells}-cell blocks, "
                      f"fold = blockId % {k}); random = randomFolds (seed {random_seed})"),
            "refit": ("each model refitted on the training folds only: WofE weights on the training-fold deposits; "
                      "the MLP on the training-fold deposits plus distance-buffered negatives sampled from the "
                      "training folds"),
            "aggregation": (f"held-out scores pooled over the {k} folds into one rank (Mann-Whitney) ROC AUC over "
                            f"all {n_cells} map cells ({n_dep} deposit cells)"),
            **baseline,
        },
        "spatial_cv": spatial_cv,
        "random_cv": random_cv,
        "inflation_gap": rd(mlp_cv["random"] - mlp_cv["spatial"]),
        "nocv": {
            "wofe_roc_auc": rd(case_result["rocAuc"]),
            "protocol": (f"WofE weights fitted on all {n_dep} deposit cells and scored on the same {n_cells} cells, "
                         "no hold-out: a fitting AUC, not comparable with the cross-validated values"),
        },
        "labelled_sample_cv": {
            "protocol": (f"MLP only, on its labelled sample ({labelled['n_pos']} deposit cells and "
                         f"{labelled['n_neg']} non-deposit cells sampled more than 2 cells from any deposit); "
                         f"spatial folds are {block_cells}x{block_cells}-cell blocks assigned to {k} folds in shuffled "
                         f"order, random folds a uniform {k}-way split; the AUC is the mean of the per-fold AUCs. "
                         "No WofE value exists on this basis, so it is not a comparison with WofE"),
            "n_pos": int(labelled["n_pos"]),
            "n_neg": int(labelled["n_neg"]),
            "spatial_mlp_roc_auc": rd(labelled["spatial"]),
            "random_mlp_roc_auc": rd(labelled["random"]),
            "inflation_gap": rd(labelled["random"] - labelled["spatial"]),
        },
        "mlp_roc_auc": mlp_spatial,  # flat alias of spatial_cv.mlp_roc_auc (the replay manifests read it)
        "nFolds": int(k),
        "nEval": n_cells,
    }
