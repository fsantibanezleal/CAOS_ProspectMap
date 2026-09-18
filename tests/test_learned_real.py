"""The real-lane learned metrics (data/derived/pm-learned-real.json) and the real case's replay records.

GitHub issue #41: `classifier.spatial_cv.wofe_roc_auc` held the Weights-of-Evidence AUC WITHOUT cross-validation, and
`winner` compared an MLP AUC on a labelled sample (the mean of per-fold AUCs) with a WofE AUC pooled over every map
cell. These tests pin the fix: the two values of the head-to-head share one protocol and cell set, the no-CV AUC lives
under `nocv`, `winner` is derived only from the like-for-like pair, and the real case's trace and manifest carry the
real lane's metrics instead of the synthetic lane's. They only read the committed artifacts."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import pipeline
from pipeline.model import head_to_head as h2h

REAL = "REAL-USMVT"


def _read(rel: str) -> dict:
    return json.loads((pipeline.DERIVED / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real() -> dict:
    return _read("pm-learned-real.json")


@pytest.fixture(scope="module")
def bake() -> dict:
    return _read("case-results.json")["cases"][REAL]


# ---------------------------------------------------------------------------------------------------------------------
# the committed artifact
# ---------------------------------------------------------------------------------------------------------------------
def test_spatial_cv_wofe_is_the_cross_validated_auc_not_the_fit_auc(real, bake):
    clf = real["classifier"]
    assert real["schema"] == h2h.LEARNED_REAL_SCHEMA
    assert clf["spatial_cv"]["wofe_roc_auc"] == round(bake["cv"]["spatialAuc"], h2h.AUC_DIGITS)
    assert clf["spatial_cv"]["wofe_roc_auc"] != round(bake["rocAuc"], h2h.AUC_DIGITS), "the fitting AUC is back under spatial_cv"
    assert clf["random_cv"]["wofe_roc_auc"] == round(bake["cv"]["randomAuc"], h2h.AUC_DIGITS)
    assert clf["nocv"]["wofe_roc_auc"] == round(bake["rocAuc"], h2h.AUC_DIGITS)


def test_head_to_head_is_one_protocol_over_every_map_cell(real, bake):
    clf = real["classifier"]
    p = clf["protocol"]
    assert p["cell_set"] == h2h.CELL_SET
    assert p["n_cells"] == bake["nCells"] == clf["nEval"]
    assert p["n_deposit_cells"] == bake["nDeposits"]
    assert (p["k"], p["block_cells"]) == (bake["cv"]["k"], bake["cv"]["blockCells"]) == (clf["nFolds"], 20)
    for block in ("spatial_cv", "random_cv"):
        assert {"mlp_roc_auc", "wofe_roc_auc"} <= set(clf[block]), f"{block} must hold both models"
        assert 0.5 <= clf[block]["distance_null_roc_auc"] <= 1.0, f"{block} must carry the proximity baseline"
    assert "distance_null" in p["baseline"]
    gap = clf["random_cv"]["mlp_roc_auc"] - clf["spatial_cv"]["mlp_roc_auc"]
    assert clf["inflation_gap"] == pytest.approx(gap, abs=1.5e-4)
    assert clf["mlp_roc_auc"] == clf["spatial_cv"]["mlp_roc_auc"]


def test_winner_only_compares_the_like_for_like_pair(real):
    clf = real["classifier"]
    s = clf["spatial_cv"]
    assert s["winner"] == h2h.winner(s["mlp_roc_auc"], s["wofe_roc_auc"])
    for key, block in clf.items():
        if key != "spatial_cv" and isinstance(block, dict):
            assert "winner" not in block, f"{key} is not a like-for-like pair and must not name a winner"
    labelled = clf["labelled_sample_cv"]
    assert not any(k.startswith("wofe") for k in labelled), "the labelled-sample CV has no WofE counterpart"
    assert labelled["n_pos"] == real["classifier"]["protocol"]["n_deposit_cells"]


# ---------------------------------------------------------------------------------------------------------------------
# the helpers
# ---------------------------------------------------------------------------------------------------------------------
def test_rank_auc_is_the_mann_whitney_statistic_with_tied_ranks():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(5, 60))
        s = rng.integers(0, 6, size=n).astype(float)  # heavy ties
        y = (rng.random(n) < 0.3).astype(float)
        if y.sum() in (0, n):
            continue
        pos, neg = s[y > 0.5], s[y < 0.5]
        brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
        assert h2h.rank_auc(s, y) == pytest.approx(brute, abs=1e-12)
        checked += 1
    assert checked > 20
    assert h2h.rank_auc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert h2h.rank_auc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    assert h2h.rank_auc([0.5] * 4, [0, 1, 0, 1]) == 0.5
    assert h2h.rank_auc([0.1, 0.2], [0, 0]) == 0.5
    with pytest.raises(ValueError):
        h2h.rank_auc([0.1, float("nan")], [0, 1])


_BAKE = {"nCells": 100, "nDeposits": 10, "rocAuc": 0.7321,
         "cv": {"k": 5, "blockCells": 20, "spatialAuc": 0.6365, "randomAuc": 0.7235}}


def _block(mlp_spatial: float, labelled_spatial: float, null_spatial: float | None = None) -> dict:
    null_cv = None if null_spatial is None else {"spatial": null_spatial, "random": null_spatial + 0.05}
    return h2h.build_classifier_block(
        case_result=_BAKE, wofe_cv={"spatial": 0.6365, "random": 0.7235},
        mlp_cv={"spatial": mlp_spatial, "random": mlp_spatial + 0.05},
        labelled={"spatial": labelled_spatial, "random": 0.98, "n_pos": 10, "n_neg": 30},
        k=5, block_cells=20, random_seed=17, null_cv=null_cv,
    )


def test_build_classifier_block_keeps_the_fit_auc_out_of_the_cv_blocks():
    blk = _block(0.62, 0.9456)
    assert blk["spatial_cv"]["wofe_roc_auc"] == 0.6365
    assert blk["random_cv"]["wofe_roc_auc"] == 0.7235
    assert blk["nocv"]["wofe_roc_auc"] == 0.7321
    assert 0.7321 not in (blk["spatial_cv"]["wofe_roc_auc"], blk["random_cv"]["wofe_roc_auc"])


def test_winner_ignores_the_labelled_sample_value():
    # the #41 pattern: a high labelled-sample MLP AUC (0.9456) beside a lower WofE value produced winner "mlp";
    # the verdict must come from the MLP value under the WofE's own protocol
    assert _block(0.62, 0.9456)["spatial_cv"]["winner"] == "wofe"
    assert _block(0.70, 0.9456)["spatial_cv"]["winner"] == "mlp"
    assert _block(0.6385, 0.9456)["spatial_cv"]["winner"] == "tie"


def test_distance_baseline_is_a_reference_not_a_contestant():
    blk = _block(0.62, 0.9456, null_spatial=0.90)
    assert blk["spatial_cv"]["distance_null_roc_auc"] == 0.90
    assert blk["random_cv"]["distance_null_roc_auc"] == 0.95
    assert blk["spatial_cv"]["winner"] == "wofe", "the baseline must never take part in the verdict"
    assert "baseline" in blk["protocol"]
    plain = _block(0.62, 0.9456)
    assert "distance_null_roc_auc" not in plain["spatial_cv"] and "baseline" not in plain["protocol"]


def _oof(wofe_spatial: np.ndarray, wofe_random: np.ndarray, deposits: np.ndarray, k: int = 5) -> dict:
    n = wofe_spatial.size
    cells = np.arange(n)
    labels = np.isin(cells, deposits)

    def scheme(folds: np.ndarray, wofe: np.ndarray) -> dict:
        null = wofe[::-1].copy()
        return {"folds": folds, "wofe": wofe, "engine_auc": h2h.rank_auc(wofe, labels),
                "distance_null": null, "distance_null_engine_auc": h2h.rank_auc(null, labels)}

    return {
        "case_id": REAL, "nx": n, "ny": 1, "k": k, "block_cells": 20, "random_seed": 17, "layer_ids": [],
        "null_scale_cells": 4, "cells": cells, "deposit_cells": deposits,
        "spatial": scheme(cells % k, wofe_spatial), "random": scheme((cells * 3) % k, wofe_random),
    }


def test_check_oof_matches_bake_accepts_the_bake_and_rejects_anything_else():
    rng = np.random.default_rng(1)
    s, r = rng.random(100), rng.random(100)
    deposits = np.arange(0, 100, 10)
    oof = _oof(s, r, deposits)
    bake = {"nCells": 100, "nDeposits": 10, "rocAuc": 0.9,
            "cv": {"k": 5, "blockCells": 20, "spatialAuc": oof["spatial"]["engine_auc"],
                   "randomAuc": oof["random"]["engine_auc"]}}
    aucs = h2h.check_oof_matches_bake(oof, bake)
    assert aucs == {"spatial": bake["cv"]["spatialAuc"], "random": bake["cv"]["randomAuc"]}
    with pytest.raises(ValueError):  # another WofE (e.g. the fitting AUC) is not the bake's cross-validation
        h2h.check_oof_matches_bake(oof, {**bake, "cv": {**bake["cv"], "spatialAuc": 0.9}})
    with pytest.raises(ValueError):  # another fold scheme
        h2h.check_oof_matches_bake(oof, {**bake, "cv": {**bake["cv"], "k": 10}})
    with pytest.raises(ValueError):  # another cell set
        h2h.check_oof_matches_bake(oof, {**bake, "nCells": 99})
    nulls = h2h.distance_null_aucs(oof)
    assert nulls["spatial"] == oof["spatial"]["distance_null_engine_auc"]
    bad = {**oof, "spatial": {**oof["spatial"], "distance_null_engine_auc": 0.99}}
    with pytest.raises(ValueError):  # the baseline must be the engine's own
        h2h.distance_null_aucs(bad)


def test_negatives_come_from_the_training_pool_and_ignore_held_out_deposits():
    nx = ny = 12
    n = nx * ny
    rows, cols = np.arange(n) // nx, np.arange(n) % nx
    y = np.zeros(n)
    y[[2 * nx + 2, 9 * nx + 3]] = 1.0          # two deposits in the training half (cols < 6)
    y[5 * nx + 7] = 1.0                          # a held-out deposit just across the split (col 7)
    pool = np.flatnonzero(cols < 6)
    neg = h2h.sample_negatives_in_pool(y, rows, cols, pool, np.random.default_rng(0), nx=nx, ny=ny, ratio=10_000)
    assert set(neg) <= set(pool) and not y[neg].any()
    train_dep = pool[y[pool] > 0.5]
    cheb = np.max(np.abs(np.stack([rows[neg][:, None] - rows[train_dep][None, :],
                                   cols[neg][:, None] - cols[train_dep][None, :]])), axis=0)
    assert cheb.min() > 2, "a negative sits inside the buffer of a training deposit"
    assert 5 * nx + 5 in set(neg), "a held-out deposit must not shape the training negatives"
