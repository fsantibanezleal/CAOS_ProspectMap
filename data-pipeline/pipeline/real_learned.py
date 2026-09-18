"""Train ProspectMap's two learned models ON THE REAL cube (the US Midcontinent MVT lane), so the
What-if (MLP) and Anomaly (AE) tabs are honest learned tools on real data, not the synthetic-trained
4-feature models (which have a different feature space and MUST NOT be silently applied here).

  1. mpm-classifier-real : a small MLP over the 6 real evidence features -> P(deposit). Presence-only
     labels (positives = real Pb-Zn deposit cells; negatives are SAMPLED, distance-buffered from the
     positives, never observed).
  2. geology-ood-real : an undercomplete autoencoder over the standardized 6-feature stack; the ONNX
     returns the per-cell reconstruction MSE (out 'xr' [N,1]) = the "outside the trained envelope"
     anomaly score. Feature standardization is baked into the graph so the browser feeds raw cube
     values (the SAME [0,1] arrays it renders).

Evaluation, recorded in data/derived/pm-learned-real.json (schema prospectmap.learned/v3):

  - classifier.spatial_cv / random_cv: the MLP and the white-box WofE under ONE protocol, the one behind the WofE
    cross-validation AUCs of the committed bake (case-results.json): the engine's folds, each model fitted on the
    training folds' cells only (for WofE: thresholds, weights and prior), every map cell scored once while held out,
    the held-out scores pooled into one rank ROC AUC
    (pipeline/model/head_to_head.py). The engine's held-out WofE posterior and folds come from
    science/real_wofe_oof.mjs; the run stops if the AUCs re-derived from that export differ from the bake. `winner`
    compares only these two like-for-like values. Beside them, `lr_roc_auc` is the engine's out-of-fold logistic
    regression and `distance_null_roc_auc` the engine's distance-to-known-deposit baseline under the same folds (the
    ranking skill proximity alone reaches; the spatial folds interleave blocks): references, never contestants.
  - classifier.nocv: the WofE AUC without cross-validation (weights fitted on every deposit, scored on the same
    cells). A fitting AUC, kept apart from every cross-validated value (GitHub issue #41).
  - classifier.labelled_sample_cv: the MLP-only CV on its labelled sample (mean of per-fold AUCs, shuffled blocks);
    a different cell set with no WofE counterpart, kept for continuity with the published technical report.

I/O mirrors the synthetic models exactly so ort.ts can load either: classifier in 'x'[N,6] out 'p'[N,1];
OOD in 'x'[N,6] out 'xr'[N,1]. Run (isolated venv), after the engine export:
    (frontend/)      node --import tsx ../data-pipeline/pipeline/science/real_wofe_oof.mjs
    (data-pipeline/) ../.venv-precompute/Scripts/python.exe -m pipeline.real_learned
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model.head_to_head import (
    LEARNED_REAL_SCHEMA,
    build_classifier_block,
    check_oof_matches_bake,
    distance_null_aucs,
    load_wofe_oof,
    reference_aucs,
    rank_auc,
    sample_negatives_in_pool,
)

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
RAW = ROOT / "data" / "raw"
CASE_ID = "REAL-USMVT"
CUBE = DERIVED / CASE_ID / "cube.json"
CASE_RESULTS = DERIVED / "case-results.json"
WOFE_OOF = RAW / f"{CASE_ID}-wofe-oof.json"
FEATURES = ["mag", "grav", "lab", "satgrav", "faultprox", "marginprox"]
SEED = 17
BLOCK = 20  # spatial block side in cells (mirrors the engine's spatialBlockFolds default)
K = 5

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def load_cube():
    d = json.loads(CUBE.read_text(encoding="utf-8"))
    nx, ny = d["nx"], d["ny"]
    n = nx * ny
    byid = {L["id"]: L for L in d["layers"]}
    X = np.zeros((n, len(FEATURES)), dtype="float64")
    finite = np.ones(n, dtype=bool)
    for j, fid in enumerate(FEATURES):
        vals = byid[fid]["values"]
        col = np.array([np.nan if v is None else v for v in vals], dtype="float64")
        finite &= np.isfinite(col)
        col = np.nan_to_num(col, nan=0.0)  # missing -> 0 (matches the browser feed)
        X[:, j] = col
    y = np.zeros(n, dtype="float64")
    y[np.array(d["depositIdx"], dtype=int)] = 1.0
    rows = np.arange(n) // nx
    cols = np.arange(n) % nx
    return X, y, nx, ny, rows, cols, finite


def sample_negatives(y, rows, cols, ratio=3, buffer_cells=2):
    """Distance-buffered presence-only negatives: cells at least `buffer_cells` away from any positive."""
    pos = np.flatnonzero(y > 0.5)
    pr, pc = rows[pos], cols[pos]
    cand = np.flatnonzero(y < 0.5)
    # a coarse buffer: reject candidates within buffer of a positive (grid Chebyshev distance)
    posset = set(zip(pr.tolist(), pc.tolist()))
    keep = []
    for i in cand:
        r, c = rows[i], cols[i]
        near = any((r + dr, c + dc) in posset for dr in range(-buffer_cells, buffer_cells + 1)
                   for dc in range(-buffer_cells, buffer_cells + 1))
        if not near:
            keep.append(i)
    keep = np.array(keep)
    n_neg = min(len(keep), ratio * len(pos))
    neg = rng.choice(keep, size=n_neg, replace=False)
    return pos, neg


def spatial_folds(rows, cols, block=BLOCK, k=K):
    """Shuffled block folds for the labelled-sample CV only (NOT the engine's blockId % k)."""
    br = rows // block
    bc = cols // block
    block_id = br * (cols.max() // block + 2) + bc
    uniq = np.unique(block_id)
    rng.shuffle(uniq)
    fold_of_block = {b: (i % k) for i, b in enumerate(uniq)}
    return np.array([fold_of_block[b] for b in block_id])


class MLP(nn.Module):
    def __init__(self, mean, std, d_in=6, h=16):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        z = (x - self.mean) / self.std
        return torch.sigmoid(self.net(z))


class AE(nn.Module):
    def __init__(self, mean, std, d_in=6, latent=3):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.enc = nn.Sequential(nn.Linear(d_in, 8), nn.ReLU(), nn.Linear(8, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 8), nn.ReLU(), nn.Linear(8, d_in))

    def forward(self, x):
        z = (x - self.mean) / self.std
        xr = self.dec(self.enc(z))
        mse = ((xr - z) ** 2).mean(dim=1, keepdim=True)  # per-row reconstruction MSE (the anomaly score)
        return mse


def roc_auc(scores, labels):
    """Trapezoidal ROC AUC without tie averaging: the estimator of the labelled-sample CV only. The head-to-head uses
    head_to_head.rank_auc, the engine's estimator."""
    order = np.argsort(-scores)
    lab = labels[order]
    P = lab.sum()
    N = len(lab) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    tpr = tp / P
    fpr = fp / N
    return float(np.trapezoid(tpr, fpr))


def train_mlp(Xtr, ytr):
    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-6
    model = MLP(mean, std)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32)
    loss_fn = nn.BCELoss(reduction="none")
    for _ in range(400):
        opt.zero_grad()
        p = model(xt)
        wt = torch.where(yt > 0.5, w, torch.ones_like(w))
        loss = (loss_fn(p, yt) * wt).mean()
        loss.backward()
        opt.step()
    return model


def cv_auc(X, y, folds):
    """Labelled-sample CV: the mean of the per-fold AUCs over the labelled rows of each held-out fold."""
    aucs = []
    for f in range(K):
        tr = folds != f
        te = folds == f
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        m = train_mlp(X[tr], y[tr])
        with torch.no_grad():
            s = m(torch.tensor(X[te], dtype=torch.float32)).numpy().ravel()
        aucs.append(roc_auc(s, y[te]))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def heldout_mlp_scores(X, y, rows, cols, nx, ny, cells, folds, k):
    """The MLP under the head-to-head protocol: for each fold f, train on the training folds' deposits plus
    distance-buffered negatives sampled from the training folds only, then score every map cell of fold f (missing
    values fed as 0, as the browser does). Returns one held-out score per entry of `cells`."""
    neg_rng = np.random.default_rng(SEED)  # independent of the module rng, so this block never shifts the others
    scores = np.full(len(cells), np.nan)
    for f in range(k):
        held = folds == f
        train_pool = cells[~held]
        neg = sample_negatives_in_pool(y, rows, cols, train_pool, neg_rng, nx=nx, ny=ny)
        pos = train_pool[y[train_pool] > 0.5]
        idx = np.concatenate([pos, neg])
        torch.manual_seed(SEED)  # the same initialisation for every fold, whatever ran before
        model = train_mlp(X[idx], y[idx])
        with torch.no_grad():
            scores[held] = model(torch.tensor(X[cells[held]], dtype=torch.float32)).numpy().ravel()
    return scores


def export_onnx(model, path, out_name):
    model.eval()
    dummy = torch.zeros(1, len(FEATURES), dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["x"], output_names=[out_name],
        dynamic_axes={"x": {0: "batch"}, out_name: {0: "batch"}},
        opset_version=17,
    )


def main():
    X, y, nx, ny, rows, cols, finite = load_cube()
    pos, neg = sample_negatives(y, rows, cols)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    Xtr, ytr = X[idx], y[idx]
    tr_rows, tr_cols = rows[idx], cols[idx]

    # MLP-only CV on its labelled sample: spatial (shuffled blocks) vs random folds, mean of per-fold AUCs
    sfolds = spatial_folds(tr_rows, tr_cols)
    rfolds = rng.integers(0, K, size=len(idx))
    ls_spatial = cv_auc(Xtr, ytr, sfolds)
    ls_random = cv_auc(Xtr, ytr, rfolds)

    # final classifier on all labelled data -> ONNX
    clf = train_mlp(Xtr, ytr)
    export_onnx(clf, DERIVED / "mpm-classifier-real.onnx", "p")

    # OOD autoencoder on the background (non-deposit) envelope -> per-row MSE ONNX
    bg = X[(y < 0.5) & finite]
    mean = bg.mean(0)
    std = bg.std(0) + 1e-6
    ae = AE(mean, std)
    opt = torch.optim.Adam(ae.parameters(), lr=5e-3, weight_decay=1e-5)
    xt = torch.tensor(bg, dtype=torch.float32)
    for _ in range(500):
        opt.zero_grad()
        mse = ae(xt).mean()
        mse.backward()
        opt.step()
    export_onnx(ae, DERIVED / "geology-ood-real.onnx", "xr")
    with torch.no_grad():
        in_mse = ae(torch.tensor(X[finite], dtype=torch.float32)).numpy().ravel()
    threshold = float(np.percentile(in_mse, 95))

    # the head-to-head: the MLP under exactly the protocol of the bake's WofE cross-validation AUCs
    if not WOFE_OOF.exists():
        raise SystemExit(f"missing {WOFE_OOF}: run science/real_wofe_oof.mjs first (from frontend/: "
                         "node --import tsx ../data-pipeline/pipeline/science/real_wofe_oof.mjs)")
    case_result = json.loads(CASE_RESULTS.read_text(encoding="utf-8"))["cases"][CASE_ID]
    oof = load_wofe_oof(WOFE_OOF)
    if oof["case_id"] != CASE_ID or (oof["nx"], oof["ny"]) != (nx, ny) or list(oof["layer_ids"]) != FEATURES:
        raise SystemExit(f"{WOFE_OOF} does not describe this cube ({CASE_ID}, {nx}x{ny}, {FEATURES})")
    if not np.array_equal(np.flatnonzero(y > 0.5), oof["deposit_cells"]):
        raise SystemExit("the engine export and cube.json disagree on the deposit cells")
    wofe_cv = check_oof_matches_bake(oof, case_result)
    null_cv = distance_null_aucs(oof)  # the proximity-only reference under the same folds
    lr_cv = reference_aucs(oof, "lr")  # the engine's out-of-fold logistic regression, the CI-free reference
    cells = oof["cells"]
    labels = y[cells]
    mlp_cv = {}
    for scheme in ("spatial", "random"):
        held = heldout_mlp_scores(X, y, rows, cols, nx, ny, cells, oof[scheme]["folds"], oof["k"])
        mlp_cv[scheme] = rank_auc(held, labels)

    classifier = build_classifier_block(
        case_result=case_result, wofe_cv=wofe_cv, mlp_cv=mlp_cv,
        labelled={"spatial": ls_spatial, "random": ls_random, "n_pos": len(pos), "n_neg": len(neg)},
        k=oof["k"], block_cells=oof["block_cells"], random_seed=oof["random_seed"],
        null_cv=null_cv, null_scale_cells=oof["null_scale_cells"], lr_cv=lr_cv,
    )
    out = {
        "schema": LEARNED_REAL_SCHEMA,
        "case_id": CASE_ID,
        "classifier": classifier,
        "ood": {"auc": None, "nEval": int(finite.sum()), "threshold": round(threshold, 4)},
        "scope": (
            "Trained on the real US Midcontinent MVT cube (6 real evidence features), not the synthetic "
            "4-feature models. Deposit labels are presence-only; negatives are sampled (distance-buffered), not "
            "observed. The MLP and the white-box WofE are compared under one protocol: the engine's spatial-block "
            "folds, each model fitted on the training folds' cells only (for WofE: thresholds, weights and prior), "
            "every map cell scored once while held out, the held-out scores pooled into one ROC AUC; the engine's "
            "random folds give the random-CV values and the inflation gap. The WofE AUC without cross-validation "
            "(nocv) is a fitting number and is not compared with a cross-validated value. The labelled-sample CV is "
            "an MLP-only measurement on a different cell set, with no WofE counterpart. MVT occurrences are strongly "
            "clustered and the engine's spatial folds interleave blocks, so every held-out block borders training "
            "blocks: the distance-to-known-deposit baseline scored under the same folds (distance_null_roc_auc) is "
            "the ranking skill proximity alone reaches, and a model must beat it to claim it learned geology. The "
            "contiguous-fold head-to-head in pu-conformal.json is the stricter transfer test. The random-CV values "
            "are inflated by spatial autocorrelation. The OOD AE flags cells outside the labelled geology envelope."
        ),
    }
    # LF on every platform, so a re-run is byte-identical on Windows and Linux
    (DERIVED / "pm-learned-real.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    s = classifier["spatial_cv"]
    r = classifier["random_cv"]
    print(f"head-to-head (all {classifier['nEval']} map cells, engine folds, pooled): spatial-CV AUC "
          f"MLP {s['mlp_roc_auc']:.4f} vs WofE {s['wofe_roc_auc']:.4f} ({s['winner']}), LR {s['lr_roc_auc']:.4f}, "
          f"distance null "
          f"{s['distance_null_roc_auc']:.4f}; random-CV MLP {r['mlp_roc_auc']:.4f} vs WofE {r['wofe_roc_auc']:.4f}, "
          f"distance null {r['distance_null_roc_auc']:.4f}; WofE without CV {classifier['nocv']['wofe_roc_auc']:.4f}")
    print(f"labelled-sample CV (MLP only): spatial {ls_spatial:.4f} random {ls_random:.4f}; "
          f"OOD p95 threshold={threshold:.4f}")
    print("wrote mpm-classifier-real.onnx, geology-ood-real.onnx, pm-learned-real.json")


if __name__ == "__main__":
    main()
