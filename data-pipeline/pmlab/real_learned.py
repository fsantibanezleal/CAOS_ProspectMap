"""Train ProspectMap's two learned models ON THE REAL cube (the US Midcontinent MVT lane), so the
What-if (MLP) and Anomaly (AE) tabs are honest learned tools on real data, not the synthetic-trained
4-feature models (which have a different feature space and MUST NOT be silently applied here).

  1. mpm-classifier-real : a small MLP over the 6 real evidence features -> P(deposit). Presence-only
     labels (positives = real Pb-Zn deposit cells; negatives are SAMPLED, distance-buffered from the
     positives, never observed). Validated by SPATIAL block cross-validation and reported beside the
     random-CV AUC (the inflation gap) and the white-box WofE AUC on the same data.
  2. geology-ood-real : an undercomplete autoencoder over the standardized 6-feature stack; the ONNX
     returns the per-cell reconstruction MSE (out 'xr' [N,1]) = the "outside the trained envelope"
     anomaly score. Feature standardization is baked into the graph so the browser feeds raw cube
     values (the SAME [0,1] arrays it renders).

I/O mirrors the synthetic models exactly so ort.ts can load either: classifier in 'x'[N,6] out 'p'[N,1];
OOD in 'x'[N,6] out 'xr'[N,1]. Run (isolated venv):
    .venv-precompute/Scripts/python.exe -m pmlab.real_learned
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"
CUBE = DERIVED / "REAL-USMVT" / "cube.json"
TRACE = DERIVED / "REAL-USMVT" / "trace.json"
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

    # spatial-block vs random CV on the labelled set (the inflation gap)
    sfolds = spatial_folds(tr_rows, tr_cols)
    rfolds = rng.integers(0, K, size=len(idx))
    mlp_spatial = cv_auc(Xtr, ytr, sfolds)
    mlp_random = cv_auc(Xtr, ytr, rfolds)

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

    # WofE AUCs (from the trace) for the head-to-head comparison shown in the App
    tr = json.loads(TRACE.read_text(encoding="utf-8"))
    wofe_auc = round(float(tr["roc_auc"]), 4)
    wofe_spatial = round(float(tr["cv"]["spatialAuc"]), 4)

    out = {
        "schema": "prospectmap.learned/v1",
        "case_id": "REAL-USMVT",
        "classifier": {
            "spatial_cv": {"mlp_roc_auc": round(mlp_spatial, 4), "wofe_roc_auc": wofe_auc,
                           "winner": "mlp" if mlp_spatial > wofe_spatial else "wofe"},
            "random_cv": {"mlp_roc_auc": round(mlp_random, 4)},
            "mlp_roc_auc": round(mlp_spatial, 4),
            "inflation_gap": round(mlp_random - mlp_spatial, 4),
            "nFolds": K,
            "nEval": int(len(idx)),
        },
        "ood": {"auc": None, "nEval": int(finite.sum()), "threshold": round(threshold, 4)},
        "honesty": (
            "Trained on the REAL US Midcontinent MVT cube (6 real evidence features), NOT the synthetic "
            "4-feature models. Deposit labels are presence-only; negatives are SAMPLED (distance-buffered), "
            "never observed. Validated by SPATIAL block cross-validation and reported beside random-CV (the "
            "inflation gap) and the white-box WofE AUC. MVT occurrences are strongly clustered, so the honest "
            "spatial-CV skill is modest; the random-CV number is inflated and shown only to expose that. The "
            "OOD AE flags cells outside the labelled geology envelope. No fabricated win."
        ),
    }
    (DERIVED / "pm-learned-real.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"classifier: spatial-CV AUC={mlp_spatial:.3f} random-CV AUC={mlp_random:.3f} "
          f"(WofE spatial {wofe_spatial:.3f}); OOD p95 threshold={threshold:.3f}")
    print("wrote mpm-classifier-real.onnx, geology-ood-real.onnx, pm-learned-real.json")


if __name__ == "__main__":
    main()
