"""HEAVY lane (local-only), train ProspectMap's two learned models and export them to ONNX. Run inside the
.venv-precompute (torch) AFTER gen_train.mjs has written data/raw/{mpm-train,mpm-eval}.json:

    python data-pipeline/pmlab/science/train_mpm.py

1. mpm-classifier, a small MLP over the per-cell evidence feature vector -> P(deposit). Presence-only labels
   (positives = known deposit cells; negatives are SAMPLED, distance-buffered, NEVER observed). It is benchmarked
   head-to-head against the white-box WofE posterior on the IDENTICAL SPATIAL holdout (block K-fold); the random-CV
   AUC is computed too, to surface the inflation gap. The standardisation is folded into the export wrapper, so the
   ONNX takes RAW features and returns P(deposit). The white-box WofE is the interpretable authority; no fabricated win.
2. geology-ood, a small autoencoder over the standardized feature vector; the reconstruction MSE separates
   in-envelope geology from out-of-envelope (the "the classifier is extrapolating under cover" flag). AUC reported here.

Outputs: data/derived/{mpm-classifier.onnx, geology-ood.onnx} + data/raw/learned-partial.json (eval_mpm.mjs assembles
the final data/derived/pm-learned.json). Deterministic (seeded).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
rng = np.random.default_rng(0)


def _auc(label: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the rank statistic (no sklearn). label 1 = positive."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = float((label > 0.5).sum())
    n_neg = float(len(label) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[label > 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class MpmMLP(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)  # logit


def _train_one(X: np.ndarray, y: np.ndarray, mu: np.ndarray, sd: np.ndarray, epochs: int = 80) -> MpmMLP:
    """train a classifier on standardized X (mu,sd computed on the TRAINING rows only -> no leakage)."""
    Xs = (X - mu) / sd
    net = MpmMLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n_pos = float((y > 0.5).sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt = torch.from_numpy(Xs.astype(np.float32))
    Yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), bs):
            sel = perm[b:b + bs]
            opt.zero_grad()
            loss = lossf(net(Xt[sel]), Yt[sel])
            loss.backward()
            opt.step()
    net.eval()
    return net


def _predict(net: MpmMLP, X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(((X - mu) / sd).astype(np.float32))
        return torch.sigmoid(net(t)).numpy().ravel()


def _cv_auc(X: np.ndarray, y: np.ndarray, fold: np.ndarray, k: int) -> float:
    """out-of-fold AUC: for each fold f, train on fold != f, predict the held-out fold; AUC over the pooled OOF preds."""
    oof = np.full(len(y), np.nan)
    for f in range(k):
        tr = fold != f
        te = fold == f
        if tr.sum() == 0 or te.sum() == 0 or (y[tr] > 0.5).sum() == 0:
            continue
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-9
        net = _train_one(X[tr], y[tr], mu, sd)
        oof[te] = _predict(net, X[te], mu, sd)
    m = ~np.isnan(oof)
    return _auc(y[m], oof[m])


class OODAE(nn.Module):
    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 8), nn.ReLU(), nn.Linear(8, 2), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ood(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, in_eval: np.ndarray, ood: np.ndarray) -> dict:
    Xs = (X - mu) / sd
    net = OODAE(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt = torch.from_numpy(Xs.astype(np.float32))
    bs = 256
    for _ in range(150):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), bs):
            sel = perm[b:b + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt[sel]), Xt[sel])
            loss.backward()
            opt.step()
    net.eval()

    def mse(arr: np.ndarray) -> np.ndarray:
        s = (arr - mu) / sd
        with torch.no_grad():
            t = torch.from_numpy(s.astype(np.float32))
            return ((net(t) - t) ** 2).mean(dim=1).numpy()

    in_scores = mse(in_eval)
    threshold = float(np.percentile(in_scores, 95))
    ood_scores = mse(ood)
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    auc = _auc(labels, scores)

    # export wrapper: RAW features -> standardise -> AE -> the standardized-space reconstruction MSE (the anomaly score
    # itself, [batch, 1]). Computing it INSIDE the ONNX means the browser reads an interpretable, correctly-scaled score
    # directly (the SAME quantity used for the AUC + threshold above), no client-side scaler needed.
    class AEExport(nn.Module):
        def __init__(self, core: OODAE) -> None:
            super().__init__()
            self.core = core
            self.register_buffer("mu_x", torch.from_numpy(mu.astype(np.float32)))
            self.register_buffer("sd_x", torch.from_numpy(sd.astype(np.float32)))

        def forward(self, x):
            xs = (x - self.mu_x) / self.sd_x
            r = self.core(xs)
            return ((r - xs) ** 2).mean(dim=1, keepdim=True)

    return {"model": AEExport(net), "auc": round(auc, 4), "nEval": int(len(scores)), "threshold": round(threshold, 6)}


def _strip_metadata(path: Path) -> None:
    """Remove any machine-specific provenance an ONNX exporter may bake in (node metadata_props / doc_strings can carry
    absolute source paths), keeps the committed ONNX clean (base-integrity guard) and reproducible across machines."""
    import onnx
    m = onnx.load(str(path))
    m.doc_string = ""
    m.graph.doc_string = ""
    for node in m.graph.node:
        del node.metadata_props[:]
        node.doc_string = ""
    onnx.save(m, str(path))


def export_onnx(model: nn.Module, n_in: int, in_name: str, out_name: str, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, n_in)
    torch.onnx.export(model, dummy, str(path), input_names=[in_name], output_names=[out_name],
                      dynamic_axes={in_name: {0: "batch"}, out_name: {0: "batch"}}, opset_version=17)
    _strip_metadata(path)


def main() -> None:
    d = json.loads((RAW / "mpm-train.json").read_text())
    rows = d["rows"]
    k = int(d["k"])
    X = np.asarray([r["feat"] for r in rows], dtype=np.float64)
    y = np.asarray([r["y"] for r in rows], dtype=np.float64)
    s_fold = np.asarray([r["sFold"] for r in rows], dtype=np.int64)
    r_fold = np.asarray([r["rFold"] for r in rows], dtype=np.int64)
    p_wofe = np.asarray([r["pWofe"] for r in rows], dtype=np.float64)

    # the honest head-to-head + the inflation gap
    mlp_spatial = _cv_auc(X, y, s_fold, k)
    mlp_random = _cv_auc(X, y, r_fold, k)
    wm = ~np.isnan(p_wofe)
    wofe_spatial = _auc(y[wm], p_wofe[wm])
    inflation = round(mlp_random - mlp_spatial, 4)
    winner = "mlp" if mlp_spatial > wofe_spatial + 0.005 else ("wofe" if wofe_spatial > mlp_spatial + 0.005 else "tie")

    # the shipped classifier: train on ALL rows (standardisation folded into the export)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-9
    net = _train_one(X, y, mu, sd, epochs=120)

    class MpmExport(nn.Module):
        def __init__(self, core: MpmMLP) -> None:
            super().__init__()
            self.core = core
            self.register_buffer("mu_x", torch.from_numpy(mu.astype(np.float32)))
            self.register_buffer("sd_x", torch.from_numpy(sd.astype(np.float32)))

        def forward(self, x):
            return torch.sigmoid(self.core((x - self.mu_x) / self.sd_x))

    ev = json.loads((RAW / "mpm-eval.json").read_text())
    in_eval = np.asarray(ev["inDist"], dtype=np.float64)
    ood = np.asarray(ev["ood"], dtype=np.float64)
    ae = train_ood(X, mu, sd, in_eval, ood)

    export_onnx(MpmExport(net), X.shape[1], "x", "p", DERIVED / "mpm-classifier.onnx")
    export_onnx(ae["model"], X.shape[1], "x", "xr", DERIVED / "geology-ood.onnx")

    partial = {
        "classifier": {
            "mlp_spatial_auc": round(mlp_spatial, 4), "wofe_spatial_auc": round(wofe_spatial, 4),
            "mlp_random_auc": round(mlp_random, 4), "inflation_gap": inflation, "winner": winner,
            "nFolds": k, "nEval": int(len(y)),
        },
        "ood": {"auc": ae["auc"], "nEval": ae["nEval"], "threshold": ae["threshold"]},
        "scaler": {"mu": mu.ravel().round(6).tolist(), "sd": sd.ravel().round(6).tolist()},
        "honesty": ("Deposit labels are presence-only; negatives are SAMPLED (distance-buffered), never observed. The "
                    "classifier is validated by SPATIAL block cross-validation and benchmarked against the white-box "
                    "WofE posterior on the SAME spatial holdout; random-CV is reported beside it to show the inflation "
                    "gap. The geology autoencoder flags out-of-envelope cells. The white-box WofE is the interpretable "
                    "authority. Reported whichever way the numbers land. No fabricated win."),
    }
    (RAW / "learned-partial.json").write_text(json.dumps(partial, indent=2))
    print(f"mpm-classifier spatial-CV AUC {mlp_spatial:.3f} vs WofE {wofe_spatial:.3f} (winner {winner}) - "
          f"random-CV {mlp_random:.3f} (inflation +{inflation:.3f}) - OOD AUC {ae['auc']}")
    print(f"wrote mpm-classifier.onnx + geology-ood.onnx + learned-partial.json -> {DERIVED} / {RAW}")


if __name__ == "__main__":
    main()
