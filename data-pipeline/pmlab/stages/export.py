"""Stage 6, export (CONTRACT 2): build the compact per-case trace from the committed bake (case-results.json, baked
by the SAME TS engine the browser runs) + the learned-model metrics (pm-learned.json, when trained), run the lane
gate, and write the manifest. No torch/node, so the contract + replay regenerate deterministically anywhere, and CI
stays fast. The HEAVY export (baking case-results.json + training the ONNX) is done by the preserved science
(pmlab/science/bake_cases.mjs + train_mpm.py), invoked by pipeline.retrain."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.gate import classify_lane
from ..core.manifest import build_case_manifest
from ..core.trace import build_trace
from ..io.formats import write_json

_RUN_MS = 60.0   # a teaching-scale WofE recompute, tens of ms; deterministic gate budget
_RUNTIMES = {"ts-mpm", "onnxruntime-web"}


def _case_metrics(case_result: dict, learned: dict | None) -> dict:
    ci = case_result.get("ci", {}) or {}
    cap = (case_result.get("capture", {}) or {})
    success = cap.get("success", {}) or {}
    pred = cap.get("prediction", {}) or {}
    cv = case_result.get("cv", {}) or {}
    lr = case_result.get("lr", {}) or {}
    m = {
        "roc_auc": float(case_result.get("rocAuc", 0.0)),
        "lr_roc_auc": float(lr.get("rocAuc", 0.0)),
        "capture_at_10_success": float(success.get("captureAt10", 0.0)),
        "capture_at_10_prediction": float(pred.get("captureAt10", 0.0)),
        "ci_ratio": float(ci.get("ciRatio", 0.0)),
        "ci_z": float(ci.get("z", 0.0)),
        "cv_random_auc": float(cv.get("randomAuc", 0.0)),
        "cv_spatial_auc": float(cv.get("spatialAuc", 0.0)),
        "cv_inflation_gap": float(cv.get("inflationGap", 0.0)),
        "n_deposits": int(case_result.get("nDeposits", 0)),
    }
    if learned:
        clf = (learned.get("classifier") or {})
        m["clf_spatial_cv_auc"] = float(clf.get("mlp_roc_auc", 0.0))
        m["ood_auc"] = float((learned.get("ood") or {}).get("auc", 0.0))
    return m


def build_replay(case: Any, *, derived_dir: str, manifests_dir: str,
                 case_results: dict, learned: dict | None, contract_flags: list[dict], seed: int) -> dict:
    cr = case_results["cases"][case.id]
    trace = build_trace(case, case_result=cr, learned=learned)
    artifact_rel = f"{case.id}/trace.json"
    trace_bytes = write_json(Path(derived_dir) / artifact_rel, trace)
    gate = classify_lane(client_side=True, runtimes=_RUNTIMES, run_ms=_RUN_MS, trace_bytes=trace_bytes)
    manifest = build_case_manifest(
        case=case, seed=seed, artifact_rel=artifact_rel, trace_bytes=trace_bytes,
        gate=gate, flags=contract_flags, metrics=_case_metrics(cr, learned),
    )
    write_json(Path(manifests_dir) / f"{case.id}.json", manifest)
    return manifest
