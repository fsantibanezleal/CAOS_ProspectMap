"""CONTRACT 2, artifact (pipeline -> web). The manifest is the authoritative, versioned record of a baked case: its
category, the engine + version, the shared learned-model ONNX of its lane, the compact per-case trace pointer + byte
size, the lane/gate verdict, the CONTRACT-1 flags, the case metrics and the case's scope statement. The web loads ONLY
manifests + traces + the shared artifacts; frontend/src/lib/contract.types.ts mirrors this schema. The committed
case-results.json (baked by the SAME TS engine the browser runs) IS the real output of the offline lane; the learned
classifier is measured against the white-box WofE posterior under one shared cross-validation protocol."""
from __future__ import annotations

from typing import Any

from .. import __version__
from .trace import TRACE_SCHEMA

MANIFEST_SCHEMA = "prospectmap.manifest/v3"  # v3: the global `honesty` text became the case-specific `scope`
INDEX_SCHEMA = "prospectmap.index/v1"

ENGINE_NOTE = ("Weights-of-Evidence mineral prospectivity mapping: per-layer W+/W-/contrast/studentized-C at the "
               "maximizing-contrast threshold, the posterior log-odds under conditional independence, the "
               "Agterberg-Cheng omnibus CI test + CI ratio, logistic regression (the CI-free generalization), and "
               "validation by success and prediction-rate capture curves and ROC AUC under spatial-block and random "
               "cross-validation, each fold fitted on its training folds only (thresholds, weights and prior). The "
               "same TS engine runs live in the browser and in the offline bake. The learned classifier + the geology "
               "OOD autoencoder (torch->ONNX) run live via onnxruntime-web; the white-box WofE posterior is the "
               "interpretable authority.")

_SCOPE_COMMON = ("Deposit labels are presence-only: a cell without a known deposit may host an undiscovered one, and "
                 "the negatives used by the learned models are sampled, not observed. The held-out skill is "
                 "capture@10% and the ROC AUC under spatial-block cross-validation; the random-versus-spatial gap and "
                 "the conditional-independence test are reported beside them. The outputs support exploration "
                 "targeting; they are not a JORC or NI 43-101 resource estimate.")


def case_scope(case: Any) -> str:
    """The scope statement of one case's manifest: what its data are, and what the outputs are not."""
    kind = str(case.real_or_synthetic)
    if kind.startswith("real"):
        lead = (f"Real open data: {case.name}. {case.n_layers} evidence layers on a {case.nx} x {case.ny} grid of "
                f"about {case.cell_km:.1f} km cells, with {case.n_deposits} deposit cells. The posterior is this "
                "tool's Weights-of-Evidence recomputation, not the model published with the data.")
    elif kind == "analytic control":
        lead = ("Synthetic control case with known ground truth, generated from a seeded specification; it tests "
                "one property of the method, not a geological setting.")
    else:
        lead = ("Synthetic study area with known ground truth, generated from a seeded specification; it contains "
                "no real geology.")
    return f"{lead} {_SCOPE_COMMON}"


# the learned models + metrics file of each lane: the synthetic cases share the 4-feature models; a real case has its
# own 6-feature models (pipeline/real_learned.py). A manifest points at the files that actually describe its case.
LEARNED_METRICS_FILE = {"synthetic": "pm-learned.json", "real": "pm-learned-real.json"}
_MODEL_FILES = {
    "synthetic": ("mpm-classifier.onnx", "geology-ood.onnx"),
    "real": ("mpm-classifier-real.onnx", "geology-ood-real.onnx"),
}


def shared_artifacts(source: str = "synthetic") -> dict:
    clf_file, ood_file = _MODEL_FILES[source]
    return {
        "models": [
            {"id": "mpm-classifier", "file": clf_file, "opset": 17, "kind": "presence-only prospectivity MLP"},
            {"id": "geology-ood", "file": ood_file, "opset": 17, "kind": "geology novelty autoencoder"},
        ],
        "learned_metrics": LEARNED_METRICS_FILE[source],
        "case_results": "case-results.json",
    }


def build_case_manifest(*, case: Any, seed: int, artifact_rel: str, trace_bytes: int,
                        gate: dict, flags: list[dict], metrics: dict, source: str = "synthetic") -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "case_id": case.id,
        "name": case.name,
        "category": case.category,
        "real_or_synthetic": case.real_or_synthetic,
        "expected_band": case.expected_band,
        "validation_anchor": case.validation_anchor,
        "engine": {"package": "pipeline", "version": __version__, "model": ENGINE_NOTE},
        "seed": seed,
        "shared": shared_artifacts(source),
        "artifact": {"path": artifact_rel, "format": "json", "trace_schema": TRACE_SCHEMA, "bytes": trace_bytes},
        "lane": gate["lane"],
        "gate": gate,
        "flags": flags,
        "metrics": metrics,
        "scope": case_scope(case),
    }


def build_index(entries: list[dict]) -> dict:
    return {
        "schema": INDEX_SCHEMA,
        "engine_version": __version__,
        "n_cases": len(entries),
        "cases": sorted(entries, key=lambda e: e["case_id"]),
    }
