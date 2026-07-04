"""CONTRACT 2, artifact (pipeline -> web). The manifest is the authoritative, versioned record of a baked case: its
category, the engine + version, the shared learned-model ONNX, the compact per-case trace pointer + byte size, the
lane/gate verdict, the CONTRACT-1 flags, and the case metrics. The web loads ONLY manifests + traces + the shared
artifacts; frontend/src/lib/contract.types.ts mirrors this schema so a drift fails the build. The committed
case-results.json (baked by the SAME TS engine the browser runs) IS the real output of the offline lane; the learned
classifier is honest, measured against the white-box WofE posterior on the SAME spatial holdout, never a fabricated
win."""
from __future__ import annotations

from typing import Any

from .. import __version__
from .trace import TRACE_SCHEMA

MANIFEST_SCHEMA = "prospectmap.manifest/v2"
INDEX_SCHEMA = "prospectmap.index/v1"

ENGINE_NOTE = ("Weights-of-Evidence mineral prospectivity mapping: per-layer W+/W-/contrast/studentized-C at the "
               "maximizing-contrast threshold, the posterior log-odds under conditional independence, the "
               "Agterberg-Cheng omnibus CI test + CI ratio, logistic regression (the CI-free generalization), and "
               "honest validation (success vs prediction-rate capture curves under SPATIAL cross-validation). The "
               "same TS engine runs live in the browser and in the offline bake. The learned classifier + the geology "
               "OOD autoencoder (torch->ONNX) run live via onnxruntime-web; the white-box WofE posterior is the "
               "interpretable authority.")
HONESTY = ("The study areas are SYNTHETIC (geostatistically-grounded, clearly labelled); C-NEGATIVE, C-CIVIOLATE, "
           "C-RECOVER and C-SATURATE are the known-ground-truth controls. Deposit labels are presence-only (negatives "
           "are sampled, never observed). The default reported metric is capture@10% under SPATIAL cross-validation; "
           "the random-vs-spatial inflation gap and the conditional-independence omnibus test are surfaced, not "
           "hidden. Outputs are exploration target generation, NOT a JORC/NI-43-101 resource estimate. No fabricated "
           "win.")


def shared_artifacts() -> dict:
    return {
        "models": [
            {"id": "mpm-classifier", "file": "mpm-classifier.onnx", "opset": 17, "kind": "presence-only prospectivity MLP"},
            {"id": "geology-ood", "file": "geology-ood.onnx", "opset": 17, "kind": "geology novelty autoencoder"},
        ],
        "learned_metrics": "pm-learned.json",
        "case_results": "case-results.json",
    }


def build_case_manifest(*, case: Any, seed: int, artifact_rel: str, trace_bytes: int,
                        gate: dict, flags: list[dict], metrics: dict) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "case_id": case.id,
        "name": case.name,
        "category": case.category,
        "real_or_synthetic": case.real_or_synthetic,
        "expected_band": case.expected_band,
        "validation_anchor": case.validation_anchor,
        "engine": {"package": "pmlab", "version": __version__, "model": ENGINE_NOTE},
        "seed": seed,
        "shared": shared_artifacts(),
        "artifact": {"path": artifact_rel, "format": "json", "trace_schema": TRACE_SCHEMA, "bytes": trace_bytes},
        "lane": gate["lane"],
        "gate": gate,
        "flags": flags,
        "metrics": metrics,
        "honesty": HONESTY,
    }


def build_index(entries: list[dict]) -> dict:
    return {
        "schema": INDEX_SCHEMA,
        "engine_version": __version__,
        "n_cases": len(entries),
        "cases": sorted(entries, key=lambda e: e["case_id"]),
    }
