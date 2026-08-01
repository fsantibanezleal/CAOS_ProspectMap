"""The compact per-case TRACE = the web-replay artifact. Part of CONTRACT 2: its shape is mirrored by
frontend/src/lib/contract.types.ts, so a drift fails the web build. Built deterministically from the committed bake
(case-results.json, produced by the SAME TS engine the browser runs) + the learned-model metrics (pm-learned.json,
when present). Carries the case SPEC so the browser can REGENERATE the synthetic cube + recompute WofE LIVE, the
per-layer weights, the combined-posterior summary, the conditional-independence diagnostics, the success +
prediction-rate capture curves, the random-vs-spatial-CV inflation gap, the logistic-regression comparison, and the
learned-model metrics."""
from __future__ import annotations

from typing import Any

TRACE_SCHEMA = "prospectmap.trace/v1"


def _learned_block(learned: dict | None) -> dict:
    if not learned:
        return {"status": "pending-training", "classifier": None, "ood": None}
    return {
        "status": "trained",
        "classifier": learned.get("classifier"),
        "ood": learned.get("ood"),
    }


def build_trace(case: Any, *, case_result: dict, learned: dict | None) -> dict:
    return {
        "schema": TRACE_SCHEMA,
        "case_id": case.id,
        "name": case.name,
        "category": case.category,
        "real_or_synthetic": case.real_or_synthetic,
        "expected_band": case.expected_band,
        "validation_anchor": case.validation_anchor,
        "spec": case_result.get("spec"),
        "layer_ids": case_result.get("layerIds"),
        "n_cells": case_result.get("nCells"),
        "n_deposits": case_result.get("nDeposits"),
        "prior_prob": case_result.get("priorProb"),
        "layers": case_result.get("layers"),
        "posterior_summary": case_result.get("posteriorSummary"),
        "ci": case_result.get("ci"),
        "roc_auc": case_result.get("rocAuc"),
        "capture": case_result.get("capture"),
        "cv": case_result.get("cv"),
        "lr": case_result.get("lr"),
        "learned": _learned_block(learned),
    }
