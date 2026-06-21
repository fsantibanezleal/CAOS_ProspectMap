# 01 - Overview

ProspectMap is instantiated on the CAOS **product-repo archetype** (ADR-0057). The archetype's frozen base is the
layout, the two data contracts, the staged pipeline, the live/precompute gate, the manifest/trace, and the CI guards.
The per-product surface is the **science engine** (Weights of Evidence), the visualisations, the cases, and the content.

## The lanes

| Lane | Where | What |
|---|---|---|
| **Live (client)** | `frontend/src/mpm/` + onnxruntime-web | the WofE/CI/logistic/validation engine recomputes the posterior raster on every control; the 2 ONNX run the learned probe + the OOD mask |
| **Offline (precompute)** | `pmlab/science/*.mjs` + `train_mpm.py` | the Node bake runs the SAME TS engine over the cases -> `case-results.json`; the heavy `--retrain` lane (torch) trains the 2 models -> ONNX |
| **Replay (light)** | `pmlab.pipeline` (numpy) | reshapes the committed bake into per-case CONTRACT-2 traces + manifests; no torch/Node, so CI is fast |
| **API** | `app/` | dormant (activate only on an ADR-0002 trigger) |

## The flow

1. The cases are SYNTHETIC study-area SPECs (`frontend/src/mpm/cases.ts`). The Node bake (`science/bake_cases.mjs`)
   regenerates each cube deterministically and runs `analyze.ts` (the whole WofE pipeline + the CI diagnostics + the
   capture curves + the random-vs-spatial-CV inflation gap + the logistic comparison) -> `data/derived/case-results.json`.
2. The light pipeline (`pmlab.pipeline all`) applies CONTRACT 1 to the case descriptors, reshapes `case-results.json`
   into per-case `trace.json` + `manifests/*.json` (CONTRACT 2), and runs the lane gate.
3. The SPA reads the manifests/traces + the shared artifacts (the 2 ONNX + `pm-learned.json`), and ALSO recomputes the
   WofE posterior LIVE from the regenerated cube (the case SPEC is carried in the trace) so the App is fully reactive.

## Frozen vs rework

Frozen (never re-litigated): the directory layout, the two contracts, the staged pipeline, the gate, the
manifest/trace, the CI guards. Reworked per product: `io/contract.py` (the case-bundle gate), `core/{trace,manifest}`
(the schemas), `cases/`, `model/learned.py` (the feature contract), the `frontend/src/mpm/` engine, the pages + viz.
