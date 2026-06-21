# ProspectMap - repository structure

Instantiated from the CAOS product-repo archetype ([ADR-0057](docs/architecture/01_overview.md)). The **frozen base**
(layout, the two contracts, the staged pipeline, the lane gate, the manifest/trace, CI guards) is never re-litigated;
the **per-product surface** is the WofE engine + the visualisations + the cases + content.

```
CAOS_ProspectMap/
+- README.md . CHANGELOG.md . STRUCTURE.md . LICENSE . LICENSES.md . ATTRIBUTION.md
+- pyproject.toml . .env.example . .gitignore . .gitattributes
+- requirements*.txt . data-pipeline/requirements*.txt (incl. requirements-precompute.txt: torch+onnx)
+- scripts/            setup . precompute . smoke . dev (.sh + .ps1)
+- data-pipeline/
|  +- pmlab/                          # the two contracts + the staged pipeline (the WofE engine itself is TS, below)
|     +- __init__.py (version) . pipeline.py (orchestrator+CLI, numpy-light + --retrain) . registry.py
|     +- io/     contract.py (CONTRACT 1: a case bundle) . schema.py . formats.py
|     +- core/   gate.py (live/precompute gate) . trace.py + manifest.py (CONTRACT 2) . rng.py
|     +- model/  learned.py (MPM_FEATURES - the SOURCE OF TRUTH the SPA reproduces)
|     +- cases/  mpm_cases.py (the 10 K-/D-/C- cases, mirroring frontend/src/mpm/cases.ts)
|     +- stages/ preprocess . feature_extraction . train . infer . evaluate . export (thin over the science)
|     +- science/  bake_cases.mjs . gen_train.mjs . eval_mpm.mjs (Node+tsx, the SAME TS engine) . train_mpm.py (torch -> ONNX)
|     +- live.py  (dormant - the live lane is TypeScript, not Pyodide)
+- data/
|  +- examples/  cases.csv (a tiny committed CONTRACT-1 sample)
|  +- derived/   case-results.json + per-case <case>/trace.json + manifests/ + mpm-classifier.onnx + geology-ood.onnx + pm-learned.json  (committed)
|  +- raw/       (git-ignored - regenerable training rows)
+- frontend/
|  +- src/mpm/   THE ENGINE: wofe . binarize . ci . logreg . validate . cv . synth . analyze . cases . types . index
|  +- src/pages/ Tool (App) . Introduction . Methodology . Implementation . Experiments . Benchmark
|  +- src/viz/   MapView (canvas raster) . CurveChart + UPlotChart (uPlot)
|  +- src/lib/   contract.types.ts (CONTRACT 2 mirror) . artifacts.ts . ort.ts (the 2 models) . learned.ts
|  +- public/svg/tech/  the 5 themed Architecture-modal SVGs (ADR-0058)
|  +- src/architecture.ts  the i Architecture modal config (ADR-0058)
|  +- test/      mpm.test.ts (11 oracles)   (node:test + tsx)
|  +- copy-data.mjs . vite.config.ts . package.json
+- app/           (dormant FastAPI - activate only on an ADR-0002 trigger)
+- docs/          the navigable wiki (architecture . frameworks . cases . guides)
+- .github/workflows/  ci.yml (python + frontend) . deploy-pages.yml
```

## The lanes

| Lane | Where | Deps |
|---|---|---|
| **Live (client)** | `frontend/src/mpm/` (WofE + CI + logistic + validation) + onnxruntime-web (the 2 models) | web npm |
| **Offline (precompute)** | `pmlab/science/` (Node bake of the TS engine + torch training) | `requirements-precompute.txt` |
| **Replay (light)** | `pmlab.pipeline` reshapes the committed bake -> traces/manifests | `data-pipeline/requirements.txt` (numpy) |
| **API** | `app/` | dormant |
