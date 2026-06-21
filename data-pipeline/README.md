# data-pipeline/ — the offline engine (`pmlab`)

Rename `pmlab` → `<slug>lab` per product. The **single source of physics/algorithm truth**; `frontend/` and
`app/` consume it, never re-implement it. Its own venv: **`.venv-pipeline`** (heavy SOTA engines, local-only).

## Layout (the package lives directly under `data-pipeline/`)
- `pmlab/pipeline.py` — orchestrator + CLI (`python -m pmlab.pipeline [all|<case>] [--seed N]`)
- `pmlab/registry.py` — cases grouped by CATEGORY · `pmlab/live.py` — Pyodide live entrypoint
- `pmlab/io/` — `contract.py` (**CONTRACT 1**) · `formats.py` (standard readers/writers) · `schema.py` (types)
- `pmlab/core/` — `rng.py` (seeded determinism) · `trace.py` · `manifest.py` (**CONTRACT 2**) · `gate.py`
- `pmlab/model/` — the shared pure-Python core (Pyodide-safe); EXAMPLE = SIR
- `pmlab/stages/` — `preprocess → feature_extraction → train → infer → evaluate → export`
- `pmlab/cases/` — documented cases

Setup + run: `scripts/setup.{sh,ps1}` then `scripts/precompute.{sh,ps1}`. See
[../docs/architecture/05_precompute-pipeline.md](../docs/architecture/05_precompute-pipeline.md).
