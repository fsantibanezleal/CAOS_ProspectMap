# data-pipeline/, the offline pipeline (`pmlab`)

`pmlab` is ProspectMap's offline pipeline package: the two data contracts + the staged pipeline + the lane
gate. The WofE/CI/logistic/validation **algorithm truth is the TypeScript engine** in `frontend/src/mpm/` , 
the bake runs the SAME TS engine via tsx (no Python re-port). Its own venv: **`.venv-pipeline`**
(the heavy `--retrain` lane adds torch via `.venv-precompute`, local-only).

## Layout (the package lives directly under `data-pipeline/`)
- `pmlab/pipeline.py`, orchestrator + CLI (`python -m pmlab.pipeline [all|<case>] [--seed N]`)
- `pmlab/registry.py`, cases grouped by CATEGORY · `pmlab/live.py`, Pyodide live entrypoint
- `pmlab/io/`, `contract.py` (**CONTRACT 1**) · `formats.py` (standard readers/writers) · `schema.py` (types)
- `pmlab/core/`, `rng.py` (seeded determinism) · `trace.py` · `manifest.py` (**CONTRACT 2**) · `gate.py`
- `pmlab/model/`, `learned.py`: the feature contracts for the two learned models (shared by the offline
  trainer and the in-browser inference)
- `pmlab/science/`, the two-language bake + learned lane (`bake_cases.mjs`, `gen_train.mjs`, `train_mpm.py`,
  `eval_mpm.mjs`)
- `pmlab/stages/`, `preprocess → feature_extraction → train → infer → evaluate → export`
- `pmlab/cases/`, documented cases

Setup + run: `scripts/setup.{sh,ps1}` then `scripts/precompute.{sh,ps1}`. See
[../docs/architecture/05_precompute-pipeline.md](../docs/architecture/05_precompute-pipeline.md).
