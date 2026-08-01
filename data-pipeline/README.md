# data-pipeline/, the offline pipeline (`pipeline`)

`pipeline` is ProspectMap's offline pipeline package: the two data contracts + the staged pipeline + the lane
gate. The WofE/CI/logistic/validation **algorithm truth is the TypeScript engine** in `frontend/src/mpm/` , 
the bake runs the SAME TS engine via tsx (no Python re-port). Its own venv: **`.venv-pipeline`**
(the heavy `--retrain` lane adds torch via `.venv-precompute`, local-only).

## Layout (the package lives directly under `data-pipeline/`)
- `pipeline/pipeline.py`, orchestrator + CLI (`python data-pipeline/run.py [all|<case>] [--seed N]`)
- `pipeline/registry.py`, cases grouped by CATEGORY · `pipeline/live.py`, Pyodide live entrypoint
- `pipeline/io/`, `contract.py` (**CONTRACT 1**) · `formats.py` (standard readers/writers) · `schema.py` (types)
- `pipeline/core/`, `rng.py` (seeded determinism) · `trace.py` · `manifest.py` (**CONTRACT 2**) · `gate.py`
- `pipeline/model/`, `learned.py`: the feature contracts for the two learned models (shared by the offline
  trainer and the in-browser inference)
- `pipeline/science/`, the two-language bake + learned lane (`bake_cases.mjs`, `gen_train.mjs`, `train_mpm.py`,
  `eval_mpm.mjs`)
- `pipeline/stages/`, `preprocess → feature_extraction → train → infer → evaluate → export`
- `pipeline/cases/`, documented cases

Setup + run: `scripts/setup.{sh,ps1}` then `scripts/precompute.{sh,ps1}`. See
[../docs/architecture/05_precompute-pipeline.md](../docs/architecture/05_precompute-pipeline.md).
