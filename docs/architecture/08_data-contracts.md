# 08 - The two data contracts

## Contract 1 - ingestion (`pipeline/io/contract.py`)

The *bring-your-own-evidence* gate. Validates a **case bundle descriptor**: a co-registered evidence cube (grid +
layers) + a presence-only deposit point pattern + a study-area mask. Required: `case_id, nx, ny, cell_km, n_layers,
n_deposits`. A record is accepted iff it passes; ill-formed records are rejected with a reason (non-positive grid/cell,
n_layers < 1, n_deposits < 1); records that limit what a result can support are flagged (accepted; the flag
travels into the manifest):
**presence-only-tiny** (< 10 known deposits -> a black-box classifier overfits; trust the white-box WofE + the OOD
mask), **single-layer** (no fusion, no CI to test), **synthetic** study area. A tiny `data/examples/cases.csv` passes
it.

## Contract 2 - artifact (`pipeline/core/{trace,manifest}.py`)

The pipeline -> web artifact contract, mirrored exactly by `frontend/src/lib/contract.types.ts` (a drift fails `tsc`).
The **trace** (`prospectmap.trace/v1`) carries the case spec (so the browser regenerates the cube + recomputes WofE
live), the per-layer weights, the posterior summary, the CI diagnostics, the success + prediction-rate capture curves,
the random-vs-spatial-CV inflation gap, the logistic comparison, and the learned-model block. The **manifest**
(`prospectmap.manifest/v3`) records the category, the engine + version, the learned-model ONNX and metrics file of
the case's own lane (synthetic or real), the trace pointer + byte size, the lane/gate verdict, the Contract-1 flags,
the case metrics, and a case-specific `scope` statement (what the case's data are and what the outputs are not; v2
carried one global text for every case, which described the real case as synthetic). A flat
`index.json` inventories every case; `scripts/check_artifacts.py` enforces manifest <-> artifact consistency in CI.
