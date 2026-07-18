# 05 - The precompute pipeline (two-language)

The default lane is numpy-light and reshapes the committed bake; the heavy `--retrain` lane is two-language (Node bake
of the TS engine + torch training).

## The bake (Node + tsx, the same engine)

`pmlab/science/bake_cases.mjs` imports the TS engine and runs `analyzeCase(spec, layerIds)` over each case ->
`data/derived/case-results.json` (schema `prospectmap.case-results/v1`). The cubes are synthetic and regenerated from
each case's spec (committed in case-results), so the artifact stays compact - no raster blobs. Because the bake and the
browser run the identical engine, the live and offline numbers agree by construction.

## The light pipeline (numpy)

`python -m pmlab.pipeline all` (default lane): applies Contract 1 to the case descriptors, reads `case-results.json` +
`pm-learned.json` (when present), builds the per-case `trace.json` + `manifests/*.json` (Contract 2) via
`stages/export.build_replay`, runs the lane gate, and writes the flat `index.json`. No torch / no Node, so CI is fast
and the artifacts regenerate deterministically (byte-identical re-run).

## The heavy lane (--retrain, two-language)

`python -m pmlab.pipeline all --retrain`:
1. `science/bake_cases.mjs` (Node) re-bakes `case-results.json`.
2. `science/gen_train.mjs` (Node) samples presence cells + distance-buffered informed negatives over the terrane/rich
   cubes, with spatial-block + random folds, and the held-out WofE posterior per fold -> `data/raw/mpm-train.json` +
   `mpm-eval.json` (git-ignored, regenerable).
3. `science/train_mpm.py` (torch, `.venv-precompute`) fits the classifier + the OOD-AE -> `mpm-classifier.onnx` +
   `geology-ood.onnx` + `learned-partial.json`.
4. `science/eval_mpm.mjs` (onnxruntime-web in Node) runs the exported classifier in the engine's runtime + assembles
   `data/derived/pm-learned.json`.
