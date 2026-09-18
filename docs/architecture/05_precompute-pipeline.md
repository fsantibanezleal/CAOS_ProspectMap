# 05 - The precompute pipeline (two-language)

The default lane is numpy-light and reshapes the committed bake; the heavy `--retrain` lane is two-language (Node bake
of the TS engine + torch training).

## The bake (Node + tsx, the same engine)

`pipeline/science/bake_cases.mjs` imports the TS engine and runs `analyzeCase(spec, layerIds)` over each case ->
`data/derived/case-results.json` (schema `prospectmap.case-results/v1`). The synthetic cubes are regenerated from each
case's spec (committed in case-results), so the artifact stays compact - no raster blobs; the one real cube is
committed once (`data/derived/REAL-USMVT/cube.json`) and analyzed by `science/bake_real.mjs`. Because the bake and the
browser run the identical engine, the live and offline numbers agree by construction.

## The light pipeline (numpy)

`python data-pipeline/run.py all` (default lane): applies Contract 1 to the case descriptors, reads `case-results.json` +
the learned metrics of each case's own lane (`pm-learned.json` for the synthetic cases, `pm-learned-real.json` for the
real case, when present; a real case never carries the synthetic models' numbers), builds the per-case `trace.json` +
`manifests/*.json` (Contract 2) via
`stages/export.build_replay`, runs the lane gate, and writes the flat `index.json`. No torch / no Node, so CI is fast
and the artifacts regenerate deterministically (byte-identical re-run).

## The heavy lane (--retrain, two-language)

`python data-pipeline/run.py all --retrain`:
1. `science/bake_cases.mjs` (Node) re-bakes `case-results.json`.
2. `science/gen_train.mjs` (Node) samples presence cells + distance-buffered informed negatives over the terrane/rich
   cubes, with spatial-block + random folds, and the held-out WofE posterior per fold -> `data/raw/mpm-train.json` +
   `mpm-eval.json` (git-ignored, regenerable).
3. `science/train_mpm.py` (torch, `.venv-precompute`) fits the classifier + the OOD-AE -> `mpm-classifier.onnx` +
   `geology-ood.onnx` + `learned-partial.json`.
4. `science/eval_mpm.mjs` (onnxruntime-web in Node) runs the exported classifier in the engine's runtime + assembles
   `data/derived/pm-learned.json`.

## The real-data lane (part of `--retrain`, two-language)

The real US-MVT cube has its own 6-feature models. `pipeline/real_usmvt.py` builds the cube, run by hand from the
CMMI files listed in `data/derived/REAL-USMVT/provenance.json` and fetched into the git-ignored
`data/raw/REAL-USMVT/`. ScienceBase answers scripted requests with a browser challenge, so the files are downloaded
through a browser. With the versions pinned in `data-pipeline/requirements-precompute.txt` (rasterio 1.5.0, pyshp
3.1.4, scipy 1.18.0, numpy 2.4.6) the build reproduces the committed `cube.json` byte for byte. After the cube,
`--retrain` runs `science/bake_real.mjs` (its WofE analysis, merged into `case-results.json` after `bake_cases.mjs`
rewrote it with the synthetic cases), then:

1. `science/real_wofe_oof.mjs` (Node, from `frontend/`) runs the engine's cross-validation of the real case with the
   bake's defaults (`spatialBlockFolds` with 20x20-cell blocks, `randomFolds` with seed 17, k = 5, and per fold the
   fully out-of-fold WofE fit of `wofeFoldScoreFn`) and writes the held-out WofE posterior and the folds of every map
   cell to `data/raw/REAL-USMVT-wofe-oof.json` (git-ignored, regenerable). The same export carries the engine's
   out-of-fold logistic regression (`lrFoldScoreFn`) and the distance-to-known-deposit baseline under the same folds.
2. `pipeline/real_learned.py` (torch, `.venv-precompute`, run as `python -m pipeline.real_learned` from
   `data-pipeline/`) trains `mpm-classifier-real.onnx` + `geology-ood-real.onnx` and writes `pm-learned-real.json`.
   It scores the MLP under exactly the protocol of that export: the same folds, each fold fitted on the training
   folds only, every map cell scored once while held out, the held-out scores pooled into one rank ROC AUC
   (`pipeline/model/head_to_head.py`). It re-derives the WofE AUCs from the export with the same estimator and stops
   unless they equal the bake, so `spatial_cv` and `random_cv` hold like-for-like pairs. The WofE AUC without
   cross-validation is written under `nocv`, never under a cross-validation key.
3. `python data-pipeline/run.py all` rebuilds the replay traces + manifests, so the real case carries these metrics.
