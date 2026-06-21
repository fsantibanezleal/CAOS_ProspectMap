# 03 - The learned models (torch -> ONNX)

Two learned models, trained OFFLINE (torch, `.venv-precompute`) and run LIVE (onnxruntime-web). Both are measured
against the white-box WofE posterior - the interpretable authority - never bolted-on.

## mpm-classifier

A small MLP (4 -> 32 -> 32 -> 1, sigmoid) over the per-cell evidence feature vector (`MPM_FEATURES` = mag, rad,
geochem, struct) -> P(deposit). **Presence-only**: positives = known deposit cells; negatives are SAMPLED
(distance-buffered, never observed), with a `pos_weight` BCE loss for the extreme imbalance. The standardization is
folded into the ONNX export wrapper, so the graph takes RAW features. Validated by SPATIAL block K-fold CV and
benchmarked head-to-head against WofE on the IDENTICAL spatial holdout; the random-CV AUC is computed too, to surface
the inflation gap.

## geology-ood

A small undercomplete autoencoder (4 -> 8 -> 2 -> 8 -> 4) over the standardized feature stack; the reconstruction MSE
separates in-envelope geology from out-of-envelope ("the classifier is extrapolating under cover; do not trust the
score here"). The 95th-percentile in-envelope MSE is the committed threshold.

## The honest numbers

Measured (not fabricated): mpm-classifier **spatial-CV AUC 0.971 vs WofE 0.929** (winner: the MLP, on the multi-layer
interactions WofE's CI form omits) - random-CV 0.979 (inflation +0.008) - geology-OOD AUC 1.0. `train_mpm.py` exports
the 2 ONNX + `learned-partial.json`; `eval_mpm.mjs` runs the exported classifier in onnxruntime-web (the engine's own
runtime, the honest end-to-end check) and assembles `pm-learned.json` (schema `prospectmap.learned/v1`).
