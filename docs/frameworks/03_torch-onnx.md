# 03 - The learned models (torch -> ONNX)

Two learned models, trained offline (torch, `.venv-precompute`) and run live (onnxruntime-web). Both are measured
against the white-box WofE posterior - the interpretable authority - never bolted-on.

## mpm-classifier

A small MLP (4 -> 32 -> 32 -> 1, sigmoid) over the per-cell evidence feature vector (`MPM_FEATURES` = mag, rad,
geochem, struct) -> P(deposit). **Presence-only**: positives = known deposit cells; negatives are sampled
(distance-buffered, never observed), with a `pos_weight` BCE loss for the extreme imbalance. The standardization is
folded into the ONNX export wrapper, so the graph takes raw features. Validated by spatial block K-fold CV and
benchmarked head-to-head against WofE on the identical spatial holdout; the random-CV AUC is computed too, to surface
the inflation gap.

## geology-ood

A small undercomplete autoencoder (4 -> 8 -> 2 -> 8 -> 4) over the standardized feature stack; the reconstruction MSE
separates in-envelope geology from out-of-envelope ("the classifier is extrapolating under cover; do not trust the
score here"). The 95th-percentile in-envelope MSE is the committed threshold.

## The numbers

Measured on the synthetic lane's labelled rows under one spatial-block CV protocol: mpm-classifier **spatial-CV AUC
0.971 vs WofE 0.868** (winner: the MLP; WofE was 0.929 before release 0.11.000 made the cross-validation fully out of
fold) - random-CV 0.979 (inflation +0.008) - geology-OOD AUC 1.0 (on a synthetic
out-of-band eval set, separable by construction - not a field-detection claim). `train_mpm.py` exports
the 2 ONNX + `learned-partial.json`; `eval_mpm.mjs` runs the exported classifier in onnxruntime-web (the engine's own
runtime, the end-to-end check) and assembles `pm-learned.json` (schema `prospectmap.learned/v3`).

The real US-MVT lane has its own 6-feature models (`mpm-classifier-real.onnx`, `geology-ood-real.onnx`), trained by
`pipeline/real_learned.py` and scored against WofE on all map cells under the engine's own folds, beside a
distance-to-known-deposit baseline (`pm-learned-real.json`, schema `prospectmap.learned/v2`); the protocol and the
numbers are in [06 - model evaluation](../architecture/06_model-evaluation.md).
