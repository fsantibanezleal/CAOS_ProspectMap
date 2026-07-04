"""Stage 5, evaluate (the TEST stage, heavy lane): the honest head-to-head, the learned classifier's prediction-rate
AUC + capture vs the white-box WofE posterior on the IDENTICAL SPATIAL holdout (block + buffered leave-one-deposit-out;
random-CV reported too, to surface the inflation gap), plus the geology OOD autoencoder AUC separating in-envelope from
out-of-envelope geology. Metrics land in pm-learned.json; invoked by pipeline.retrain (pmlab/science/eval_mpm.mjs)."""
