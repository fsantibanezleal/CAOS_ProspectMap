"""Stage 5, evaluate (the TEST stage, heavy lane): the head-to-head, the learned classifier's held-out AUC vs the
white-box WofE posterior under the IDENTICAL spatial-block folds (each fold fitted on its training folds only;
random-CV reported too, to surface the inflation gap), plus the geology OOD autoencoder AUC separating in-envelope from
out-of-envelope geology. Metrics land in pm-learned.json; invoked by pipeline.retrain (pipeline/science/eval_mpm.mjs)."""
