"""Stage 2, feature_extraction (heavy lane): assemble the learned-model training data, sampled presence cells +
informed/background negatives over the case cubes, each cell's evidence feature vector built in the SOURCE-OF-TRUTH
order (pmlab/model/learned.py :: MPM_FEATURES) via the SAME TS engine (pmlab/science/gen_train.mjs), plus the
in-envelope vectors for the geology OOD autoencoder and the spatial-block fold assignments (so train + eval share
folds and no statistic leaks across the spatial holdout)."""
