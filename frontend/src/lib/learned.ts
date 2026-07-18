// The learned-model feature contract (browser side), mirrors data-pipeline/pmlab/model/learned.py :: MPM_FEATURES.
// The per-cell evidence feature vector fed to the mpm-classifier ONNX + the geology-ood autoencoder. The single source
// of truth for the order; a drift vs the Python contract would mis-feed the model.
export const MPM_FEATURES = ['mag', 'rad', 'geochem', 'struct'] as const;
export const N_FEATURES = MPM_FEATURES.length;

// The real-data lane (US Midcontinent MVT cube, pmlab/real_usmvt.py) has its own 6-feature space, so the synthetic
// 4-feature ONNX cannot be applied to it. The real models (pmlab/real_learned.py -> *-real.onnx) are trained on these.
export const REAL_FEATURES = ['mag', 'grav', 'lab', 'satgrav', 'faultprox', 'marginprox'] as const;
export const N_REAL_FEATURES = REAL_FEATURES.length;
