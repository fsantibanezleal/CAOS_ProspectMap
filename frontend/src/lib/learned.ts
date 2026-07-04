// The learned-model feature contract (browser side), mirrors data-pipeline/pmlab/model/learned.py :: MPM_FEATURES.
// The per-cell evidence feature vector fed to the mpm-classifier ONNX + the geology-ood autoencoder. SINGLE SOURCE OF
// TRUTH for the order; a drift vs the Python contract would mis-feed the model.
export const MPM_FEATURES = ['mag', 'rad', 'geochem', 'struct'] as const;
export const N_FEATURES = MPM_FEATURES.length;
