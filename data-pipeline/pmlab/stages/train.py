"""Stage 3, train (OFFLINE, heavy lane): fit the two learned models, the presence-only mpm-classifier (MLP, with the
standardization folded into the export) and the geology OOD autoencoder, and export them to ONNX. Deterministic
(seeded). Delegates to pmlab/science/train_mpm.py (torch); writes mpm-classifier.onnx, geology-ood.onnx and the
partial metrics to data/derived/."""
