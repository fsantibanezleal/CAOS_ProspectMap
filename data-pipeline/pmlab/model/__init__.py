"""ProspectMap's model package: `learned.py` holds the feature contracts for the two learned models
(mpm-classifier MLP + geology-ood autoencoder), shared by the offline trainer (science/train_mpm.py) and the
in-browser ONNX inference. The WofE/CI/logistic algorithm itself lives in the TypeScript engine
(frontend/src/mpm/), which the bake runs via tsx — no Python re-port."""
