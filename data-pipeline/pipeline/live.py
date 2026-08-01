"""LIVE lane note. ProspectMap's live lane is the dependency-free TypeScript engine (frontend/src/mpm/) running in the
browser, the WofE/CI/logistic-regression/validation recompute on every control, plus the learned classifier + OOD
autoencoder via onnxruntime-web. There is NO Python live lane (no Pyodide): this module is intentionally dormant. The
offline Python side is the two data contracts + the numpy-light replay pipeline + the torch->ONNX retrain lane only."""
from __future__ import annotations
