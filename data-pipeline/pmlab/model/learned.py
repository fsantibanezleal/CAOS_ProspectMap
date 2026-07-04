"""Feature contracts for ProspectMap's two learned models (the SINGLE SOURCE OF TRUTH shared by the offline trainer
pmlab/science/train_mpm.py and the in-browser inference). Trained OFFLINE (torch -> ONNX), run LIVE (onnxruntime-web).
The white-box Weights-of-Evidence posterior is the interpretable AUTHORITY; the learned classifier is benchmarked
against it on the SAME spatial holdout and earns its place only on fused non-linear gains. (Stub until commit 4b.)

1. mpm-classifier, a small MLP over the per-cell evidence feature vector -> P(deposit). Presence-only labels
   (positives = known deposit cells; negatives are SAMPLED, never observed). Validated by SPATIAL cross-validation and
   compared head-to-head with the WofE posterior on the identical spatial holdout. No fabricated win.

2. geology-ood, an undercomplete autoencoder over the standardized feature stack; a high reconstruction MSE = a cell
   whose multivariate geology is OUTSIDE the labelled training envelope (the classifier is extrapolating under cover;
   do not trust the score there). The strongest operational honesty signal in exploration.
"""
from __future__ import annotations

# the per-cell evidence feature vector (mirrors the synthetic case layers in frontend/src/mpm/cases.ts; the real-data
# superset adds gravity/ASTER-alteration/engineered terms, see the learned research note).
MPM_FEATURES = ("mag", "rad", "geochem", "struct")
N_FEATURES = len(MPM_FEATURES)

CLF_INPUT_NAME = "x"
CLF_OUTPUT_NAME = "p"          # P(deposit) in [0,1]

OOD_INPUT_NAME = "x"
OOD_OUTPUT_NAME = "xr"          # standardized reconstruction (MSE computed downstream)
