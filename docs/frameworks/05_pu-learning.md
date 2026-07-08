# 05 - PU learning (the false-negative-bias correction)

The heart of the beyond-SOTA lane. In exploration a cell with no known deposit is not a confirmed absence: it is
UNLABELED, and may host an undiscovered deposit. Training a classifier on pseudo-negatives bakes that false-negative
bias into the score. Positive-Unlabeled (PU) learning fixes it.

## The true-negative fallacy

The App's current `mpm-classifier` (and the RF/GBM/logistic baselines) draw negatives by SAMPLING non-deposit cells.
Those "negatives" include future discoveries, so the learned boundary is pulled toward the exploration history, not the
geology. The first PU algorithm for mineral prospectivity (Xiong & Zuo 2021, Computers & Geosciences 147, 104667,
doi:10.1016/j.cageo.2020.104667) showed PU beats one-class SVM and pseudo-negative ANN precisely because it stops
treating the unlabeled majority as negatives.

## Elkan-Noto and the SCAR assumption

Under **Selected Completely At Random** (SCAR), labeled positives are a random subset of all positives. Then the
observed "is-labeled" score and the true posterior differ only by a constant label frequency `c` (Elkan & Noto 2008,
KDD, 213-220, doi:10.1145/1401890.1401920):

```
p(s=1 | x) = c * p(y=1 | x),    c = p(s=1 | y=1)
```

so a classifier trained to predict "labeled vs unlabeled" recovers the true posterior up to `1/c`. `c` (or the class
prior `pi = p(y=1)`) is what must be estimated, and under exploration bias SCAR is doubtful, so we **sweep** `pi`
rather than trust one value.

## nnPU: the non-negative risk estimator

For a flexible model on very few positives, the unbiased PU risk can go negative and the model overfits. The
non-negative estimator clamps it (Kiryo, Niu, du Plessis & Sugiyama 2017, NeurIPS, arXiv:1703.00593). With the sigmoid
surrogate loss `l(+1,g)=sigmoid(-g)`, `l(-1,g)=sigmoid(g)` on the raw score `g`:

```
R_p^+ = mean over POSITIVES of l(+1, g)
R_p^- = mean over POSITIVES of l(-1, g)
R_u^- = mean over UNLABELED of l(-1, g)      (unlabeled = ALL cells, SCAR)

R_nnPU = pi * R_p^+ + max(0, R_u^- - pi * R_p^-)
```

The `max(0, .)` clamp is the correction: when the empirical negative-risk term drops below zero (overfitting), the
optimizer ascends it back toward zero instead of descending further. Implemented in `pu_conformal.py::train_nnpu`
(torch, standardization + sigmoid baked into the graph, exported as `mpm-puconformal-real.onnx`).

## What PU is and is NOT

- It **is** the removal of the pseudo-negative bias: deposits are positives, everything else is unlabeled.
- It is **not** a magic ranking booster. On the clustered real MVT belt PU-Conformal's block-CV AUC (0.656) does not
  exceed WofE (0.732); PU corrects the LABEL model, not the underlying regional signal. Its value is bias-corrected,
  calibrated uncertainty (see [06 - uncertainty and conformal](06_uncertainty-and-conformal.md)), not a higher AUC.
- SCAR is likely violated by exploration bias, so `pi` is a swept sensitivity parameter and the map's dependence on it
  is reported, not hidden.

## Related PU refinements (cited, not implemented)

Bagging-PU with cost-sensitive Bayesian logistic regression (Yang, Cheng et al. 2022,
doi:10.1007/s11053-022-10120-0) and the large-scale porphyry-Cu PU proof over the American Cordillera (Alfonso, Muller,
Mather & Anthony 2024, GSA Bulletin, doi:10.1130/B37614.1) are the field context for the PU family.
