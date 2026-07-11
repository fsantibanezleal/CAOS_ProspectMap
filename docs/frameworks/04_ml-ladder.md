# 04 - The ML ladder (tabular RF/GBM, the SOTA-classical rung)

Weights of Evidence is additive in log-odds under conditional independence: it cannot represent an interaction where
two layers only matter jointly. The tabular machine-learning rung closes that gap and is the SOTA-classical baseline
the PU-Conformal lane must be judged against, not just WofE and logistic regression.

## Why a tabular ML rung

In the canonical head-to-head for mineral prospectivity mapping (epithermal Au, Rodalquilar), random forest beat
artificial neural networks, regression trees and support vector machines on success rate and ROC, and was the most
stable to training-parameter choices (Rodriguez-Galiano, Sanchez-Castillo, Chica-Olmo & Chica-Rivas 2015, Ore Geology
Reviews 71, 804-818, [doi:10.1016/j.oregeorev.2015.01.001](https://doi.org/10.1016/j.oregeorev.2015.01.001)). So RF/GBM is the honest "which ML model for MPM" reference.

## Gradient boosting

A stagewise additive model of shallow regression trees, each fit to the negative gradient of the logistic loss:

```
F_0(x) = logit(prior)
F_m(x) = F_{m-1}(x) + nu * h_m(x),   h_m ~= argmin_h sum_i loss(y_i, F_{m-1}(x_i) + h(x_i))
p_hat(x) = sigmoid(F_M(x))
```

with learning rate `nu` (shrinkage) and `M` trees. In `pu_conformal.py`: `GradientBoostingClassifier(n_estimators=200,
max_depth=3, learning_rate=0.05)`.

## Random forest

An average of `B` decorrelated bagged trees, each grown on a bootstrap sample with a random feature subset per split;
the ensemble reduces variance without materially increasing bias. In `pu_conformal.py`:
`RandomForestClassifier(n_estimators=300, max_depth=8)`.

## The training contract on this data

Both are trained **presence-only**: positives = deposit cells, negatives = distance-buffered SAMPLED cells (never
observed absences). Every model in the benchmark is scored on the SAME contiguous spatial folds (see
[06 - uncertainty and conformal](06_uncertainty-and-conformal.md) and
[architecture/06 - model evaluation](../architecture/06_model-evaluation.md)).

## What this rung is and is NOT

- It **is** the non-linear tabular frontier that exposes interactions WofE's additive form omits, and the honest
  yardstick for any proposed method.
- It is **not** a licence to ignore leakage: without strict spatial control these fine-grained models overfit
  autocorrelation and report inflated held-out AUC (the exact failure the App warns about). Under the strict
  contiguous folds the tabular models do **not** dominate the trivial distance-to-deposit null on the real MVT belt.
- It still needs pseudo-negatives, the bias that PU learning ([05 - PU learning](05_pu-learning.md)) removes.

## The deep learning ceiling (named, not implemented)

Beyond tabular ML the field uses deep autoencoders (Xiong, Zuo & Carranza 2018, [doi:10.1016/j.oregeorev.2018.10.006](https://doi.org/10.1016/j.oregeorev.2018.10.006)),
CNN/GeoCNN with augmentation for tiny positive sets (Li, Zuo, Zhao & Zhao 2022, [doi:10.1016/j.oregeorev.2022.104693](https://doi.org/10.1016/j.oregeorev.2022.104693)),
and self-supervised geospatial foundation models (GFM4MPM, [arXiv:2406.12756](https://arxiv.org/abs/2406.12756)). Our grid is small, so a CNN is optional;
these are cited as the honest ceiling this build does not train.
