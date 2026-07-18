# Frameworks

The research-chosen methods + libraries ProspectMap actually uses (each one is used by the code, not aspirational).

- [01 - Weights of Evidence + CI + logistic](frameworks/01_wofe.md) - the per-layer weights, the conditional-
  independence machinery, the logistic-regression generalization, and the honest spatial-CV validation.
- [02 - the visualisation stack](frameworks/02_viz.md) - the canvas prospectivity raster + the capture/ROC charts
  (uPlot) and the shared `@fasl-work/caos-app-shell` (+ the ⓘ Architecture modal).
- [03 - the learned models](frameworks/03_torch-onnx.md) - the mpm-classifier + the geology-ood AE,
  torch -> ONNX -> onnxruntime-web, with the honest spatial-CV-vs-WofE head-to-head.
- [04 - the ML ladder (tabular RF/GBM)](frameworks/04_ml-ladder.md) - the SOTA-classical rung: gradient
  boosting + random forest, the interactions WofE's additive form omits, and the deep-learning ceiling named.
- [05 - PU learning](frameworks/05_pu-learning.md) - the false-negative-bias correction: the true-negative
  fallacy, Elkan-Noto + SCAR, and the nnPU non-negative risk estimator behind `mpm-puconformal-real.onnx`.
- [06 - uncertainty and conformal](frameworks/06_uncertainty-and-conformal.md) - split conformal prediction,
  the positive-class coverage-guaranteed prospective set, Brier/ECE, and the honest near-vacuous-set finding.
