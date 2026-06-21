# Frameworks

The research-chosen methods + libraries ProspectMap actually uses (each one is used by the code, not aspirational).

- [01 - Weights of Evidence + CI + logistic](frameworks/01_wofe.md) - the per-layer weights, the conditional-
  independence machinery, the logistic-regression generalization, and the honest spatial-CV validation.
- [02 - the visualisation stack](frameworks/02_viz.md) - the canvas prospectivity raster + the capture/ROC charts
  (uPlot) and the shared `@fasl-work/caos-app-shell` (+ the i Architecture modal).
- [03 - the learned models](frameworks/03_torch-onnx.md) - the mpm-classifier + the geology-ood AE,
  torch -> ONNX -> onnxruntime-web, with the honest spatial-CV-vs-WofE head-to-head.
