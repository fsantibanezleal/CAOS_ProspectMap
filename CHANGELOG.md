# Changelog

All notable changes to ProspectMap. Format: [Keep a Changelog](https://keepachangelog.com); versions are X.XX.XXX.

## [0.06.000] - 2026-06-21

First full build on the product-repo archetype (ADR-0057). Weights-of-Evidence mineral prospectivity mapping.

### Added
- **The WofE engine** (`frontend/src/mpm/`, dependency-free TypeScript, live in-browser + Node-bakeable): WofE
  (W+/W-/contrast/studentized-C/posterior under CI, Haldane 0.5 guard) . the maximizing-contrast threshold sweep .
  the CI machinery (pairwise chi-square + the Agterberg-Cheng omnibus + the CI ratio) . logistic regression (IRLS +
  ridge) . honest validation (success/prediction-rate capture + ROC) . spatial vs random cross-validation . the
  deterministic synthetic study-area generator. 11 node:test oracles.
- **The Python core** (`data-pipeline/pmlab/`): CONTRACT 1 (a case-bundle ingestion gate) + CONTRACT 2
  (`prospectmap.trace/v1` + `manifest/v2`) + the 10 K-/D-/C- cases + the numpy-light pipeline + the two-language bake.
- **The 6-page SPA** on `@fasl-work/caos-app-shell`: the interactive canvas prospectivity map + 10 tabs + the 5 doc
  pages + the i Architecture modal (ADR-0058).
- **The 2 learned models** (torch -> ONNX): `mpm-classifier` (presence-only MLP) + `geology-ood` (autoencoder), run
  live via onnxruntime-web. Honest spatial-CV: classifier 0.971 vs WofE 0.929; OOD AUC 1.0.
- The docs/ wiki (ADR-0056), CI + deploy-pages, the root files.

[0.06.000]: https://github.com/fsantibanezleal/CAOS_ProspectMap/releases/tag/v0.06.000
