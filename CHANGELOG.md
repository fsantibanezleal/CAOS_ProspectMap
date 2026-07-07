# Changelog

All notable changes to ProspectMap. Format: [Keep a Changelog](https://keepachangelog.com); versions are X.XX.XXX.

## [0.07.000] - 2026-07-07

First-level Synthetic | Real Source selector (the Faena "Real sample" lane) on real, openly-licensed data.

### Added
- **Real-data lane on the US Midcontinent MVT Zn-Pb belt** from the Lawley et al. (2022) Tri-National
  Critical Minerals Mapping Initiative (CMMI) open release (USGS ScienceBase item
  6193e9f3d34eb622f68f13a5, data DOI 10.5066/P970GDD5, paper DOI 10.1016/j.oregeorev.2021.104635;
  US public domain). New offline builder `data-pipeline/pmlab/real_usmvt.py` clips + rasterizes the
  evidential layers to a 144x176 grid (~5.4 km cells) over W-97/E-88/S35/N43.5: REAL measured
  geophysics (`mag`, `grav`, `lab` depth-to-LAB tomography, `satgrav`), DERIVED proximity layers
  (`faultprox`, `marginprox`), and `depositIdx` = cells with a real Pb-Zn (MVT/CD) occurrence (858
  deposit cells from 5837 occurrences). Bakes `data/derived/REAL-USMVT/cube.json` + `provenance.json`.
- **FIRST-LEVEL Source selector** (`Synthetic | Real sample`) at the top of the Tool sidebar. In Real
  mode the synthetic case knobs (planted-weight generators) disable, you pick the real datum, and all
  App tools run live on the real Cube. The `realOrSynthetic: 'real (open dataset)'` contract path is
  now wired.
- **Two new method tabs** (method-tab floor 9 -> 11): a fuzzy-logic / index-overlay combiner (gamma
  operator, no fitting, no CI assumption; Bonham-Carter 1994, Carranza 2009) and a calibration /
  reliability readout (decile reliability diagram + Brier + ECE; the C-SATURATE case had flagged it
  missing).
- **Real learned models** (`data-pipeline/pmlab/real_learned.py`): `mpm-classifier-real.onnx` +
  `geology-ood-real.onnx` retrained on the real 6-feature cube (the synthetic 4-feature ONNX are NOT
  silently applied to the real cube). Spatial-block CV, distance-buffered presence-only negatives.
- **In-app honesty block** in the real lane: REAL / DERIVED / RECOMPUTED legend, the Lawley 2022
  citation + license, and the conditional-independence caveat (real geophysics is physically correlated,
  so the omnibus test fires; route to logistic regression). The posterior is stated as OUR browser WofE
  recomputation, NOT the published H3 + gradient-boosting model.
- Engine: `analyzeCube(cube, layerIds)` runs the full WofE/CI/validation/LR analysis on an already-built
  Cube; `analyzeCase` now delegates to it. Node bake `science/bake_real.mjs` emits the real
  `trace.json` + merges the real case into `case-results.json` through the SAME TS engine.

### Changed
- Version sources normalized to `0.07.000` (X.XX.XXX): `frontend/package.json`, root `VERSION`,
  `CHANGELOG`, and the footer (the shell version is now derived from `package.json`, no longer a
  hard-coded string that drifted to `0.06.000`).

## [0.06.001] - 2026-07-04

### Changed
- Content standards (ADR-0067): removed every em-dash from tracked content (replaced with commas, or
  "n/a" in table cells). No behaviour change. Added `scripts/check_content_standards.py` + wired it
  into the CI `guards` job so the repo cannot regress on em-dashes or emojis.

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
- **The 6-page SPA** on `@fasl-work/caos-app-shell`: the interactive canvas prospectivity map + 9 tabs + the 5 doc
  pages + the i Architecture modal (ADR-0058).
- **The 2 learned models** (torch -> ONNX): `mpm-classifier` (presence-only MLP) + `geology-ood` (autoencoder), run
  live via onnxruntime-web. Honest spatial-CV: classifier 0.971 vs WofE 0.929; OOD AUC 1.0 (on a synthetic
  out-of-band eval set, separable by construction).
- The docs/ wiki (ADR-0056), CI + deploy-pages, the root files.

[0.06.000]: https://github.com/fsantibanezleal/CAOS_ProspectMap/releases/tag/v0.06.000
