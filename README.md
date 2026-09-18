# ProspectMap, Weights-of-Evidence mineral prospectivity

[![CI](https://img.shields.io/github/actions/workflow/status/fsantibanezleal/CAOS_ProspectMap/ci.yml?branch=main&label=CI)](https://github.com/fsantibanezleal/CAOS_ProspectMap/actions)
[![License](https://img.shields.io/github/license/fsantibanezleal/CAOS_ProspectMap)](LICENSE)
[![Live demo](https://img.shields.io/badge/demo-live-2ea44f)](https://prospectmap.fasl-work.com)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21519007-blue)](https://doi.org/10.5281/zenodo.21519007)

Technical report (CC-BY-4.0): *"ProspectMap: The Two Ways a Mineral Prospectivity Map Lies, Quantified on the Real
US Mississippi-Valley-Type Belt"*, concept DOI [10.5281/zenodo.21519007](https://doi.org/10.5281/zenodo.21519007)
(source in [`manuscripts/prospectivity/`](manuscripts/prospectivity/)); its version 1.1 predates the fully out-of-fold
cross-validation of release 0.11.000. With the current release, on the real US-MVT belt the Weights-of-Evidence
ROC-AUC is 0.732 fitted without cross-validation, 0.739 under random and 0.648 under spatial-block cross-validation
(each fold fitted on its training folds only); under the same spatial folds a learned MLP reaches 0.908 and a
distance-to-known-deposit baseline, which learns no geology, 0.899.

[![CI](https://github.com/fsantibanezleal/CAOS_ProspectMap/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/CAOS_ProspectMap/actions)
**Live:** https://prospectmap.fasl-work.com

ProspectMap answers *"where is the next deposit most likely to be?"*, it stacks open geophysical / geochemical /
structural evidence layers over a study-area grid and computes a **posterior prospectivity map** P(deposit | evidence)
per cell by **Weights of Evidence** (a Bayesian log-odds update), recomputed **live in the browser** on every control.
It makes first-class the two ways a prospectivity map lies: **conditional-independence
violation** inflating the posterior, and **random-CV vs spatial-CV** inflating the AUC.

A CAOS/Faena mining web-app instantiated on the **product-repo archetype** ([ADR-0057](docs/architecture/01_overview.md)),
with the in-app ⓘ **Architecture modal** ([ADR-0058](docs/frameworks/02_viz.md)).

## What it does

- **Weights of Evidence**, per binary evidence pattern: W+/W-, the contrast C = W+ - W-, the studentized contrast
  C/s(C); the posterior log-odds = the prior logit + the sum of the present/absent weights, under conditional
  independence; the maximizing-contrast threshold t* binarizes a continuous layer.
- **The conditional-independence machinery**, the pairwise chi-square + the Agterberg-Cheng omnibus test (sum of the
  posterior ~ N(D) under CI) + the CI ratio. When correlated layers double-count, the posterior inflates and the app
  says so.
- **Logistic regression**, the CI-free generalization (IRLS + ridge); on independent patterns its coefficients match
  the WofE contrasts; under CI violation its jointly-fit coefficients are expected not to double-count (the in-app
  omnibus check runs on the WofE posterior; an LR calibration readout is not yet shown).
- **Validation**, the success (fitting) vs prediction-rate (held-out) capture curves; capture@10% under **spatial**
  cross-validation is the headline, each fold fitted on its training folds only (thresholds, weights and prior); the
  random-vs-spatial inflation gap is shown beside it.
- **mpm-classifier (learned)**, a presence-only MLP over the evidence vector, compared with the white-box WofE under
  one shared cross-validation protocol; trained offline (torch -> ONNX), run **live** (onnxruntime-web).
- **geology-ood (learned)**, an autoencoder that flags cells whose geology is outside the labelled training envelope.
- **Bring your own evidence**, Contract 1 validates a case bundle (a co-registered evidence cube + a presence-only
  deposit pattern + a study-area mask).

## Scope

Ten study areas are **synthetic** (smooth value-noise fields, geostatistical in spirit, not variogram-controlled,
with planted per-layer weights; deposits rejection-sampled on the known latent prospectivity, fixed count per case),
clearly labelled. They carry known ground truth, so the controls are exact: `C-NEGATIVE` (uninformative -> AUC ~ 0.5),
`C-CIVIOLATE` (a correlated duplicate -> the omnibus test fails), `C-RECOVER` (recovers the planted weight ordering),
`C-SATURATE` (the analytic limit). One case, **REAL-USMVT**, is real open data: the US Midcontinent
Mississippi-Valley-type Zn-Pb belt from the Lawley et al. (2022) Critical Minerals Mapping Initiative release (US
public domain; see [ATTRIBUTION](ATTRIBUTION.md)), 858 deposit cells on a 144 x 176 grid of about 5.4 km cells; its
posterior is this tool's recomputation, not the published model.

The white-box WofE is the interpretable authority; each learned classifier is compared with it under one shared
cross-validation protocol, each fold fitted on its training folds only. On the labelled rows of the five synthetic
training cases (spatial-block folds, pooled held-out AUC): **mpm-classifier 0.971 vs WofE 0.868**. On all map cells
of REAL-USMVT: **MLP 0.908 vs WofE 0.648**, beside a distance-to-known-deposit baseline of 0.899 that learns no
geology. Under the stricter contiguous k-means regions of the PU-Conformal lane (`data/derived/pu-conformal.json`), an
MLP trained on sampled pseudo-negatives reaches 0.783, level with the distance-to-deposit baseline there (0.783). The
geology-OOD AUC of 1.0 is measured on a synthetic out-of-band eval set, separable by construction, not a
field-detection claim. Deposit labels are presence-only (negatives are sampled, not observed). Outputs are
exploration **target generation**, not a JORC / NI 43-101 resource estimate.

## Quickstart

```bash
# light lane (numpy only) - rebuild the replay artifacts + run the checks
python -m venv .venv-pipeline && .venv-pipeline/Scripts/pip install -r data-pipeline/requirements.txt -r requirements-dev.txt
.venv-pipeline/Scripts/python data-pipeline/run.py all      # 11 cases -> traces + manifests
.venv-pipeline/Scripts/python scripts/check_artifacts.py # Contract 2 OK

# the SPA (the WofE engine + the learned models run live in the browser)
cd frontend && npm ci && npm run dev                     # http://localhost:5173
npm test                                                 # the engine tests (17)

# heavy lane (local only) - re-bake + train the learned models (torch -> ONNX)
python -m venv .venv-precompute && .venv-precompute/Scripts/pip install -r data-pipeline/requirements-precompute.txt
.venv-pipeline/Scripts/python data-pipeline/run.py all --retrain
```

## Layout

See [STRUCTURE.md](STRUCTURE.md) and the wiki in [docs/](docs/README.md). The WofE engine is the TypeScript code in
[`frontend/src/mpm/`](frontend/src/mpm/) (it runs in the browser **and** in the offline Node bake - no Python re-port);
`data-pipeline/pipeline/` is the two contracts + the staged pipeline + the lane gate.

## License

MIT - see [LICENSE](LICENSE). Third-party components in [LICENSES.md](LICENSES.md); attributions in
[ATTRIBUTION.md](ATTRIBUTION.md).
