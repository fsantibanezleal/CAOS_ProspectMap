# ProspectMap — Weights-of-Evidence mineral prospectivity

[![CI](https://github.com/fsantibanezleal/CAOS_ProspectMap/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/CAOS_ProspectMap/actions)
**Live:** https://prospectmap.fasl-work.com

ProspectMap answers *"where is the next deposit most likely to be?"* — it stacks open geophysical / geochemical /
structural evidence layers over a study-area grid and computes a **posterior prospectivity map** P(deposit | evidence)
per cell by **Weights of Evidence** (a Bayesian log-odds update), recomputed **live in your browser** on every control.
Its reason to exist is honesty: it makes first-class the two ways a prospectivity map lies — **conditional-independence
violation** inflating the posterior, and **random-CV vs spatial-CV** inflating the AUC.

A CAOS/Faena mining web-app instantiated on the **product-repo archetype** ([ADR-0057](docs/architecture/01_overview.md)),
with the in-app i **Architecture modal** ([ADR-0058](docs/frameworks/02_viz.md)).

## What it does

- **Weights of Evidence** — per binary evidence pattern: W+/W-, the contrast C = W+ - W-, the studentized contrast
  C/s(C); the posterior log-odds = the prior logit + the sum of the present/absent weights, under conditional
  independence; the maximizing-contrast threshold t* binarizes a continuous layer.
- **The conditional-independence machinery** — the pairwise chi-square + the Agterberg-Cheng omnibus test (sum of the
  posterior ~ N(D) under CI) + the CI ratio. When correlated layers double-count, the posterior inflates and the app
  says so.
- **Logistic regression** — the CI-free generalization (IRLS + ridge); on independent patterns its coefficients match
  the WofE contrasts; under CI violation it does not over-estimate.
- **Honest validation** — the success (fitting) vs prediction-rate (held-out) capture curves; capture@10% under
  **spatial** cross-validation is the headline; the random-vs-spatial inflation gap is shown, not hidden.
- **mpm-classifier (learned)** — a presence-only MLP over the evidence vector, benchmarked head-to-head against the
  white-box WofE on the SAME spatial holdout; trained offline (torch -> ONNX), run **live** (onnxruntime-web).
- **geology-ood (learned)** — an autoencoder that flags cells whose geology is outside the labelled training envelope.
- **Bring your own evidence** — CONTRACT 1 validates a case bundle (a co-registered evidence cube + a presence-only
  deposit pattern + a study-area mask).

## Honesty

The study areas are **synthetic** (geostatistically-grounded smooth fields + a planted fault network + deposits by an
inhomogeneous-Poisson process on a known latent prospectivity), clearly labelled. They are the only data with known
ground truth, so the controls are exact: `C-NEGATIVE` (uninformative -> AUC ~ 0.5), `C-CIVIOLATE` (a correlated
duplicate -> the omnibus test fails), `C-RECOVER` (recovers the planted weight ordering), `C-SATURATE` (the analytic
limit). The white-box WofE is the interpretable authority; the learned classifier earns its place only on the spatial
holdout: **mpm-classifier spatial-CV AUC 0.971 vs WofE 0.929**, geology-OOD AUC 1.0. Deposit labels are presence-only
(negatives are sampled, never observed). Outputs are exploration **target generation**, NOT a JORC / NI 43-101 resource
estimate. No fabricated wins. Real open datasets (Lawley et al. 2022 Zn-Pb from USGS ScienceBase, Geoscience Australia
CC-BY) are a documented next step; the pipeline accepts a real cube identically.

## Quickstart

```bash
# light lane (numpy only) - rebuild the replay artifacts + run the checks
python -m venv .venv-pipeline && .venv-pipeline/Scripts/pip install -r data-pipeline/requirements.txt -r requirements-dev.txt -e .
.venv-pipeline/Scripts/python -m pmlab.pipeline all      # 10 cases -> traces + manifests
.venv-pipeline/Scripts/python scripts/check_artifacts.py # CONTRACT 2 OK

# the SPA (the WofE engine + the learned models run live in the browser)
cd frontend && npm ci && npm run dev                     # http://localhost:5173
npm test                                                 # the engine oracles (11)

# heavy lane (local only) - re-bake + train the learned models (torch -> ONNX)
python -m venv .venv-precompute && .venv-precompute/Scripts/pip install -r data-pipeline/requirements-precompute.txt
.venv-pipeline/Scripts/python -m pmlab.pipeline all --retrain
```

## Layout

See [STRUCTURE.md](STRUCTURE.md) and the wiki in [docs/](docs/README.md). The WofE engine is the TypeScript code in
[`frontend/src/mpm/`](frontend/src/mpm/) (it runs in the browser **and** in the offline Node bake - no Python re-port);
`data-pipeline/pmlab/` is the two contracts + the staged pipeline + the lane gate.

## License

MIT - see [LICENSE](LICENSE). Third-party components in [LICENSES.md](LICENSES.md); attributions in
[ATTRIBUTION.md](ATTRIBUTION.md).
