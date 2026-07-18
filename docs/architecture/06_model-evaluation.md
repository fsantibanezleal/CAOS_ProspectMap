# 06 - Model evaluation (the oracles + the two learned models)

## The engine oracles (`frontend/test/mpm.test.ts`, node:test + tsx)

The science is pinned against closed forms + WofE theory + the synthetic controls (known ground truth):

- **WofE closed-form 2x2** - a hand-computed contingency gives the exact W+/W-/contrast/s(C)/studC.
- **The Haldane guard** - a zero-count class does not blow up the log.
- **Posterior monotonicity** - a cell inside a favourable pattern has a higher posterior than outside.
- **Synthetic weight recovery** (positive control) - WofE recovers the planted weight ordering; ROC AUC clearly > 0.5.
- **Negative control** - an uninformative layer gives contrast ~ 0 and ROC ~ 0.5.
- **WofE <-> logistic equivalence** - on CI-true binary patterns the logistic coefficients match the WofE contrasts in
  sign + ordering.
- **The omnibus CI test** - T ~ N(D) on CI-true data; T > N(D) (z > 0) on a planted CI violation.
- **The capture curves** - a perfect ranking captures all deposits in minimal area; a random ranking gives the diagonal.
- **The spatial-CV inflation** - the same model has a higher random-CV AUC than spatial-CV AUC.

## The two learned models

The **mpm-classifier** (a presence-only MLP) and the **geology-ood** (an autoencoder) are honest, value-adding ML
measured against the white-box WofE posterior, not bolted-on. The WofE posterior is the interpretable authority. The
classifier is validated by spatial block cross-validation and benchmarked head-to-head against WofE on the identical
spatial holdout; the random-CV AUC is reported beside it to surface the inflation gap. Measured (not fabricated):
**mpm-classifier spatial-CV AUC 0.971 vs WofE 0.929** (winner: the MLP, on the multi-layer interactions WofE's CI form
omits), random-CV 0.979 (inflation +0.008), **geology-OOD AUC 1.0** (on a synthetic out-of-band eval set - uniform
features pushed outside the training band, separable by construction; not a field-detection claim). Deposit labels are
presence-only; negatives are sampled, never observed. Reported whichever way the numbers land. No fabricated win.

## The PU-Conformal head-to-head + negative controls (`data-pipeline/pmlab/pu_conformal.py`)

The beyond-SOTA lane scores six models on the real US MVT cube under **identical contiguous spatial folds** and reads
the ranking verdict directly off bootstrap CIs.

### Spatial-block protocol

Folds are **contiguous** geographic regions (k-means on cell coordinates, k=5), deliberately stricter than the App's
interleaved `blockId % k`: interleaving 20-cell blocks leaves every held-out block adjacent to training blocks, which
lets a fine-grained learned model memorize the autocorrelated local feature signature and inflate the held-out AUC.
Under contiguous holdout a held-out region is spatially separated from its training, so the transfer question is honest
(Roberts et al. 2017, [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)). AUC is reported with a 95% bootstrap CI.

### Negative controls (must pass, on the real cube)

- **Label permutation** - shuffle the deposit labels; every model must collapse to ~0.5 (measured: WofE 0.506, PU
  0.490). A high score on permuted labels would mean leakage.
- **Uninformative layer** - append a pure-noise feature; it must not lift AUC (measured: 0.638 with noise vs 0.656
  without).
- **Distance-to-deposit null** - score each cell by proximity to the nearest training deposit; any real model must beat
  this trivial autocorrelation baseline to claim it learned geology. Measured: **0.783**, which beats WofE/RF/GBM/PU,
  so most apparent skill is spatial proximity, not geology.

### The honest result

Under strict contiguous holdout PU-Conformal (block-CV AUC 0.656) does not beat classical WofE (0.732). PU corrects the
label bias, not the regional signal; its advance is calibrated, bias-corrected, coverage-guaranteed uncertainty (see
[frameworks/06 - uncertainty and conformal](../frameworks/06_uncertainty-and-conformal.md)) that passes the controls,
not a higher AUC. Committed numbers: `data/derived/pu-conformal.json`. No fabricated win.
