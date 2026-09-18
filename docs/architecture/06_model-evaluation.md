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

The **mpm-classifier** (a presence-only MLP) and the **geology-ood** (an autoencoder) are measured against the
white-box WofE posterior, the interpretable authority. Each lane compares the classifier with WofE under ONE protocol
(the same folds, held-out cells, labels and aggregation) and states it beside every value; an AUC measured under
another protocol is never set against it. Deposit labels are presence-only; negatives are sampled, never observed.
Reported whichever way the numbers land.

### Synthetic lane (`pm-learned.json`, `science/gen_train.mjs` + `science/train_mpm.py`)

Cells: the labelled rows (deposit cells plus negatives sampled at least 6 cells from any deposit, 8 per deposit) of
the five training cases (K-PORPHYRY, K-OROGENIC, K-VMS, K-IOCG, D-RICH). Folds: the engine's `spatialBlockFolds`
(20x20-cell blocks, fold = blockId % 5). WofE: the engine's held-out posterior, weights refitted per fold; MLP: trained
on the other folds' rows. One pooled held-out AUC over the rows. Measured: **MLP 0.971 vs WofE 0.929** (winner: the
MLP), random-CV MLP 0.979 (inflation +0.008), **geology-OOD AUC 1.0** (on a synthetic out-of-band eval set, uniform
features pushed outside the training band, separable by construction; not a field-detection claim).

### Real lane (`pm-learned-real.json`, `science/real_wofe_oof.mjs` + `pipeline/real_learned.py`)

Cells: all 25344 map cells of the US Midcontinent MVT cube (858 deposit cells). Folds: the engine's `spatialBlockFolds`
(20x20-cell blocks, fold = blockId % 5) and `randomFolds` (seed 17), the folds behind the WofE cross-validation AUCs of
`case-results.json`. WofE: weights refitted per fold on the training deposits; MLP: trained per fold on the
training-fold deposits plus buffered negatives sampled from the training folds only. Every cell is scored once while
held out, and the held-out scores are pooled into one rank (Mann-Whitney) AUC. The engine's distance-to-known-deposit
score (`nearestDepositScore`, exp(-d/4) to the nearest training deposit) is scored under the same folds as a reference
that learns no geology.

| protocol (all 25344 cells, pooled held-out AUC) | MLP | WofE | distance baseline |
|---|---|---|---|
| spatial-block CV | 0.908 | 0.637 | 0.899 |
| random CV | 0.939 | 0.723 | 0.960 |

Under the engine's folds the MLP ranks held-out cells better than WofE, but the baseline reaches 0.899, within 0.010 of
the MLP, so this protocol cannot show that the MLP learned more than proximity to known deposits. The engine's folds
interleave blocks, so every held-out block borders training blocks; the contiguous-fold head-to-head below is the
stricter transfer test. The WofE AUC without cross-validation (0.732, fitted and scored on the same cells) is a fitting
number, stored apart under `nocv`. The MLP's CV on its labelled sample (858 deposit and 2574 buffered cells, shuffled
blocks, mean of per-fold AUCs: 0.946 spatial, 0.982 random) is kept under `labelled_sample_cv`; it has no WofE
counterpart and is not a comparison. Before version 0.10.001 the file stored the no-CV WofE value under
`spatial_cv.wofe_roc_auc`, beside a winner flag that compared values from different cell sets (GitHub issue #41).

## The PU-Conformal head-to-head + negative controls (`data-pipeline/pipeline/pu_conformal.py`)

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
