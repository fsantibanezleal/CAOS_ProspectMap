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
- **The spatial-CV inflation** - the SAME model has a higher random-CV AUC than spatial-CV AUC.

## The two learned models

The **mpm-classifier** (a presence-only MLP) and the **geology-ood** (an autoencoder) are honest, value-adding ML
measured against the white-box WofE posterior - NOT bolted-on. The WofE posterior is the interpretable AUTHORITY. The
classifier is validated by SPATIAL block cross-validation and benchmarked head-to-head against WofE on the IDENTICAL
spatial holdout; the random-CV AUC is reported beside it to surface the inflation gap. Measured (not fabricated):
**mpm-classifier spatial-CV AUC 0.971 vs WofE 0.929** (winner: the MLP, on the multi-layer interactions WofE's CI form
omits), random-CV 0.979 (inflation +0.008), **geology-OOD AUC 1.0** (on a synthetic out-of-band eval set - uniform
features pushed outside the training band, separable by construction; not a field-detection claim). Deposit labels are
presence-only; negatives are sampled, never observed. Reported whichever way the numbers land. No fabricated win.
