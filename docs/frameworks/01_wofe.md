# 01 - Weights of Evidence + CI + logistic

## Weights of Evidence

For a binary evidence pattern B over a grid of unit cells with deposit set D, the weights are the log-likelihood
ratios W+ = ln[P(B|D)/P(B|D-bar)] and W- = ln[P(B-bar|D)/P(B-bar|D-bar)], estimated from the 2x2 contingency counts.
The contrast C = W+ - W- measures the spatial association; the studentized contrast C/s(C) (with s^2(W+) = 1/n(B,D) +
1/n(B,D-bar)) is the significance guide. The posterior log-odds of a cell is the prior logit plus the sum of the
present/absent weights, under conditional independence; the Haldane 0.5 correction guards zero-count classes. A
continuous layer is binarized at the maximizing-contrast threshold t* = argmax C(t) (subject to a studC floor).

## Conditional independence

The posterior sum is only valid if the patterns are conditionally independent given D. Correlated favourable layers
double-count and OVER-ESTIMATE the posterior. ProspectMap surfaces this: the pairwise chi-square (Yates-corrected) +
the **Agterberg-Cheng omnibus test** (T = sum of the posterior ~ N(D) under CI; z = (T - N(D))/s(T)) + the CI ratio
N(D)/T (~ 1 ok; < 0.85 a problematic violation). The `C-CIVIOLATE` control demonstrates it on purpose.

## Logistic regression

The CI-free generalization: logit P = beta_0 + sum beta_j x_j, fit by IRLS + ridge. On conditionally-independent binary
patterns beta_j ~ C_j (the WofE <-> LR equivalence); under CI violation the jointly-fit coefficients are expected to
shrink rather than double-count (the theoretical fix), so logistic is the comparison to reach for when the omnibus
test fails. Note: the app's omnibus readout runs on the WofE posterior; an LR calibration readout (sum P_LR vs N(D))
is not yet shown in-app.

## Honest validation

A model fitting its own training deposits is NOT evidence of predictive skill; only spatially held-out capture is. The
prediction-rate capture curve (% deposits captured vs % area, ranked by prospectivity, under SPATIAL block CV +
buffered leave-one-deposit-out) is the headline; capture@10% under spatial CV is the reported number; ROC/AUC is
secondary (presence-only). The mandatory demonstration: the SAME model collapses from random-CV to spatial-CV - the
inflation gap.
