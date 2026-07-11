# 06 - Uncertainty and conformal prediction

Most mineral-prospectivity maps ship a single probability per cell with no calibrated band. The beyond-SOTA lane adds a
distribution-free, coverage-guaranteed uncertainty layer under honest spatial blocking, and reports its limits.

## Split conformal prediction

Conformal prediction turns any pre-trained score into prediction sets with finite-sample coverage, distribution-free
(Angelopoulos & Bates 2021, [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)). Split (inductive) conformal: fit the model on a TRAIN split, then on a
held-out CALIBRATION split compute a nonconformity score per example and take an empirical quantile.

We use a **positive-class (Mondrian) construction** suited to prospectivity: the nonconformity of a known deposit is
`s = 1 - p_hat(x)` (a deposit with a low score is nonconforming). With `n` calibration deposits and miscoverage `alpha`:

```
q_hat = Quantile( {s_i}, ceil((n+1)(1-alpha)) / n )
Prospective set C_alpha = { x : p_hat(x) >= 1 - q_hat }
Guarantee: Pr( a held-out deposit in C_alpha ) >= 1 - alpha      (under exchangeability)
```

`set size` = the fraction of the belt inside `C_alpha` (the exploration cost of the guarantee). The browser applies the
exported threshold `1 - q_hat` live; no heavy compute in-page.

## Spatial blocking breaks exchangeability (the honest caveat)

The coverage guarantee assumes the calibration and test points are exchangeable. Spatial autocorrelation breaks that.
Under spatial blocking the guarantee is only marginal over blocks and degrades under block-to-block distribution shift
(Roberts et al. 2017, [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)). On strongly clustered MVT the consequence is concrete: to guarantee
coverage the prospective set becomes **near-vacuous**, flagging almost the entire belt. That wide set IS the honest
finding: regional geophysics cannot localize MVT under spatial transfer. It is reported, not hidden.

## Calibration metrics (all models)

Alongside conformal coverage the benchmark reports, per model:

- **Brier score** = mean squared error of the probability vs the 0/1 label.
- **Expected Calibration Error (ECE)** = weighted mean gap between mean predicted probability and observed frequency
  over equal-count bins.
- A **reliability diagram** (the App's Calibration tab): predicted vs observed frequency by decile.

Presence-only labels bias the ABSOLUTE calibration (the "absences" include undiscovered deposits), so these read as
RELATIVE reliability across bins, not absolute probabilities.

## Measured result (real US MVT cube)

Positive-class split conformal, spatially-separated WEST(train)/CENTER(calib)/EAST(test) longitude bands, `pi = 0.05`:

| nominal | empirical coverage | set size (area) |
|---|---|---|
| 90% | ~98% | ~88% |
| 80% | ~97% | ~78% |

Coverage meets (over-satisfies) the guarantee, but only by flagging most of the belt: the near-vacuous set that
honestly reports the localization limit. See `data/derived/pu-conformal.json` for the committed numbers and the
class-prior `pi` sensitivity sweep.

## Precedents

Quantile-regression forests gave per-pixel uncertainty in geochemical mapping (Kirkwood et al. 2016,
[doi:10.1016/j.gexplo.2016.05.003](https://doi.org/10.1016/j.gexplo.2016.05.003)); a review situates UQ choices for prospectivity mapping (Wang et al. 2024,
[doi:10.1029/2023GC011301](https://doi.org/10.1029/2023GC011301)).
