# Cases

Eleven study areas, grouped by category: ten synthetic (clearly labelled) and one of real open data (`REAL-USMVT`).
Each synthetic area is generated deterministically from a spec by
`frontend/src/mpm/synth.ts` (smooth value-noise evidence layers + a planted latent prospectivity + deposits
rejection-sampled on it, fixed count per case), so the controls have known ground truth. The App shows one case; Experiments/Benchmark
summarize across categories. The Python registry (`pipeline/cases/mpm_cases.py`) mirrors the TS cases; a test cross-checks
the ids against the baked `case-results.json`.

## K - deposit-type terrane (the geological setting)

| Case | What it shows |
|---|---|
| `K-PORPHYRY` | porphyry-Cu-like: magnetic high + geochem anomaly + proximity to structure favourable; radiometrics uninformative (AUC ~ 0.91) |
| `K-OROGENIC` | orogenic-Au-like: structure dominates; geochem secondary; magnetics uninformative (AUC ~ 0.71) |
| `K-VMS` | VMS-like: magnetic + geochem co-located; structure uninformative |
| `K-IOCG` | IOCG-like: a strong magnetic signature with multi-layer support |

## D - data / validation regime (evidence richness)

| Case | What it shows |
|---|---|
| `D-RICH` | all four layers informative -> a high-skill posterior (but watch the CI + the spatial-CV gap) |
| `D-SPARSE` | only a weak geochem signal -> little real skill; a low AUC is the expected result (AUC ~ 0.71) |

## C - control (oracle / negative control), known ground truth

| Case | What it shows (the exact oracle) |
|---|---|
| `C-NEGATIVE` | uninformative layers -> all contrasts ~ 0, ROC AUC ~ 0.499 - no skill from noise |
| `C-CIVIOLATE` | a correlated duplicate of a favourable layer -> the omnibus test fails (CI ratio 0.65, z 4.1); the WofE posterior is inflated (logistic is the CI-free alternative; its calibration is not yet read out in-app) |
| `C-RECOVER` | well-separated planted weights -> WofE recovers the ordering contrast(mag) > contrast(geochem) > contrast(struct) |
| `C-SATURATE` | a near-perfect single layer -> a near-saturated posterior, no numerical blow-up (the Haldane guard), AUC ~ 0.97 |

## R - real open dataset (a published mineral system)

| Case | What it shows |
|---|---|
| `REAL-USMVT` | the US Midcontinent Mississippi-Valley-type Zn-Pb belt from the Lawley et al. (2022) Critical Minerals Mapping Initiative data release (DOI [10.5066/P970GDD5](https://doi.org/10.5066/P970GDD5), US public domain): four measured geophysics grids (mag, grav, LAB tomography, satellite gravity) and two proximity layers derived from its fault and passive-margin vectors, 858 deposit cells on a 144 x 176 grid of about 5.4 km cells. The layers are physically correlated, so the omnibus test fails (CI ratio 0.76, z 8.3); the occurrences are strongly clustered (the Tri-State district), so the spatial-CV WofE AUC (0.648) sits well below the random-CV (0.739) and fitting (0.732) AUCs. The posterior is this tool's WofE recomputation, not the published model. |

The real cube is built by `data-pipeline/pipeline/real_usmvt.py` from the files listed in
`data/derived/REAL-USMVT/provenance.json` (see [ATTRIBUTION](../../ATTRIBUTION.md)). Other open datasets (USGS MRDS and
NURE-HSSR, Geoscience Australia grids under CC-BY) remain candidates; the pipeline accepts a real cube identically,
and the synthetic cases stay the verifiable controls (the CutoffGrade precedent of synthetic data + real method).
