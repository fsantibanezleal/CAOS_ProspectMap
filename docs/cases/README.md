# Cases

Ten synthetic study areas (clearly labelled), grouped by category. Each is generated deterministically from a spec by
`frontend/src/mpm/synth.ts` (smooth value-noise evidence layers + a planted latent prospectivity + deposits
rejection-sampled on it, fixed count per case), so the controls have known ground truth. The App shows one case; Experiments/Benchmark
summarize across categories. The Python registry (`pmlab/cases/mpm_cases.py`) mirrors the TS cases; a test cross-checks
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
| `D-SPARSE` | only a weak geochem signal -> little real skill; the product does not manufacture confidence (AUC ~ 0.71) |

## C - control (oracle / negative control), known ground truth

| Case | What it shows (the exact oracle) |
|---|---|
| `C-NEGATIVE` | uninformative layers -> all contrasts ~ 0, ROC AUC ~ 0.499 - no skill from noise |
| `C-CIVIOLATE` | a correlated duplicate of a favourable layer -> the omnibus test fails (CI ratio 0.65, z 4.1); the WofE posterior is inflated (logistic is the CI-free alternative; its calibration is not yet read out in-app) |
| `C-RECOVER` | well-separated planted weights -> WofE recovers the ordering contrast(mag) > contrast(geochem) > contrast(struct) |
| `C-SATURATE` | a near-perfect single layer -> a near-saturated posterior, no numerical blow-up (the Haldane guard), AUC ~ 0.97 |

Real open datasets (Lawley et al. 2022 Zn-Pb from USGS ScienceBase, Geoscience Australia CC-BY) are the documented
next step; the pipeline accepts a real cube identically (the synthetic cases are the verifiable controls, exactly the
CutoffGrade precedent of synthetic data + real method).
