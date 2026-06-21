# 04 - The live lane (TypeScript)

ProspectMap's live lane is a **dependency-free TypeScript engine** (`frontend/src/mpm/`) - no DOM, no npm runtime deps -
so the SAME code runs live in the browser (recompute on a control) **and** offline under Node via `tsx` (the bake that
produces the committed artifacts). There is no Pyodide lane.

## Modules

| Module | Computes |
|---|---|
| `types.ts` | the typed raster Cube + result types (the CONTRACT-2 mirror) |
| `rng.ts` | the seeded RNG (mulberry32) + erf/normCdf/normInv + sigmoid |
| `grid.ts` | the active-cell mask, the deposit set, N(S)/N(D), per-layer access |
| `binarize.ts` | the maximizing-contrast threshold sweep + best-weights per layer |
| `wofe.ts` | the 2x2 contingency, W+/W-/contrast/variances/studentized-C, the posterior (with the Haldane 0.5 guard) |
| `ci.ts` | pairwise chi-square + the Agterberg-Cheng omnibus T/z + the CI ratio |
| `logreg.ts` | logistic regression by IRLS + ridge (the CI-free generalization) |
| `validate.ts` | the success/prediction-rate capture curves + capture@a + ROC AUC |
| `cv.ts` | random vs spatial-block folds + the inflation demonstrator |
| `synth.ts` | the deterministic synthetic study-area generator with planted weights |
| `analyze.ts` | the per-case analysis the bake + the App consume |
| `cases.ts` | the 10 K-/D-/C- case SPECs |

## The live recompute path

When the user toggles a layer or switches method: `binarize.bestWeights` -> updated patterns + the C(t) sweep ->
`wofe.posterior` over the cube -> the prospectivity raster re-renders (canvas + ImageData) + the per-layer weights
table -> `ci.ciCheck` -> the CI heatmap + omnibus readout -> `validate.captureCurve` under the selected CV scheme.
All pure TS over the in-memory cube; no server, no Python at runtime. The 2 ONNX (the classifier + the OOD-AE) run via
onnxruntime-web for the learned probe + the trust mask.
