# ProspectMap - documentation

The navigable wiki for ProspectMap: **Weights-of-Evidence mineral prospectivity** - stack open geophysical /
geochemical / structural evidence layers over a study-area grid into a posterior P(deposit | evidence) map, with the
whole computation running live in the browser, and the failure modes (conditional-independence violation, random-CV
inflation) made first-class. Instantiated on the CAOS product-repo archetype (ADR-0057).

- **[Architecture](architecture.md)** - the archetype, the lanes, the gate, the two data contracts, determinism, deploy.
- **[Frameworks](frameworks.md)** - the WofE/CI/logistic method, the viz stack, the learned models (torch -> ONNX).
- **[Cases](cases.md)** - the 10 cases by category + their validation anchors.
- **[Guides](guides.md)** - run the precompute/retrain lane, bring your own evidence stack.

## One-paragraph orientation

The engine is the **TypeScript code** in [`frontend/src/mpm/`](../frontend/src/mpm/): Weights of Evidence (per-layer
W+/W-/contrast/studentized-C at the maximizing-contrast threshold, the posterior log-odds under conditional
independence), the conditional-independence machinery (pairwise chi-square + the Agterberg-Cheng omnibus + the CI
ratio), logistic regression (the CI-free generalization), and honest validation (success/prediction-rate capture
curves under spatial cross-validation). It runs *live in the browser* (the App recomputes the posterior raster on every
layer toggle or method switch) **and** in the offline Node bake (no Python re-port). The Python package
[`pmlab`](../data-pipeline/pmlab/) is the two data contracts + the staged pipeline + the lane gate; its default lane is
numpy-light, and a `--retrain` lane re-bakes the cases and trains the **mpm-classifier** + the **geology-ood** AE
(torch -> ONNX). The `.onnx` run live via onnxruntime-web.
