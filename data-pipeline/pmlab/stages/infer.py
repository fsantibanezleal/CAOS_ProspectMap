"""Stage 4 — infer (heavy lane): run every case through the SAME TS engine the browser runs (frontend/src/mpm/, via
tsx) — regenerate the synthetic cube and compute the per-layer WofE weights, the combined posterior, the CI
diagnostics, the success + prediction-rate capture curves, the random-vs-spatial-CV inflation gap and the logistic-
regression comparison — and bake the deterministic per-case outputs to data/derived/case-results.json. Delegates to
pmlab/science/bake_cases.mjs."""
