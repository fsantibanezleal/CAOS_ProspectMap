# Determinism + the trace

**A run is a pure function of `(params, seed)`.** Use `core/rng.py :: make_rng(seed)`, never a global/implicit
RNG. Same inputs ⇒ byte-identical artifact (asserted in `tests/test_pipeline_smoke.py`). This is what makes the
committed artifact a trustworthy source-of-truth the SPA merely animates (ADR-0052 / ADR-0054).

**The trace** (`core/trace.py`, schema `prospectmap.trace/v1`) is the compact per-case replay artifact, not the
raw solver state. `build_trace()` carries the case SPEC (so the browser can regenerate the synthetic cube and
recompute WofE live), the per-layer weights, the posterior summary, the CI diagnostics, the capture curves, the
random-vs-spatial-CV gap, the logistic-regression comparison and the learned-model metrics. The learned-lane
intermediates stay local (git-ignored `data/raw/`); only the compact trace is committed and shipped. Its shape is
mirrored by `frontend/src/lib/contract.types.ts` (CONTRACT 2).
