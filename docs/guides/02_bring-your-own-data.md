# Guide — bring your own evidence stack

The product is **applicable to NEW data**, not just the synthetic cases — that is what makes it a tool. The door is
**CONTRACT 1** (`data-pipeline/pmlab/io/contract.py`).

1. Reduce your study area OFFLINE to a co-registered evidence **cube** (a common CRS/grid/extent, one band per
   evidence layer) + a presence-only **deposit** point pattern + a study-area **mask**, and describe it as a
   case-bundle row: `case_id, nx, ny, cell_km, n_layers, n_deposits[, real_or_synthetic, deposit_type]` (see
   [`data/examples/cases.csv`](../../data/examples/cases.csv)). The raw national grids stay out of the repo
   (`data/raw/`, git-ignored); only the compact reduced cube is committed.
2. Run CONTRACT 1 over the descriptor (the pipeline does this for the built-in cases). A record is **rejected** with a
   reason if it violates the schema (non-positive grid/cell, `n_layers < 1`, `n_deposits < 1`); **flagged** if
   honesty-relevant — `presence-only-tiny` (< 10 known deposits => a black-box overfits; trust WofE + the OOD mask),
   `single-layer` (no fusion, no conditional-independence to test), or `synthetic`. Nothing is silently coerced.
3. The bake (`science/bake_cases.mjs`) runs the SAME WofE engine over the cube and the pipeline produces a compact
   trace + manifest you can replay in the SPA, exactly like the built-in cases.
4. **Honesty stays mandatory:** report capture@10% under SPATIAL cross-validation (never the fitting curve, never the
   random-CV AUC), keep `N(D)` next to every metric, and present outputs as exploration target generation, not a
   resource estimate.

If your data legitimately doesn't fit, extend CONTRACT 1 (and its tests) **deliberately** — never loosen it just to
make bad data pass. Real open datasets to start from: USGS public-domain (MRDS, NURE-HSSR, the Lawley-2022 Zn-Pb
ScienceBase grids), Geoscience Australia CC-BY (national geophysics + OZMIN).
"""
