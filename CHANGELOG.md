# Changelog

All notable changes to ProspectMap. Format: [Keep a Changelog](https://keepachangelog.com); versions are X.XX.XXX.

## [0.11.000] · 2026-09-18

### Changed - the Weights-of-Evidence cross-validation is fully out of fold (#44)

- Each fold's model is fitted on the training folds' cells only: every layer's maximizing-contrast threshold, its
  weights and the prior. Before, the thresholds were chosen on all deposits, so a held-out fold's labels moved the
  thresholds that scored it; and the refit counted the whole held-out fold as non-deposit cells (label-free, but it
  lowered the result).
- Every case's `cv` block and prediction capture curve change. REAL-USMVT: spatial-block CV 0.637 to 0.648, random CV
  0.723 to 0.739. The largest random-minus-spatial gap is now K-IOCG (0.110, REAL-USMVT 0.091); spatial exceeds
  random in K-OROGENIC and D-SPARSE. The fitting AUCs, the weights and the omnibus test are unchanged.
- Learned lanes, same folds: on the synthetic labelled rows the MLP stays at 0.971 and WofE moves from 0.929 to 0.868
  (its AUC now averages tied ranks, as the engine does); on all REAL-USMVT cells the MLP stays at 0.908 against WofE
  0.648. The real lane adds the engine's out-of-fold logistic regression under the same folds (0.644 spatial, 0.744
  random) as a reference beside the distance-to-known-deposit baseline (0.899, 0.960). All ONNX models are unchanged.
- The committed bake was stale against the pairwise conditional-independence test fixed in 0.08.000 (the table read 0
  for most pairs, and for all 15 REAL-USMVT pairs); the re-bake carries the fixed values. The omnibus test is computed
  on the full-data fit and is unchanged.
- Engine tests for the out-of-fold protocol (a held-out fold's labels cannot move its scores; the fit counts the
  training cells only), and a CI job that runs the engine tests and the typecheck, which did not run in CI before.
- `--retrain` now also bakes the real case and runs the real learned lane; it used to drop REAL-USMVT from
  `case-results.json`.

### Changed - scope statements and wording (#44)

- Manifests carry a case-specific `scope` statement instead of one global text that described every case, the real
  one included, as synthetic (schema `prospectmap.manifest/v3`). The learned-metrics files (`prospectmap.learned/v3`)
  and `pu-conformal.json` (`prospectmap.puconformal/v2`) rename their descriptive field to `scope`.
- `pu-conformal.json` states the k-means regions its folds use; it said "blockId % k, identical to the live TS
  engine". Its numbers are unchanged (the lane reproduces its committed output exactly).
- Self-assessment wording is removed from the App pages, the Architecture modal and its diagrams, and the docs; the
  facts they carried stay.
- The CV tab and the Focus view describe a random-minus-spatial gap as inflation only when it is positive; under the
  new cross-validation D-SPARSE has a negative gap. The Implementation page lists the current engine tests (17) and
  cases (11).
- README, ATTRIBUTION, `data/README.md` and the cases docs describe the real REAL-USMVT data and its licence; they
  described all data as synthetic or the real data as a next step (#31). The README headline quotes the current
  values with their protocols, and the contiguous-region MLP (0.783) beside the distance baseline (0.783).
- The real-data cube builder's dependencies are pinned in `data-pipeline/requirements-precompute.txt` (rasterio
  1.5.0, pyshp 3.1.4, scipy 1.18.0, numpy 2.4.6); with them the build reproduced the committed `cube.json` byte for
  byte from the CMMI files (checked on Windows with Python 3.12) (#31).

### Added - data for the technical report (#44)

- `manuscripts/prospectivity/data/build_pm.py` builds `pm.json` from the artifacts, with a key per protocol: per
  layer the studentized contrast with its 2x2 counts, s(C), a second small-count correction and exact statistics
  (Fisher's exact test and the exact interval for C, which need no correction); the like-for-like pairs, the
  logistic regression and the distance baseline; the omnibus and pairwise conditional-independence tests of the
  full-data fit. A test keeps `pm.json` equal to a fresh build.
- `figures/make_figs.py` labels every AUC with its protocol and plots the studentized contrast with the 1.645 and 1.96
  levels. Version 1.1 of the report plotted the binarization threshold tStar under that label.

## [0.10.001] · 2026-09-18

### Fixed - the real lane set a cross-validated MLP AUC against a WofE AUC fitted without cross-validation (#41)

`pm-learned-real.json` stored the Weights-of-Evidence AUC fitted on all deposits and scored on the same cells (0.732,
no cross-validation) as `classifier.spatial_cv.wofe_roc_auc`, and set `winner: "mlp"` by comparing the MLP's
labelled-sample CV (0.946: 858 deposit plus 2574 buffered cells, shuffled blocks, mean of per-fold AUCs) with the WofE
spatial-CV AUC pooled over all 25344 map cells (0.637). Different cell sets and aggregations: the flag compared
nothing.

- The MLP is now scored under exactly the protocol of the WofE cross-validation AUCs in `case-results.json`: the
  engine's folds (exported through the TS engine itself by the new `science/real_wofe_oof.mjs`), each model refitted
  on the training folds only, every map cell scored once while held out, one pooled rank AUC. `real_learned.py` stops
  unless the WofE AUCs it re-derives from the export equal the bake. Spatial CV: MLP 0.908, WofE 0.637. Random CV:
  MLP 0.939, WofE 0.723.
- The engine's distance-to-known-deposit baseline is scored under the same folds and reported beside the pair: 0.899
  spatial, 0.960 random. It learns no geology and comes within 0.010 of the MLP, so this protocol cannot show that the
  MLP learned more than proximity to known deposits. The engine's folds interleave blocks; the contiguous-fold
  benchmark (`pu-conformal.json`) stays the stricter transfer test.
- Schema `prospectmap.learned/v2`: a machine-readable `protocol`; `winner` from the like-for-like spatial pair only;
  `nocv.wofe_roc_auc` (0.732) for the fitting AUC; the old labelled-sample values under `labelled_sample_cv` with their
  protocol and no WofE counterpart; AUCs written at 6 decimals so the App's 3-decimal display matches the live engine.
  Both real ONNX models are byte-identical (the original computations keep their RNG order).
- The replay pipeline (`run.py all`, run on every deploy) wrote the synthetic lane's `pm-learned.json` (0.971 vs
  0.929, winner "mlp") into the REAL-USMVT trace and manifest. Each case now carries its own lane's learned metrics and
  ONNX pointers.

### Changed - every AUC in the App states its protocol

- What-if (MLP): the MLP and WofE values under the one shared protocol, stated in EN/ES, with the like-for-like
  verdict, the distance baseline and the interleaved-folds caveat. A real-lane metrics file that does not declare its
  protocol shows no pairing and no verdict.
- Method compare gains a protocol column; the cross-validated MLP value no longer sits in the ranking column of the
  fitting AUCs. The Map, ROC, CV, overlay, PU-Conformal, Benchmark and Experiments labels name their protocol (fit and
  no CV, spatial CV, random CV, contiguous folds). Benchmark shows each lane's like-for-like pair with its protocol.
- The pipeline's version constant follows `VERSION` again (it had stayed at 0.08.000).

## [0.10.000] · 2026-08-01

### Fixed - up to 54% of the map was deleted, and nothing could scroll to it

`MapView` sized its canvas from container WIDTH alone (`scale = clientWidth / nx`), never reading the height
of the box it was in. Measured live at 1600x900: the synthetic lane drew 1238x1238, **106.4% of the
viewport**; the real Lawley US-MVT lane (a taller 144x176 cube) drew 1238x1513, **130.1%**, with 816px past
the clipped shell bottom. Twelve wheel ticks and the End key left every ancestor at `scrollTop` 0. For a
product whose single output IS the map, between 42% and 54% of the instrument was unreachable.

- The canvas now contain-fits: `scale = min(availW / nx, availH / ny)`, centred.
- The height constraint is propagated down `.pf-main / .pf-tabs / .pf-tabpanel`, because sizing the layout
  root alone never reached the canvas: every descendant in between grew to content.
- The rail scrolls itself. The method chips (WofE / logistic) sat at y 881-913 against a 900px shell, so
  the control that switches the whole analysis method was 13px out of reach.
- Prose routes get their own scroll (floor v2).

**A bug the fix itself introduced, caught before shipping:** `onMove` derived the hovered cell from the
WRAPPER rect. That was correct only while the canvas filled the wrapper edge to edge; once it is
contain-fitted and centred the two differ by the letterbox margin and every readout would report the wrong
cell. The readout is the only way to interrogate a value on this map, so a silent offset is worse than no
readout. Rebased on the canvas rect.

### Changed - tabs regularized per ADR-0071

Twelve flat tabs are now FIVE groups on one 45px row (Map, Evidence, Skill, Compare, Learned), sub-views
revealed on hover from the same tab.

### Added - ADR-0070 focus mode, with the honest negative as the headline

A full-viewport view of the selected area, recomputing WofE LIVE over the active evidence layers. The state
NAMED on the stage is the spatially-blocked CV verdict, and the inflation gap is a first-class HUD value
beside the spatial and random AUC.

This is deliberate. The finding this product exists to report is that regional geophysics alone has little
spatial-transfer skill on a clustered MVT belt: random CV looks good because it splits neighbouring cells
across train and test. A focus view showing a confident-looking posterior map WITHOUT that number beside it
would be the most misleading screen in the line. The rail also states that the posterior is our own
recomputation, not the published model.

## [0.09.000] · 2026-07-30

### Fixed
- **Version coherence.** Every version source in this repo now declares the same number. They had drifted
  apart (0.08.002), which `conventions/versioning.md` forbids: `VERSION`, the manifests, the CHANGELOG and the
  git tag are required to move together on every release.
- A line-wide sweep on 2026-07-30 found 79 tags across 9 CAOS repos pointing at commits declaring a
  different version, plus 13 repos whose working tree was internally incoherent. The cause is one habit: a
  release gets merged, tagged and deployed while the version files stay where they were. The cost is not
  cosmetic, since a product footer reads its version from a manifest, so a deployed app reported a version
  older than the release it was running.
- This is a MINOR bump rather than a patch: it puts the whole repo onto one clean number regardless of
  development stage, so the numbering is in order from here rather than carrying the drift forward.
- Historical tags are left untouched. A published tag is the accurate record of a release that happened, so
  drift is fixed by moving the files forward, never by rewriting or deleting a tag.
- Guarded going forward by `tools/version-audit/check_version_coherence.py` in CAOS_MANAGE.

## [0.08.002] - 2026-07-11

### Added
- Per-panel error boundary (`viz/PanelBoundary.tsx`, mirroring the RotorVitals reference): a crash inside one
  prospectivity view now renders a small inline message instead of unmounting the whole App to a blank page; the
  tab bar stays usable.

## [0.08.001] - 2026-07-11

### Fixed
- Reference integrity (ADR-0017 §4): all 13 `citations.ts` entries were link-less, so every on-page citation
  rendered as non-clickable text. Populated a real, verified `doi` or `url` for the 11 that have one (Agterberg
  & Cheng 2002, Chung & Fabbri 2003, Roberts 2017, Rodriguez-Galiano 2015, Xiong & Zuo 2021, Elkan & Noto 2008,
  Lawley 2022, Kiryo 2017, Angelopoulos & Bates 2021, Bonham-Carter 1994, Carranza 2009). The two genuinely
  pre-DOI print sources (Bonham-Carter et al. 1989 GSC Paper 89-9; Agterberg & Bonham-Carter 1990 APCOM
  proceedings) are explicitly marked as having no open landing page (never a fabricated link).
- Docs wiki: plain-text `doi:` / `arXiv:` citations in `docs/architecture/06` and `docs/frameworks/04,05,06`
  now render as clickable `doi.org` / `arxiv.org` links.

## [0.08.000] - 2026-07-07

The PU-Conformal beyond-SOTA lane, the RF/GBM SOTA-classical rung, a validated honest benchmark with
negative controls, deeper docs, and a pre-existing conditional-independence table fix.

### Added
- **PU-Conformal lane** (`data-pipeline/pmlab/pu_conformal.py`), trained OFFLINE on the real US
  Midcontinent MVT cube and exported as `mpm-puconformal-real.onnx` + `pu-conformal.json`. It composes
  three verified ingredients the WofE/logistic ladder lacks: a non-negative PU risk estimator (nnPU,
  Kiryo et al. 2017, arXiv:1703.00593; deposits = positives, ALL other cells = UNLABELED, fixing the
  pseudo-negative bias; Elkan & Noto 2008, doi:10.1145/1401890.1401920), spatially-blocked evaluation
  (Roberts et al. 2017, doi:10.1111/ecog.02881), and a distribution-free positive-class split-conformal
  band (Angelopoulos & Bates 2021, arXiv:2107.07511). The class prior pi is a swept sensitivity
  parameter.
- **SOTA-classical tabular rung**: random forest + gradient boosting (Rodriguez-Galiano et al. 2015,
  doi:10.1016/j.oregeorev.2015.01.001), plus logistic and a naive pseudo-negative MLP, all scored on
  IDENTICAL contiguous spatial folds so PU-Conformal is judged against the real ML frontier.
- **New App tab "PU-Conformal"** (11 -> 12 tabs): the calibrated nnPU posterior map, the
  coverage-guaranteed prospective set, a coverage/set-size readout, and a class-prior pi sensitivity
  selector. Live via onnxruntime-web on the real cube.
- **Benchmark + Experiments** now carry the six-model head-to-head (block-CV AUC with bootstrap CIs, AP,
  Brier, ECE), the mandatory negative-control matrix (label permutation, uninformative layer,
  distance-to-deposit null), and the conformal coverage table.
- **Deep docs**: `docs/frameworks/{04_ml-ladder,05_pu-learning,06_uncertainty-and-conformal}.md`, an
  extended `docs/architecture/06_model-evaluation.md`, three new Methodology SubTabs (tabular ML / PU /
  conformal) with term-by-term math, and inline real-DOI citations across the pages.

### Fixed
- **Conditional-independence pairwise table read 0.00** on every case. `ci.ts::pairwiseChi2` built its
  2x2 over the DEPOSIT cells only, which degenerates at the maximizing-contrast threshold (favourable
  patterns are present at nearly every deposit). Now the proper stratified (2x2x2) conditional test,
  summing the deposit and non-deposit strata; regression test added.

### Honesty
- Under strict contiguous spatial holdout PU-Conformal (block-CV AUC 0.656) does NOT beat classical
  WofE (0.732); the trivial distance-to-deposit null alone reaches 0.783, so most apparent skill is
  spatial proximity, not learned geology. The negative controls collapse as required and the conformal
  band meets its coverage guarantee (0.98 >= 0.90) but only by flagging ~88% of the belt: an honest,
  near-vacuous prospective set. The genuine advance is calibrated, bias-corrected uncertainty, not a
  higher AUC. No fabricated "beats SOTA" number.

## [0.07.000] - 2026-07-07

First-level Synthetic | Real Source selector (the Faena "Real sample" lane) on real, openly-licensed data.

### Added
- **Real-data lane on the US Midcontinent MVT Zn-Pb belt** from the Lawley et al. (2022) Tri-National
  Critical Minerals Mapping Initiative (CMMI) open release (USGS ScienceBase item
  6193e9f3d34eb622f68f13a5, data DOI 10.5066/P970GDD5, paper DOI 10.1016/j.oregeorev.2021.104635;
  US public domain). New offline builder `data-pipeline/pmlab/real_usmvt.py` clips + rasterizes the
  evidential layers to a 144x176 grid (~5.4 km cells) over W-97/E-88/S35/N43.5: REAL measured
  geophysics (`mag`, `grav`, `lab` depth-to-LAB tomography, `satgrav`), DERIVED proximity layers
  (`faultprox`, `marginprox`), and `depositIdx` = cells with a real Pb-Zn (MVT/CD) occurrence (858
  deposit cells from 5837 occurrences). Bakes `data/derived/REAL-USMVT/cube.json` + `provenance.json`.
- **FIRST-LEVEL Source selector** (`Synthetic | Real sample`) at the top of the Tool sidebar. In Real
  mode the synthetic case knobs (planted-weight generators) disable, you pick the real datum, and all
  App tools run live on the real Cube. The `realOrSynthetic: 'real (open dataset)'` contract path is
  now wired.
- **Two new method tabs** (method-tab floor 9 -> 11): a fuzzy-logic / index-overlay combiner (gamma
  operator, no fitting, no CI assumption; Bonham-Carter 1994, Carranza 2009) and a calibration /
  reliability readout (decile reliability diagram + Brier + ECE; the C-SATURATE case had flagged it
  missing).
- **Real learned models** (`data-pipeline/pmlab/real_learned.py`): `mpm-classifier-real.onnx` +
  `geology-ood-real.onnx` retrained on the real 6-feature cube (the synthetic 4-feature ONNX are NOT
  silently applied to the real cube). Spatial-block CV, distance-buffered presence-only negatives.
- **In-app honesty block** in the real lane: REAL / DERIVED / RECOMPUTED legend, the Lawley 2022
  citation + license, and the conditional-independence caveat (real geophysics is physically correlated,
  so the omnibus test fires; route to logistic regression). The posterior is stated as OUR browser WofE
  recomputation, NOT the published H3 + gradient-boosting model.
- Engine: `analyzeCube(cube, layerIds)` runs the full WofE/CI/validation/LR analysis on an already-built
  Cube; `analyzeCase` now delegates to it. Node bake `science/bake_real.mjs` emits the real
  `trace.json` + merges the real case into `case-results.json` through the SAME TS engine.

### Changed
- Version sources normalized to `0.07.000` (X.XX.XXX): `frontend/package.json`, root `VERSION`,
  `CHANGELOG`, and the footer (the shell version is now derived from `package.json`, no longer a
  hard-coded string that drifted to `0.06.000`).

## [0.06.001] - 2026-07-04

### Changed
- Content standards (ADR-0067): removed every em-dash from tracked content (replaced with commas, or
  "n/a" in table cells). No behaviour change. Added `scripts/check_content_standards.py` + wired it
  into the CI `guards` job so the repo cannot regress on em-dashes or emojis.

## [0.06.000] - 2026-06-21

First full build on the product-repo archetype (ADR-0057). Weights-of-Evidence mineral prospectivity mapping.

### Added
- **The WofE engine** (`frontend/src/mpm/`, dependency-free TypeScript, live in-browser + Node-bakeable): WofE
  (W+/W-/contrast/studentized-C/posterior under CI, Haldane 0.5 guard) . the maximizing-contrast threshold sweep .
  the CI machinery (pairwise chi-square + the Agterberg-Cheng omnibus + the CI ratio) . logistic regression (IRLS +
  ridge) . honest validation (success/prediction-rate capture + ROC) . spatial vs random cross-validation . the
  deterministic synthetic study-area generator. 11 node:test oracles.
- **The Python core** (`data-pipeline/pmlab/`): CONTRACT 1 (a case-bundle ingestion gate) + CONTRACT 2
  (`prospectmap.trace/v1` + `manifest/v2`) + the 10 K-/D-/C- cases + the numpy-light pipeline + the two-language bake.
- **The 6-page SPA** on `@fasl-work/caos-app-shell`: the interactive canvas prospectivity map + 9 tabs + the 5 doc
  pages + the i Architecture modal (ADR-0058).
- **The 2 learned models** (torch -> ONNX): `mpm-classifier` (presence-only MLP) + `geology-ood` (autoencoder), run
  live via onnxruntime-web. Honest spatial-CV: classifier 0.971 vs WofE 0.929; OOD AUC 1.0 (on a synthetic
  out-of-band eval set, separable by construction).
- The docs/ wiki (ADR-0056), CI + deploy-pages, the root files.

[0.06.000]: https://github.com/fsantibanezleal/CAOS_ProspectMap/releases/tag/v0.06.000
