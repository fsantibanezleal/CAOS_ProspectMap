# 02 - The visualisation stack

ProspectMap's viz is dependency-light, lib-per-data-type, and reads live values on hover (the interactive-visualization
rubric).

- **The prospectivity raster** (`viz/MapView.tsx`) - a scalar field over the nx*ny grid painted with a
  perceptually-uniform viridis colormap into an offscreen `ImageData` at native resolution, then `drawImage`-scaled
  with `imageSmoothingEnabled=false` (the FragmentIQ `SceneView` pattern). Known deposits are a vector overlay; OOD
  ("do-not-trust") cells are dimmed/cross-hatched; hover reads the cell value. Pure canvas - no GPU/map dependency for
  a teaching-scale grid (deck.gl / MapLibre are the documented escalation path for large georeferenced cases).
- **The curves** (`viz/CurveChart.tsx` + `UPlotChart.tsx`) - the success/prediction-rate capture curves and the ROC,
  on uPlot (theme-aware, crosshair readout, the 1:1 random diagonal).
- **The math** - KaTeX (via the shell's `<Equation>` / `<InlineMath>`), in the Methodology page.
- **The shell** - `@fasl-work/caos-app-shell` provides the header/nav/theme/lang chrome, the doc-kit
  (`Tabs`/`Callout`/`Cite`/`Equation`/`ReferenceList`), and the i **Architecture modal** (ADR-0058) configured by
  `src/architecture.ts` (5 themed SVGs + bilingual bodies).
