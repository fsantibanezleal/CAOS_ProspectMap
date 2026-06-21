// The typed raster-cube + result types for the Weights-of-Evidence MPM engine. A study area is a regular grid of
// nx*ny unit cells; each evidence Layer is one value per cell; deposits are the indices of occupied cells. Everything
// downstream (WofE weights, the posterior raster, the CI checks, the capture/ROC curves, the CV folds) is computed
// over this in-memory cube — the SAME structure the browser renders and the Node bake consumes (CONTRACT-2 mirror).

export type LayerKind = 'continuous' | 'binary' | 'categorical';

/** One evidence layer: a scalar field over the grid (nx*ny values, row-major). */
export interface Layer {
  id: string;
  name: string;
  kind: LayerKind;
  /** per-cell value, length nx*ny, row-major (NaN = missing/no-data). */
  values: Float64Array;
  /** for continuous layers: does a HIGH value favour deposits (true) or a LOW value (false)? drives the sweep direction. */
  highIsFavourable?: boolean;
}

/** A study area: the grid + its evidence layers + the known-deposit point pattern. */
export interface Cube {
  nx: number;
  ny: number;
  /** unit-cell side length in km (sets the area scale; all WofE variances depend on the cell count). */
  cellKm: number;
  layers: Layer[];
  /** cell indices [0, nx*ny) that contain a known deposit (deduped to one per cell). */
  depositIdx: number[];
  /** active cells (the study-area mask); if omitted, all nx*ny cells are active. */
  maskIdx?: number[];
  /** real | synthetic — honesty flag carried from the case metadata. */
  realOrSynthetic?: 'real' | 'synthetic' | 'analytic control';
}

/** A binarized evidence pattern: 1 = inside the pattern (present), 0 = outside, 255 = missing. */
export interface Binarized {
  layerId: string;
  threshold: number;
  present: Uint8Array; // length nx*ny
}

/** WofE weights for one binary pattern, with the 2x2 counts they came from. */
export interface WofEWeights {
  layerId: string;
  threshold: number;
  wPlus: number;
  wMinus: number;
  contrast: number;
  sWPlus: number;
  sWMinus: number;
  sContrast: number;
  studC: number;
  // the 2x2 contingency counts (B vs D), for transparency + the inspector
  nBD: number;
  nBDbar: number;
  nBbarD: number;
  nBbarDbar: number;
  haldane: boolean; // was the 0.5 zero-count correction applied?
}

/** The per-cell posterior prospectivity (logit + probability + logit uncertainty). */
export interface Posterior {
  priorLogit: number;
  logit: Float64Array; // length nx*ny
  prob: Float64Array; // length nx*ny
  sLogit: Float64Array; // length nx*ny (sqrt of the posterior-logit variance)
}

/** Conditional-independence diagnostics. */
export interface CICheck {
  pairwise: { a: string; b: string; chi2: number; cramersV: number }[];
  /** Agterberg-Cheng new omnibus test: T = sum of posterior over active cells. */
  T: number;
  nD: number;
  ciRatio: number; // nD / T  (≈1 ok; <~0.85 problematic)
  sT: number;
  z: number; // (T - nD)/sT  (z >> 0 ⇒ reject CI)
}

/** A success/prediction-rate capture curve + its area-under. */
export interface CaptureCurve {
  areaFrac: number[];
  captureFrac: number[];
  aucCapture: number; // area under the capture curve (1.0 = perfect, 0.5 = random)
  captureAt10: number; // fraction of deposits captured in the top 10% of the area
}

export type FoldId = Int32Array; // per active-cell fold assignment (-1 = excluded by a buffer)
