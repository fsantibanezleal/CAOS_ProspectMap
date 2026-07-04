// CONTRACT 2 mirror (frontend side). MUST stay in lock-step with the Python schemas in
// data-pipeline/pmlab/core/{trace.py, manifest.py}. A drift here makes `tsc` fail -> the contract is enforced at
// BUILD time (the web cannot ship reading a shape the pipeline does not produce).

export interface LayerResult {
  id: string;
  tStar: number;
  wPlus: number;
  wMinus: number;
  contrast: number;
  studC: number;
  nBD: number;
  nBDbar: number;
  nBbarD: number;
  nBbarDbar: number;
}

export interface CICheck {
  pairwise: Array<{ a: string; b: string; chi2: number; cramersV: number }>;
  T: number;
  nD: number;
  ciRatio: number;
  sT: number;
  z: number;
}

export interface CaptureCurve {
  areaFrac: number[];
  captureFrac: number[];
  aucCapture: number;
  captureAt10: number;
}

export interface LearnedBlock {
  status: 'pending-training' | 'trained';
  classifier: Record<string, number> | null;
  ood: Record<string, number> | null;
}

export interface Trace {
  schema: string; // "prospectmap.trace/v1"
  case_id: string;
  name: string;
  category: string;
  real_or_synthetic: string;
  expected_band: string;
  validation_anchor: string;
  spec: unknown; // the synthetic SynthSpec, the browser regenerates the cube + recomputes WofE live
  layer_ids: string[];
  n_cells: number;
  n_deposits: number;
  prior_prob: number;
  layers: LayerResult[];
  posterior_summary: { min: number; max: number; mean: number };
  ci: CICheck;
  roc_auc: number;
  capture: { success: CaptureCurve; prediction: CaptureCurve };
  cv: { k: number; blockCells: number; randomAuc: number; spatialAuc: number; inflationGap: number };
  lr: { rocAuc: number; betas: Array<{ id: string; beta: number }> };
  learned: LearnedBlock;
}

export interface ArtifactRef {
  path: string;
  format: string;
  trace_schema: string;
  bytes: number;
}

export interface GateVerdict {
  lane: string;
  client_side: boolean;
  runtimes: string[];
  trace_bytes: number;
  run_ms_budget: number;
  trace_bytes_budget: number;
  reasons: string[];
}

export interface SharedArtifacts {
  models: Array<{ id: string; file: string; opset: number; kind: string }>;
  learned_metrics: string;
  case_results: string;
}

export interface CaseManifest {
  schema: string; // "prospectmap.manifest/v2"
  case_id: string;
  name: string;
  category: string;
  real_or_synthetic: string;
  expected_band: string;
  validation_anchor: string;
  engine: { package: string; version: string; model: string };
  seed: number;
  shared: SharedArtifacts;
  artifact: ArtifactRef;
  lane: 'live' | 'precompute';
  gate: GateVerdict;
  flags: Array<Record<string, unknown>>;
  metrics: Record<string, number>;
  honesty: string;
}

export interface CaseIndexEntry {
  case_id: string;
  category: string;
  manifest_path: string;
}

export interface CaseIndex {
  schema: string; // "prospectmap.index/v1"
  engine_version: string;
  n_cases: number;
  cases: CaseIndexEntry[];
}
