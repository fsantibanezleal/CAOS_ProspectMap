// Load the committed Contract-2 artifacts (overlaid into public/ by copy-data.mjs). The App runs the WofE engine live
// (src/mpm) for full reactivity; these baked outputs are the replay fallback + the cross-case data Benchmark/
// Experiments summarise. Paths are relative to the Vite base.
import type { CaseIndex, CaseManifest, Trace } from './contract.types.ts';
import type { RealCubeFile } from '../mpm/real.ts';

const base = () => import.meta.env.BASE_URL || '/';

async function getJSON<T>(rel: string): Promise<T> {
  const r = await fetch(`${base()}${rel}`);
  if (!r.ok) throw new Error(`fetch ${rel} -> ${r.status}`);
  return (await r.json()) as T;
}

export interface CaseResultsFile {
  schema: string;
  nCases: number;
  cases: Record<string, unknown>;
}

/** One cross-validation scheme of the learned-vs-WofE head-to-head: both values come from ONE protocol. */
export interface LearnedPair {
  mlp_roc_auc?: number;
  wofe_roc_auc?: number;
  winner?: 'mlp' | 'wofe' | 'tie';
  /** the engine's distance-to-known-deposit baseline under the same protocol: a reference, never a contestant */
  distance_null_roc_auc?: number;
}

/** The shared head-to-head protocol, declared by the real lane (prospectmap.learned/v2). The synthetic lane's file
 * (v1) does not carry it; its protocol is fixed by science/gen_train.mjs + train_mpm.py (see lib/learned.ts). */
export interface LearnedProtocol {
  cell_set: string;
  n_cells: number;
  n_deposit_cells: number;
  k: number;
  block_cells: number;
  random_seed: number;
  folds?: string;
  refit?: string;
  aggregation?: string;
  baseline?: string;
}

export interface LearnedFile {
  schema: string;
  case_id?: string;
  classifier: {
    protocol?: LearnedProtocol;
    spatial_cv?: LearnedPair;
    random_cv?: LearnedPair;
    inflation_gap?: number;
    /** the WofE AUC WITHOUT cross-validation (fitted and scored on the same cells); never a CV value */
    nocv?: { wofe_roc_auc: number; protocol: string };
    /** the MLP-only CV on its labelled sample (a different cell set, no WofE counterpart) */
    labelled_sample_cv?: {
      protocol: string; n_pos: number; n_neg: number;
      spatial_mlp_roc_auc: number; random_mlp_roc_auc: number; inflation_gap: number;
    };
    mlp_roc_auc?: number;
    nFolds?: number;
    nEval?: number;
  };
  ood: { auc: number | null; nEval: number; threshold: number };
  honesty: string;
}

export interface ConformalLevel {
  alpha: number; nominal: number; threshold: number; empirical_coverage: number; set_size_frac: number;
}
export interface BenchmarkRow {
  model: string; label: string; auc: number; ap: number; brier: number; ece: number; auc_ci95: [number, number];
}
export interface PuConformalFile {
  schema: string;
  case_id: string;
  features: string[];
  protocol: { folds: number; block_cells: number; scheme: string; pi_default: number };
  benchmark: BenchmarkRow[];
  inflation: { pu_random_cv_auc: number; pu_spatial_cv_auc: number };
  conformal: { pi: number; nCalibPos: number; nTestPos: number; levels: ConformalLevel[] };
  pi_sensitivity: { pi: number; block_cv_auc: number; conformal: ConformalLevel[] }[];
  negative_controls: {
    label_permutation: { wofe_auc: number; pu_auc: number; expectation: string };
    uninformative_layer: { pu_auc_with_noise_layer: number; pu_auc_without: number; expectation: string };
    distance_to_deposit_null: { distance_to_deposit_auc: number; expectation: string };
  };
  verdict: { ranking_win: boolean; coverage_within_tolerance: boolean; text: string };
  references: Record<string, string>;
  honesty: string;
}

export const loadCaseResults = () => getJSON<CaseResultsFile>('case-results.json');
export const loadLearned = () => getJSON<LearnedFile>('pm-learned.json');
export const loadLearnedReal = () => getJSON<LearnedFile>('pm-learned-real.json');
export const loadPuConformal = () => getJSON<PuConformalFile>('pu-conformal.json');
export const loadRealCube = (rel: string) => getJSON<RealCubeFile>(rel);
export const loadIndex = () => getJSON<CaseIndex>('data/manifests/index.json');
export const loadManifest = (caseId: string) => getJSON<CaseManifest>(`data/manifests/${caseId}.json`);
export const loadTrace = (caseId: string) => getJSON<Trace>(`data/${caseId}/trace.json`);
