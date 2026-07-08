// Load the committed CONTRACT-2 artifacts (overlaid into public/ by copy-data.mjs). The App runs the WofE engine LIVE
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

export interface LearnedFile {
  schema: string;
  classifier: {
    spatial_cv?: Record<string, number | string>;
    random_cv?: Record<string, number>;
    inflation_gap?: number;
    nFolds?: number;
    nEval?: number;
  };
  ood: { auc: number; nEval: number; threshold: number };
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
