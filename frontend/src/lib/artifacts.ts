// Load the committed CONTRACT-2 artifacts (overlaid into public/ by copy-data.mjs). The App runs the WofE engine LIVE
// (src/mpm) for full reactivity; these baked outputs are the replay fallback + the cross-case data Benchmark/
// Experiments summarise. Paths are relative to the Vite base.
import type { CaseIndex, CaseManifest, Trace } from './contract.types.ts';

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

export const loadCaseResults = () => getJSON<CaseResultsFile>('case-results.json');
export const loadLearned = () => getJSON<LearnedFile>('pm-learned.json');
export const loadIndex = () => getJSON<CaseIndex>('data/manifests/index.json');
export const loadManifest = (caseId: string) => getJSON<CaseManifest>(`data/manifests/${caseId}.json`);
export const loadTrace = (caseId: string) => getJSON<Trace>(`data/${caseId}/trace.json`);
