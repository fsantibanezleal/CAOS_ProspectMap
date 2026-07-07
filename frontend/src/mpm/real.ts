// The "Real sample" lane: load a baked real-data Cube (built offline by data-pipeline/pmlab/real_usmvt.py from the
// openly-licensed Lawley 2022 CMMI release) and run the SAME live WofE engine on it. Unlike the synthetic cases (which
// regenerate their cube deterministically from a SynthSpec), a real case has no generator, so the per-cell evidence
// arrays + the deposit cells are baked to public/data/<id>/cube.json and loaded here into a Cube. The honesty metadata
// (REAL vs DERIVED layers, the citation, the conditional-independence caveat) rides along and is surfaced in the App.

import type { Cube, Layer, LayerKind } from './types.ts';

export interface RealLayerMeta {
  id: string;
  name: string;
  kind: LayerKind;
  highIsFavourable: boolean;
  provenance: 'REAL' | 'DERIVED';
  values: (number | null)[];
}

export interface RealCubeFile {
  schema: string;
  case_id: string;
  name: string;
  real_or_synthetic: string;
  nx: number;
  ny: number;
  cellKm: number;
  box: { west: number; east: number; south: number; north: number; crs: string };
  layer_ids: string[];
  layers: RealLayerMeta[];
  depositIdx: number[];
  n_deposits: number;
  citation: string;
  license: string;
  honesty: string;
}

export interface RealCase {
  id: string;
  name: string;
  category: string;
  layerIds: string[];
  realOrSynthetic: 'real (open dataset)';
  file: string; // relative to the Vite base
  expectedBand: string;
  validationAnchor: string;
}

// The registered real cases (start with one; the pipeline accepts more from the same release, e.g. an Australian belt).
export const REAL_CASES: RealCase[] = [
  {
    id: 'REAL-USMVT',
    name: 'US Midcontinent MVT Zn-Pb belt (Lawley 2022, CMMI)',
    category: 'real open dataset (US public domain)',
    layerIds: ['mag', 'grav', 'lab', 'satgrav', 'faultprox', 'marginprox'],
    realOrSynthetic: 'real (open dataset)',
    file: 'data/REAL-USMVT/cube.json',
    expectedBand:
      'real geophysics + fault/margin proximity over the US Midcontinent MVT belt; a genuine, non-toy prospectivity signal',
    validationAnchor:
      'faultprox/marginprox carry real positive contrast; the CI omnibus is EXPECTED to fire on correlated geophysics, routing to logistic regression',
  },
];

const _byId = new Map(REAL_CASES.map((c) => [c.id, c]));
export function realCaseById(id: string): RealCase {
  const c = _byId.get(id);
  if (!c) throw new Error(`unknown real case: ${id}`);
  return c;
}

/** Build a live Cube from a baked real-cube file (NaN for missing cells). */
export function cubeFromFile(f: RealCubeFile): Cube {
  const n = f.nx * f.ny;
  const layers: Layer[] = f.layers.map((l) => {
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = l.values[i];
      values[i] = v == null ? NaN : v;
    }
    return { id: l.id, name: l.name, kind: l.kind, values, highIsFavourable: l.highIsFavourable };
  });
  return {
    nx: f.nx,
    ny: f.ny,
    cellKm: f.cellKm,
    layers,
    depositIdx: f.depositIdx,
    realOrSynthetic: 'real',
  };
}
