// Deterministic synthetic study-area generator (CLEARLY LABELLED synthetic). Smooth value-noise evidence layers + a
// known latent prospectivity (a weighted sum of the informative layers) + deposits rejection-sampled on that latent
// field (probability-weighted, fixed count per case). This is the only case family with KNOWN ground-truth → the exact oracles: WofE/LR
// must recover the planted weight ORDERING (positive control), a duplicated correlated layer makes the omnibus test
// fail on purpose (the CI-trap), and an uninformative layer is the negative control. Used by the tests now and the
// synthetic cases later. Geostatistically-grounded in spirit (smooth correlated fields), not a real deposit model.

import type { Cube, Layer } from './types.ts';
import { mulberry32, sigmoid } from './rng.ts';

export interface SynthLayerSpec {
  id: string;
  /** planted weight in the latent logit (0 = uninformative/noise control). */
  weight: number;
  /** does a HIGH value favour deposits? */
  favHigh: boolean;
  /** spatial smoothness (coarse-node spacing in cells; larger = smoother). */
  coarse?: number;
}

export interface SynthSpec {
  nx: number;
  ny: number;
  seed: number;
  layers: SynthLayerSpec[];
  nDeposits: number;
  /** base latent logit (calibrated internally to hit nDeposits). */
  gain?: number;
  /** add a correlated near-duplicate of this layer id (the CI-violation trap). */
  duplicate?: { srcId: string; id: string; noise?: number };
}

/** a smooth value-noise field in [0,1] over nx*ny (coarse random nodes, bilinear interpolation). */
function valueNoise(nx: number, ny: number, coarse: number, rnd: () => number): Float64Array {
  const gx = Math.ceil(nx / coarse) + 2;
  const gy = Math.ceil(ny / coarse) + 2;
  const node = new Float64Array(gx * gy);
  for (let i = 0; i < node.length; i++) node[i] = rnd();
  const out = new Float64Array(nx * ny);
  for (let row = 0; row < ny; row++) {
    for (let col = 0; col < nx; col++) {
      const fx = col / coarse;
      const fy = row / coarse;
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const tx = fx - x0;
      const ty = fy - y0;
      const n00 = node[y0 * gx + x0];
      const n10 = node[y0 * gx + x0 + 1];
      const n01 = node[(y0 + 1) * gx + x0];
      const n11 = node[(y0 + 1) * gx + x0 + 1];
      const a = n00 * (1 - tx) + n10 * tx;
      const b = n01 * (1 - tx) + n11 * tx;
      out[row * nx + col] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

/** build a synthetic Cube with planted weights + rejection-sampled deposits. Deterministic given the seed. */
export function makeSyntheticArea(spec: SynthSpec): { cube: Cube; planted: Record<string, number> } {
  const { nx, ny, seed } = spec;
  const n = nx * ny;
  const rnd = mulberry32(seed);
  const layers: Layer[] = [];
  const fields: Record<string, Float64Array> = {};

  for (const ls of spec.layers) {
    const f = valueNoise(nx, ny, ls.coarse ?? 12, rnd);
    fields[ls.id] = f;
    layers.push({ id: ls.id, name: ls.id, kind: 'continuous', values: f, highIsFavourable: ls.favHigh });
  }
  // CI-trap: a near-duplicate (correlated) copy of an informative layer
  if (spec.duplicate) {
    const src = fields[spec.duplicate.srcId];
    const noise = spec.duplicate.noise ?? 0.08;
    const dup = new Float64Array(n);
    for (let i = 0; i < n; i++) dup[i] = Math.min(1, Math.max(0, src[i] + (rnd() - 0.5) * 2 * noise));
    fields[spec.duplicate.id] = dup;
    const srcSpec = spec.layers.find((l) => l.id === spec.duplicate!.srcId)!;
    layers.push({ id: spec.duplicate.id, name: spec.duplicate.id, kind: 'continuous', values: dup, highIsFavourable: srcSpec.favHigh });
  }

  // latent logit = Σ weightⱼ · centred favourable signal
  const gain = spec.gain ?? 6;
  const latent = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = 0;
    for (const ls of spec.layers) {
      if (ls.weight === 0) continue;
      const v = fields[ls.id][i] - 0.5;
      z += ls.weight * (ls.favHigh ? v : -v);
    }
    latent[i] = z;
  }
  // calibrate the intercept so the expected deposit count ≈ nDeposits, then place deposits by probability-weighted
  // rejection sampling until exactly nDeposits are placed (a conditioned draw, not a Poisson count)
  let lo = -20;
  let hi = 20;
  for (let it = 0; it < 60; it++) {
    const mid = (lo + hi) / 2;
    let exp = 0;
    for (let i = 0; i < n; i++) exp += sigmoid(gain * latent[i] + mid);
    if (exp > spec.nDeposits) hi = mid;
    else lo = mid;
  }
  const intercept = (lo + hi) / 2;
  const depositIdx: number[] = [];
  const used = new Set<number>();
  let guard = 0;
  while (depositIdx.length < spec.nDeposits && guard < n * 20) {
    guard++;
    const i = Math.floor(rnd() * n);
    if (used.has(i)) continue;
    const p = sigmoid(gain * latent[i] + intercept);
    if (rnd() < p) {
      used.add(i);
      depositIdx.push(i);
    }
  }

  const planted: Record<string, number> = {};
  for (const ls of spec.layers) planted[ls.id] = ls.weight;
  if (spec.duplicate) planted[spec.duplicate.id] = planted[spec.duplicate.srcId];

  return { cube: { nx, ny, cellKm: 1, layers, depositIdx, realOrSynthetic: 'synthetic' }, planted };
}
