// Converting a continuous evidence layer to the binary pattern WofE consumes, by a threshold. The defensible,
// data-driven choice is the maximizing-contrast threshold t* = argmax_t C(t) s.t. studC(t) ≥ floor (Bonham-Carter
// 1994 ch.9; Carranza 2009). The App always binarizes at t* (a live C(t) curve + threshold control is planned).

import type { Binarized, Cube, Layer, WofEWeights } from './types.ts';
import { getLayer, layerRange } from './grid.ts';
import { contingency2x2, weightsFromCounts } from './wofe.ts';

/** binarize a layer at threshold t. highIsFavourable ⇒ present = value ≥ t; else present = value ≤ t. NaN ⇒ missing(255). */
export function binarize(layer: Layer, t: number): Binarized {
  const n = layer.values.length;
  const present = new Uint8Array(n);
  const high = layer.highIsFavourable !== false; // default: high favours
  for (let i = 0; i < n; i++) {
    const v = layer.values[i];
    if (Number.isNaN(v)) {
      present[i] = 255;
    } else {
      present[i] = (high ? v >= t : v <= t) ? 1 : 0;
    }
  }
  return { layerId: layer.id, threshold: t, present };
}

/** a binary layer (kind 'binary') is already a pattern: present = value > 0.5. */
export function binaryPattern(layer: Layer): Binarized {
  const n = layer.values.length;
  const present = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = layer.values[i];
    present[i] = Number.isNaN(v) ? 255 : v > 0.5 ? 1 : 0;
  }
  return { layerId: layer.id, threshold: 0.5, present };
}

export interface SweepPoint {
  t: number;
  contrast: number;
  studC: number;
  wPlus: number;
  wMinus: number;
}

/** sweep the threshold over a continuous layer's value range, recomputing C(t) and studC(t) at each step. */
export function thresholdSweep(cube: Cube, layerId: string, steps = 40): SweepPoint[] {
  const layer = getLayer(cube, layerId);
  const { min, max } = layerRange(cube, layerId);
  const pts: SweepPoint[] = [];
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return pts;
  for (let s = 0; s < steps; s++) {
    // interior thresholds (avoid the degenerate all-in / all-out endpoints)
    const t = min + ((max - min) * (s + 0.5)) / steps;
    const w = weightsFromCounts(layerId, t, contingency2x2(cube, binarize(layer, t)));
    pts.push({ t, contrast: w.contrast, studC: w.studC, wPlus: w.wPlus, wMinus: w.wMinus });
  }
  return pts;
}

/** the maximizing-contrast threshold t* = argmax_t C(t) subject to studC(t) ≥ floor (default 1.5). */
export function maximizingContrast(cube: Cube, layerId: string, studCFloor = 1.5, steps = 40): { tStar: number; sweep: SweepPoint[] } {
  const sweep = thresholdSweep(cube, layerId, steps);
  let best: SweepPoint | null = null;
  for (const p of sweep) {
    if (p.studC < studCFloor) continue;
    if (!best || p.contrast > best.contrast) best = p;
  }
  // fall back to the max-contrast point ignoring the floor if nothing clears it
  if (!best) for (const p of sweep) if (!best || p.contrast > best!.contrast) best = p;
  return { tStar: best ? best.t : NaN, sweep };
}

/** weights at the maximizing-contrast threshold for a layer (the App's default per-layer pattern). */
export function bestWeights(cube: Cube, layerId: string, studCFloor = 1.5): { weights: WofEWeights; pattern: Binarized; tStar: number } {
  const layer = getLayer(cube, layerId);
  if (layer.kind === 'binary') {
    const pattern = binaryPattern(layer);
    return { weights: weightsFromCounts(layerId, 0.5, contingency2x2(cube, pattern)), pattern, tStar: 0.5 };
  }
  const { tStar } = maximizingContrast(cube, layerId, studCFloor);
  const pattern = binarize(layer, tStar);
  return { weights: weightsFromCounts(layerId, tStar, contingency2x2(cube, pattern)), pattern, tStar };
}
