// Cube helpers: the active-cell mask, the deposit set, the counts N(S)/N(D), and per-layer value access. The unit
// cell is the area assumed to hold at most one deposit; all WofE counts/variances are in unit-cell counts (Bonham-
// Carter 1994), so deposits are deduped to one occupied cell each.

import type { Cube, Layer } from './types.ts';

/** active cell indices (the study-area mask); defaults to every cell. */
export function maskCells(cube: Cube): number[] {
  if (cube.maskIdx && cube.maskIdx.length) return cube.maskIdx;
  const n = cube.nx * cube.ny;
  const all = new Array<number>(n);
  for (let i = 0; i < n; i++) all[i] = i;
  return all;
}

/** N(S) — number of active unit cells. */
export function nCells(cube: Cube): number {
  return maskCells(cube).length;
}

/** the deposit cells that fall inside the mask, deduped (a Set for O(1) membership). */
export function depositSet(cube: Cube): Set<number> {
  const inMask = new Set(maskCells(cube));
  const s = new Set<number>();
  for (const d of cube.depositIdx) if (inMask.has(d)) s.add(d);
  return s;
}

/** N(D) — number of occupied (deposit) cells in the mask. */
export function nDeposits(cube: Cube): number {
  return depositSet(cube).size;
}

/** look up a layer by id (throws if unknown — a silent miss would corrupt every weight). */
export function getLayer(cube: Cube, layerId: string): Layer {
  const l = cube.layers.find((x) => x.id === layerId);
  if (!l) throw new Error(`unknown layer: ${layerId}. have: ${cube.layers.map((x) => x.id).join(', ')}`);
  return l;
}

/** cell (col,row) → flat index. */
export function idx(cube: Cube, col: number, row: number): number {
  return row * cube.nx + col;
}

/** flat index → (col,row). */
export function colRow(cube: Cube, i: number): { col: number; row: number } {
  return { col: i % cube.nx, row: Math.floor(i / cube.nx) };
}

/** finite min/max of a layer over the active cells (ignores NaN). */
export function layerRange(cube: Cube, layerId: string): { min: number; max: number } {
  const l = getLayer(cube, layerId);
  let min = Infinity;
  let max = -Infinity;
  for (const i of maskCells(cube)) {
    const v = l.values[i];
    if (Number.isNaN(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}
