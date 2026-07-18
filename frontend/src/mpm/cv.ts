// Spatial cross-validation, the single most important honesty mechanism. Deposit cells and their neighbours are
// not independent (spatial autocorrelation), so a random train/test split leaks neighbours of training deposits into
// the test set and the AUC/capture look fantastic and are a lie. Spatial block CV holds out whole contiguous blocks
// so train/test are spatially separated. The mandatory demonstration: the same model collapses from random-CV to
// spatial-CV (Roberts et al. 2017; Valavi et al. 2019).

import type { Cube, FoldId } from './types.ts';
import { maskCells, depositSet, colRow } from './grid.ts';
import { mulberry32 } from './rng.ts';
import { rocAuc } from './validate.ts';

/** assign each active cell to one of k folds at random (seeded). Non-mask cells = -1. */
export function randomFolds(cube: Cube, k: number, seed = 1): FoldId {
  const n = cube.nx * cube.ny;
  const folds = new Int32Array(n).fill(-1);
  const rnd = mulberry32(seed);
  for (const i of maskCells(cube)) folds[i] = Math.floor(rnd() * k) % k;
  return folds;
}

/** assign cells to k folds by contiguous spatial blocks of blockCells×blockCells (a checkerboard of folds). */
export function spatialBlockFolds(cube: Cube, k: number, blockCells: number): FoldId {
  const n = cube.nx * cube.ny;
  const folds = new Int32Array(n).fill(-1);
  const nbx = Math.ceil(cube.nx / blockCells);
  for (const i of maskCells(cube)) {
    const { col, row } = colRow(cube, i);
    const bx = Math.floor(col / blockCells);
    const by = Math.floor(row / blockCells);
    const blockId = by * nbx + bx;
    folds[i] = blockId % k;
  }
  return folds;
}

/**
 * A model-agnostic CV driver: for each fold f, fit a score from the training deposits (deposits not in fold f) via
 * scoreFn, then record the held-out cells' scores. Returns the held-out score array (assembled across folds) and its
 * ROC AUC vs the true deposit labels, the honest, leakage-controlled skill estimate.
 */
export function crossValScores(cube: Cube, folds: FoldId, k: number, scoreFn: (trainDeposits: Set<number>) => Float64Array): Float64Array {
  const n = cube.nx * cube.ny;
  const dep = depositSet(cube);
  const heldOut = new Float64Array(n).fill(NaN);
  for (let f = 0; f < k; f++) {
    const trainDep = new Set<number>();
    for (const d of dep) if (folds[d] !== f) trainDep.add(d);
    if (trainDep.size === 0) continue;
    const score = scoreFn(trainDep);
    for (const i of maskCells(cube)) if (folds[i] === f) heldOut[i] = score[i];
  }
  return heldOut;
}

export function crossValAuc(cube: Cube, folds: FoldId, k: number, scoreFn: (trainDeposits: Set<number>) => Float64Array): number {
  const heldOut = crossValScores(cube, folds, k, scoreFn);
  // AUC over the cells that got a held-out score
  const evalCube: Cube = { ...cube, maskIdx: maskCells(cube).filter((i) => !Number.isNaN(heldOut[i])) };
  return rocAuc(evalCube, heldOut, depositSet(cube));
}

/**
 * The demonstrator "model": each cell's score = exp(-d_nearest_training_deposit / scale). It captures spatial
 * autocorrelation literally, "close to a known deposit = prospective". Under random CV it LEAKS (held-out deposits
 * have training neighbours → high score → inflated AUC); under spatial-block CV it collapses (held-out blocks have no
 * nearby training deposits). The same model, two schemes → the inflation gap. (O(cells·deposits); fine for the grid.)
 */
export function nearestDepositScore(cube: Cube, scaleCells = 4): (trainDeposits: Set<number>) => Float64Array {
  const cells = maskCells(cube);
  return (trainDeposits: Set<number>) => {
    const n = cube.nx * cube.ny;
    const score = new Float64Array(n);
    const tds = [...trainDeposits].map((d) => colRow(cube, d));
    for (const i of cells) {
      const { col, row } = colRow(cube, i);
      let best = Infinity;
      for (const t of tds) {
        const dd = Math.hypot(col - t.col, row - t.row);
        if (dd < best) best = dd;
      }
      score[i] = Math.exp(-best / scaleCells);
    }
    return score;
  };
}
