// Honest validation — the product's spine. The capture (prediction-rate) curve: rank cells by descending
// prospectivity, plot % of deposits captured vs % of area (Chung & Fabbri 2003). capture@10% is the headline
// target-generation number. ROC/AUC is secondary (presence-only caveat) via the Mann-Whitney rank statistic.

import type { CaptureCurve, Cube } from './types.ts';
import { maskCells, depositSet } from './grid.ts';

/**
 * The capture / prediction-rate curve for a per-cell score over the active cells. Cells are ranked by descending
 * score; the curve is (fraction of area selected, fraction of deposits captured). aucCapture is the area under it
 * (1 = perfect ranking, 0.5 = random). `evalDeposits` lets the caller pass a HELD-OUT deposit set (spatial CV); it
 * defaults to all deposits in the mask (the fitting / success-rate curve).
 */
export function captureCurve(cube: Cube, score: Float64Array, evalDeposits?: Set<number>, nPts = 200): CaptureCurve {
  const cells = maskCells(cube);
  const dep = evalDeposits ?? depositSet(cube);
  const order = [...cells].sort((a, b) => score[b] - score[a]);
  const nS = order.length;
  const nD = dep.size;
  if (nS === 0 || nD === 0) return { areaFrac: [0, 1], captureFrac: [0, 0], aucCapture: 0.5, captureAt10: 0 };

  // walk the full ranking; accumulate AUC (trapezoid) and capture@10%; sample the curve at nPts for charting.
  const areaFrac: number[] = [0];
  const captureFrac: number[] = [0];
  let captured = 0;
  let auc = 0;
  let prevArea = 0;
  let prevCap = 0;
  let captureAt10 = 0;
  const sampleEvery = Math.max(1, Math.floor(nS / nPts));
  for (let r = 0; r < nS; r++) {
    if (dep.has(order[r])) captured++;
    const area = (r + 1) / nS;
    const cap = captured / nD;
    auc += ((cap + prevCap) / 2) * (area - prevArea);
    if (prevArea < 0.1 && area >= 0.1) captureAt10 = cap;
    prevArea = area;
    prevCap = cap;
    if (r % sampleEvery === 0 || r === nS - 1) {
      areaFrac.push(area);
      captureFrac.push(cap);
    }
  }
  if (captureAt10 === 0) captureAt10 = prevCap >= 1 ? 1 : captured / nD; // tiny grids
  return { areaFrac, captureFrac, aucCapture: auc, captureAt10 };
}

/**
 * ROC AUC via the Mann-Whitney U rank statistic (no sklearn). `positives` = the deposit cells; the rest of the mask
 * are pseudo-absences (presence-only caveat: "negatives" may host undiscovered deposits). Ties get average ranks.
 */
export function rocAuc(cube: Cube, score: Float64Array, positives?: Set<number>): number {
  const cells = maskCells(cube);
  const pos = positives ?? depositSet(cube);
  const order = [...cells].sort((a, b) => score[a] - score[b]); // ascending
  // average ranks (1-based), handling ties
  const rank = new Map<number, number>();
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && score[order[j + 1]] === score[order[i]]) j++;
    const avg = (i + 1 + (j + 1)) / 2;
    for (let k = i; k <= j; k++) rank.set(order[k], avg);
    i = j + 1;
  }
  let sumPos = 0;
  let nPos = 0;
  for (const c of cells) {
    if (pos.has(c)) {
      sumPos += rank.get(c)!;
      nPos++;
    }
  }
  const nNeg = cells.length - nPos;
  if (nPos === 0 || nNeg === 0) return 0.5;
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/** capture fraction at a given area budget (e.g. captureAt(curve, 0.1)). */
export function captureAt(curve: CaptureCurve, areaBudget: number): number {
  for (let i = 0; i < curve.areaFrac.length; i++) {
    if (curve.areaFrac[i] >= areaBudget) return curve.captureFrac[i];
  }
  return curve.captureFrac[curve.captureFrac.length - 1];
}
