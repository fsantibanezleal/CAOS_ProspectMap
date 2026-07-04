// Weights-of-Evidence, the canonical data-driven Bayesian MPM method (Bonham-Carter, Agterberg & Wright 1988-1994;
// Bonham-Carter 1994 ch.9). For a binary evidence pattern B over the study area with deposit set D:
//   W+ = ln[ P(B|D) / P(B|D̄) ],  W- = ln[ P(B̄|D) / P(B̄|D̄) ],  contrast C = W+ - W-
//   posterior logit = prior logit + Σ W^(±)  (under conditional independence).
// Counts are unit-cell counts; the Haldane 0.5 correction guards ln(0) zero-count classes (flagged in the trace).

import type { Binarized, Cube, Posterior, WofEWeights } from './types.ts';
import { maskCells, nCells, depositSet } from './grid.ts';

export interface Counts {
  nBD: number;
  nBDbar: number;
  nBbarD: number;
  nBbarDbar: number;
}

/**
 * the 2x2 contingency table of a binary pattern B vs the deposit set D, over the active cells (missing cells excluded).
 * `deposits` overrides the cube's deposit set (used for spatial-holdout / training-fold WofE).
 */
export function contingency2x2(cube: Cube, pattern: Binarized, deposits?: Set<number>): Counts {
  const dep = deposits ?? depositSet(cube);
  let nBD = 0;
  let nBDbar = 0;
  let nBbarD = 0;
  let nBbarDbar = 0;
  for (const i of maskCells(cube)) {
    const p = pattern.present[i];
    if (p === 255) continue; // missing, contributes nothing
    const isDep = dep.has(i);
    if (p === 1) {
      if (isDep) nBD++;
      else nBDbar++;
    } else {
      if (isDep) nBbarD++;
      else nBbarDbar++;
    }
  }
  return { nBD, nBDbar, nBbarD, nBbarDbar };
}

/** WofE weights + variances + studentized contrast from a 2x2 table (Haldane 0.5 correction on any zero count). */
export function weightsFromCounts(layerId: string, threshold: number, c: Counts): WofEWeights {
  const zero = c.nBD === 0 || c.nBDbar === 0 || c.nBbarD === 0 || c.nBbarDbar === 0;
  const k = zero ? 0.5 : 0;
  const nBD = c.nBD + k;
  const nBDbar = c.nBDbar + k;
  const nBbarD = c.nBbarD + k;
  const nBbarDbar = c.nBbarDbar + k;
  const nD = nBD + nBbarD;
  const nDbar = nBDbar + nBbarDbar;

  const wPlus = Math.log(nBD / nD / (nBDbar / nDbar));
  const wMinus = Math.log(nBbarD / nD / (nBbarDbar / nDbar));
  const contrast = wPlus - wMinus;
  const sWPlus = Math.sqrt(1 / nBD + 1 / nBDbar);
  const sWMinus = Math.sqrt(1 / nBbarD + 1 / nBbarDbar);
  const sContrast = Math.sqrt(sWPlus * sWPlus + sWMinus * sWMinus);
  const studC = sContrast > 0 ? contrast / sContrast : 0;

  return {
    layerId,
    threshold,
    wPlus,
    wMinus,
    contrast,
    sWPlus,
    sWMinus,
    sContrast,
    studC,
    nBD: c.nBD,
    nBDbar: c.nBDbar,
    nBbarD: c.nBbarD,
    nBbarDbar: c.nBbarDbar,
    haldane: zero,
  };
}

/** convenience: weights for a binarized pattern over a cube (optional deposit-set override for holdout folds). */
export function weights(cube: Cube, pattern: Binarized, deposits?: Set<number>): WofEWeights {
  return weightsFromCounts(pattern.layerId, pattern.threshold, contingency2x2(cube, pattern, deposits));
}

/** prior log-odds of a deposit in a random cell: ln( N(D) / (N(S) - N(D)) ). */
export function priorLogit(cube: Cube): number {
  const nS = nCells(cube);
  const nD = depositSet(cube).size;
  return Math.log(nD / (nS - nD));
}

/**
 * The posterior prospectivity over every cell: prior logit + Σ (W+ if inside pattern j, W- if outside, 0 if missing).
 * The logit variance adds the prior-logit variance + the variance of each weight actually used (for the uncertainty
 * overlay). Optional per-layer `weightScale` (the App's transparent re-weighting slider) multiplies a layer's weight.
 */
export function posterior(
  cube: Cube,
  patterns: Binarized[],
  ws: WofEWeights[],
  weightScale?: Record<string, number>,
  deposits?: Set<number>,
): Posterior {
  const n = cube.nx * cube.ny;
  const nS = nCells(cube);
  const nD = (deposits ?? depositSet(cube)).size;
  const pLogit = Math.log(nD / (nS - nD));
  const sPrior2 = 1 / nD + 1 / (nS - nD);

  const logit = new Float64Array(n);
  const prob = new Float64Array(n);
  const sLogit = new Float64Array(n);
  const wById = new Map(ws.map((w) => [w.layerId, w]));

  for (let i = 0; i < n; i++) {
    let lg = pLogit;
    let var_ = sPrior2;
    for (const pat of patterns) {
      const w = wById.get(pat.layerId);
      if (!w) continue;
      const scale = weightScale?.[pat.layerId] ?? 1;
      const p = pat.present[i];
      if (p === 255) continue; // missing, contributes nothing
      if (p === 1) {
        lg += scale * w.wPlus;
        var_ += w.sWPlus * w.sWPlus;
      } else {
        lg += scale * w.wMinus;
        var_ += w.sWMinus * w.sWMinus;
      }
    }
    logit[i] = lg;
    prob[i] = 1 / (1 + Math.exp(-lg));
    sLogit[i] = Math.sqrt(var_);
  }
  return { priorLogit: pLogit, logit, prob, sLogit };
}
