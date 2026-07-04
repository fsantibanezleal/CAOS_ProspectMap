// Logistic regression, the CI-free generalization of WofE. logit P = β₀ + Σ βⱼ xⱼ, fit by IRLS / Newton-Raphson on
// the penalized log-likelihood (L2 ridge on the slopes, not the intercept) for the tiny imbalanced positive set.
// On conditionally-independent binary patterns βⱼ ≈ the WofE contrast Cⱼ (the WofE↔LR equivalence). Cheap → runs live.

import { sigmoid } from './rng.ts';

export interface LRFit {
  beta: number[]; // [intercept, β₁ … β_k]
  se: number[]; // standard errors (inverse Fisher information diagonal)
  iters: number;
  converged: boolean;
  ll: number; // final log-likelihood
}

/** solve A x = b for a small dense system (Gaussian elimination, partial pivoting). Returns null if singular. */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/**
 * Fit logistic regression by IRLS. X = rows of feature vectors (WITHOUT the intercept column, added here).
 * ridge = L2 penalty on the slopes (not the intercept). Returns coefficients + their standard errors.
 */
export function fitLR(
  X: number[][],
  y: number[],
  opts: { ridge?: number; maxIter?: number; tol?: number } = {},
): LRFit {
  const ridge = opts.ridge ?? 1e-6;
  const maxIter = opts.maxIter ?? 100;
  const tol = opts.tol ?? 1e-8;
  const n = X.length;
  const k = (X[0]?.length ?? 0) + 1; // + intercept
  // design matrix with a leading 1
  const D = X.map((row) => [1, ...row]);
  let beta = new Array(k).fill(0);
  let iters = 0;
  let converged = false;
  let H: number[][] = [];

  for (; iters < maxIter; iters++) {
    // gradient g = Xᵀ(y - p) - ridge·β(slopes); Hessian Hn = Xᵀ W X + ridge·I(slopes)
    const g = new Array(k).fill(0);
    H = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      let eta = 0;
      for (let j = 0; j < k; j++) eta += beta[j] * D[i][j];
      const p = sigmoid(eta);
      const w = Math.max(p * (1 - p), 1e-9);
      const resid = y[i] - p;
      for (let a = 0; a < k; a++) {
        g[a] += resid * D[i][a];
        for (let b = a; b < k; b++) H[a][b] += w * D[i][a] * D[i][b];
      }
    }
    // ridge on slopes (index ≥ 1); symmetrize H
    for (let a = 1; a < k; a++) {
      g[a] -= ridge * beta[a];
      H[a][a] += ridge;
    }
    for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) H[a][b] = H[b][a];

    const step = solve(H, g);
    if (!step) break;
    let maxStep = 0;
    for (let j = 0; j < k; j++) {
      beta[j] += step[j];
      maxStep = Math.max(maxStep, Math.abs(step[j]));
    }
    if (maxStep < tol) {
      converged = true;
      iters++;
      break;
    }
  }

  // standard errors = sqrt(diag(H⁻¹)); log-likelihood
  const se = new Array(k).fill(NaN);
  const inv = invert(H);
  if (inv) for (let j = 0; j < k; j++) se[j] = Math.sqrt(Math.max(inv[j][j], 0));
  let ll = 0;
  for (let i = 0; i < n; i++) {
    let eta = 0;
    for (let j = 0; j < k; j++) eta += beta[j] * D[i][j];
    const p = sigmoid(eta);
    ll += y[i] * Math.log(Math.max(p, 1e-12)) + (1 - y[i]) * Math.log(Math.max(1 - p, 1e-12));
  }
  return { beta, se, iters, converged, ll };
}

/** predict P(deposit) for a feature row (without the intercept column). */
export function predictLR(beta: number[], x: number[]): number {
  let eta = beta[0];
  for (let j = 0; j < x.length; j++) eta += beta[j + 1] * x[j];
  return sigmoid(eta);
}

/** invert a small dense matrix via Gauss-Jordan (returns null if singular). */
function invert(A: number[][]): number[][] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    for (let c = 0; c < 2 * n; c++) M[col][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = 0; c < 2 * n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row.slice(n));
}
