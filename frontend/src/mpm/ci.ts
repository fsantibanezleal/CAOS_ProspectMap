// The conditional-independence (CI) machinery — the honesty core. WofE's posterior sum is only valid if the evidence
// patterns are conditionally independent given D; correlated favourable layers double-count and INFLATE the posterior.
// We surface this: the pairwise χ² heatmap (CI given D), and the Agterberg-Cheng (2002) new omnibus test
// (T = Σ posterior ≈ N(D) under CI; z = (T-N(D))/s(T)) + the CI ratio N(D)/T.

import type { Binarized, CICheck, Cube, Posterior } from './types.ts';
import { maskCells, depositSet } from './grid.ts';

/** Pearson χ² (Yates-corrected) of two binary patterns over the DEPOSIT cells (conditional independence given D). */
export function pairwiseChi2(cube: Cube, a: Binarized, b: Binarized): { chi2: number; cramersV: number; n: number } {
  const dep = depositSet(cube);
  // 2x2 of (a present/absent) × (b present/absent) over deposit cells
  let n11 = 0;
  let n10 = 0;
  let n01 = 0;
  let n00 = 0;
  for (const i of dep) {
    const pa = a.present[i];
    const pb = b.present[i];
    if (pa === 255 || pb === 255) continue;
    if (pa === 1 && pb === 1) n11++;
    else if (pa === 1) n10++;
    else if (pb === 1) n01++;
    else n00++;
  }
  const n = n11 + n10 + n01 + n00;
  if (n === 0) return { chi2: 0, cramersV: 0, n: 0 };
  const r1 = n11 + n10;
  const r0 = n01 + n00;
  const c1 = n11 + n01;
  const c0 = n10 + n00;
  // Yates continuity correction for the 2x2
  const obs = [n11, n10, n01, n00];
  const exp = [(r1 * c1) / n, (r1 * c0) / n, (r0 * c1) / n, (r0 * c0) / n];
  let chi2 = 0;
  for (let k = 0; k < 4; k++) {
    if (exp[k] <= 0) continue;
    const d = Math.max(0, Math.abs(obs[k] - exp[k]) - 0.5);
    chi2 += (d * d) / exp[k];
  }
  const cramersV = Math.sqrt(chi2 / n); // 2x2 ⇒ min(r-1,c-1)=1
  return { chi2, cramersV, n };
}

/**
 * The Agterberg-Cheng new omnibus test: under full conditional independence the sum of the posterior probabilities
 * over all unit cells equals the observed number of deposits, T = Σ P_post ≈ N(D). T > N(D) ⇒ posteriors inflated ⇒
 * CI violated. The variance of T is the Poisson-binomial variance Σ P(1-P) (the standard approximation); z = (T-N(D))/s(T).
 */
export function omnibus(cube: Cube, post: Posterior): { T: number; nD: number; sT: number; z: number; ciRatio: number } {
  const nD = depositSet(cube).size;
  let T = 0;
  let varT = 0;
  for (const i of maskCells(cube)) {
    const p = post.prob[i];
    T += p;
    varT += p * (1 - p);
  }
  const sT = Math.sqrt(varT);
  const z = sT > 0 ? (T - nD) / sT : 0;
  const ciRatio = T > 0 ? nD / T : 1;
  return { T, nD, sT, z, ciRatio };
}

/** the full CI diagnostic bundle: pairwise χ² over all layer pairs + the omnibus test. */
export function ciCheck(cube: Cube, patterns: Binarized[], post: Posterior): CICheck {
  const pairwise: CICheck['pairwise'] = [];
  for (let i = 0; i < patterns.length; i++) {
    for (let j = i + 1; j < patterns.length; j++) {
      const r = pairwiseChi2(cube, patterns[i], patterns[j]);
      pairwise.push({ a: patterns[i].layerId, b: patterns[j].layerId, chi2: r.chi2, cramersV: r.cramersV });
    }
  }
  const om = omnibus(cube, post);
  return { pairwise, T: om.T, nD: om.nD, ciRatio: om.ciRatio, sT: om.sT, z: om.z };
}
