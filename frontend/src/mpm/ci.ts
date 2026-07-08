// The conditional-independence (CI) machinery, the honesty core. WofE's posterior sum is only valid if the evidence
// patterns are conditionally independent given D; correlated favourable layers double-count and INFLATE the posterior.
// We surface this: the pairwise χ² heatmap (CI given D), and the Agterberg-Cheng (2002) new omnibus test
// (T = Σ posterior ≈ N(D) under CI; z = (T-N(D))/s(T)) + the CI ratio N(D)/T.

import type { Binarized, CICheck, Cube, Posterior } from './types.ts';
import { maskCells, depositSet } from './grid.ts';

/** One D-stratum's Yates-corrected 2x2 chi-square of patterns a,b (a present/absent × b present/absent) over the
 * cells whose deposit membership equals `inDep`. Missing cells (255) are dropped. */
function stratumChi2(cube: Cube, a: Binarized, b: Binarized, dep: Set<number>, inDep: boolean): { chi2: number; n: number } {
  let n11 = 0;
  let n10 = 0;
  let n01 = 0;
  let n00 = 0;
  for (const i of maskCells(cube)) {
    if (dep.has(i) !== inDep) continue;
    const pa = a.present[i];
    const pb = b.present[i];
    if (pa === 255 || pb === 255) continue;
    if (pa === 1 && pb === 1) n11++;
    else if (pa === 1) n10++;
    else if (pb === 1) n01++;
    else n00++;
  }
  const n = n11 + n10 + n01 + n00;
  if (n === 0) return { chi2: 0, n: 0 };
  const r1 = n11 + n10;
  const r0 = n01 + n00;
  const c1 = n11 + n01;
  const c0 = n10 + n00;
  const obs = [n11, n10, n01, n00];
  const exp = [(r1 * c1) / n, (r1 * c0) / n, (r0 * c1) / n, (r0 * c0) / n];
  let chi2 = 0;
  for (let k = 0; k < 4; k++) {
    if (exp[k] <= 0) continue;
    const d = Math.max(0, Math.abs(obs[k] - exp[k]) - 0.5); // Yates continuity correction
    chi2 += (d * d) / exp[k];
  }
  return { chi2, n };
}

/**
 * Conditional-independence-given-D test for two binary patterns: the STRATIFIED (2x2x2) Yates-corrected chi-square,
 * summing the association of a,b WITHIN the deposit stratum (D=1) AND within the non-deposit stratum (D=0), df=2.
 * Testing ONLY the deposit stratum (the previous implementation) is degenerate at the maximizing-contrast threshold:
 * favourable patterns are present at nearly every deposit, so that 2x2 collapses (n11≈n, the rest≈0) and chi2 -> 0,
 * which is exactly why the pairwise table read 0.00. The non-deposit stratum, which holds the overwhelming majority
 * of the cells, carries the evidence-layer correlation that double-counts and inflates the WofE posterior. Cramer's V
 * is the pooled effect size over both strata (per-stratum min(r-1,c-1)=1).
 */
export function pairwiseChi2(cube: Cube, a: Binarized, b: Binarized): { chi2: number; cramersV: number; n: number } {
  const dep = depositSet(cube);
  const sD = stratumChi2(cube, a, b, dep, true);
  const sDbar = stratumChi2(cube, a, b, dep, false);
  const chi2 = sD.chi2 + sDbar.chi2;
  const n = sD.n + sDbar.n;
  if (n === 0) return { chi2: 0, cramersV: 0, n: 0 };
  const cramersV = Math.sqrt(chi2 / n);
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
