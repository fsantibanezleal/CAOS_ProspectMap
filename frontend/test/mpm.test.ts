// Engine correctness, run with: node --import tsx --test test/mpm.test.ts
//
// The science is pinned against closed forms + WofE theory + the synthetic controls (known ground truth):
//   · a hand-computed 2x2 gives the exact W+/W-/contrast/studentized-C;
//   · the posterior is higher inside a favourable pattern than outside;
//   · on a planted-weights synthetic area WofE recovers the weight ORDERING + ranks deposits well (positive control);
//   · an uninformative layer has contrast ≈ 0 and ROC ≈ 0.5 (negative control);
//   · on conditionally-independent binary patterns the logistic-regression coefficients match the WofE contrasts in
//     sign + ordering (the WofE↔LR equivalence);
//   · the Agterberg-Cheng omnibus test gives ciRatio ≈ 1 when CI holds and T > N(D) (z > 0) on a planted CI-violation;
//   · a perfect ranking captures all deposits in minimal area, a random ranking gives the diagonal;
//   · the SAME spatial-autocorrelation model has a HIGHER random-CV AUC than spatial-CV AUC (the inflation gap).
// Everything is deterministic (seeded).

import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  bestWeights, captureCurve, ciCheck, contingency2x2, crossValAuc, makeSyntheticArea, nearestDepositScore,
  normCdf, omnibus, pairwiseChi2, posterior, randomFolds, rocAuc, spatialBlockFolds, weightsFromCounts, fitLR,
  binarize, getLayer,
} from '../src/mpm/index.ts';
import type { Cube } from '../src/mpm/index.ts';

test('normCdf is a valid CDF', () => {
  assert.ok(Math.abs(normCdf(0) - 0.5) < 1e-6);
  assert.ok(Math.abs(normCdf(1.959964) - 0.975) < 1e-3);
  assert.ok(normCdf(-3) < normCdf(0) && normCdf(0) < normCdf(3));
});

test('WofE closed-form 2x2 oracle', () => {
  // nBD=10, nBDbar=20, nBbarD=5, nBbarDbar=965  →  hand-computed weights
  const w = weightsFromCounts('L', 0, { nBD: 10, nBDbar: 20, nBbarD: 5, nBbarDbar: 965 });
  assert.ok(Math.abs(w.wPlus - 3.49144) < 1e-3, `W+ ${w.wPlus}`);
  assert.ok(Math.abs(w.wMinus - -1.07825) < 1e-3, `W- ${w.wMinus}`);
  assert.ok(Math.abs(w.contrast - 4.56969) < 1e-3, `C ${w.contrast}`);
  assert.ok(Math.abs(w.sContrast - 0.592484) < 1e-3, `sC ${w.sContrast}`);
  assert.ok(Math.abs(w.studC - 7.7128) < 1e-2, `studC ${w.studC}`);
  assert.equal(w.haldane, false);
});

test('Haldane 0.5 correction guards a zero-count class', () => {
  const w = weightsFromCounts('L', 0, { nBD: 8, nBDbar: 0, nBbarD: 2, nBbarDbar: 990 });
  assert.equal(w.haldane, true);
  assert.ok(Number.isFinite(w.wPlus) && Number.isFinite(w.contrast), 'no ln(0) blow-up');
  assert.ok(w.contrast > 0, 'all deposits in-pattern ⇒ strong positive contrast');
});

test('posterior is higher inside a favourable pattern than outside', () => {
  // 4x4 grid; a binary layer present in the left two columns; all deposits in the left columns.
  const nx = 4;
  const ny = 4;
  const vals = new Float64Array(nx * ny);
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) vals[r * nx + c] = c < 2 ? 1 : 0;
  const depositIdx = [0, 4, 8, 1]; // all in the left half
  const cube: Cube = { nx, ny, cellKm: 1, layers: [{ id: 'L', name: 'L', kind: 'binary', values: vals }], depositIdx };
  const { weights, pattern } = bestWeights(cube, 'L');
  assert.ok(weights.contrast > 0, 'left-half pattern is favourable');
  const post = posterior(cube, [pattern], [weights]);
  const left = post.prob[0]; // a present cell
  const right = post.prob[3]; // an absent cell (col 3)
  const prior = 1 / (1 + Math.exp(-post.priorLogit));
  assert.ok(left > prior, `posterior inside ${left} > prior ${prior}`);
  assert.ok(left > right, `posterior inside ${left} > outside ${right}`);
});

test('synthetic positive control, WofE recovers the planted weight ordering + ranks deposits', () => {
  const { cube, planted } = makeSyntheticArea({
    nx: 100, ny: 100, seed: 7, nDeposits: 90, gain: 7,
    layers: [
      { id: 'strong', weight: 2.2, favHigh: true, coarse: 14 },
      { id: 'medium', weight: 1.2, favHigh: true, coarse: 14 },
      { id: 'weak', weight: 0.6, favHigh: true, coarse: 14 },
      { id: 'noise', weight: 0.0, favHigh: true, coarse: 14 },
    ],
  });
  const wStrong = bestWeights(cube, 'strong').weights;
  const wMedium = bestWeights(cube, 'medium').weights;
  const wWeak = bestWeights(cube, 'weak').weights;
  const wNoise = bestWeights(cube, 'noise').weights;
  assert.ok(planted.strong > planted.weak, 'sanity: planted ordering');
  // informative layers have positive contrast; the ordering tracks the planted weights
  assert.ok(wStrong.contrast > 0 && wMedium.contrast > 0, 'informative layers positively associated');
  assert.ok(wStrong.contrast > wWeak.contrast, `strong ${wStrong.contrast} > weak ${wWeak.contrast}`);
  assert.ok(wStrong.studC > Math.abs(wNoise.studC), 'strong layer more significant than noise');
  assert.ok(Math.abs(wNoise.contrast) < wStrong.contrast, 'noise contrast << strong contrast');
  // the combined WofE posterior ranks deposits clearly better than random
  const pats = ['strong', 'medium', 'weak'].map((id) => bestWeights(cube, id));
  const post = posterior(cube, pats.map((p) => p.pattern), pats.map((p) => p.weights));
  const auc = rocAuc(cube, post.prob);
  assert.ok(auc > 0.62, `WofE ROC AUC ${auc} should beat random`);
});

test('negative control, an uninformative layer makes no skill from noise', () => {
  const { cube } = makeSyntheticArea({
    nx: 90, ny: 90, seed: 3, nDeposits: 70, gain: 7,
    layers: [
      { id: 'real', weight: 2.0, favHigh: true, coarse: 12 },
      { id: 'noise', weight: 0.0, favHigh: true, coarse: 9 },
    ],
  });
  const wNoise = bestWeights(cube, 'noise');
  const postNoise = posterior(cube, [wNoise.pattern], [wNoise.weights]);
  const aucNoise = rocAuc(cube, postNoise.prob);
  assert.ok(Math.abs(aucNoise - 0.5) < 0.12, `noise-only ROC AUC ${aucNoise} ≈ 0.5`);
  assert.ok(Math.abs(wNoise.weights.studC) < 3.0, `noise studC ${wNoise.weights.studC} not strongly significant`);
});

test('WofE ↔ logistic-regression equivalence (sign + ordering on binary patterns)', () => {
  const { cube } = makeSyntheticArea({
    nx: 100, ny: 100, seed: 11, nDeposits: 100, gain: 7,
    layers: [
      { id: 'a', weight: 2.0, favHigh: true, coarse: 16 },
      { id: 'b', weight: 1.0, favHigh: true, coarse: 11 },
    ],
  });
  const ids = ['a', 'b'];
  const bw = ids.map((id) => bestWeights(cube, id));
  // build the binary design matrix over all cells + the deposit label
  const dep = new Set(cube.depositIdx);
  const X: number[][] = [];
  const y: number[] = [];
  const n = cube.nx * cube.ny;
  for (let i = 0; i < n; i++) {
    X.push(bw.map((b) => (b.pattern.present[i] === 1 ? 1 : 0)));
    y.push(dep.has(i) ? 1 : 0);
  }
  const fit = fitLR(X, y, { ridge: 1e-4 });
  // β slopes match the WofE contrasts in sign and ordering
  for (let j = 0; j < ids.length; j++) {
    assert.ok(Math.sign(fit.beta[j + 1]) === Math.sign(bw[j].weights.contrast), `sign(β_${ids[j]}) == sign(C)`);
  }
  const betaOrder = fit.beta[1] > fit.beta[2];
  const contrastOrder = bw[0].weights.contrast > bw[1].weights.contrast;
  assert.equal(betaOrder, contrastOrder, 'LR coefficient ordering matches WofE contrast ordering');
});

test('Agterberg-Cheng omnibus, CI≈1 when independent, T>N(D) on a planted CI violation', () => {
  // CI-true: two independent informative layers
  const ind = makeSyntheticArea({
    nx: 100, ny: 100, seed: 21, nDeposits: 90, gain: 6,
    layers: [
      { id: 'x', weight: 1.6, favHigh: true, coarse: 17 },
      { id: 'y', weight: 1.4, favHigh: true, coarse: 9 },
    ],
  });
  const indPats = ['x', 'y'].map((id) => bestWeights(ind.cube, id));
  const indPost = posterior(ind.cube, indPats.map((p) => p.pattern), indPats.map((p) => p.weights));
  const omInd = omnibus(ind.cube, indPost);

  // CI-violation: include a near-duplicate (correlated) copy of 'x' alongside 'x' → double counting
  const dep = makeSyntheticArea({
    nx: 100, ny: 100, seed: 21, nDeposits: 90, gain: 6,
    layers: [
      { id: 'x', weight: 1.6, favHigh: true, coarse: 17 },
      { id: 'y', weight: 1.4, favHigh: true, coarse: 9 },
    ],
    duplicate: { srcId: 'x', id: 'xdup', noise: 0.05 },
  });
  const depPats = ['x', 'xdup', 'y'].map((id) => bestWeights(dep.cube, id));
  const depPost = posterior(dep.cube, depPats.map((p) => p.pattern), depPats.map((p) => p.weights));
  const omDep = omnibus(dep.cube, depPost);

  assert.ok(Math.abs(omInd.ciRatio - 1) < 0.35, `CI-true ratio ${omInd.ciRatio} ≈ 1`);
  assert.ok(omDep.ciRatio < omInd.ciRatio, `violation ratio ${omDep.ciRatio} < independent ${omInd.ciRatio}`);
  assert.ok(omDep.T > omDep.nD, `violation inflates posterior: T ${omDep.T} > N(D) ${omDep.nD}`);
  assert.ok(omDep.z > 1.5, `violation z ${omDep.z} flags CI failure`);
});

test('pairwise CI chi-square is non-degenerate: correlated patterns >> independent patterns', () => {
  // A CI-violation area: 'x' and its near-duplicate 'xdup' are strongly correlated; 'y' is independent of 'x'.
  const { cube } = makeSyntheticArea({
    nx: 100, ny: 100, seed: 21, nDeposits: 90, gain: 6,
    layers: [
      { id: 'x', weight: 1.6, favHigh: true, coarse: 17 },
      { id: 'y', weight: 1.4, favHigh: true, coarse: 9 },
    ],
    duplicate: { srcId: 'x', id: 'xdup', noise: 0.05 },
  });
  const px = bestWeights(cube, 'x').pattern;
  const pxdup = bestWeights(cube, 'xdup').pattern;
  const py = bestWeights(cube, 'y').pattern;
  const corr = pairwiseChi2(cube, px, pxdup);
  const indep = pairwiseChi2(cube, px, py);
  // the stratified (2x2x2) test must return real, non-zero values (the previous deposit-only test collapsed to 0)
  assert.ok(corr.n > cube.depositIdx.length, 'the stratified test spans both D strata (far more than the deposit cells)');
  assert.ok(corr.chi2 > 5 && corr.cramersV > 0.05, `correlated pair has real association: chi2 ${corr.chi2}, V ${corr.cramersV}`);
  assert.ok(corr.cramersV > indep.cramersV, `duplicate pair V ${corr.cramersV} > independent pair V ${indep.cramersV}`);
});

test('capture curve, perfect ranking vs random', () => {
  const { cube } = makeSyntheticArea({
    nx: 60, ny: 60, seed: 5, nDeposits: 40, gain: 6,
    layers: [{ id: 'a', weight: 2.0, favHigh: true, coarse: 10 }],
  });
  const dep = new Set(cube.depositIdx);
  const n = cube.nx * cube.ny;
  // perfect: deposits score highest
  const perfect = new Float64Array(n);
  for (let i = 0; i < n; i++) perfect[i] = dep.has(i) ? 1 : 0;
  const cPerfect = captureCurve(cube, perfect);
  assert.ok(cPerfect.aucCapture > 0.95, `perfect capture AUC ${cPerfect.aucCapture}`);
  // random
  const rnd = new Float64Array(n);
  let s = 12345;
  for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; rnd[i] = s / 0x7fffffff; }
  const cRand = captureCurve(cube, rnd);
  assert.ok(Math.abs(cRand.aucCapture - 0.5) < 0.12, `random capture AUC ${cRand.aucCapture} ≈ 0.5`);
});

test('spatial-CV inflation, the SAME model scores higher under random CV than spatial-block CV', () => {
  const { cube } = makeSyntheticArea({
    nx: 100, ny: 100, seed: 33, nDeposits: 80, gain: 7,
    layers: [{ id: 'a', weight: 2.2, favHigh: true, coarse: 12 }],
  });
  const model = nearestDepositScore(cube, 4);
  const aucRandom = crossValAuc(cube, randomFolds(cube, 5, 9), 5, model);
  const aucSpatial = crossValAuc(cube, spatialBlockFolds(cube, 5, 20), 5, model);
  assert.ok(aucRandom > aucSpatial + 0.04, `random-CV AUC ${aucRandom} should exceed spatial-CV AUC ${aucSpatial} (leakage)`);
});

test('contingency counts are conserved', () => {
  const { cube } = makeSyntheticArea({
    nx: 50, ny: 50, seed: 2, nDeposits: 30, gain: 6,
    layers: [{ id: 'a', weight: 1.5, favHigh: true, coarse: 10 }],
  });
  const layer = getLayer(cube, 'a');
  const pat = binarize(layer, 0.5);
  const c = contingency2x2(cube, pat);
  assert.equal(c.nBD + c.nBDbar + c.nBbarD + c.nBbarDbar, cube.nx * cube.ny, 'all cells accounted for');
  assert.equal(c.nBD + c.nBbarD, cube.depositIdx.length, 'deposit counts conserved');
});
