// The per-case analysis the Node bake (science/bake_cases.mjs → case-results.json) and the App both consume. Given a
// case (a synthetic study-area SPEC + the evidence layers to use), it regenerates the cube deterministically and runs
// the WHOLE WofE pipeline: per-layer weights at the maximizing-contrast threshold, the combined posterior, the CI
// diagnostics, the success (fitting) + prediction (spatial-holdout) capture curves + ROC, the random-vs-spatial-CV
// inflation gap, and the logistic-regression comparison. Everything is deterministic given the spec's seed.

import type { Binarized, CaptureCurve, CICheck, Cube } from './types.ts';
import { makeSyntheticArea, type SynthSpec } from './synth.ts';
import { bestWeights } from './binarize.ts';
import { maskCells, nCells, nDeposits, depositSet } from './grid.ts';
import { posterior, weights } from './wofe.ts';
import { ciCheck } from './ci.ts';
import { captureCurve, rocAuc } from './validate.ts';
import { crossValAuc, crossValScores, randomFolds, spatialBlockFolds } from './cv.ts';
import { fitLR, predictLR } from './logreg.ts';

export interface LayerResult {
  id: string;
  tStar: number;
  wPlus: number;
  wMinus: number;
  contrast: number;
  studC: number;
  nBD: number;
  nBDbar: number;
  nBbarD: number;
  nBbarDbar: number;
}

export interface CaseAnalysis {
  spec: SynthSpec;
  layerIds: string[];
  nx: number;
  ny: number;
  cellKm: number;
  nCells: number;
  nDeposits: number;
  priorProb: number;
  layers: LayerResult[];
  posteriorSummary: { min: number; max: number; mean: number };
  ci: CICheck;
  rocAuc: number;
  capture: { success: CaptureCurve; prediction: CaptureCurve };
  cv: { k: number; blockCells: number; randomAuc: number; spatialAuc: number; inflationGap: number };
  lr: { rocAuc: number; betas: { id: string; beta: number }[] };
}

/** a WofE-posterior scoring function refit on a TRAINING deposit subset (for spatial-holdout CV). */
function wofeScoreFn(cube: Cube, pats: Binarized[]): (train: Set<number>) => Float64Array {
  return (train: Set<number>) => {
    const ws = pats.map((p) => weights(cube, p, train));
    return posterior(cube, pats, ws, undefined, train).prob;
  };
}

export function analyzeCase(spec: SynthSpec, layerIds: string[], k = 5, blockCells = 20): CaseAnalysis {
  const { cube } = makeSyntheticArea(spec);
  const best = layerIds.map((id) => ({ id, ...bestWeights(cube, id) }));
  const pats = best.map((b) => b.pattern);
  const ws = best.map((b) => b.weights);
  const post = posterior(cube, pats, ws);

  // posterior summary over active cells
  let mn = Infinity;
  let mx = -Infinity;
  let sum = 0;
  const cells = maskCells(cube);
  for (const i of cells) {
    const p = post.prob[i];
    if (p < mn) mn = p;
    if (p > mx) mx = p;
    sum += p;
  }

  const ci = ciCheck(cube, pats, post);
  const auc = rocAuc(cube, post.prob);
  const success = captureCurve(cube, post.prob);

  // prediction (spatial holdout): WofE refit per spatial block, predict held-out cells
  const scoreFn = wofeScoreFn(cube, pats);
  const spatialFolds = spatialBlockFolds(cube, k, blockCells);
  const heldOut = crossValScores(cube, spatialFolds, k, scoreFn);
  const predMask = cells.filter((i) => !Number.isNaN(heldOut[i]));
  const predCube: Cube = { ...cube, maskIdx: predMask };
  const prediction = captureCurve(predCube, heldOut, depositSet(predCube));

  // the inflation gap: the SAME WofE model under random vs spatial-block CV
  const randomAuc = crossValAuc(cube, randomFolds(cube, k, 17), k, scoreFn);
  const spatialAuc = crossValAuc(cube, spatialFolds, k, scoreFn);

  // logistic-regression comparison on the binary patterns
  const dep = depositSet(cube);
  const X: number[][] = [];
  const y: number[] = [];
  for (const i of cells) {
    X.push(pats.map((p) => (p.present[i] === 1 ? 1 : 0)));
    y.push(dep.has(i) ? 1 : 0);
  }
  const fit = fitLR(X, y, { ridge: 1e-3 });
  const lrScore = new Float64Array(cube.nx * cube.ny);
  for (let r = 0; r < cells.length; r++) lrScore[cells[r]] = predictLR(fit.beta, X[r]);
  const lrAuc = rocAuc(cube, lrScore);

  return {
    spec,
    layerIds,
    nx: cube.nx,
    ny: cube.ny,
    cellKm: cube.cellKm,
    nCells: nCells(cube),
    nDeposits: nDeposits(cube),
    priorProb: 1 / (1 + Math.exp(-post.priorLogit)),
    layers: best.map((b) => ({
      id: b.id,
      tStar: b.tStar,
      wPlus: b.weights.wPlus,
      wMinus: b.weights.wMinus,
      contrast: b.weights.contrast,
      studC: b.weights.studC,
      nBD: b.weights.nBD,
      nBDbar: b.weights.nBDbar,
      nBbarD: b.weights.nBbarD,
      nBbarDbar: b.weights.nBbarDbar,
    })),
    posteriorSummary: { min: mn, max: mx, mean: sum / cells.length },
    ci,
    rocAuc: auc,
    capture: { success, prediction },
    cv: { k, blockCells, randomAuc, spatialAuc, inflationGap: randomAuc - spatialAuc },
    lr: { rocAuc: lrAuc, betas: layerIds.map((id, j) => ({ id, beta: fit.beta[j + 1] })) },
  };
}
