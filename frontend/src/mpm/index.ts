// The ProspectMap MPM engine barrel — dependency-free TypeScript, runs LIVE in the browser AND in the Node bake
// (under tsx). Weights-of-Evidence + the conditional-independence machinery + logistic regression + honest
// validation + spatial cross-validation. The white-box WofE posterior is the interpretable authority; the learned
// classifier (commit 4b) is benchmarked against it on the SAME spatial holdout. Mirrors CutoffGrade's `lane/`.

export * from './types.ts';
export { mulberry32, erf, normCdf, normInv, sigmoid, mean, std } from './rng.ts';
export { maskCells, nCells, depositSet, nDeposits, getLayer, idx, colRow, layerRange } from './grid.ts';
export { binarize, binaryPattern, thresholdSweep, maximizingContrast, bestWeights } from './binarize.ts';
export type { SweepPoint } from './binarize.ts';
export { contingency2x2, weightsFromCounts, weights, priorLogit, posterior } from './wofe.ts';
export type { Counts } from './wofe.ts';
export { pairwiseChi2, omnibus, ciCheck } from './ci.ts';
export { fitLR, predictLR } from './logreg.ts';
export type { LRFit } from './logreg.ts';
export { captureCurve, rocAuc, captureAt } from './validate.ts';
export { randomFolds, spatialBlockFolds, crossValAuc, nearestDepositScore } from './cv.ts';
export { makeSyntheticArea } from './synth.ts';
export type { SynthSpec, SynthLayerSpec } from './synth.ts';
export { analyzeCase } from './analyze.ts';
export type { CaseAnalysis, LayerResult } from './analyze.ts';
export { CASES, caseById, CAT_TERRANE, CAT_DATA, CAT_CONTROL } from './cases.ts';
export type { MPMCase } from './cases.ts';
