// Generate the learned-model training data by running the SAME TypeScript engine the browser runs, so the classifier
// trains on EXACTLY the cubes the App shows, and is benchmarked against the SAME white-box WofE posterior on the SAME
// spatial holdout. Writes to data/raw/ (git-ignored, regenerable). Invoked by pipeline.retrain before train_mpm.py.
// Run (from frontend/ so tsx resolves):  node --import tsx ../data-pipeline/pmlab/science/gen_train.mjs
//
// For each training case (the terrane + evidence-rich synthetic areas) we regenerate the cube, assign SPATIAL-BLOCK
// folds + RANDOM folds (so train can show the inflation gap), sample presence cells + distance-buffered informed
// negatives, and record each cell's evidence feature vector + label + folds + the held-out WofE posterior (the
// white-box authority, refit per spatial fold by the EXACT engine). Plus the in-envelope vs out-of-envelope feature
// rows for the geology OOD autoencoder.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  bestWeights, CASES, colRow, crossValScores, depositSet, getLayer, makeSyntheticArea, maskCells, posterior,
  randomFolds, spatialBlockFolds, weights,
} from '../../../frontend/src/mpm/index.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const RAW = resolve(HERE, '../../../data/raw');
mkdirSync(RAW, { recursive: true });

const FEATURES = ['mag', 'rad', 'geochem', 'struct']; // MPM_FEATURES (model/learned.py)
const TRAIN_CASES = ['K-PORPHYRY', 'K-OROGENIC', 'K-VMS', 'K-IOCG', 'D-RICH'];
const K = 5;
const BLOCK = 20;
const NEG_RATIO = 8;     // informed negatives per positive
const BUFFER = 6;        // cells: negatives must be at least this far from any deposit
const SEED = 20260621;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(SEED);

function featRow(cube, i) {
  return FEATURES.map((id) => {
    const v = getLayer(cube, id).values[i];
    return Number.isNaN(v) ? 0 : Math.round(v * 1e5) / 1e5;
  });
}

// a WofE-posterior scoring function refit on a TRAINING deposit subset (the spatial-holdout authority)
function wofeScoreFn(cube, ids) {
  const best = ids.map((id) => bestWeights(cube, id));
  const pats = best.map((b) => b.pattern);
  return (train) => {
    const ws = pats.map((p) => weights(cube, p, train));
    return posterior(cube, pats, ws, undefined, train).prob;
  };
}

const rows = [];     // {feat, y, sFold, rFold, pWofe, case}
const inEnv = [];    // in-envelope feature rows (for the OOD-AE training + the in-distribution eval)

for (const cid of TRAIN_CASES) {
  const c = CASES.find((x) => x.id === cid);
  const { cube } = makeSyntheticArea(c.spec);
  const ids = c.layerIds;
  const sFolds = spatialBlockFolds(cube, K, BLOCK);
  const rFolds = randomFolds(cube, K, 17 + cid.length);
  const wofeOOF = crossValScores(cube, sFolds, K, wofeScoreFn(cube, ids)); // held-out WofE posterior per cell

  const cells = maskCells(cube);
  const dep = depositSet(cube);
  const pos = [...dep];
  // distance to nearest deposit (for the buffered negatives)
  const dpos = pos.map((d) => colRow(cube, d));
  const farEnough = (i) => {
    const { col, row } = colRow(cube, i);
    for (const t of dpos) if (Math.hypot(col - t.col, row - t.row) < BUFFER) return false;
    return true;
  };
  const candidates = cells.filter((i) => !dep.has(i) && farEnough(i));
  // sample NEG_RATIO * nPos informed negatives
  const negTarget = Math.min(candidates.length, NEG_RATIO * pos.length);
  const neg = [];
  const taken = new Set();
  let guard = 0;
  while (neg.length < negTarget && guard < candidates.length * 20) {
    guard++;
    const i = candidates[(rnd() * candidates.length) | 0];
    if (taken.has(i)) continue;
    taken.add(i);
    neg.push(i);
  }

  for (const i of pos) rows.push({ feat: featRow(cube, i), y: 1, sFold: sFolds[i], rFold: rFolds[i], pWofe: wofeOOF[i], case: cid });
  for (const i of neg) rows.push({ feat: featRow(cube, i), y: 0, sFold: sFolds[i], rFold: rFolds[i], pWofe: wofeOOF[i], case: cid });
  for (const i of [...pos, ...neg]) inEnv.push(featRow(cube, i));
}

// out-of-envelope feature rows for the OOD AUC: the training features live in ~[0,1] (value-noise); push components
// well outside that band (each in [1.3,2.6] or [-1.6,-0.3]) so the AE reconstructs them poorly.
const ood = [];
const u = (lo, hi) => lo + (hi - lo) * rnd();
for (let i = 0; i < 600; i++) {
  ood.push(FEATURES.map(() => (rnd() < 0.5 ? u(1.3, 2.6) : u(-1.6, -0.3))));
}

writeFileSync(resolve(RAW, 'mpm-train.json'), JSON.stringify({ features: FEATURES, k: K, rows }));
writeFileSync(resolve(RAW, 'mpm-eval.json'), JSON.stringify({ inDist: inEnv, ood }));
const nPos = rows.filter((r) => r.y === 1).length;
console.log(`gen_train: ${rows.length} rows (${nPos} presence) across ${TRAIN_CASES.length} cases - ${ood.length} OOD -> ${RAW}`);
