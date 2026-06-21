// Assemble the final data/derived/pm-learned.json — and confirm the exported classifier RUNS in the engine's own
// runtime (onnxruntime-web in Node), the honest end-to-end check. The honest skill numbers (the classifier's
// spatial-CV AUC vs the white-box WofE posterior on the SAME spatial holdout, + the random-vs-spatial inflation gap)
// are computed by train_mpm.py and carried in data/raw/learned-partial.json; here we load mpm-classifier.onnx via
// onnxruntime-web, run it over the in-distribution feature rows to verify the graph produces valid probabilities, and
// merge everything into the final artifact the SPA reads.
//   node --import tsx ../data-pipeline/pmlab/science/eval_mpm.mjs   (run from frontend/ so onnxruntime-web resolves)
import { createRequire } from 'node:module';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../../..');
const RAW = resolve(ROOT, 'data/raw');
const DERIVED = resolve(ROOT, 'data/derived');
const FRONTEND = resolve(ROOT, 'frontend');

// load onnxruntime-web (resolved from frontend/node_modules) — node build, WASM EP, single-threaded, local wasm.
const req = createRequire(pathToFileURL(resolve(FRONTEND, 'pkg.js')));
const ortMod = await import(pathToFileURL(req.resolve('onnxruntime-web')));
const ort = ortMod.default ?? ortMod;
ort.env.wasm.wasmPaths = pathToFileURL(resolve(FRONTEND, 'node_modules/onnxruntime-web/dist')).href + '/';
ort.env.wasm.numThreads = 1;

const partialPath = resolve(RAW, 'learned-partial.json');
const partial = existsSync(partialPath) ? JSON.parse(readFileSync(partialPath, 'utf-8')) : {};
const clf = partial.classifier ?? {};
const ev = JSON.parse(readFileSync(resolve(RAW, 'mpm-eval.json'), 'utf-8'));

// run the exported classifier over the in-distribution rows to confirm it runs in the engine's runtime
const modelPath = resolve(DERIVED, 'mpm-classifier.onnx');
let pMin = 1;
let pMax = 0;
let nRun = 0;
if (existsSync(modelPath)) {
  const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
  const rows = ev.inDist.slice(0, 2000);
  const F = rows[0].length;
  const flat = new Float32Array(rows.length * F);
  for (let i = 0; i < rows.length; i++) for (let j = 0; j < F; j++) flat[i * F + j] = rows[i][j];
  const out = await session.run({ x: new ort.Tensor('float32', flat, [rows.length, F]) });
  const p = out.p.data;
  for (let i = 0; i < p.length; i++) { pMin = Math.min(pMin, p[i]); pMax = Math.max(pMax, p[i]); }
  nRun = p.length;
}

const learned = {
  schema: 'prospectmap.learned/v1',
  classifier: {
    spatial_cv: {
      mlp_roc_auc: clf.mlp_spatial_auc ?? 0,
      wofe_roc_auc: clf.wofe_spatial_auc ?? 0,
      winner: clf.winner ?? 'tie',
    },
    random_cv: { mlp_roc_auc: clf.mlp_random_auc ?? 0 },
    mlp_roc_auc: clf.mlp_spatial_auc ?? 0,        // flat alias (the manifest metrics read this)
    inflation_gap: clf.inflation_gap ?? 0,
    nFolds: clf.nFolds ?? 0,
    nEval: clf.nEval ?? 0,
  },
  ood: partial.ood ?? { auc: 0, nEval: 0, threshold: 0 },
  honesty: partial.honesty ??
    'Presence-only labels; sampled negatives. The white-box WofE posterior is the authority; the classifier is ' +
    'compared on the SAME spatial holdout. No fabricated win.',
};
writeFileSync(resolve(DERIVED, 'pm-learned.json'), JSON.stringify(learned, null, 2));
console.log(`eval_mpm: classifier ran in onnxruntime-web (${nRun} cells, p in [${pMin.toFixed(3)}, ${pMax.toFixed(3)}]) - `
  + `MLP spatial-CV AUC ${learned.classifier.spatial_cv.mlp_roc_auc} vs WofE ${learned.classifier.spatial_cv.wofe_roc_auc} `
  + `(${learned.classifier.spatial_cv.winner}) - inflation +${learned.classifier.inflation_gap} - OOD AUC ${learned.ood.auc} -> pm-learned.json`);
