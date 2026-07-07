// Bake the REAL-data case (the "Real sample" lane) through the SAME TypeScript WofE engine the browser runs, so the
// live and offline numbers stay identical by construction (as for the synthetic cases in bake_cases.mjs). Unlike the
// synthetic bake, there is no SynthSpec: the real Cube is loaded from data/derived/<id>/cube.json (built offline by
// pmlab/real_usmvt.py), analysed with analyzeCube, and written back as <id>/trace.json. It is ALSO merged into
// case-results.json so Experiments/Benchmark include it. Run (from frontend/ so tsx resolves the engine):
//   node --import tsx ../data-pipeline/pmlab/science/bake_real.mjs
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyzeCube, cubeFromFile } from '../../../frontend/src/mpm/index.ts';
import { REAL_CASES } from '../../../frontend/src/mpm/real.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const DERIVED = resolve(HERE, '../../../data/derived');

const CR_PATH = resolve(DERIVED, 'case-results.json');
const caseResults = existsSync(CR_PATH)
  ? JSON.parse(readFileSync(CR_PATH, 'utf-8'))
  : { schema: 'prospectmap.case-results/v1', nCases: 0, cases: {} };

for (const rc of REAL_CASES) {
  const cubePath = resolve(DERIVED, rc.id, 'cube.json');
  if (!existsSync(cubePath)) {
    console.warn(`[bake_real] missing ${cubePath}; run pmlab.real_usmvt first`);
    continue;
  }
  const f = JSON.parse(readFileSync(cubePath, 'utf-8'));
  const cube = cubeFromFile(f);
  const a = analyzeCube(cube, rc.layerIds);

  const learned = readLearnedReal();
  const trace = {
    schema: 'prospectmap.trace/v1',
    case_id: rc.id,
    name: rc.name,
    category: rc.category,
    real_or_synthetic: rc.realOrSynthetic,
    expected_band: rc.expectedBand,
    validation_anchor: rc.validationAnchor,
    spec: { source: 'real-open-dataset', file: rc.file, nx: a.nx, ny: a.ny, cellKm: a.cellKm },
    layer_ids: a.layerIds,
    n_cells: a.nCells,
    n_deposits: a.nDeposits,
    prior_prob: a.priorProb,
    layers: a.layers,
    posterior_summary: a.posteriorSummary,
    ci: a.ci,
    roc_auc: a.rocAuc,
    capture: a.capture,
    cv: a.cv,
    lr: a.lr,
    learned,
  };
  writeFileSync(resolve(DERIVED, rc.id, 'trace.json'), JSON.stringify(trace), 'utf-8');

  caseResults.cases[rc.id] = {
    name: rc.name,
    category: rc.category,
    realOrSynthetic: rc.realOrSynthetic,
    expectedBand: rc.expectedBand,
    validationAnchor: rc.validationAnchor,
    spec: trace.spec,
    layerIds: a.layerIds,
    nx: a.nx,
    ny: a.ny,
    cellKm: a.cellKm,
    nCells: a.nCells,
    nDeposits: a.nDeposits,
    priorProb: a.priorProb,
    layers: a.layers,
    posteriorSummary: a.posteriorSummary,
    ci: a.ci,
    rocAuc: a.rocAuc,
    capture: a.capture,
    cv: a.cv,
    lr: a.lr,
  };
  console.log(
    `[bake_real] ${rc.id}: nCells=${a.nCells} nDep=${a.nDeposits} AUC(WofE)=${a.rocAuc.toFixed(3)} ` +
      `AUC(LR)=${a.lr.rocAuc.toFixed(3)} CIratio=${a.ci.ciRatio.toFixed(2)}`,
  );
}

caseResults.nCases = Object.keys(caseResults.cases).length;
writeFileSync(CR_PATH, JSON.stringify(caseResults), 'utf-8');
console.log(`[bake_real] merged real case(s) -> ${CR_PATH} (nCases=${caseResults.nCases})`);

function readLearnedReal() {
  const p = resolve(DERIVED, 'pm-learned-real.json');
  if (!existsSync(p)) return { status: 'pending-training', classifier: null, ood: null };
  const l = JSON.parse(readFileSync(p, 'utf-8'));
  return { status: 'trained', classifier: l.classifier ?? null, ood: l.ood ?? null };
}
