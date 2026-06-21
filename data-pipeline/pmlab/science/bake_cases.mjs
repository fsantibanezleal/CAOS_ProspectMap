// Bake the per-case WofE/CI/validation analysis through the SAME TypeScript engine the browser runs, and write
// data/derived/case-results.json — the committed, deterministic per-case outputs the LIGHT Python pipeline reshapes
// into per-case replay traces + manifests (CONTRACT 2). No Python re-port of the WofE engine. The cubes are SYNTHETIC
// and regenerated from each case's SPEC (committed in case-results), so the artifact stays compact (no raster blobs);
// the learned-model metrics are added by --retrain once trained. Run (from frontend/ so tsx resolves):
//   node --import tsx ../data-pipeline/pmlab/science/bake_cases.mjs
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CASES } from '../../../frontend/src/mpm/cases.ts';
import { analyzeCase } from '../../../frontend/src/mpm/index.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const DERIVED = resolve(HERE, '../../../data/derived');
mkdirSync(DERIVED, { recursive: true });

const cases = {};
for (const c of CASES) {
  const a = analyzeCase(c.spec, c.layerIds);
  cases[c.id] = {
    name: c.name,
    category: c.category,
    realOrSynthetic: c.realOrSynthetic,
    expectedBand: c.expectedBand,
    validationAnchor: c.validationAnchor,
    spec: a.spec,
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
}

const out = { schema: 'prospectmap.case-results/v1', nCases: CASES.length, cases };
writeFileSync(resolve(DERIVED, 'case-results.json'), JSON.stringify(out), 'utf-8');
console.log(`baked ${CASES.length} cases -> ${resolve(DERIVED, 'case-results.json')}`);
