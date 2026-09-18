// Export the engine's held-out Weights-of-Evidence posterior for each registered real case, so
// pipeline/real_learned.py can score the learned classifier under the SAME protocol as the WofE spatial- and
// random-CV AUCs in case-results.json: identical folds, identical held-out cells, identical pooled aggregation.
// Everything here is the TypeScript engine the browser runs and the bake uses: the folds (spatialBlockFolds,
// randomFolds), the fully out-of-fold WofE fit of each fold (wofeFoldScoreFn: thresholds, weights and prior fitted on
// the training folds' cells only) and the pooled AUC (crossValScores + crossValAuc), with the
// defaults analyzeCube applies to the committed bake (k = 5, 20x20-cell blocks, random-fold seed 17). Nothing is
// re-implemented in Python; real_learned.py re-derives the two pooled AUCs from these arrays and refuses to run if
// they differ from the committed bake.
//
// The same folds also score the engine's distance-to-known-deposit baseline (nearestDepositScore: exp(-d / 4), d the
// cell distance to the nearest TRAINING-fold deposit). It learns no geology, so it measures how much of a held-out AUC
// is proximity to known deposits; a model must beat it to claim more.
//
// Writes data/raw/<case>-wofe-oof.json (git-ignored, regenerable). Run from frontend/ so tsx resolves the engine:
//   node --import tsx ../data-pipeline/pipeline/science/real_wofe_oof.mjs
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  crossValAuc, crossValScores, cubeFromFile, depositSet, maskCells, nearestDepositScore, randomFolds,
  spatialBlockFolds, wofeFoldScoreFn, REAL_CASES,
} from '../../../frontend/src/mpm/index.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const DERIVED = resolve(HERE, '../../../data/derived');
const RAW = resolve(HERE, '../../../data/raw');

// analyzeCube's defaults (frontend/src/mpm/analyze.ts); the committed cv block of every case uses exactly these.
const K = 5;
const BLOCK_CELLS = 20;
const RANDOM_SEED = 17;
const NULL_SCALE_CELLS = 4; // nearestDepositScore's default scale, as in the engine's cv.ts

mkdirSync(RAW, { recursive: true });

for (const rc of REAL_CASES) {
  const cubePath = resolve(DERIVED, rc.id, 'cube.json');
  if (!existsSync(cubePath)) {
    console.warn(`[real_wofe_oof] missing ${cubePath}; run pipeline.real_usmvt first`);
    continue;
  }
  const cube = cubeFromFile(JSON.parse(readFileSync(cubePath, 'utf-8')));
  // the same fold model analyzeCube cross-validates (each fold: thresholds, weights and prior from the training folds)
  const scoreFn = wofeFoldScoreFn(cube, rc.layerIds);
  const nullFn = nearestDepositScore(cube, NULL_SCALE_CELLS);
  const cells = maskCells(cube);

  const scheme = (name, folds) => {
    const heldOut = crossValScores(cube, folds, K, scoreFn);
    const auc = crossValAuc(cube, folds, K, scoreFn);
    const nullHeld = crossValScores(cube, folds, K, nullFn);
    const nullAuc = crossValAuc(cube, folds, K, nullFn);
    const foldOf = new Array(cells.length);
    const wofe = new Array(cells.length);
    const distanceNull = new Array(cells.length);
    for (let r = 0; r < cells.length; r++) {
      const i = cells[r];
      if (folds[i] < 0 || Number.isNaN(heldOut[i]) || Number.isNaN(nullHeld[i])) {
        throw new Error(`[real_wofe_oof] ${rc.id} ${name}: cell ${i} has no held-out score`);
      }
      foldOf[r] = folds[i];
      wofe[r] = heldOut[i];
      distanceNull[r] = nullHeld[i];
    }
    return { folds: foldOf, wofe, engine_auc: auc, distance_null: distanceNull, distance_null_engine_auc: nullAuc };
  };

  const out = {
    schema: 'prospectmap.wofe-oof/v1',
    case_id: rc.id,
    nx: cube.nx,
    ny: cube.ny,
    layer_ids: rc.layerIds,
    k: K,
    block_cells: BLOCK_CELLS,
    random_seed: RANDOM_SEED,
    null_scale_cells: NULL_SCALE_CELLS,
    cells, // flat cell indices, in the order of every per-cell array below
    deposit_cells: [...depositSet(cube)].sort((a, b) => a - b),
    spatial: scheme('spatial', spatialBlockFolds(cube, K, BLOCK_CELLS)),
    random: scheme('random', randomFolds(cube, K, RANDOM_SEED)),
  };
  const path = resolve(RAW, `${rc.id}-wofe-oof.json`);
  writeFileSync(path, JSON.stringify(out));
  console.log(
    `[real_wofe_oof] ${rc.id}: ${cells.length} cells, ${out.deposit_cells.length} deposit cells; pooled AUC ` +
      `spatial WofE ${out.spatial.engine_auc.toFixed(4)} / distance null ${out.spatial.distance_null_engine_auc.toFixed(4)}, ` +
      `random WofE ${out.random.engine_auc.toFixed(4)} / distance null ${out.random.distance_null_engine_auc.toFixed(4)} -> ${path}`,
  );
}
