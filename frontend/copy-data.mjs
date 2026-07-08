// Prebuild: overlay the committed CONTRACT-2 artifacts (../data/derived) into the SPA's public/ so the static site
// loads them. Canonical copies live in ../data/derived, public/ is a build-time overlay (git-ignored). ProspectMap's
// live lane is the TypeScript WofE engine (frontend/src/mpm/) + onnxruntime-web; there is no Pyodide lane to inline.
import { copyFileSync, cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const PUB = join(HERE, 'public');
const derived = join(ROOT, 'data', 'derived');

if (!existsSync(derived)) {
  console.warn('[copy-data] no data/derived, run `npm run bake` (or `python -m pmlab.pipeline all`) first');
} else {
  mkdirSync(join(PUB, 'data'), { recursive: true });
  cpSync(derived, join(PUB, 'data'), { recursive: true });
  for (const f of ['case-results.json', 'pm-learned.json', 'mpm-classifier.onnx', 'geology-ood.onnx',
    'pm-learned-real.json', 'mpm-classifier-real.onnx', 'geology-ood-real.onnx',
    'pu-conformal.json', 'mpm-puconformal-real.onnx']) {
    const src = join(derived, f);
    if (existsSync(src)) copyFileSync(src, join(PUB, f));
  }
  console.log('[copy-data] data/derived -> public/data (+ root-level case-results / onnx)');
}
