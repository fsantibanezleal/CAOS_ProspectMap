// Live in-browser inference of ProspectMap's two learned models (onnxruntime-web). GRACEFUL: until they are trained
// (science/train_mpm.py -> mpm-classifier.onnx + geology-ood.onnx) the files are absent; the loader resolves to null
// and the App uses the white-box WofE engine (which runs live anyway) + shows the honest "pending training" state.
// The classifier's value is a per-cell "what-if" probe + the cross-case benchmark vs WofE; the OOD AE is the
// "where-not-to-trust" mask. WASM EP, single-threaded; the npm package + CDN wasmPaths are pinned to 1.27.
import * as ort from 'onnxruntime-web';
import { MPM_FEATURES, N_FEATURES } from './learned.ts';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
ort.env.wasm.numThreads = 1;

const base = () => import.meta.env.BASE_URL || '/';
const sessions: Record<string, Promise<ort.InferenceSession | null>> = {};

function get(file: string): Promise<ort.InferenceSession | null> {
  return (sessions[file] ??= (async () => {
    try {
      const head = await fetch(`${base()}${file}`, { method: 'HEAD' });
      if (!head.ok) return null;
      return await ort.InferenceSession.create(`${base()}${file}`, { executionProviders: ['wasm'] });
    } catch {
      return null;
    }
  })());
}

export const classifierAvailable = async () => (await get('mpm-classifier.onnx')) != null;

// onnxruntime-web sessions are not re-entrant; serialise runs per file.
const runChain: Record<string, Promise<unknown>> = {};
async function runSerial(file: string, s: ort.InferenceSession, feeds: Record<string, ort.Tensor>) {
  const prev = runChain[file] ?? Promise.resolve();
  let release!: () => void;
  runChain[file] = new Promise<void>((r) => { release = r; });
  try { await prev.catch(() => {}); return await s.run(feeds); } finally { release(); }
}

/** the per-cell evidence feature vector in the SOURCE-OF-TRUTH order (model/learned.py :: MPM_FEATURES). */
export function featureVec(values: Record<string, number>): Float32Array {
  return Float32Array.from(MPM_FEATURES.map((k) => values[k] ?? 0));
}

/** Classifier forward over a batch of cells: feature rows -> P(deposit). null if the model isn't trained. */
export async function runClassifier(rows: Float32Array, nRows: number): Promise<Float32Array | null> {
  const s = await get('mpm-classifier.onnx');
  if (!s) return null;
  const out = await runSerial('mpm-classifier.onnx', s, { x: new ort.Tensor('float32', rows, [nRows, N_FEATURES]) });
  return out.p.data as Float32Array;
}
