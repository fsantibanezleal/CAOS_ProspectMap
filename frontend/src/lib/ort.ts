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

// ONE global serialization chain for ALL onnxruntime-web work (session creation AND inference). The WASM EP runs
// single-threaded and ships TWO models here (the mpm-classifier + the geology OOD autoencoder) that the App queries
// when you switch the map method; without a global lock their concurrent create()/run() calls race the single WASM
// runtime and throw "Session already started" / "Session mismatch". Serialising every op end-to-end removes the race.
let ortChain: Promise<unknown> = Promise.resolve();
function serial<T>(fn: () => Promise<T>): Promise<T> {
  const next = ortChain.then(fn, fn);
  ortChain = next.catch(() => {});
  return next;
}

function get(file: string): Promise<ort.InferenceSession | null> {
  return (sessions[file] ??= serial(async () => {
    try {
      const head = await fetch(`${base()}${file}`, { method: 'HEAD' });
      if (!head.ok) return null;
      return await ort.InferenceSession.create(`${base()}${file}`, { executionProviders: ['wasm'] });
    } catch {
      return null;
    }
  }));
}

export const classifierAvailable = async () => (await get('mpm-classifier.onnx')) != null;

async function runSerial(_file: string, s: ort.InferenceSession, feeds: Record<string, ort.Tensor>) {
  return serial(() => s.run(feeds));
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

/** geology OOD autoencoder over a batch of cells: feature rows -> the standardized-space reconstruction MSE (the
 * per-cell anomaly score, computed inside the ONNX). High = the evidence is out-of-envelope (the classifier is
 * extrapolating "under cover"). null if the model isn't trained. */
export async function runOod(rows: Float32Array, nRows: number): Promise<Float32Array | null> {
  const s = await get('geology-ood.onnx');
  if (!s) return null;
  const out = await runSerial('geology-ood.onnx', s, { x: new ort.Tensor('float32', rows, [nRows, N_FEATURES]) });
  return out.xr.data as Float32Array;
}
