import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useShellLang } from '@fasl-work/caos-app-shell';
import {
  analyzeCube, bestWeights, CASES, caseById, ciCheck, cubeFromFile, depositSet, fitLR, getLayer, layerRange,
  makeSyntheticArea, maskCells, posterior, predictLR, REAL_CASES, realCaseById, rocAuc,
  type Cube, type MPMCase, type RealCase, type RealCubeFile,
} from '../mpm/index.ts';
import { loadLearned, loadLearnedReal, loadPuConformal, loadRealCube, type LearnedFile, type PuConformalFile } from '../lib/artifacts.ts';
import { headToHeadContext, headToHeadDeclared, headToHeadProtocol, headToHeadVerdict } from '../lib/learned.ts';
import { runClassifier, runOod, runPuConformal, type Lane } from '../lib/ort.ts';
import { MapView } from '../viz/MapView.tsx';
import { PanelBoundary } from '../viz/PanelBoundary.tsx';
import { CurveChart } from '../viz/CurveChart.tsx';

const CATS = [
  'deposit-type terrane (the geological setting)',
  'data / validation regime (evidence richness)',
  'control (oracle / negative control)',
];
type Method = 'wofe' | 'logistic';
type Source = 'synthetic' | 'real';
const pct = (v: number, n = 1) => `${(v * 100).toFixed(n)}%`;
const num = (v: number | string | undefined | null, d = 3) => (typeof v === 'number' ? v.toFixed(d) : 'n/a');

// resample a capture curve (areaFrac, captureFrac) onto a common x grid for the chart
function resample(areaFrac: number[], captureFrac: number[], xs: number[]): number[] {
  return xs.map((x) => {
    for (let i = 0; i < areaFrac.length; i++) if (areaFrac[i] >= x) return captureFrac[i];
    return captureFrac[captureFrac.length - 1] ?? 0;
  });
}

// fuzzy-logic / index-overlay combiner (Bonham-Carter 1994; Carranza 2009): a data-model-free MPM method. Each active
// layer maps to a fuzzy membership in [0,1] (its favourable normalized value); the gamma operator blends the fuzzy-OR
// (increasive) and fuzzy-AND (decreasive) combinations. A genuine alternative to WofE/LR, no CI assumption, no fitting.
function fuzzyOverlay(cube: Cube, activeIds: string[], gamma = 0.8): Float64Array {
  const cells = maskCells(cube);
  const mems = activeIds.map((id) => {
    const layer = getLayer(cube, id);
    const { min, max } = layerRange(cube, id);
    const span = max - min || 1;
    const high = layer.highIsFavourable !== false;
    return (i: number) => {
      const v = layer.values[i];
      if (Number.isNaN(v)) return 0.5;
      const t = (v - min) / span;
      return high ? t : 1 - t;
    };
  });
  const f = new Float64Array(cube.nx * cube.ny);
  for (const i of cells) {
    let prod = 1;
    let orComp = 1;
    for (const m of mems) {
      const u = Math.max(0, Math.min(1, m(i)));
      prod *= u;
      orComp *= 1 - u;
    }
    const fAnd = prod;
    const fOr = 1 - orComp;
    f[i] = Math.pow(fOr, gamma) * Math.pow(fAnd, 1 - gamma);
  }
  return f;
}

// calibration / reliability: equal-count (decile) bins of the predicted score; per bin the mean predicted probability
// vs the observed deposit frequency. A well-calibrated model sits on the diagonal. Brier score + expected calibration
// error (ECE) summarize it. The C-SATURATE case flagged this readout was missing in-app; it lands here.
function calibration(cube: Cube, field: Float64Array, bins = 10) {
  const cells = maskCells(cube);
  const dep = depositSet(cube);
  const rows = cells.map((i) => ({ p: field[i], y: dep.has(i) ? 1 : 0 })).sort((a, b) => a.p - b.p);
  const out: { meanPred: number; obsFreq: number; n: number }[] = [];
  let brier = 0;
  for (const r of rows) brier += (r.p - r.y) ** 2;
  brier /= Math.max(1, rows.length);
  let ece = 0;
  const per = Math.ceil(rows.length / bins);
  for (let b = 0; b < bins; b++) {
    const slice = rows.slice(b * per, (b + 1) * per);
    if (!slice.length) continue;
    const mp = slice.reduce((s, r) => s + r.p, 0) / slice.length;
    const of = slice.reduce((s, r) => s + r.y, 0) / slice.length;
    out.push({ meanPred: mp, obsFreq: of, n: slice.length });
    ece += (slice.length / rows.length) * Math.abs(mp - of);
  }
  return { bins: out, brier, ece };
}


/** ADR-0071 rules 4+5. Twelve flat sibling tabs is a list, not an architecture: the user reads every label
 *  to find one view, and the row cannot fit a narrow window, so the rightmost tabs become unreachable
 *  under the shell's `overflow: hidden`. Grouped by the question being asked; the sub-views are revealed
 *  from the same tab. */
const TAB_GROUPS: { id: string; en: string; es: string; members: string[] }[] = [
  { id: 'map',      en: 'Map',        es: 'Mapa',        members: ['map', 'weights'] },
  { id: 'evidence', en: 'Evidence',   es: 'Evidencia',   members: ['rates', 'ci'] },
  { id: 'skill',    en: 'Skill',      es: 'Desempeño',   members: ['roc', 'cv', 'calib'] },
  { id: 'compare',  en: 'Compare',    es: 'Comparar',    members: ['method', 'overlay'] },
  { id: 'learned',  en: 'Learned',    es: 'Aprendido',   members: ['whatif', 'anomaly', 'puconformal'] },
];

export default function Tool() {
  const es = useShellLang() === 'es';
  const [source, setSource] = useState<Source>('synthetic');
  const [caseId, setCaseId] = useState('K-PORPHYRY');
  const [realCaseId, setRealCaseId] = useState(REAL_CASES[0].id);
  const [layerOff, setLayerOff] = useState<Record<string, boolean>>({});
  const [method, setMethod] = useState<Method>('wofe');

  const [realCube, setRealCube] = useState<Cube | null>(null);
  const [realMeta, setRealMeta] = useState<RealCubeFile | null>(null);
  const [realErr, setRealErr] = useState<string | null>(null);

  const [learned, setLearned] = useState<LearnedFile | null>(null);
  const [learnedReal, setLearnedReal] = useState<LearnedFile | null>(null);
  const [puConformal, setPuConformal] = useState<PuConformalFile | null>(null);

  const isReal = source === 'real';
  const theCase = useMemo<MPMCase>(() => caseById(caseId), [caseId]);
  const theReal = useMemo<RealCase>(() => realCaseById(realCaseId), [realCaseId]);
  const synthCube = useMemo(() => makeSyntheticArea(theCase.spec).cube, [theCase]);

  const cube: Cube | null = isReal ? realCube : synthCube;
  const layerIds = isReal ? realMeta?.layer_ids ?? theReal.layerIds : theCase.layerIds;
  const activeIds = useMemo(() => layerIds.filter((id) => !layerOff[id]), [layerIds, layerOff]);
  const lane: Lane = isReal ? 'real' : 'synthetic';
  const activeLearned = isReal ? learnedReal : learned;

  useEffect(() => { setLayerOff({}); }, [caseId, realCaseId, source]);
  useEffect(() => { loadLearned().then(setLearned).catch(() => setLearned(null)); }, []);
  useEffect(() => { loadLearnedReal().then(setLearnedReal).catch(() => setLearnedReal(null)); }, []);
  useEffect(() => { loadPuConformal().then(setPuConformal).catch(() => setPuConformal(null)); }, []);

  // load the baked real cube when the Real lane is picked (arrays only; the WofE engine runs live on it)
  useEffect(() => {
    if (!isReal) return;
    let cancel = false;
    setRealCube(null);
    setRealErr(null);
    loadRealCube(theReal.file)
      .then((f) => { if (!cancel) { setRealMeta(f); setRealCube(cubeFromFile(f)); } })
      .catch((e) => { if (!cancel) setRealErr(String(e)); });
    return () => { cancel = true; };
  }, [isReal, theReal]);

  return (
    <div className="page-body pf-layout">
      <aside className="pf-side">
        {/* ADR-0070 entry: without a visible control the focus route is an orphan that ships and nobody
            reaches. It carries the SELECTED scenario on whichever lane is active. */}
        <Link className="pf-focus-enter" to={`/focus/${isReal ? realCaseId : caseId}`}>
          <span className="pf-focus-enter-t">{es ? 'Modo enfoque' : 'Focus mode'}</span>
          <span className="pf-focus-enter-d">
            {es ? 'Abrir este area a pantalla completa' : 'Open this area full screen'}
          </span>
        </Link>
        <div className="pf-card">
          <div className="pf-card-t">{es ? 'Fuente' : 'Source'}</div>
          <div className="pf-chips">
            <button className={`chip ${!isReal ? 'on' : ''}`} onClick={() => setSource('synthetic')}>{es ? 'Sintético' : 'Synthetic'}</button>
            <button className={`chip ${isReal ? 'on' : ''}`} onClick={() => setSource('real')}>{es ? 'Muestra real' : 'Real sample'}</button>
          </div>
          <div className="pf-cap pf-muted">{isReal
            ? (es ? 'dataset abierto real; los knobs sintéticos se desactivan, se elige el dato y todas las herramientas se ejecutan sobre él' : 'a real open dataset; the synthetic knobs disable, the datum is selected and every tool runs on it')
            : (es ? 'áreas sintéticas con verdad conocida (los knobs plantan los pesos)' : 'synthetic areas with known ground truth (the knobs plant the weights)')}</div>
        </div>

        {!isReal ? (
          <div className="pf-card">
            <div className="pf-card-t">{es ? 'Caso (sintético)' : 'Case (synthetic)'}</div>
            {CATS.map((cat) => (
              <div key={cat} className="pf-catgroup">
                <div className="pf-catlabel">{cat.split(' (')[0]}</div>
                <div className="pf-chips">
                  {CASES.filter((c) => c.category === cat).map((c) => (
                    <button key={c.id} className={`chip ${caseId === c.id ? 'on' : ''}`} title={c.name} onClick={() => setCaseId(c.id)}>{c.id}</button>
                  ))}
                </div>
              </div>
            ))}
            <div className="pf-cap">{theCase.name}</div>
            <div className="pf-cap pf-muted">{theCase.realOrSynthetic} · {theCase.validationAnchor}</div>
          </div>
        ) : (
          <div className="pf-card">
            <div className="pf-card-t">{es ? 'Dato real' : 'Real datum'}</div>
            <div className="pf-chips">
              {REAL_CASES.map((c) => (
                <button key={c.id} className={`chip ${realCaseId === c.id ? 'on' : ''}`} title={c.name} onClick={() => setRealCaseId(c.id)}>{c.id}</button>
              ))}
            </div>
            <div className="pf-cap">{theReal.name}</div>
            <div className="pf-cap pf-muted">{theReal.realOrSynthetic} · {theReal.category}</div>
            {realMeta && (
              <div className="pf-provbox">
                <div className="pf-prov-legend">
                  <span><b className="pf-real">REAL</b> {es ? 'geofísica medida: mag · grav · lab(tomografía) · satgrav' : 'measured geophysics: mag · grav · lab(tomography) · satgrav'}</span>
                  <span><b className="pf-derived">DERIVED</b> {es ? 'por nosotros desde vectores reales: faultprox · marginprox' : 'by us from real vectors: faultprox · marginprox'}</span>
                  <span><b className="pf-recomp">RECOMPUTED</b> {es ? 'nuestro posterior WofE en el navegador, no el modelo H3+gradient-boosting publicado' : 'our browser WofE posterior, not the published H3 + gradient-boosting model'}</span>
                </div>
                <div className="pf-cite">{realMeta.citation}</div>
                <div className="pf-cap pf-muted">{realMeta.license}</div>
              </div>
            )}
          </div>
        )}

        {cube && (
          <div className="pf-card">
            <div className="pf-card-t">{es ? 'Capas (en vivo)' : 'Layers (live)'}</div>
            <div className="pf-chips">
              {layerIds.map((id) => (
                <button key={id} className={`chip ${!layerOff[id] ? 'on' : ''}`} onClick={() => setLayerOff((s) => ({ ...s, [id]: !s[id] }))}>{id}</button>
              ))}
            </div>
            <div className="pf-cap pf-muted">{es ? 'las capas de evidencia se activan y desactivan, el posterior se recalcula en vivo' : 'toggle evidence layers, the posterior recomputes live'}</div>
            <div className="pf-card-t" style={{ marginTop: 12 }}>{es ? 'Método' : 'Method'}</div>
            <div className="pf-chips">
              {(['wofe', 'logistic'] as Method[]).map((m) => (
                <button key={m} className={`chip ${method === m ? 'on' : ''}`} onClick={() => setMethod(m)}>{m === 'wofe' ? 'WofE' : (es ? 'logística' : 'logistic')}</button>
              ))}
            </div>
          </div>
        )}
      </aside>
      <main className="pf-main">
        {isReal && !cube && !realErr && <div className="pf-pending"><strong>{es ? 'Cargando el dato real...' : 'Loading the real datum...'}</strong></div>}
        {isReal && realErr && <div className="pf-pending"><strong>{es ? 'No se pudo cargar el cubo real' : 'Could not load the real cube'}</strong><p>{realErr}</p></div>}
        {cube && (
          <CubeViews
            cube={cube} activeIds={activeIds} method={method} lane={lane}
            learned={activeLearned} isReal={isReal} es={es} puConformal={puConformal}
          />
        )}
      </main>
    </div>
  );
}

function CubeViews({ cube, activeIds, method, lane, learned, isReal, es, puConformal }: {
  cube: Cube; activeIds: string[]; method: Method; lane: Lane; learned: LearnedFile | null;
  isReal: boolean; es: boolean; puConformal: PuConformalFile | null;
}) {
  const [learnedField, setLearnedField] = useState<Float64Array | null>(null);
  const [oodField, setOodField] = useState<{ field: Float64Array; max: number; offFrac: number } | null>(null);
  const [puField, setPuField] = useState<Float64Array | null>(null);
  const [piIdx, setPiIdx] = useState(1); // default -> pi = 0.05 (pi_sensitivity index 1)
  const [alphaIdx, setAlphaIdx] = useState(0); // default -> alpha = 0.10 (nominal coverage 0.90)
  const oodThr = (learned?.ood as { threshold?: number } | undefined)?.threshold ?? null;

  const best = useMemo(() => activeIds.map((id) => ({ id, ...bestWeights(cube, id) })), [cube, activeIds]);
  const pats = useMemo(() => best.map((b) => b.pattern), [best]);
  const post = useMemo(() => posterior(cube, pats, best.map((b) => b.weights)), [cube, pats, best]);
  const ci = useMemo(() => ciCheck(cube, pats, post), [cube, pats, post]);
  const analysis = useMemo(() => analyzeCube(cube, activeIds.length ? activeIds : cube.layers.map((l) => l.id)), [cube, activeIds]);

  const lrField = useMemo(() => {
    const cells = maskCells(cube);
    const dep = depositSet(cube);
    const X = cells.map((i) => pats.map((p) => (p.present[i] === 1 ? 1 : 0)));
    const y = cells.map((i) => (dep.has(i) ? 1 : 0));
    const fit = fitLR(X, y, { ridge: 1e-3 });
    const f = new Float64Array(cube.nx * cube.ny);
    for (let r = 0; r < cells.length; r++) f[cells[r]] = predictLR(fit.beta, X[r]);
    return f;
  }, [cube, pats]);

  const fuzzyField = useMemo(() => fuzzyOverlay(cube, activeIds.length ? activeIds : cube.layers.map((l) => l.id)), [cube, activeIds]);
  const fuzzyAuc = useMemo(() => rocAuc(cube, fuzzyField), [cube, fuzzyField]);

  const map = useMemo(() => {
    if (method === 'logistic') return { field: lrField, range: [0, 1] as [number, number], label: 'P(LR)' };
    return { field: post.prob, range: [0, 1] as [number, number], label: 'P' };
  }, [method, post, lrField]);

  const calib = useMemo(() => calibration(cube, map.field), [cube, map]);

  // the two learned models, run live over the cube's cells on the lane's feature space + model file (never the
  // synthetic model on the real cube). Graceful: absent ONNX -> null fields -> honest pending state.
  useEffect(() => {
    let cancel = false;
    setLearnedField(null);
    setOodField(null);
    const feats = lane === 'real' ? ['mag', 'grav', 'lab', 'satgrav', 'faultprox', 'marginprox'] : ['mag', 'rad', 'geochem', 'struct'];
    const cells = maskCells(cube);
    const F = feats.length;
    const rows = new Float32Array(cells.length * F);
    for (let r = 0; r < cells.length; r++) {
      for (let j = 0; j < F; j++) {
        const layer = cube.layers.find((l) => l.id === feats[j]);
        const v = layer ? layer.values[cells[r]] : 0;
        rows[r * F + j] = Number.isNaN(v) ? 0 : v;
      }
    }
    Promise.all([runClassifier(rows, cells.length, lane), runOod(rows, cells.length, lane)]).then(([pc, mse]) => {
      if (cancel) return;
      if (pc) {
        const f = new Float64Array(cube.nx * cube.ny);
        for (let r = 0; r < cells.length; r++) f[cells[r]] = pc[r];
        setLearnedField(f);
      }
      if (mse) {
        const f = new Float64Array(cube.nx * cube.ny);
        let max = 0;
        let off = 0;
        const thr = (learned?.ood as { threshold?: number } | undefined)?.threshold ?? Infinity;
        for (let r = 0; r < cells.length; r++) { f[cells[r]] = mse[r]; if (mse[r] > max) max = mse[r]; if (mse[r] > thr) off++; }
        setOodField({ field: f, max, offFrac: off / Math.max(1, cells.length) });
      }
    });
    return () => { cancel = true; };
  }, [cube, activeIds, lane, learned]);

  // PU-Conformal nnPU score, run live over the real cube (the offline lane trains only the 6-feature real model).
  useEffect(() => {
    let cancel = false;
    setPuField(null);
    if (lane !== 'real') return;
    const feats = ['mag', 'grav', 'lab', 'satgrav', 'faultprox', 'marginprox'];
    const cells = maskCells(cube);
    const F = feats.length;
    const rows = new Float32Array(cells.length * F);
    for (let r = 0; r < cells.length; r++) {
      for (let j = 0; j < F; j++) {
        const layer = cube.layers.find((l) => l.id === feats[j]);
        const v = layer ? layer.values[cells[r]] : 0;
        rows[r * F + j] = Number.isNaN(v) ? 0 : v;
      }
    }
    runPuConformal(rows, cells.length).then((p) => {
      if (cancel || !p) return;
      const f = new Float64Array(cube.nx * cube.ny).fill(NaN);
      for (let r = 0; r < cells.length; r++) f[cells[r]] = p[r];
      setPuField(f);
    });
    return () => { cancel = true; };
  }, [cube, lane]);

  // the nnPU score compresses near 0, so an adaptive [min,max] range reveals its spatial structure (the [0,1] scale
  // would render an almost-uniform map). This is a display stretch only; the values read at the cursor are unchanged.
  const puRange = useMemo<[number, number]>(() => {
    if (!puField) return [0, 1];
    let mn = Infinity;
    let mx = -Infinity;
    for (const v of puField) { if (Number.isNaN(v)) continue; if (v < mn) mn = v; if (v > mx) mx = v; }
    return Number.isFinite(mn) && mx > mn ? [mn, mx] : [0, 1];
  }, [puField]);

  // the selected pi row of the offline sensitivity sweep + its conformal level at the selected alpha
  const piRow = puConformal?.pi_sensitivity?.[piIdx] ?? null;
  const confLevel = piRow?.conformal?.[alphaIdx] ?? puConformal?.conformal?.levels?.[alphaIdx] ?? null;
  const puSet = useMemo(() => {
    if (!puField || !confLevel || !Number.isFinite(confLevel.threshold)) return null;
    const thr = confLevel.threshold;
    const f = new Float64Array(cube.nx * cube.ny).fill(NaN);
    let inSet = 0;
    let n = 0;
    for (const i of maskCells(cube)) {
      if (Number.isNaN(puField[i])) continue;
      n++;
      const member = puField[i] >= thr ? 1 : 0;
      f[i] = member;
      inSet += member;
    }
    return { field: f, frac: n ? inSet / n : 0 };
  }, [puField, confLevel, cube]);

  const Kpi = ({ label, value }: { label: string; value: string }) => (
    <div className="pf-kpi"><div className="pf-kpi-v">{value}</div><div className="pf-kpi-l">{label}</div></div>
  );

  const xs = useMemo(() => Array.from({ length: 51 }, (_, i) => i / 50), []);
  const captureSeries = useMemo(() => [
    { label: 'prediction (spatial CV)', y: resample(analysis.capture.prediction.areaFrac, analysis.capture.prediction.captureFrac, xs), color: undefined },
    { label: 'fitting (success)', y: resample(analysis.capture.success.areaFrac, analysis.capture.success.captureFrac, xs), dash: [4, 4] as number[] },
  ], [analysis, xs]);

  const roc = useMemo(() => {
    const cells = maskCells(cube);
    const dep = depositSet(cube);
    const ranked = [...cells].sort((a, b) => map.field[b] - map.field[a]);
    const P = dep.size;
    const N = cells.length - P;
    const tpr: number[] = [0];
    const fpr: number[] = [0];
    let tp = 0;
    let fp = 0;
    const step = Math.max(1, Math.floor(ranked.length / 50));
    for (let i = 0; i < ranked.length; i++) {
      if (dep.has(ranked[i])) tp++; else fp++;
      if (i % step === 0 || i === ranked.length - 1) { tpr.push(tp / P); fpr.push(fp / N); }
    }
    return { tpr, fpr };
  }, [cube, map]);

  const clf = learned?.classifier;
  // the WofE value of the head-to-head is shown only when the file declares the pair's protocol
  const h2hDeclared = headToHeadDeclared(lane, learned);
  const h2hVerdict = headToHeadVerdict(lane, learned, es);

  // every AUC below states its protocol; values computed under different protocols are never set up as a contest
  const FIT = es ? 'ajuste, sin CV' : 'fit, no CV';
  const fitAuc = (model: string) => (es ? `AUC ${model} (${FIT})` : `${model} AUC (${FIT})`);

  const recomputeNote = isReal
    ? (es ? 'El posterior es nuestra recomputación WofE en el navegador sobre una sub-región rasterizada, no el modelo H3+gradient-boosting publicado (Lawley 2022).' : 'The posterior is our browser WofE recomputation over a rasterized sub-region, not the published Lawley 2022 H3 + gradient-boosting model.')
    : '';

  const tabs = [
    {
      id: 'map', label: es ? 'Mapa' : 'Map',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'Mapa posterior de prospectividad P(depósito | evidencia) por celda (colormap viridis). Los depósitos conocidos como marcadores; al pasar el cursor se lee el valor.'
            : 'Posterior prospectivity map P(deposit | evidence) per cell (viridis colormap). Known deposits as markers; hover to read the value.'}
            {isReal && <> {recomputeNote}</>}</div>
          <MapView nx={cube.nx} ny={cube.ny} field={map.field} range={map.range} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel={map.label} />
          <div className="pf-kpis">
            <Kpi label={fitAuc('WofE')} value={analysis.rocAuc.toFixed(3)} />
            <Kpi label="capture@10%" value={pct(analysis.capture.prediction.captureAt10)} />
            <Kpi label="CI ratio" value={ci.ciRatio.toFixed(2)} />
            <Kpi label={es ? 'depósitos' : 'deposits'} value={`${cube.depositIdx.length}`} />
          </div>
          {isReal && (
            <p className="pf-note">{es
              ? `Dato real: ${cube.depositIdx.length} celdas con ocurrencia Pb-Zn (MVT) en ${cube.nx}x${cube.ny} celdas (~${cube.cellKm} km/celda). REAL = geofísica medida; DERIVED = proximidad a fallas/márgenes calculada por nosotros.`
              : `Real datum: ${cube.depositIdx.length} cells with a Pb-Zn (MVT) occurrence over a ${cube.nx}x${cube.ny} grid (~${cube.cellKm} km/cell). REAL = measured geophysics; DERIVED = fault/margin proximity computed by us.`}</p>
          )}
        </div>
      ),
    },
    {
      id: 'weights', label: es ? 'Pesos' : 'Layer weights',
      content: (
        <div className="pf-vizstack">
          <table className="cmp-table">
            <thead><tr><th>{es ? 'capa' : 'layer'}</th><th>t*</th><th>W+</th><th>W-</th><th>{es ? 'contraste' : 'contrast'}</th><th>studC</th></tr></thead>
            <tbody>
              {analysis.layers.map((l) => (
                <tr key={l.id}>
                  <td>{l.id}</td><td>{l.tStar.toFixed(2)}</td><td>{l.wPlus.toFixed(2)}</td><td>{l.wMinus.toFixed(2)}</td>
                  <td><b>{l.contrast.toFixed(2)}</b></td>
                  <td style={{ color: Math.abs(l.studC) < 1.96 ? 'var(--color-fg-faint)' : undefined }}>{l.studC.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="pf-note">{es
            ? 'W+/W- son los log-likelihood ratios de presencia/ausencia del patrón dado un depósito. |studC| < 1.96 (gris) = no significativo. t* es el umbral de contraste máximo.'
            : 'W+/W- are the log-likelihood ratios of the pattern present/absent given a deposit. |studC| < 1.96 (greyed) = not significant. t* is the maximizing-contrast threshold.'}
            {isReal && <> {es ? 'En el dato real, lab (tomografía) y faultprox suelen llevar el mayor contraste, consistente con Lawley 2022.' : 'On the real datum, lab (tomography) and faultprox tend to carry the largest contrast, consistent with Lawley 2022.'}</>}</p>
        </div>
      ),
    },
    {
      id: 'rates', label: es ? 'Tasas de captura' : 'Capture rates',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? '% de depósitos capturados vs % del área (ranking por prospectividad). prediction = held-out espacial; fitting = sobre los datos de entrenamiento (optimista).'
            : '% of deposits captured vs % of area (ranked by prospectivity). prediction = spatial held-out; fitting = on the training data (optimistic).'}</div>
          <CurveChart x={xs} series={captureSeries} xLabel={es ? '% área' : '% area'} yLabel={es ? '% depósitos' : '% deposits'} diagonal />
          <p className="pf-note">{es
            ? `El top 10% del área captura ${pct(analysis.capture.prediction.captureAt10)} de los depósitos held-out (spatial CV). El gap fitting-prediction mide el sobreajuste.`
            : `The top 10% of the area captures ${pct(analysis.capture.prediction.captureAt10)} of the held-out deposits (spatial CV). The fitting-prediction gap measures over-fit.`}</p>
        </div>
      ),
    },
    {
      id: 'roc', label: 'ROC',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es ? 'Curva ROC del posterior vs las etiquetas de depósito. Con datos presence-only el AUC es secundario (las "ausencias" pueden tener depósitos no descubiertos).' : 'ROC of the posterior vs the deposit labels. With presence-only data AUC is secondary (the "absences" may host undiscovered deposits).'}</div>
          <CurveChart x={roc.fpr} series={[{ label: method === 'logistic' ? 'logistic' : 'WofE', y: roc.tpr }]} xLabel="FPR" yLabel="TPR" diagonal />
          <div className="pf-kpis">
            <Kpi label={fitAuc('WofE')} value={analysis.rocAuc.toFixed(3)} />
            <Kpi label={fitAuc('LR')} value={analysis.lr.rocAuc.toFixed(3)} />
          </div>
        </div>
      ),
    },
    {
      id: 'ci', label: es ? 'Independencia cond.' : 'Conditional indep.',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'El test omnibus de Agterberg-Cheng: bajo independencia condicional la suma del posterior ~ N(D). T > N(D) implica que el posterior está inflado por capas correlacionadas.'
            : 'The Agterberg-Cheng omnibus test: under conditional independence the posterior sum ~ N(D). T > N(D) means the posterior is inflated by correlated layers.'}</div>
          <div className="pf-kpis">
            <Kpi label="T (sum posterior)" value={ci.T.toFixed(1)} />
            <Kpi label="N(D)" value={`${ci.nD}`} />
            <Kpi label="CI ratio" value={ci.ciRatio.toFixed(2)} />
            <Kpi label="z" value={ci.z.toFixed(1)} />
          </div>
          <table className="cmp-table">
            <thead><tr><th>{es ? 'par de capas' : 'layer pair'}</th><th>chi2</th><th>Cramer V</th></tr></thead>
            <tbody>
              {ci.pairwise.map((p) => (
                <tr key={`${p.a}-${p.b}`}><td>{p.a} · {p.b}</td><td>{p.chi2.toFixed(2)}</td><td>{p.cramersV.toFixed(2)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="pf-note">{ci.ciRatio < 0.85
            ? (es ? `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 implica violación: el posterior WofE está sobre-estimado. Usar el ranking relativo, o la regresión logística como alternativa sin supuesto de CI (AUC ${analysis.lr.rocAuc.toFixed(3)}).` : `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 means violation: the WofE posterior is over-estimated. Use the relative ranking, or logistic regression as the CI-free alternative (AUC ${analysis.lr.rocAuc.toFixed(3)}).`)
            : (es ? `CI ratio ${ci.ciRatio.toFixed(2)} ~ 1 implica consistencia con independencia condicional; el posterior WofE es razonable.` : `CI ratio ${ci.ciRatio.toFixed(2)} ~ 1 means consistent with conditional independence; the WofE posterior is reasonable.`)}
            {isReal && <> {es ? 'Esperado en dato real: mag/grav/lab son físicamente correlacionadas, así que el test se dispara. No es una trampa plantada, es la realidad geofísica; por eso la logística es la ruta honesta.' : 'Expected on real data: mag/grav/lab are physically correlated, so the test fires. This is not a planted trap, it is the geophysical reality; hence logistic regression is the honest route.'}</>}</p>
        </div>
      ),
    },
    {
      id: 'cv', label: es ? 'Validación CV' : 'CV inflation',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? `El mismo modelo (WofE) bajo CV aleatorio vs CV espacial-por-bloques: ${analysis.cv.k} folds, bloques de ${analysis.cv.blockCells}x${analysis.cv.blockCells} celdas, pesos reajustados con los folds de entrenamiento, cada celda del mapa puntuada una vez mientras está retenida, un único AUC sobre los scores agrupados. El gap es la inflación por autocorrelación espacial (el AUC aleatorio miente).`
            : `The same model (WofE) under random CV vs spatial-block CV: ${analysis.cv.k} folds, ${analysis.cv.blockCells}x${analysis.cv.blockCells}-cell blocks, weights refitted on the training folds, every map cell scored once while held out, one AUC over the pooled scores. The gap is the inflation from spatial autocorrelation (the random AUC lies).`}</div>
          <div className="pf-kpis">
            <Kpi label={es ? 'AUC WofE, CV aleatorio' : 'WofE AUC, random CV'} value={analysis.cv.randomAuc.toFixed(3)} />
            <Kpi label={es ? 'AUC WofE, CV espacial' : 'WofE AUC, spatial CV'} value={analysis.cv.spatialAuc.toFixed(3)} />
            <Kpi label={es ? 'gap de inflación' : 'inflation gap'} value={analysis.cv.inflationGap.toFixed(3)} />
          </div>
          <p className="pf-note">{es
            ? 'El número reportado por defecto es siempre el espacial (honesto). El aleatorio se muestra solo para exponer la inflación.'
            : 'The default reported number is always the spatial one (honest). The random one is shown only to expose the inflation.'}
            {isReal && <> {es ? 'En el belt MVT real las ocurrencias están muy agrupadas (distrito Tri-State), así que el CV espacial cae cerca del azar: la geofísica regional sola tiene poca habilidad de transferencia espacial. Es un hallazgo honesto, no un defecto.' : 'On the real MVT belt the occurrences are strongly clustered (the Tri-State district), so spatial CV drops near chance: regional geophysics alone has little spatial-transfer skill. That is an honest finding, not a defect.'}</>}</p>
        </div>
      ),
    },
    {
      id: 'method', label: es ? 'Comparar método' : 'Method compare',
      content: (
        <div className="pf-vizstack">
          <table className="cmp-table">
            <thead><tr><th>{es ? 'método' : 'method'}</th><th>ROC AUC</th><th>{es ? 'protocolo' : 'protocol'}</th><th>{es ? 'nota' : 'note'}</th></tr></thead>
            <tbody>
              <tr><td><b>WofE</b></td><td>{analysis.rocAuc.toFixed(3)}</td><td>{es ? 'ajustado y evaluado en las mismas celdas, sin CV' : 'fitted and scored on the same cells, no CV'}</td><td>{es ? 'caja blanca, la autoridad' : 'white-box, the authority'}</td></tr>
              <tr><td>{es ? 'logística' : 'logistic'}</td><td>{analysis.lr.rocAuc.toFixed(3)}</td><td>{es ? 'ajustado y evaluado en las mismas celdas, sin CV' : 'fitted and scored on the same cells, no CV'}</td><td>{es ? 'sin supuesto de independencia condicional' : 'no conditional-independence assumption'}</td></tr>
              <tr><td>{es ? 'fuzzy / index-overlay' : 'fuzzy / index-overlay'}</td><td>{fuzzyAuc.toFixed(3)}</td><td>{es ? 'sin ajuste (no usa las etiquetas), evaluado en las mismas celdas' : 'no fitting (labels unused), scored on the same cells'}</td><td>{es ? 'operador gamma, sin supuesto de CI' : 'gamma operator, no CI assumption'}</td></tr>
            </tbody>
          </table>
          <p className="pf-note">{es
            ? 'Cuando se cumple la independencia condicional, la logística ~ WofE. Cuando se viola, la logística ajusta las capas en conjunto y no doble-cuenta. El overlay fuzzy es un método MPM clásico sin ajuste.'
            : 'When conditional independence holds, logistic ~ WofE. When it is violated, logistic fits the layers jointly and does not double-count. The fuzzy overlay is a classic fit-free MPM method.'}</p>
          <p className="pf-note">{es
            ? 'Las tres filas comparten protocolo: todas las celdas del mapa con las capas activas, sin datos retenidos, así que WofE y la logística son optimistas. La habilidad con datos retenidos está en Desempeño > Validación CV. El MLP aprendido tiene otro protocolo (validación cruzada), por eso no está en esta tabla: se compara con WofE bajo un único protocolo compartido en Aprendido > What-if.'
            : 'The three rows share one protocol: every map cell of the active layers, nothing held out, so WofE and logistic are optimistic. Held-out skill is in Skill > CV inflation. The learned MLP is measured under another protocol (cross-validation), so it is not in this table: it is compared with WofE under one shared protocol in Learned > What-if.'}</p>
        </div>
      ),
    },
    {
      id: 'overlay', label: es ? 'Fuzzy / overlay' : 'Fuzzy / overlay',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'Combinador fuzzy-logic / index-overlay (Bonham-Carter 1994; Carranza 2009): cada capa mapea a una pertenencia difusa favorable en [0,1]; el operador gamma mezcla el fuzzy-AND (producto) y el fuzzy-OR. Sin ajuste de datos ni supuesto de independencia condicional, un método MPM distinto de WofE/LR.'
            : 'Fuzzy-logic / index-overlay combiner (Bonham-Carter 1994; Carranza 2009): each layer maps to a favourable fuzzy membership in [0,1]; the gamma operator blends the fuzzy-AND (product) and the fuzzy-OR. No data fitting, no conditional-independence assumption, a distinct MPM method from WofE/LR.'}</div>
          <MapView nx={cube.nx} ny={cube.ny} field={fuzzyField} range={[0, 1]} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel={es ? 'favorabilidad' : 'favourability'} />
          <div className="pf-kpis">
            <Kpi label={es ? 'AUC fuzzy (sin ajuste)' : 'fuzzy AUC (no fitting)'} value={fuzzyAuc.toFixed(3)} />
            <Kpi label={fitAuc('WofE')} value={analysis.rocAuc.toFixed(3)} />
            <Kpi label="gamma" value="0.80" />
          </div>
          <p className="pf-note">{es
            ? 'El overlay fuzzy no aprende pesos: expone el juicio del experto (elección de pertenencias + gamma). Útil como sanity-check independiente frente al WofE data-driven.'
            : 'The fuzzy overlay learns no weights: it encodes expert judgement (the membership choice + gamma). Useful as an independent sanity-check against the data-driven WofE.'}</p>
        </div>
      ),
    },
    {
      id: 'calib', label: es ? 'Calibración' : 'Calibration',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'Diagrama de fiabilidad: en deciles del score, la probabilidad media predicha vs la frecuencia de depósitos observada. Un modelo bien calibrado se sitúa sobre la diagonal. Brier + ECE lo resumen. (El caso C-SATURATE señalaba que faltaba este readout.)'
            : 'Reliability diagram: over deciles of the score, the mean predicted probability vs the observed deposit frequency. A well-calibrated model sits on the diagonal. Brier + ECE summarize it. (The C-SATURATE case flagged this readout was missing.)'}</div>
          <CurveChart
            x={calib.bins.map((b) => b.meanPred)}
            series={[{ label: es ? 'observado vs predicho' : 'observed vs predicted', y: calib.bins.map((b) => b.obsFreq) }]}
            xLabel={es ? 'prob. media predicha' : 'mean predicted prob.'} yLabel={es ? 'frecuencia observada' : 'observed frequency'} diagonal
          />
          <div className="pf-kpis">
            <Kpi label={es ? 'Brier' : 'Brier'} value={calib.brier.toFixed(4)} />
            <Kpi label="ECE" value={calib.ece.toFixed(4)} />
            <Kpi label={es ? 'método' : 'method'} value={method === 'logistic' ? 'LR' : 'WofE'} />
          </div>
          <p className="pf-note">{es
            ? 'El presence-only sesga la calibración absoluta (las "ausencias" incluyen depósitos no descubiertos), así que se lee como fiabilidad relativa entre bins más que como probabilidades absolutas.'
            : 'Presence-only biases the absolute calibration (the "absences" include undiscovered deposits), so read it as relative reliability across bins rather than absolute probabilities.'}</p>
        </div>
      ),
    },
    {
      id: 'whatif', label: es ? 'What-if (MLP)' : 'What-if (MLP)',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'El clasificador MLP aprendido pinta un mapa de prospectividad alternativo, comparable en vivo con el WofE de caja blanca.'
            : 'The learned MLP classifier paints an alternative prospectivity map, compared live to the white-box WofE.'}
            {isReal && <> {es ? 'Entrenado sobre el cubo real de 6 capas (no el modelo sintético de 4 capas).' : 'Trained on the real 6-layer cube (not the synthetic 4-layer model).'}</>}</div>
          {!learnedField ? (
            <div className="pf-pending">
              <strong>{es ? 'Clasificador: pendiente de entrenamiento' : 'Classifier: pending training'}</strong>
              <p>{es ? 'Ejecutar el pipeline de aprendidos (torch -> ONNX). El WofE de caja blanca se ejecuta en vivo mientras tanto.' : 'Run the learned pipeline (torch -> ONNX). The white-box WofE runs live meanwhile.'}</p>
            </div>
          ) : (
            <>
              <MapView nx={cube.nx} ny={cube.ny} field={learnedField} range={[0, 1]} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel="P(MLP)" />
              <div className="pf-kpis">
                <Kpi label={es ? 'AUC MLP, CV espacial' : 'MLP AUC, spatial CV'} value={num(clf?.spatial_cv?.mlp_roc_auc)} />
                <Kpi label={es ? 'AUC WofE, CV espacial' : 'WofE AUC, spatial CV'} value={h2hDeclared ? num(clf?.spatial_cv?.wofe_roc_auc) : 'n/a'} />
                <Kpi label={es ? 'AUC MLP, CV aleatorio (inflado)' : 'MLP AUC, random CV (inflated)'} value={num(clf?.random_cv?.mlp_roc_auc)} />
                <Kpi label={es ? 'gap de inflación del MLP' : 'MLP inflation gap'} value={num(clf?.inflation_gap)} />
                {h2hDeclared && typeof clf?.spatial_cv?.distance_null_roc_auc === 'number' && (
                  <Kpi label={es ? 'AUC línea base de distancia, CV espacial' : 'distance baseline AUC, spatial CV'} value={num(clf.spatial_cv.distance_null_roc_auc)} />
                )}
              </div>
              <p className="pf-note">{headToHeadProtocol(lane, learned, es)}{h2hVerdict && <> {h2hVerdict}</>} {headToHeadContext(lane, learned, es)} {isReal
                ? (es ? 'Son valores offline con las seis capas; activar o desactivar capas no los cambia.' : 'These are offline values over all six layers; toggling layers does not change them.')
                : (es ? 'Son valores offline agrupados sobre los casos de entrenamiento; ni el caso elegido ni las capas activas los cambian.' : 'These are offline values pooled over the training cases; neither the selected case nor the active layers change them.')}</p>
              <p className="pf-note">{es
                ? 'El WofE de caja blanca es la autoridad interpretable. El CV aleatorio se muestra solo para exponer la inflación (aleatorio menos espacial). No hay victoria fabricada: solo se comparan valores medidos con el mismo protocolo.'
                : 'The white-box WofE is the interpretable authority. Random CV is shown only to expose the inflation (random minus spatial). No fabricated win: only values measured under the same protocol are compared.'}
                {isReal && clf?.nocv && <> {es
                  ? `El AUC de WofE sin CV (ajustado y evaluado en las mismas celdas, ${num(clf.nocv.wofe_roc_auc)}) es un número de ajuste y no se compara con estos.`
                  : `The WofE AUC without CV (fitted and scored on the same cells, ${num(clf.nocv.wofe_roc_auc)}) is a fitting number and is not compared with these.`}</>}</p>
            </>
          )}
        </div>
      ),
    },
    {
      id: 'anomaly', label: es ? 'Anomalía (AE)' : 'Anomaly (AE)',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'El autoencoder pinta el mapa de anomalía por celda (error de reconstrucción): dónde la evidencia está fuera del envolvente entrenado, un aviso de no confiar en el clasificador bajo cobertura.'
            : 'The autoencoder paints the per-cell anomaly map (reconstruction error): where the evidence is outside the trained envelope, a signal not to trust the classifier under cover.'}</div>
          {!oodField ? (
            <div className="pf-pending">
              <strong>{es ? 'Autoencoder OOD: pendiente de entrenamiento' : 'OOD autoencoder: pending training'}</strong>
              <p>{es ? 'Entrenar con el pipeline. El WofE de caja blanca se ejecuta en vivo mientras tanto.' : 'Train it with the pipeline. The white-box WofE runs live meanwhile.'}</p>
            </div>
          ) : (
            <>
              <MapView nx={cube.nx} ny={cube.ny} field={oodField.field} range={[0, Math.max(oodThr ?? 0, oodField.max)]} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel="anomaly" />
              <div className="pf-kpis">
                <Kpi label={es ? 'umbral (p95 in-dist)' : 'threshold (in-dist p95)'} value={oodThr != null ? oodThr.toFixed(2) : 'n/a'} />
                <Kpi label={es ? '% celdas fuera de envolvente' : '% cells off-envelope'} value={pct(oodField.offFrac)} />
              </div>
              <p className="pf-note">{es
                ? 'El guardia OOD marca dónde la evidencia se aleja de la distribución de entrenamiento, donde el clasificador extrapola y su score es menos confiable.'
                : 'The OOD guard flags where the evidence drifts from the training distribution, where the classifier extrapolates and its score is less trustworthy.'}</p>
            </>
          )}
        </div>
      ),
    },
    {
      id: 'puconformal', label: es ? 'PU-Conformal' : 'PU-Conformal',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'La propuesta más allá del SOTA: un score nnPU (depósitos = positivos, todo lo demás = no etiquetado, no negativo) calibrado con predicción conforme espacialmente bloqueada. El mapa es el posterior corregido por sesgo; abajo, el conjunto prospectivo con cobertura garantizada y la sensibilidad al prior de clase pi.'
            : 'The beyond-SOTA proposal: an nnPU score (deposits = positives, everything else = unlabeled, not negative) calibrated with spatially-blocked conformal prediction. The map is the bias-corrected posterior; below, the coverage-guaranteed prospective set and the sensitivity to the class prior pi.'}</div>
          {!isReal ? (
            <div className="pf-pending">
              <strong>{es ? 'Carril entrenado sobre el cubo real' : 'Lane trained on the real cube'}</strong>
              <p>{es ? 'El modelo PU-Conformal se entrena solo sobre el cubo real de 6 capas (US Midcontinent MVT). Cambiar la Fuente a "Muestra real" para verlo en vivo. El WofE de caja blanca se ejecuta en vivo en modo sintético.' : 'The PU-Conformal model is trained only on the real 6-layer cube (US Midcontinent MVT). Switch Source to "Real sample" to see it live. The white-box WofE runs live in synthetic mode.'}</p>
            </div>
          ) : !puField ? (
            <div className="pf-pending">
              <strong>{es ? 'PU-Conformal: pendiente de entrenamiento' : 'PU-Conformal: pending training'}</strong>
              <p>{es ? 'Ejecutar el carril offline (pipeline/pu_conformal.py -> ONNX + JSON). El WofE se ejecuta en vivo mientras tanto.' : 'Run the offline lane (pipeline/pu_conformal.py -> ONNX + JSON). The white-box WofE runs live meanwhile.'}</p>
            </div>
          ) : (
            <>
              <MapView nx={cube.nx} ny={cube.ny} field={puField} range={puRange} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel={es ? 'P (nnPU)' : 'P (nnPU)'} />
              <div className="pf-controls-row">
                <div>
                  <div className="pf-card-t">{es ? 'Prior de clase pi (sensibilidad)' : 'Class prior pi (sensitivity)'}</div>
                  <div className="pf-chips">
                    {(puConformal?.pi_sensitivity ?? []).map((row, i) => (
                      <button key={row.pi} className={`chip ${piIdx === i ? 'on' : ''}`} onClick={() => setPiIdx(i)}>pi={row.pi.toFixed(3)}</button>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="pf-card-t">{es ? 'Nivel conforme (cobertura nominal)' : 'Conformal level (nominal coverage)'}</div>
                  <div className="pf-chips">
                    {(puConformal?.conformal?.levels ?? []).map((lv, i) => (
                      <button key={lv.alpha} className={`chip ${alphaIdx === i ? 'on' : ''}`} onClick={() => setAlphaIdx(i)}>{pct(lv.nominal, 0)}</button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="pf-kpis">
                <Kpi label={es ? 'AUC CV bloques contiguos (pi)' : 'contiguous-fold CV AUC (pi)'} value={num(piRow?.block_cv_auc)} />
                <Kpi label={es ? 'cobertura empírica' : 'empirical coverage'} value={confLevel ? pct(confLevel.empirical_coverage, 0) : 'n/a'} />
                <Kpi label={es ? 'nominal' : 'nominal'} value={confLevel ? pct(confLevel.nominal, 0) : 'n/a'} />
                <Kpi label={es ? 'tamaño del conjunto (área)' : 'set size (area)'} value={confLevel ? pct(confLevel.set_size_frac, 0) : 'n/a'} />
              </div>
              {puSet && (
                <>
                  <div className="pf-plot-t" style={{ marginTop: 8 }}>{es
                    ? `Conjunto prospectivo con cobertura garantizada al ${pct(confLevel?.nominal ?? 0, 0)}: las celdas dentro del conjunto (claras) contienen un depósito held-out con probabilidad >= nominal.`
                    : `Coverage-guaranteed prospective set at ${pct(confLevel?.nominal ?? 0, 0)}: cells inside the set (bright) contain a held-out deposit with probability >= nominal.`}</div>
                  <MapView nx={cube.nx} ny={cube.ny} field={puSet.field} range={[0, 1]} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel={es ? 'en conjunto' : 'in set'} />
                </>
              )}
              <p className="pf-note">{es
                ? `Resultado honesto: sobre el belt MVT agrupado, PU-Conformal (AUC CV bloques contiguos ${num(puConformal?.benchmark?.find((b) => b.model === 'pu_conformal')?.auc)}) no supera al WofE en las mismas particiones (${num(puConformal?.benchmark?.find((b) => b.model === 'wofe')?.auc)}) en ranking; el null de distancia-a-depósito ya alcanza ${num(puConformal?.negative_controls?.distance_to_deposit_null?.distance_to_deposit_auc)}. La cobertura conforme se cumple, pero solo marcando ~${pct(confLevel?.set_size_frac ?? 0, 0)} del belt: un conjunto casi vacío que reporta honestamente que la geofísica regional no localiza MVT bajo transferencia espacial. El avance es la incertidumbre calibrada y corregida por sesgo, no un AUC mayor.`
                : `Honest result: over the clustered MVT belt, PU-Conformal (contiguous-fold CV AUC ${num(puConformal?.benchmark?.find((b) => b.model === 'pu_conformal')?.auc)}) does not beat WofE on the same folds (${num(puConformal?.benchmark?.find((b) => b.model === 'wofe')?.auc)}) in ranking; the distance-to-deposit null alone reaches ${num(puConformal?.negative_controls?.distance_to_deposit_null?.distance_to_deposit_auc)}. Conformal coverage holds, but only by flagging ~${pct(confLevel?.set_size_frac ?? 0, 0)} of the belt: a near-vacuous set that honestly reports regional geophysics cannot localize MVT under spatial transfer. The advance is calibrated, bias-corrected uncertainty, not a higher AUC.`}</p>
            </>
          )}
        </div>
      ),
    },
  ];

  return <GroupedTabs tabs={tabs} es={es} />;
}

/** ONE row of tabs with hover sub-menus (ADR-0071 rules 4+5). */
function GroupedTabs({ tabs, es }: { tabs: { id: string; label: string; content: React.ReactNode }[]; es: boolean }) {
  const [activeTab, setActiveTab] = useState(tabs[0]?.id ?? 'map');
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (tabs.length && !tabs.some((x) => x.id === activeTab)) setActiveTab(tabs[0].id);
  }, [tabs.length, activeTab]);
  const cur = tabs.find((x) => x.id === activeTab) ?? tabs[0];
  return (
    <div className="pf-tabs">
      <div className="pf-tabrow" role="tablist" aria-label={es ? 'vistas de prospectividad' : 'prospectivity views'}>
        {TAB_GROUPS.filter((g) => tabs.some((x) => g.members.includes(x.id))).map((g) => {
          const mine = tabs.filter((x) => g.members.includes(x.id));
          const activeHere = mine.some((x) => x.id === activeTab);
          const shown = activeHere ? mine.find((x) => x.id === activeTab)! : mine[0];
          const multi = mine.length > 1;
          return (
            <div key={g.id} className="pf-tabwrap"
                 onPointerEnter={() => { if (multi) { if (closeTimer.current) clearTimeout(closeTimer.current); setOpenMenu(g.id); } }}
                 onPointerLeave={() => {
                   if (closeTimer.current) clearTimeout(closeTimer.current);
                   closeTimer.current = setTimeout(() => setOpenMenu((m) => (m === g.id ? null : m)), 240);
                 }}>
              <button role="tab" aria-selected={activeHere} className={`pf-tab ${activeHere ? 'on' : ''}`}
                      onClick={() => {
                        if (!multi) { setActiveTab(mine[0].id); setOpenMenu(null); return; }
                        setOpenMenu(openMenu === g.id ? null : g.id);
                        if (!activeHere) setActiveTab(shown.id);
                      }}>
                {activeHere ? shown.label : (es ? g.es : g.en)}{multi ? <span className="pf-caret">v</span> : null}
              </button>
              {multi && openMenu === g.id && (
                <div className="pf-tabmenu" role="menu">
                  {mine.map((x) => (
                    <button key={x.id} role="menuitem" className={x.id === activeTab ? 'on' : ''}
                            onClick={() => { setActiveTab(x.id); setOpenMenu(null); }}>{x.label}</button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="pf-tabpanel">
        {cur ? <PanelBoundary key={cur.id} lang={es ? 'es' : 'en'}>{cur.content}</PanelBoundary> : null}
      </div>
    </div>
  );
}
