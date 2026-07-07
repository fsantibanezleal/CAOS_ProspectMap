import { useEffect, useMemo, useState } from 'react';
import { Tabs, useShellLang } from '@fasl-work/caos-app-shell';
import {
  analyzeCube, bestWeights, CASES, caseById, ciCheck, cubeFromFile, depositSet, fitLR, getLayer, layerRange,
  makeSyntheticArea, maskCells, posterior, predictLR, REAL_CASES, realCaseById, rocAuc,
  type Cube, type MPMCase, type RealCase, type RealCubeFile,
} from '../mpm/index.ts';
import { loadLearned, loadLearnedReal, loadRealCube, type LearnedFile } from '../lib/artifacts.ts';
import { runClassifier, runOod, type Lane } from '../lib/ort.ts';
import { MapView } from '../viz/MapView.tsx';
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
// vs the OBSERVED deposit frequency. A well-calibrated model sits on the diagonal. Brier score + expected calibration
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

  // load the baked real cube when the Real lane is picked (arrays only; the WofE engine runs LIVE on it)
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
        <div className="pf-card">
          <div className="pf-card-t">{es ? 'Fuente' : 'Source'}</div>
          <div className="pf-chips">
            <button className={`chip ${!isReal ? 'on' : ''}`} onClick={() => setSource('synthetic')}>{es ? 'Sintético' : 'Synthetic'}</button>
            <button className={`chip ${isReal ? 'on' : ''}`} onClick={() => setSource('real')}>{es ? 'Muestra real' : 'Real sample'}</button>
          </div>
          <div className="pf-cap pf-muted">{isReal
            ? (es ? 'dataset abierto real; los knobs sintéticos se desactivan, eliges el dato y todas las herramientas corren sobre él' : 'a real open dataset; the synthetic knobs disable, you pick the datum and every tool runs on it')
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
                  <span><b className="pf-recomp">RECOMPUTED</b> {es ? 'nuestro posterior WofE en el navegador, NO el modelo H3+gradient-boosting publicado' : 'our browser WofE posterior, NOT the published H3 + gradient-boosting model'}</span>
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
            <div className="pf-cap pf-muted">{es ? 'activa/desactiva capas de evidencia, el posterior se recalcula en vivo' : 'toggle evidence layers, the posterior recomputes live'}</div>
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
            learned={activeLearned} isReal={isReal} es={es}
          />
        )}
      </main>
    </div>
  );
}

function CubeViews({ cube, activeIds, method, lane, learned, isReal, es }: {
  cube: Cube; activeIds: string[]; method: Method; lane: Lane; learned: LearnedFile | null;
  isReal: boolean; es: boolean;
}) {
  const [learnedField, setLearnedField] = useState<Float64Array | null>(null);
  const [oodField, setOodField] = useState<{ field: Float64Array; max: number; offFrac: number } | null>(null);
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

  // the two learned models, run LIVE over the cube's cells on the lane's feature space + model file (never the
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

  const clf = learned?.classifier as {
    spatial_cv?: Record<string, number>; random_cv?: Record<string, number>; inflation_gap?: number;
  } | undefined;

  const recomputeNote = isReal
    ? (es ? 'El posterior es NUESTRA recomputación WofE en el navegador sobre una sub-región rasterizada, NO el modelo H3+gradient-boosting publicado (Lawley 2022).' : 'The posterior is OUR browser WofE recomputation over a rasterized sub-region, NOT the published Lawley 2022 H3 + gradient-boosting model.')
    : '';

  const tabs = [
    {
      id: 'map', label: es ? 'Mapa' : 'Map',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'Mapa posterior de prospectividad P(depósito | evidencia) por celda (colormap viridis). Los depósitos conocidos como marcadores; pasa el cursor para leer el valor.'
            : 'Posterior prospectivity map P(deposit | evidence) per cell (viridis colormap). Known deposits as markers; hover to read the value.'}
            {isReal && <> {recomputeNote}</>}</div>
          <MapView nx={cube.nx} ny={cube.ny} field={map.field} range={map.range} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel={map.label} />
          <div className="pf-kpis">
            <Kpi label="ROC AUC" value={analysis.rocAuc.toFixed(3)} />
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
            <Kpi label="WofE AUC" value={analysis.rocAuc.toFixed(3)} />
            <Kpi label="LR AUC" value={analysis.lr.rocAuc.toFixed(3)} />
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
            ? (es ? `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 implica violación: el posterior WofE está sobre-estimado. Usa el ranking relativo, o la regresión logística como alternativa sin supuesto de CI (AUC ${analysis.lr.rocAuc.toFixed(3)}).` : `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 means violation: the WofE posterior is over-estimated. Use the relative ranking, or logistic regression as the CI-free alternative (AUC ${analysis.lr.rocAuc.toFixed(3)}).`)
            : (es ? `CI ratio ${ci.ciRatio.toFixed(2)} ~ 1 implica consistencia con independencia condicional; el posterior WofE es razonable.` : `CI ratio ${ci.ciRatio.toFixed(2)} ~ 1 means consistent with conditional independence; the WofE posterior is reasonable.`)}
            {isReal && <> {es ? 'ESPERADO en dato real: mag/grav/lab son físicamente correlacionadas, así que el test se dispara. No es una trampa plantada, es la realidad geofísica; por eso la logística es la ruta honesta.' : 'EXPECTED on real data: mag/grav/lab are physically correlated, so the test fires. This is not a planted trap, it is the geophysical reality; hence logistic regression is the honest route.'}</>}</p>
        </div>
      ),
    },
    {
      id: 'cv', label: es ? 'Validación CV' : 'CV inflation',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'La pantalla más importante: el MISMO modelo bajo CV aleatorio vs CV espacial-por-bloques. El gap es la inflación por autocorrelación espacial (el AUC aleatorio miente).'
            : 'The most important screen: the SAME model under random CV vs spatial-block CV. The gap is the inflation from spatial autocorrelation (the random AUC lies).'}</div>
          <div className="pf-kpis">
            <Kpi label={es ? 'CV aleatorio AUC' : 'random-CV AUC'} value={analysis.cv.randomAuc.toFixed(3)} />
            <Kpi label={es ? 'CV espacial AUC' : 'spatial-CV AUC'} value={analysis.cv.spatialAuc.toFixed(3)} />
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
            <thead><tr><th>{es ? 'método' : 'method'}</th><th>ROC AUC</th><th>{es ? 'nota' : 'note'}</th></tr></thead>
            <tbody>
              <tr><td><b>WofE</b></td><td>{analysis.rocAuc.toFixed(3)}</td><td>{es ? 'caja blanca, la autoridad' : 'white-box, the authority'}</td></tr>
              <tr><td>{es ? 'logística' : 'logistic'}</td><td>{analysis.lr.rocAuc.toFixed(3)}</td><td>{es ? 'sin supuesto de independencia condicional' : 'no conditional-independence assumption'}</td></tr>
              <tr><td>{es ? 'fuzzy / index-overlay' : 'fuzzy / index-overlay'}</td><td>{fuzzyAuc.toFixed(3)}</td><td>{es ? 'operador gamma, sin ajuste ni supuesto de CI' : 'gamma operator, no fitting, no CI assumption'}</td></tr>
              <tr><td>{es ? 'aprendido (MLP)' : 'learned (MLP)'}</td><td>{num(clf?.spatial_cv?.mlp_roc_auc)}</td><td>{es ? 'CV espacial sobre el set etiquetado' : 'spatial CV on the labelled set'}</td></tr>
            </tbody>
          </table>
          <p className="pf-note">{es
            ? 'Cuando se cumple la independencia condicional, la logística ~ WofE. Cuando se viola, la logística ajusta las capas EN CONJUNTO y no doble-cuenta. El overlay fuzzy es un método MPM clásico sin ajuste.'
            : 'When conditional independence holds, logistic ~ WofE. When it is violated, logistic fits the layers JOINTLY and does not double-count. The fuzzy overlay is a classic fit-free MPM method.'}</p>
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
            <Kpi label={es ? 'fuzzy ROC AUC' : 'fuzzy ROC AUC'} value={fuzzyAuc.toFixed(3)} />
            <Kpi label="WofE AUC" value={analysis.rocAuc.toFixed(3)} />
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
            ? 'Diagrama de fiabilidad: en deciles del score, la probabilidad media predicha vs la frecuencia de depósitos OBSERVADA. Un modelo bien calibrado se sitúa sobre la diagonal. Brier + ECE lo resumen. (El caso C-SATURATE señalaba que faltaba este readout.)'
            : 'Reliability diagram: over deciles of the score, the mean predicted probability vs the OBSERVED deposit frequency. A well-calibrated model sits on the diagonal. Brier + ECE summarize it. (The C-SATURATE case flagged this readout was missing.)'}</div>
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
            ? 'El presence-only sesga la calibración absoluta (las "ausencias" incluyen depósitos no descubiertos), así que léela como fiabilidad RELATIVA entre bins más que como probabilidades absolutas.'
            : 'Presence-only biases the absolute calibration (the "absences" include undiscovered deposits), so read it as RELATIVE reliability across bins rather than absolute probabilities.'}</p>
        </div>
      ),
    },
    {
      id: 'whatif', label: es ? 'What-if (MLP)' : 'What-if (MLP)',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'El clasificador MLP aprendido pinta un mapa de prospectividad ALTERNATIVO; compáralo EN VIVO con el WofE de caja blanca.'
            : 'The learned MLP classifier paints an ALTERNATIVE prospectivity map; compare it LIVE to the white-box WofE.'}
            {isReal && <> {es ? 'Entrenado sobre el cubo REAL de 6 capas (no el modelo sintético de 4 capas).' : 'Trained on the REAL 6-layer cube (not the synthetic 4-layer model).'}</>}</div>
          {!learnedField ? (
            <div className="pf-pending">
              <strong>{es ? 'Clasificador: pendiente de entrenamiento' : 'Classifier: pending training'}</strong>
              <p>{es ? 'Corre el pipeline de aprendidos (torch -> ONNX). El WofE de caja blanca corre EN VIVO mientras tanto.' : 'Run the learned pipeline (torch -> ONNX). The white-box WofE runs LIVE meanwhile.'}</p>
            </div>
          ) : (
            <>
              <MapView nx={cube.nx} ny={cube.ny} field={learnedField} range={[0, 1]} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel="P(MLP)" />
              <div className="pf-kpis">
                <Kpi label={es ? 'MLP spatial-CV AUC' : 'MLP spatial-CV AUC'} value={num(clf?.spatial_cv?.mlp_roc_auc)} />
                <Kpi label="WofE AUC" value={num(clf?.spatial_cv?.wofe_roc_auc)} />
                <Kpi label={es ? 'CV aleatorio (inflado)' : 'random-CV (inflated)'} value={num(clf?.random_cv?.mlp_roc_auc)} />
                <Kpi label={es ? 'gap de inflación' : 'inflation gap'} value={num(clf?.inflation_gap)} />
              </div>
              <p className="pf-note">{es
                ? 'El WofE de caja blanca es la autoridad interpretable; el MLP se valida por CV espacial y se reporta junto al CV aleatorio (el gap de inflación). No hay victoria fabricada.'
                : 'The white-box WofE is the interpretable authority; the MLP is validated by spatial CV and reported beside random CV (the inflation gap). No fabricated win.'}
                {isReal && <> {es ? 'Aviso honesto: el AUC del MLP se mide sobre el set etiquetado con negativos muestreados (buffer), no idéntico al eval de rejilla completa del WofE.' : 'Honest caveat: the MLP AUC is measured on the labelled set with buffered sampled negatives, not identical to the WofE full-grid eval.'}</>}</p>
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
            ? 'El autoencoder pinta el mapa de anomalía por celda (error de reconstrucción): dónde la evidencia está FUERA del envolvente entrenado, "no confíes en el clasificador bajo cobertura".'
            : 'The autoencoder paints the per-cell anomaly map (reconstruction error): where the evidence is OUTSIDE the trained envelope, "do not trust the classifier under cover".'}</div>
          {!oodField ? (
            <div className="pf-pending">
              <strong>{es ? 'Autoencoder OOD: pendiente de entrenamiento' : 'OOD autoencoder: pending training'}</strong>
              <p>{es ? 'Entrénalo con el pipeline. El WofE de caja blanca corre en vivo mientras tanto.' : 'Train it with the pipeline. The white-box WofE runs live meanwhile.'}</p>
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
  ];

  return <Tabs tabs={tabs} ariaLabel={es ? 'vistas de prospectividad' : 'prospectivity views'} />;
}
