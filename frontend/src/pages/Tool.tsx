import { useEffect, useMemo, useState } from 'react';
import { Tabs, useShellLang } from '@fasl-work/caos-app-shell';
import {
  analyzeCase, bestWeights, CASES, caseById, ciCheck, depositSet, fitLR, getLayer, makeSyntheticArea,
  maskCells, posterior, predictLR, type MPMCase,
} from '../mpm/index.ts';
import { loadLearned, type LearnedFile } from '../lib/artifacts.ts';
import { MPM_FEATURES } from '../lib/learned.ts';
import { runClassifier, runOod } from '../lib/ort.ts';
import { MapView } from '../viz/MapView.tsx';
import { CurveChart } from '../viz/CurveChart.tsx';

const CATS = [
  'deposit-type terrane (the geological setting)',
  'data / validation regime (evidence richness)',
  'control (oracle / negative control)',
];
type Method = 'wofe' | 'logistic';
const pct = (v: number, n = 1) => `${(v * 100).toFixed(n)}%`;

// resample a capture curve (areaFrac, captureFrac) onto a common x grid for the chart
function resample(areaFrac: number[], captureFrac: number[], xs: number[]): number[] {
  return xs.map((x) => {
    for (let i = 0; i < areaFrac.length; i++) if (areaFrac[i] >= x) return captureFrac[i];
    return captureFrac[captureFrac.length - 1] ?? 0;
  });
}

export default function Tool() {
  const es = useShellLang() === 'es';
  const [caseId, setCaseId] = useState('K-PORPHYRY');
  const [layerOff, setLayerOff] = useState<Record<string, boolean>>({});
  const [method, setMethod] = useState<Method>('wofe');
  const [learned, setLearned] = useState<LearnedFile | null>(null);
  const [learnedField, setLearnedField] = useState<Float64Array | null>(null);
  const [oodField, setOodField] = useState<{ field: Float64Array; max: number; offFrac: number } | null>(null);
  const oodThr = (learned?.ood as { threshold?: number } | undefined)?.threshold ?? null;

  const theCase = useMemo<MPMCase>(() => caseById(caseId), [caseId]);
  const cube = useMemo(() => makeSyntheticArea(theCase.spec).cube, [theCase]);
  const activeIds = useMemo(() => theCase.layerIds.filter((id) => !layerOff[id]), [theCase, layerOff]);

  const best = useMemo(() => activeIds.map((id) => ({ id, ...bestWeights(cube, id) })), [cube, activeIds]);
  const pats = useMemo(() => best.map((b) => b.pattern), [best]);
  const post = useMemo(() => posterior(cube, pats, best.map((b) => b.weights)), [cube, pats, best]);
  const ci = useMemo(() => ciCheck(cube, pats, post), [cube, pats, post]);
  const analysis = useMemo(() => analyzeCase(theCase.spec, activeIds.length ? activeIds : theCase.layerIds), [theCase, activeIds]);

  // logistic-regression field (the CI-free comparison)
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

  const map = useMemo(() => {
    if (method === 'logistic') return { field: lrField, range: [0, 1] as [number, number], label: 'P(LR)' };
    return { field: post.prob, range: [0, 1] as [number, number], label: 'P' };
  }, [method, post, lrField]);

  useEffect(() => { setLayerOff({}); }, [caseId]);
  useEffect(() => { loadLearned().then(setLearned).catch(() => setLearned(null)); }, []);

  // the two learned models, run LIVE over the cube's cells (onnxruntime-web, batched): the mpm-classifier paints an
  // alternative prospectivity map (the What-if tool); the geology OOD-AE paints the per-cell anomaly map (the Anomaly
  // guard). Graceful, if the ONNX are absent the fields stay null and the tools show the honest pending state.
  useEffect(() => {
    let cancel = false;
    setLearnedField(null);
    setOodField(null);
    const cells = maskCells(cube);
    const F = MPM_FEATURES.length;
    const rows = new Float32Array(cells.length * F);
    for (let r = 0; r < cells.length; r++) {
      for (let j = 0; j < F; j++) {
        const v = getLayer(cube, MPM_FEATURES[j]).values[cells[r]];
        rows[r * F + j] = Number.isNaN(v) ? 0 : v;
      }
    }
    Promise.all([runClassifier(rows, cells.length), runOod(rows, cells.length)]).then(([pc, mse]) => {
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
  }, [cube, activeIds, learned]);

  const Kpi = ({ label, value }: { label: string; value: string }) => (
    <div className="pf-kpi"><div className="pf-kpi-v">{value}</div><div className="pf-kpi-l">{label}</div></div>
  );

  const xs = useMemo(() => Array.from({ length: 51 }, (_, i) => i / 50), []);
  const captureSeries = useMemo(() => [
    { label: es ? 'prediction (spatial CV)' : 'prediction (spatial CV)', y: resample(analysis.capture.prediction.areaFrac, analysis.capture.prediction.captureFrac, xs), color: undefined },
    { label: es ? 'fitting (success)' : 'fitting (success)', y: resample(analysis.capture.success.areaFrac, analysis.capture.success.captureFrac, xs), dash: [4, 4] },
  ], [analysis, xs, es]);

  // a small ROC from the posterior field vs the deposit labels
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

  const tabs = [
    {
      id: 'map', label: es ? 'Mapa' : 'Map',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'Mapa posterior de prospectividad P(depósito | evidencia) por celda (colormap viridis). Los depósitos conocidos como marcadores; pasa el cursor para leer el valor.'
            : 'Posterior prospectivity map P(deposit | evidence) per cell (viridis colormap). Known deposits as markers; hover to read the value.'}</div>
          <MapView nx={cube.nx} ny={cube.ny} field={map.field} range={map.range} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel={map.label} />
          <div className="pf-kpis">
            <Kpi label={es ? 'ROC AUC' : 'ROC AUC'} value={analysis.rocAuc.toFixed(3)} />
            <Kpi label="capture@10%" value={pct(analysis.capture.prediction.captureAt10)} />
            <Kpi label="CI ratio" value={ci.ciRatio.toFixed(2)} />
            <Kpi label={es ? 'depósitos' : 'deposits'} value={`${cube.depositIdx.length}`} />
          </div>
        </div>
      ),
    },
    {
      id: 'weights', label: es ? 'Pesos' : 'Layer weights',
      content: (
        <div className="pf-vizstack">
          <table className="cmp-table">
            <thead><tr><th>{es ? 'capa' : 'layer'}</th><th>t*</th><th>W⁺</th><th>W⁻</th><th>{es ? 'contraste' : 'contrast'}</th><th>studC</th></tr></thead>
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
            ? 'W⁺/W⁻ son los log-likelihood ratios de presencia/ausencia del patrón dado un depósito. |studC| < 1.96 (gris) = no significativo. t* es el umbral de contraste máximo.'
            : 'W⁺/W⁻ are the log-likelihood ratios of the pattern present/absent given a deposit. |studC| < 1.96 (greyed) = not significant. t* is the maximizing-contrast threshold.'}</p>
        </div>
      ),
    },
    {
      id: 'rates', label: es ? 'Tasas de captura' : 'Capture rates',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? '% de depósitos capturados vs % del área (ranking por prospectividad). prediction = held-out espacial (W⁺/W⁻ se reajustan por fold; el umbral t* se elige con todos los depósitos, un sesgo optimista pequeño); fitting = sobre los datos de entrenamiento (optimista).'
            : '% of deposits captured vs % of area (ranked by prospectivity). prediction = spatial held-out (W⁺/W⁻ refit per fold; the binarization threshold t* is chosen on all deposits, a small optimism); fitting = on the training data (optimistic).'}</div>
          <CurveChart x={xs} series={captureSeries} xLabel={es ? '% área' : '% area'} yLabel={es ? '% depósitos' : '% deposits'} diagonal />
          <p className="pf-note">{es
            ? `El top 10% del área captura ${pct(analysis.capture.prediction.captureAt10)} de los depósitos held-out (spatial CV). El gap fitting−prediction mide el sobreajuste.`
            : `The top 10% of the area captures ${pct(analysis.capture.prediction.captureAt10)} of the held-out deposits (spatial CV). The fitting−prediction gap measures over-fit.`}</p>
        </div>
      ),
    },
    {
      id: 'roc', label: 'ROC',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es ? 'Curva ROC del posterior vs las etiquetas de depósito. Cuidado: con datos presence-only el AUC es secundario (las "ausencias" pueden tener depósitos no descubiertos).' : 'ROC of the posterior vs the deposit labels. Caveat: with presence-only data AUC is secondary (the "absences" may host undiscovered deposits).'}</div>
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
            ? 'El test omnibus de Agterberg-Cheng: bajo independencia condicional Σ posterior ≈ N(D). T > N(D) ⇒ el posterior está inflado por capas correlacionadas.'
            : 'The Agterberg-Cheng omnibus test: under conditional independence Σ posterior ≈ N(D). T > N(D) ⇒ the posterior is inflated by correlated layers.'}</div>
          <div className="pf-kpis">
            <Kpi label="T (Σ posterior)" value={ci.T.toFixed(1)} />
            <Kpi label="N(D)" value={`${ci.nD}`} />
            <Kpi label="CI ratio" value={ci.ciRatio.toFixed(2)} />
            <Kpi label="z" value={ci.z.toFixed(1)} />
          </div>
          <table className="cmp-table">
            <thead><tr><th>{es ? 'par de capas' : 'layer pair'}</th><th>χ²</th><th>Cramér V</th></tr></thead>
            <tbody>
              {ci.pairwise.map((p) => (
                <tr key={`${p.a}-${p.b}`}><td>{p.a} · {p.b}</td><td>{p.chi2.toFixed(2)}</td><td>{p.cramersV.toFixed(2)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="pf-note">{ci.ciRatio < 0.85
            ? (es ? `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 ⇒ violación: el posterior WofE está sobre-estimado. Usa el ranking relativo, o la regresión logística como alternativa sin supuesto de CI (AUC ${analysis.lr.rocAuc.toFixed(3)}, una métrica de ranking; su calibración aún no se muestra en la app).` : `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 ⇒ violation: the WofE posterior is over-estimated. Use the relative ranking, or logistic regression as the CI-free alternative (AUC ${analysis.lr.rocAuc.toFixed(3)}, a ranking metric; its calibration is not yet read out in-app).`)
            : (es ? `CI ratio ${ci.ciRatio.toFixed(2)} ≈ 1 ⇒ consistente con independencia condicional; el posterior WofE es razonable.` : `CI ratio ${ci.ciRatio.toFixed(2)} ≈ 1 ⇒ consistent with conditional independence; the WofE posterior is reasonable.`)}</p>
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
            : 'The default reported number is always the spatial one (honest). The random one is shown only to expose the inflation.'}</p>
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
              <tr><td>{es ? 'aprendido (MLP)' : 'learned (MLP)'}</td><td>{learned?.classifier?.spatial_cv ? String((learned.classifier.spatial_cv as Record<string, number>).mlp_roc_auc ?? ', ') : (es ? 'pendiente' : 'pending')}</td><td>{es ? 'mide vs WofE en el mismo holdout espacial' : 'measured vs WofE on the same spatial holdout'}</td></tr>
            </tbody>
          </table>
          <p className="pf-note">{es
            ? 'Cuando se cumple la independencia condicional, la logística ≈ WofE. Cuando se viola (caso C-CIVIOLATE), la logística ajusta las capas EN CONJUNTO y se espera que no doble-cuente, la app aún no muestra un readout de calibración que lo demuestre (el test omnibus corre sobre el posterior WofE).'
            : 'When conditional independence holds, logistic ≈ WofE. When it is violated (case C-CIVIOLATE), logistic fits the layers JOINTLY and is expected not to double-count, the app does not yet show a calibration readout demonstrating it (the omnibus test runs on the WofE posterior).'}</p>
        </div>
      ),
    },
    {
      id: 'whatif', label: es ? 'What-if (MLP)' : 'What-if (MLP)',
      content: (
        <div className="pf-vizstack">
          <div className="pf-plot-t">{es
            ? 'El clasificador MLP aprendido pinta un mapa de prospectividad ALTERNATIVO, compáralo EN VIVO con el WofE de caja blanca, medido en el MISMO holdout espacial.'
            : 'The learned MLP classifier paints an ALTERNATIVE prospectivity map, compare it LIVE to the white-box WofE, measured on the SAME spatial holdout.'}</div>
          {!learnedField ? (
            <div className="pf-pending">
              <strong>{es ? 'Clasificador: pendiente de entrenamiento' : 'Classifier: pending training'}</strong>
              <p>{es ? 'Corre `python -m pmlab.pipeline all --retrain` para entrenar el clasificador MPM (torch → ONNX). El WofE de caja blanca corre EN VIVO mientras tanto.' : 'Run `python -m pmlab.pipeline all --retrain` to train the mpm-classifier (torch → ONNX). The white-box WofE runs LIVE meanwhile.'}</p>
            </div>
          ) : (
            <>
              <MapView nx={cube.nx} ny={cube.ny} field={learnedField} range={[0, 1]} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel="P(MLP)" />
              <div className="pf-kpis">
                <Kpi label={es ? 'MLP spatial-CV AUC' : 'MLP spatial-CV AUC'} value={String(clf?.spatial_cv?.mlp_roc_auc ?? ', ')} />
                <Kpi label="WofE spatial-CV AUC" value={String(clf?.spatial_cv?.wofe_roc_auc ?? ', ')} />
                <Kpi label={es ? 'CV aleatorio (inflado)' : 'random-CV (inflated)'} value={String(clf?.random_cv?.mlp_roc_auc ?? ', ')} />
                <Kpi label={es ? 'gap de inflación' : 'inflation gap'} value={String(clf?.inflation_gap ?? ', ')} />
              </div>
              <p className="pf-note">{es
                ? 'El WofE de caja blanca es la autoridad interpretable; el MLP gana su lugar por capturar interacciones multi-capa que la forma CI de WofE omite, medido en el mismo holdout espacial, no por una victoria fabricada.'
                : 'The white-box WofE is the interpretable authority; the MLP earns its place by capturing multi-layer interactions the CI form of WofE misses, measured on the same spatial holdout, not a fabricated win.'}</p>
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
            ? 'El autoencoder de geología pinta el mapa de anomalía por celda (error de reconstrucción): dónde la evidencia está FUERA del envolvente entrenado, "no confíes en el clasificador bajo cobertura".'
            : 'The geology autoencoder paints the per-cell anomaly map (reconstruction error): where the evidence is OUTSIDE the trained envelope, "do not trust the classifier under cover".'}</div>
          {!oodField ? (
            <div className="pf-pending">
              <strong>{es ? 'Autoencoder OOD: pendiente de entrenamiento' : 'OOD autoencoder: pending training'}</strong>
              <p>{es ? 'Entrénalo con `--retrain`. El WofE de caja blanca corre en vivo mientras tanto.' : 'Train it with `--retrain`. The white-box WofE runs live meanwhile.'}</p>
            </div>
          ) : (
            <>
              <MapView nx={cube.nx} ny={cube.ny} field={oodField.field} range={[0, Math.max(oodThr ?? 0, oodField.max)]} deposits={cube.depositIdx} lang={es ? 'es' : 'en'} valueLabel="anomaly" />
              <div className="pf-kpis">
                <Kpi label={es ? 'umbral (p95 in-dist)' : 'threshold (in-dist p95)'} value={oodThr != null ? oodThr.toFixed(2) : ', '} />
                <Kpi label={es ? '% celdas fuera de envolvente' : '% cells off-envelope'} value={pct(oodField.offFrac)} />
                <Kpi label="OOD AUC" value={learned ? learned.ood.auc.toFixed(3) : ', '} />
              </div>
              <p className="pf-note">{es
                ? 'En estos casos sintéticos la geología es in-envelope (pocas celdas sobre el umbral). El guardia importa en dato real bajo cobertura, donde la evidencia se aleja de la distribución de entrenamiento.'
                : 'In these synthetic cases the geology is in-envelope (few cells above the threshold). The guard matters on real data under cover, where the evidence drifts from the training distribution.'}</p>
            </>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="page-body pf-layout">
      <aside className="pf-side">
        <div className="pf-card">
          <div className="pf-card-t">{es ? 'Caso' : 'Case'}</div>
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
        <div className="pf-card">
          <div className="pf-card-t">{es ? 'Capas (en vivo)' : 'Layers (live)'}</div>
          <div className="pf-chips">
            {theCase.layerIds.map((id) => (
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
      </aside>
      <main className="pf-main">
        <Tabs tabs={tabs} ariaLabel={es ? 'vistas de prospectividad' : 'prospectivity views'} />
      </main>
    </div>
  );
}
