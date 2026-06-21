import { useEffect, useMemo, useState } from 'react';
import { Tabs, useShellLang } from '@fasl-work/caos-app-shell';
import {
  analyzeCase, bestWeights, CASES, caseById, ciCheck, depositSet, fitLR, makeSyntheticArea,
  maskCells, posterior, predictLR, type MPMCase,
} from '../mpm/index.ts';
import { loadLearned, loadManifest, type LearnedFile } from '../lib/artifacts.ts';
import type { CaseManifest } from '../lib/contract.types.ts';
import { MapView } from '../viz/MapView.tsx';
import { CurveChart } from '../viz/CurveChart.tsx';

const CATS = [
  'deposit-type terrane (the geological setting)',
  'data / validation regime (evidence richness)',
  'control (oracle / negative control)',
];
type Method = 'wofe' | 'logistic' | 'learned' | 'ood';
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
  const [manifest, setManifest] = useState<CaseManifest | null>(null);
  const [learned, setLearned] = useState<LearnedFile | null>(null);

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
  useEffect(() => { loadManifest(caseId).then(setManifest).catch(() => setManifest(null)); }, [caseId]);
  useEffect(() => { loadLearned().then(setLearned).catch(() => setLearned(null)); }, []);

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
            ? '% de depósitos capturados vs % del área (ranking por prospectividad). prediction = held-out espacial (honesto); fitting = sobre los datos de entrenamiento (optimista).'
            : '% of deposits captured vs % of area (ranked by prospectivity). prediction = spatial held-out (honest); fitting = on the training data (optimistic).'}</div>
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
            ? (es ? `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 ⇒ violación: el posterior WofE está sobre-estimado. Usa el ranking relativo o la regresión logística (AUC ${analysis.lr.rocAuc.toFixed(3)}).` : `CI ratio ${ci.ciRatio.toFixed(2)} < 0.85 ⇒ violation: the WofE posterior is over-estimated. Use the relative ranking or logistic regression (AUC ${analysis.lr.rocAuc.toFixed(3)}).`)
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
              <tr><td>{es ? 'aprendido (MLP)' : 'learned (MLP)'}</td><td>{learned?.classifier?.spatial_cv ? String((learned.classifier.spatial_cv as Record<string, number>).mlp_roc_auc ?? '—') : (es ? 'pendiente' : 'pending')}</td><td>{es ? 'mide vs WofE en el mismo holdout espacial' : 'measured vs WofE on the same spatial holdout'}</td></tr>
            </tbody>
          </table>
          <p className="pf-note">{es
            ? 'Cuando se cumple la independencia condicional, la logística ≈ WofE. Cuando se viola (caso C-CIVIOLATE), la logística no sobre-estima — esa es la razón de preferirla.'
            : 'When conditional independence holds, logistic ≈ WofE. When it is violated (case C-CIVIOLATE), logistic does not over-estimate — that is the reason to prefer it.'}</p>
        </div>
      ),
    },
    {
      id: 'learned', label: es ? 'Modelos aprendidos' : 'Learned models',
      content: (
        <div className="pf-vizstack">
          {learned ? (
            <>
              <table className="cmp-table">
                <thead><tr><th>{es ? 'modelo' : 'model'}</th><th>{es ? 'métrica (spatial holdout)' : 'metric (spatial holdout)'}</th><th>{es ? 'valor' : 'value'}</th></tr></thead>
                <tbody>
                  <tr><td>{es ? 'clasificador MPM' : 'mpm-classifier'}</td><td>ROC AUC</td><td><b>{String((learned.classifier?.spatial_cv as Record<string, number>)?.mlp_roc_auc ?? '—')}</b></td></tr>
                  <tr><td>{es ? 'OOD de geología' : 'geology-ood'}</td><td>AUC</td><td><b>{learned.ood.auc.toFixed(3)}</b></td></tr>
                </tbody>
              </table>
              <p className="pf-cap">{learned.honesty}</p>
            </>
          ) : (
            <div className="pf-pending">
              <strong>{es ? 'Modelos aprendidos: pendientes de entrenamiento' : 'Learned models: pending training'}</strong>
              <p>{es ? 'Corre `python -m pmlab.pipeline all --retrain` para entrenar el clasificador MPM + el OOD-AE de geología (torch → ONNX). El App usa el WofE EXACTO de caja blanca EN VIVO mientras tanto.' : 'Run `python -m pmlab.pipeline all --retrain` to train the mpm-classifier + the geology OOD-AE (torch → ONNX). The App uses the EXACT white-box WofE LIVE meanwhile.'}</p>
            </div>
          )}
        </div>
      ),
    },
    {
      id: 'contract', label: es ? 'Contrato · gate' : 'Contract · gate',
      content: (
        <div className="pf-vizstack">
          {manifest ? (
            <>
              <div className="pf-kpis">
                <Kpi label="lane" value={manifest.lane} />
                <Kpi label="runtimes" value={manifest.gate.runtimes.join(', ')} />
                <Kpi label={es ? 'bytes traza' : 'trace bytes'} value={`${manifest.gate.trace_bytes}`} />
              </div>
              {manifest.flags.length > 0 && <p className="pf-note">flags: {JSON.stringify(manifest.flags)}</p>}
              <p className="pf-note">{manifest.honesty}</p>
            </>
          ) : <p className="pf-note">{es ? 'cargando manifiesto…' : 'loading manifest…'}</p>}
        </div>
      ),
    },
    {
      id: 'raw', label: es ? 'Traza' : 'Trace',
      content: (
        <pre className="codeblock" style={{ maxHeight: 360 }}>{JSON.stringify({
          case: theCase.id, activeLayers: activeIds, nDeposits: cube.depositIdx.length,
          rocAuc: analysis.rocAuc, captureAt10: analysis.capture.prediction.captureAt10,
          ci: { T: ci.T, nD: ci.nD, ciRatio: ci.ciRatio, z: ci.z },
          cv: analysis.cv, lr: analysis.lr,
        }, null, 2)}</pre>
      ),
    },
  ];

  return (
    <div className="pf-layout">
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
          <div className="pf-cap pf-muted">{es ? 'activa/desactiva capas de evidencia — el posterior se recalcula en vivo' : 'toggle evidence layers — the posterior recomputes live'}</div>
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
