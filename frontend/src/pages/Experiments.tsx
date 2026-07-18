import { useEffect, useState } from 'react';
import { Callout, useShellLang } from '@fasl-work/caos-app-shell';
import { loadCaseResults, loadPuConformal, type PuConformalFile } from '../lib/artifacts.ts';

interface Row {
  id: string; name: string; category: string; realOrSynthetic: string; rocAuc: number;
  capPred: number; ciRatio: number; z: number; inflation: number; lrAuc: number; nDeposits: number;
}

export default function Experiments() {
  const es = useShellLang() === 'es';
  const [rows, setRows] = useState<Row[] | null>(null);
  const [pu, setPu] = useState<PuConformalFile | null>(null);

  useEffect(() => {
    loadPuConformal().then(setPu).catch(() => setPu(null));
    loadCaseResults().then((cr) => {
      const out: Row[] = Object.entries(cr.cases).map(([id, raw]) => {
        const c = raw as Record<string, unknown>;
        const cap = (c.capture as Record<string, Record<string, number>>) ?? {};
        const ci = (c.ci as Record<string, number>) ?? {};
        const cv = (c.cv as Record<string, number>) ?? {};
        const lr = (c.lr as Record<string, number>) ?? {};
        return {
          id, name: String(c.name), category: String(c.category), realOrSynthetic: String(c.realOrSynthetic),
          rocAuc: Number(c.rocAuc), capPred: cap.prediction?.captureAt10 ?? 0, ciRatio: ci.ciRatio ?? 0, z: ci.z ?? 0,
          inflation: cv.inflationGap ?? 0, lrAuc: lr.rocAuc ?? 0, nDeposits: Number(c.nDeposits),
        };
      });
      setRows(out);
    }).catch(() => setRows([]));
  }, []);

  return (
    <article className="page-body prose">
      <h1>{es ? 'Experimentos' : 'Experiments'}</h1>
      <p className="lede">{es
        ? 'Los 10 casos sintéticos, precalculados por el motor WofE. Cada fila lleva su ROC AUC, capture@10% (spatial CV), el CI ratio (+ z del omnibus), el gap de inflación random−spatial y el AUC de la regresión logística.'
        : 'The 10 synthetic cases, baked by the WofE engine. Each row carries its ROC AUC, capture@10% (spatial CV), the CI ratio (+ omnibus z), the random−spatial inflation gap and the logistic-regression AUC.'}</p>

      {rows == null ? <p className="pf-note">{es ? 'cargando…' : 'loading…'}</p> : (
        <table className="cmp-table">
          <thead>
            <tr>
              <th>{es ? 'caso' : 'case'}</th><th>{es ? 'categoría' : 'category'}</th><th>{es ? 'datos' : 'data'}</th>
              <th>N(D)</th><th>ROC AUC</th><th>cap@10%</th><th>CI ratio</th><th>z</th><th>{es ? 'inflación' : 'inflation'}</th><th>LR AUC</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td><b>{r.id}</b></td>
                <td>{r.category.split(' (')[0]}</td>
                <td>{r.realOrSynthetic === 'synthetic' ? (es ? 'sintético' : 'synthetic') : (es ? 'control' : 'control')}</td>
                <td>{r.nDeposits}</td>
                <td>{r.rocAuc.toFixed(3)}</td>
                <td>{(r.capPred * 100).toFixed(0)}%</td>
                <td style={{ color: r.ciRatio < 0.85 ? 'var(--color-bad)' : undefined }}>{r.ciRatio.toFixed(2)}</td>
                <td>{r.z.toFixed(1)}</td>
                <td>{r.inflation.toFixed(3)}</td>
                <td>{r.lrAuc.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="pf-note">{es
        ? 'Anclas: C-NEGATIVE, capas no informativas ⇒ AUC ≈ 0.5. C-CIVIOLATE, un duplicado correlacionado ⇒ CI ratio < 1 (z alto), posterior inflado. C-RECOVER, recupera el orden de pesos plantados. C-SATURATE, una capa casi perfecta, sin blow-up numérico.'
        : 'Anchors: C-NEGATIVE, uninformative layers ⇒ AUC ≈ 0.5. C-CIVIOLATE, a correlated duplicate ⇒ CI ratio < 1 (high z), inflated posterior. C-RECOVER, recovers the planted weight ordering. C-SATURATE, a near-perfect single layer, no numerical blow-up.'}</p>

      <h2>{es ? 'Matriz de controles negativos (cubo real MVT)' : 'Negative-control matrix (real MVT cube)'}</h2>
      <p className="pf-note">{es
        ? 'El carril PU-Conformal debe pasar tres controles adversariales antes de creer cualquier resultado. Todos sobre las MISMAS particiones espaciales contiguas (k-means) que el head-to-head del Benchmark.'
        : 'The PU-Conformal lane must pass three adversarial controls before any result is believed. All on the SAME contiguous spatial folds (k-means) as the Benchmark head-to-head.'}</p>
      {!pu ? <p className="pf-note">{es ? 'cargando…' : 'loading…'}</p> : (
        <table className="cmp-table">
          <thead><tr><th>{es ? 'control' : 'control'}</th><th>{es ? 'diseño' : 'design'}</th><th>{es ? 'resultado' : 'result'}</th><th>{es ? 'veredicto' : 'verdict'}</th></tr></thead>
          <tbody>
            <tr>
              <td><b>{es ? 'permutación de etiquetas' : 'label permutation'}</b></td>
              <td className="pf-muted">{es ? 'baraja las etiquetas de depósito; no debe quedar señal' : 'shuffle the deposit labels; no signal must remain'}</td>
              <td>WofE {pu.negative_controls.label_permutation.wofe_auc.toFixed(3)} · PU {pu.negative_controls.label_permutation.pu_auc.toFixed(3)}</td>
              <td style={{ color: 'var(--color-ok, #2ea043)' }}>{es ? 'colapsa a ~0.5' : 'collapses to ~0.5'}</td>
            </tr>
            <tr>
              <td><b>{es ? 'capa no informativa' : 'uninformative layer'}</b></td>
              <td className="pf-muted">{es ? 'añade una 7.ª capa de puro ruido; no debe dar lift' : 'add a pure-noise 7th layer; it must not lift AUC'}</td>
              <td>{pu.negative_controls.uninformative_layer.pu_auc_with_noise_layer.toFixed(3)} {es ? 'vs' : 'vs'} {pu.negative_controls.uninformative_layer.pu_auc_without.toFixed(3)}</td>
              <td style={{ color: 'var(--color-ok, #2ea043)' }}>{es ? 'sin lift' : 'no lift'}</td>
            </tr>
            <tr>
              <td><b>{es ? 'null distancia-a-depósito' : 'distance-to-deposit null'}</b></td>
              <td className="pf-muted">{es ? 'score = proximidad al depósito de entrenamiento más cercano' : 'score = proximity to the nearest training deposit'}</td>
              <td style={{ color: 'var(--color-warn, #d29922)' }}>AUC {pu.negative_controls.distance_to_deposit_null.distance_to_deposit_auc.toFixed(3)}</td>
              <td style={{ color: 'var(--color-warn, #d29922)' }}>{es ? 'alto: la mayoría del skill es proximidad, no geología' : 'high: most skill is proximity, not geology'}</td>
            </tr>
          </tbody>
        </table>
      )}
      <Callout variant="strong" title={es ? 'Protocolo de bloques espaciales' : 'Spatial-block protocol'}>
        {es
          ? 'Particiones espaciales CONTIGUAS por k-means sobre las coordenadas de celda (k=5), más estrictas que el esquema entrelazado blockId % k del App: bajo holdout contiguo, un bloque held-out está espacialmente separado de su entrenamiento, así que la pregunta es honesta: ¿puede la geofísica regional predecir un distrito del que nunca vio un vecino? El AUC bootstrap se reporta con IC 95%.'
          : 'CONTIGUOUS spatial folds by k-means on cell coordinates (k=5), stricter than the App\'s interleaved blockId % k: under contiguous holdout a held-out block is spatially separated from its training, so the question is honest: can regional geophysics predict a district it never saw a neighbour of? Bootstrap AUC is reported with a 95% CI.'}
      </Callout>
    </article>
  );
}
