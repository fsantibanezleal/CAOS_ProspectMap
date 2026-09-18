import { useEffect, useState } from 'react';
import { Callout, Cite, ReferenceList, useShellLang } from '@fasl-work/caos-app-shell';
import { loadCaseResults, loadLearned, loadLearnedReal, loadPuConformal, type LearnedFile, type PuConformalFile } from '../lib/artifacts.ts';
import { headToHeadContext, headToHeadDeclared, headToHeadProtocol, headToHeadVerdict } from '../lib/learned.ts';

interface Row { id: string; wofe: number; lr: number; ciRatio: number; inflation: number; }

const f3 = (v: number | null | undefined) => (typeof v === 'number' ? v.toFixed(3) : 'n/a');

export default function Benchmark() {
  const es = useShellLang() === 'es';
  const [rows, setRows] = useState<Row[] | null>(null);
  const [learned, setLearned] = useState<LearnedFile | null>(null);
  const [learnedReal, setLearnedReal] = useState<LearnedFile | null>(null);
  const [pu, setPu] = useState<PuConformalFile | null>(null);

  useEffect(() => {
    loadCaseResults().then((cr) => {
      setRows(Object.entries(cr.cases).map(([id, raw]) => {
        const c = raw as Record<string, unknown>;
        return {
          id, wofe: Number(c.rocAuc), lr: Number((c.lr as Record<string, number>)?.rocAuc ?? 0),
          ciRatio: Number((c.ci as Record<string, number>)?.ciRatio ?? 0),
          inflation: Number((c.cv as Record<string, number>)?.inflationGap ?? 0),
        };
      }));
    }).catch(() => setRows([]));
    loadLearned().then(setLearned).catch(() => setLearned(null));
    loadLearnedReal().then(setLearnedReal).catch(() => setLearnedReal(null));
    loadPuConformal().then(setPu).catch(() => setPu(null));
  }, []);

  return (
    <article className="page-body prose">
      <h1>Benchmark</h1>
      <p className="lede">{es
        ? 'Comparaciones cruzadas que no dependen de un solo caso: WofE (caja blanca, la autoridad) vs regresión logística, y dónde la independencia condicional se rompe.'
        : 'Cross-case comparisons that do not depend on a single case: WofE (white-box, the authority) vs logistic regression, and where conditional independence breaks.'}</p>

      <Callout variant="honest" title={es ? 'Cómo se comparan los modelos aprendidos' : 'How the learned models are compared'}>
        {es
          ? 'El posterior WofE de caja blanca es la autoridad interpretable. Cada clasificador aprendido (entrenado con torch y exportado a ONNX, carril --retrain) se compara con WofE bajo un único protocolo de cross-validation compartido; en el belt real, junto a una línea base de distancia al depósito conocido. Un empate o una derrota se reporta tal como se mide.'
          : 'The white-box WofE posterior is the interpretable authority. Each learned classifier (trained with torch and exported to ONNX, the --retrain lane) is compared with WofE under one shared cross-validation protocol; on the real belt, beside a distance-to-known-deposit baseline. A tie or a loss is reported as measured.'}
      </Callout>

      {rows == null ? <p className="pf-note">{es ? 'cargando…' : 'loading…'}</p> : (
        <table className="cmp-table">
          <thead><tr><th>{es ? 'caso' : 'case'}</th><th>{es ? 'AUC WofE (ajuste)' : 'WofE AUC (fit)'}</th><th>{es ? 'AUC LR (ajuste)' : 'LR AUC (fit)'}</th><th>CI ratio</th><th>{es ? 'inflación CV' : 'CV inflation'}</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td><b>{r.id}</b></td>
                <td>{r.wofe.toFixed(3)}</td>
                <td>{r.lr.toFixed(3)}</td>
                <td style={{ color: r.ciRatio < 0.85 ? 'var(--color-bad)' : undefined }}>{r.ciRatio.toFixed(2)}</td>
                <td>{r.inflation.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="pf-note">{es
        ? 'Procedencia de las métricas: las columnas WofE AUC y LR AUC son números in-sample (de ajuste); la columna de inflación CV viene de cross-validation. La captura held-out espacial vive en la pestaña Tasas de captura del App.'
        : 'Metric provenance: the WofE AUC and LR AUC columns are in-sample (fitting) numbers; the CV-inflation column comes from cross-validation. Spatially held-out capture lives in the App\'s Capture-rates tab.'}</p>

      <h2>{es ? 'Clasificador aprendido vs WofE' : 'Learned classifier vs WofE'}</h2>
      <p className="pf-note">{es
        ? 'Cada carril compara el MLP con WofE bajo un único protocolo, descrito junto a sus valores. Los dos carriles usan datos y protocolos distintos, así que sus valores no se comparan entre sí.'
        : 'Each lane compares the MLP with WofE under one protocol, stated beside its values. The two lanes use different data and protocols, so their values are not compared with each other.'}</p>
      {learned ? (
        <p className="pf-note"><b>{es ? 'Carril sintético' : 'Synthetic lane'}</b> (<code>pm-learned.json</code>): {es ? 'AUC con CV espacial' : 'spatial-CV AUC'} MLP <b>{f3(learned.classifier.spatial_cv?.mlp_roc_auc)}</b> · WofE <b>{f3(learned.classifier.spatial_cv?.wofe_roc_auc)}</b>; {es ? 'CV aleatorio del MLP' : 'MLP random CV'} {f3(learned.classifier.random_cv?.mlp_roc_auc)} ({es ? 'inflación' : 'inflation'} {f3(learned.classifier.inflation_gap)}). {headToHeadProtocol('synthetic', learned, es)} {headToHeadVerdict('synthetic', learned, es)} {headToHeadContext('synthetic', learned, es)} OOD AUC {f3(learned.ood.auc)}{es ? ' (eval OOD sintético fuera de banda, separable por construcción).' : ' (synthetic out-of-band eval set, separable by construction).'}</p>
      ) : (
        <p className="pf-note">{es ? 'Modelos aprendidos sintéticos pendientes, ejecutar `python data-pipeline/run.py all --retrain`. El App usa el WofE exacto en vivo mientras tanto.' : 'Synthetic learned models pending, run `python data-pipeline/run.py all --retrain`. The App uses the exact WofE live meanwhile.'}</p>
      )}
      {learnedReal && headToHeadDeclared('real', learnedReal) && (
        <p className="pf-note"><b>{es ? 'Carril real, US MVT' : 'Real lane, US MVT'}</b> (<code>pm-learned-real.json</code>): {es ? 'AUC con CV espacial' : 'spatial-CV AUC'} MLP <b>{f3(learnedReal.classifier.spatial_cv?.mlp_roc_auc)}</b> · WofE <b>{f3(learnedReal.classifier.spatial_cv?.wofe_roc_auc)}</b>; {es ? 'CV aleatorio' : 'random CV'} MLP {f3(learnedReal.classifier.random_cv?.mlp_roc_auc)} · WofE {f3(learnedReal.classifier.random_cv?.wofe_roc_auc)}. {headToHeadProtocol('real', learnedReal, es)} {headToHeadVerdict('real', learnedReal, es)} {headToHeadContext('real', learnedReal, es)} {es
          ? `El AUC de WofE sin CV (${f3(learnedReal.classifier.nocv?.wofe_roc_auc)}, ajustado y evaluado en las mismas celdas) es un número de ajuste y no entra en la comparación.`
          : `The WofE AUC without CV (${f3(learnedReal.classifier.nocv?.wofe_roc_auc)}, fitted and scored on the same cells) is a fitting number and takes no part in the comparison.`}</p>
      )}

      <h2>{es ? 'Head-to-head PU-Conformal (cubo real MVT, CV espacial contiguo)' : 'PU-Conformal head-to-head (real MVT cube, contiguous spatial CV)'}</h2>
      {!pu ? <p className="pf-note">{es ? 'cargando el carril PU-Conformal…' : 'loading the PU-Conformal lane…'}</p> : (
        <>
          <p className="pf-note">{es
            ? `Los seis modelos sobre las mismas particiones espaciales contiguas (k-means, ${pu.protocol.folds} folds) del cubo real US MVT. La clásica-SOTA (random forest / gradient boosting) es el rung que hace que PU-Conformal se juzgue contra la frontera ML real. AUC con IC 95% bootstrap.`
            : `All six models on the same contiguous spatial folds (k-means, ${pu.protocol.folds} folds) of the real US MVT cube. The SOTA-classical rung (random forest / gradient boosting) is what makes PU-Conformal judged against the real ML frontier. AUC with 95% bootstrap CI.`} <Cite id="roberts2017" paren /> <Cite id="rodriguezgaliano2015" paren /></p>
          <table className="cmp-table">
            <thead><tr><th>{es ? 'modelo' : 'model'}</th><th>{es ? 'AUC CV-esp.' : 'block-CV AUC'}</th><th>IC 95%</th><th>AP</th><th>Brier</th><th>ECE</th></tr></thead>
            <tbody>
              {pu.benchmark.map((b) => (
                <tr key={b.model}>
                  <td><b>{b.label}</b></td>
                  <td>{b.auc.toFixed(3)}</td>
                  <td style={{ color: 'var(--color-fg-faint)' }}>[{b.auc_ci95[0].toFixed(3)}, {b.auc_ci95[1].toFixed(3)}]</td>
                  <td>{b.ap.toFixed(3)}</td><td>{b.brier.toFixed(3)}</td><td>{b.ece.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3>{es ? 'Controles negativos (deben pasar)' : 'Negative controls (must pass)'}</h3>
          <table className="cmp-table">
            <thead><tr><th>{es ? 'control' : 'control'}</th><th>{es ? 'resultado' : 'result'}</th><th>{es ? 'expectativa' : 'expectation'}</th></tr></thead>
            <tbody>
              <tr><td><b>{es ? 'permutación de etiquetas' : 'label permutation'}</b></td>
                <td>WofE {pu.negative_controls.label_permutation.wofe_auc.toFixed(3)} · PU {pu.negative_controls.label_permutation.pu_auc.toFixed(3)}</td>
                <td className="pf-muted">{es ? 'ambos colapsan a ~0.5' : 'both collapse to ~0.5'}</td></tr>
              <tr><td><b>{es ? 'capa no informativa' : 'uninformative layer'}</b></td>
                <td>{es ? 'con ruido' : 'with noise'} {pu.negative_controls.uninformative_layer.pu_auc_with_noise_layer.toFixed(3)} · {es ? 'sin' : 'without'} {pu.negative_controls.uninformative_layer.pu_auc_without.toFixed(3)}</td>
                <td className="pf-muted">{es ? 'el ruido no sube el AUC' : 'noise gives no lift'}</td></tr>
              <tr><td><b>{es ? 'null distancia-a-depósito' : 'distance-to-deposit null'}</b></td>
                <td style={{ color: 'var(--color-warn, #d29922)' }}>{pu.negative_controls.distance_to_deposit_null.distance_to_deposit_auc.toFixed(3)}</td>
                <td className="pf-muted">{es ? 'un modelo real debe superarlo' : 'a real model must beat it'}</td></tr>
            </tbody>
          </table>

          <h3>{es ? 'Cobertura conforme (split-conformal espacial, clase positiva)' : 'Conformal coverage (spatial split-conformal, positive class)'}</h3>
          <table className="cmp-table">
            <thead><tr><th>{es ? 'nominal' : 'nominal'}</th><th>{es ? 'empírica' : 'empirical'}</th><th>{es ? 'umbral' : 'threshold'}</th><th>{es ? 'tamaño del conjunto' : 'set size (area)'}</th></tr></thead>
            <tbody>
              {pu.conformal.levels.map((lv) => (
                <tr key={lv.alpha}>
                  <td>{(lv.nominal * 100).toFixed(0)}%</td>
                  <td>{(lv.empirical_coverage * 100).toFixed(1)}%</td>
                  <td>{lv.threshold.toFixed(3)}</td>
                  <td style={{ color: 'var(--color-warn, #d29922)' }}>{(lv.set_size_frac * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>

          <Callout variant="honest" title={es ? 'Resultado' : 'Result'}>
            {pu.verdict.text}
          </Callout>
        </>
      )}
      <ReferenceList />
    </article>
  );
}
