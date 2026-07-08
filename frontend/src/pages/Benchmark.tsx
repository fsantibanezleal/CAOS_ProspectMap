import { useEffect, useState } from 'react';
import { Callout, Cite, ReferenceList, useShellLang } from '@fasl-work/caos-app-shell';
import { loadCaseResults, loadLearned, loadPuConformal, type LearnedFile, type PuConformalFile } from '../lib/artifacts.ts';

interface Row { id: string; wofe: number; lr: number; ciRatio: number; inflation: number; }

export default function Benchmark() {
  const es = useShellLang() === 'es';
  const [rows, setRows] = useState<Row[] | null>(null);
  const [learned, setLearned] = useState<LearnedFile | null>(null);
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
    loadPuConformal().then(setPu).catch(() => setPu(null));
  }, []);

  return (
    <article className="page-body prose">
      <h1>Benchmark</h1>
      <p className="lede">{es
        ? 'Comparaciones cruzadas que no dependen de un solo caso: WofE (caja blanca, la autoridad) vs regresión logística, y dónde la independencia condicional se rompe.'
        : 'Cross-case comparisons that do not depend on a single case: WofE (white-box, the authority) vs logistic regression, and where conditional independence breaks.'}</p>

      <Callout variant="honest" title={es ? 'Sin victoria fabricada' : 'No fabricated win'}>
        {es
          ? 'El posterior WofE de caja blanca es la autoridad interpretable. El clasificador aprendido (torch→ONNX, carril --retrain) se mide contra WofE en el MISMO holdout espacial y gana su lugar sólo si lo supera capturando interacciones no lineales. Si empata o pierde, el benchmark lo dice, y ese es el resultado honesto.'
          : 'The white-box WofE posterior is the interpretable authority. The learned classifier (torch→ONNX, the --retrain lane) is measured against WofE on the SAME spatial holdout and earns its place only if it beats it by capturing non-linear interactions. If it ties or loses, the benchmark says so, and that is the honest result.'}
      </Callout>

      {rows == null ? <p className="pf-note">{es ? 'cargando…' : 'loading…'}</p> : (
        <table className="cmp-table">
          <thead><tr><th>{es ? 'caso' : 'case'}</th><th>WofE AUC</th><th>LR AUC</th><th>CI ratio</th><th>{es ? 'inflación CV' : 'CV inflation'}</th></tr></thead>
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
      {learned ? (
        <p className="pf-note">{es ? 'clasificador MPM AUC (spatial holdout): ' : 'mpm-classifier AUC (spatial holdout): '}<b>{String((learned.classifier?.spatial_cv as Record<string, number>)?.mlp_roc_auc ?? ', ')}</b> · OOD AUC <b>{learned.ood.auc.toFixed(3)}</b>{es ? ' (eval OOD sintético fuera de banda, separable por construcción)' : ' (synthetic out-of-band eval set, separable by construction)'}</p>
      ) : (
        <p className="pf-note">{es ? 'Modelos aprendidos pendientes, corre `python -m pmlab.pipeline all --retrain`. El App usa el WofE exacto en vivo mientras tanto.' : 'Learned models pending, run `python -m pmlab.pipeline all --retrain`. The App uses the exact WofE live meanwhile.'}</p>
      )}

      <h2>{es ? 'Head-to-head PU-Conformal (cubo real MVT, CV espacial contiguo)' : 'PU-Conformal head-to-head (real MVT cube, contiguous spatial CV)'}</h2>
      {!pu ? <p className="pf-note">{es ? 'cargando el carril PU-Conformal…' : 'loading the PU-Conformal lane…'}</p> : (
        <>
          <p className="pf-note">{es
            ? `Los seis modelos sobre las MISMAS particiones espaciales contiguas (k-means, ${pu.protocol.folds} folds) del cubo real US MVT. La clásica-SOTA (random forest / gradient boosting) es el rung que hace que PU-Conformal se juzgue contra la frontera ML real. AUC con IC 95% bootstrap.`
            : `All six models on the SAME contiguous spatial folds (k-means, ${pu.protocol.folds} folds) of the real US MVT cube. The SOTA-classical rung (random forest / gradient boosting) is what makes PU-Conformal judged against the real ML frontier. AUC with 95% bootstrap CI.`} <Cite id="roberts2017" paren /> <Cite id="rodriguezgaliano2015" paren /></p>
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

          <Callout variant="honest" title={es ? 'El resultado honesto' : 'The honest result'}>
            {pu.verdict.text}
          </Callout>
        </>
      )}
      <ReferenceList />
    </article>
  );
}
