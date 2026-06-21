import { useEffect, useState } from 'react';
import { Callout, useShellLang } from '@fasl-work/caos-app-shell';
import { loadCaseResults, loadLearned, type LearnedFile } from '../lib/artifacts.ts';

interface Row { id: string; wofe: number; lr: number; ciRatio: number; inflation: number; }

export default function Benchmark() {
  const es = useShellLang() === 'es';
  const [rows, setRows] = useState<Row[] | null>(null);
  const [learned, setLearned] = useState<LearnedFile | null>(null);

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
  }, []);

  return (
    <article className="page-body prose">
      <h1>Benchmark</h1>
      <p className="lede">{es
        ? 'Comparaciones cruzadas que no dependen de un solo caso: WofE (caja blanca, la autoridad) vs regresión logística, y dónde la independencia condicional se rompe.'
        : 'Cross-case comparisons that do not depend on a single case: WofE (white-box, the authority) vs logistic regression, and where conditional independence breaks.'}</p>

      <Callout variant="honest" title={es ? 'Sin victoria fabricada' : 'No fabricated win'}>
        {es
          ? 'El posterior WofE de caja blanca es la autoridad interpretable. El clasificador aprendido (torch→ONNX, carril --retrain) se mide contra WofE en el MISMO holdout espacial y gana su lugar sólo si lo supera capturando interacciones no lineales. Si empata o pierde, el benchmark lo dice — y ese es el resultado honesto.'
          : 'The white-box WofE posterior is the interpretable authority. The learned classifier (torch→ONNX, the --retrain lane) is measured against WofE on the SAME spatial holdout and earns its place only if it beats it by capturing non-linear interactions. If it ties or loses, the benchmark says so — and that is the honest result.'}
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

      <h2>{es ? 'Clasificador aprendido vs WofE' : 'Learned classifier vs WofE'}</h2>
      {learned ? (
        <p className="pf-note">{es ? 'clasificador MPM AUC (spatial holdout): ' : 'mpm-classifier AUC (spatial holdout): '}<b>{String((learned.classifier?.spatial_cv as Record<string, number>)?.mlp_roc_auc ?? '—')}</b> · OOD AUC <b>{learned.ood.auc.toFixed(3)}</b></p>
      ) : (
        <p className="pf-note">{es ? 'Modelos aprendidos pendientes — corre `python -m pmlab.pipeline all --retrain`. El App usa el WofE exacto en vivo mientras tanto.' : 'Learned models pending — run `python -m pmlab.pipeline all --retrain`. The App uses the exact WofE live meanwhile.'}</p>
      )}
    </article>
  );
}
