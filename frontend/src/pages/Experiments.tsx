import { useEffect, useState } from 'react';
import { useShellLang } from '@fasl-work/caos-app-shell';
import { loadCaseResults } from '../lib/artifacts.ts';

interface Row {
  id: string; name: string; category: string; realOrSynthetic: string; rocAuc: number;
  capPred: number; ciRatio: number; z: number; inflation: number; lrAuc: number; nDeposits: number;
}

export default function Experiments() {
  const es = useShellLang() === 'es';
  const [rows, setRows] = useState<Row[] | null>(null);

  useEffect(() => {
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
    <article className="pf-doc">
      <h1>{es ? 'Experimentos' : 'Experiments'}</h1>
      <p className="pf-lead">{es
        ? 'Los 10 casos sintéticos, horneados por el motor WofE. Cada fila lleva su ROC AUC, capture@10% (spatial CV), el CI ratio (+ z del omnibus), el gap de inflación random−spatial y el AUC de la regresión logística.'
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
        ? 'Anclas: C-NEGATIVE — capas no informativas ⇒ AUC ≈ 0.5. C-CIVIOLATE — un duplicado correlacionado ⇒ CI ratio < 1 (z alto), posterior inflado. C-RECOVER — recupera el orden de pesos plantados. C-SATURATE — una capa casi perfecta, sin blow-up numérico.'
        : 'Anchors: C-NEGATIVE — uninformative layers ⇒ AUC ≈ 0.5. C-CIVIOLATE — a correlated duplicate ⇒ CI ratio < 1 (high z), inflated posterior. C-RECOVER — recovers the planted weight ordering. C-SATURATE — a near-perfect single layer, no numerical blow-up.'}</p>
    </article>
  );
}
