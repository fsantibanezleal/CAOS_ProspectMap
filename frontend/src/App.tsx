// COMMIT-3 PLACEHOLDER. Proves the MPM engine (frontend/src/mpm/) bundles + runs LIVE in the browser: pick a case,
// regenerate the synthetic study area from its SPEC, recompute the full Weights-of-Evidence analysis, and show the
// per-layer weights + the honesty numbers (CI ratio, the random-vs-spatial-CV inflation gap). The full 6-page SPA on
// @fasl-work/caos-app-shell replaces this in commit 4.
import { useMemo, useState } from 'react';
import { analyzeCase, CASES } from './mpm/index.ts';

export default function App() {
  const [sel, setSel] = useState(CASES[0].id);
  const c = useMemo(() => CASES.find((x) => x.id === sel)!, [sel]);
  const a = useMemo(() => analyzeCase(c.spec, c.layerIds), [c]);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 880, margin: '2rem auto', padding: '0 1rem' }}>
      <h1>ProspectMap — Weights-of-Evidence prospectivity</h1>
      <p style={{ color: '#666' }}>
        Engine proof-of-life (commit 3). The full interactive map + 6-page app lands in commit 4.
      </p>
      <label>
        Case:{' '}
        <select value={sel} onChange={(e) => setSel(e.target.value)}>
          {CASES.map((x) => (
            <option key={x.id} value={x.id}>
              {x.id} — {x.name}
            </option>
          ))}
        </select>
      </label>
      <p>
        <b>{c.name}</b> · {c.realOrSynthetic} · {a.nDeposits} deposits over {a.nCells} cells
        <br />
        ROC AUC <b>{a.rocAuc.toFixed(3)}</b> · capture@10% {a.capture.success.captureAt10.toFixed(2)} · CI ratio{' '}
        <b>{a.ci.ciRatio.toFixed(2)}</b> (z {a.ci.z.toFixed(1)}) · random−spatial inflation{' '}
        <b>{a.cv.inflationGap.toFixed(3)}</b> · LR AUC {a.lr.rocAuc.toFixed(3)}
      </p>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid #ccc' }}>
            <th>layer</th>
            <th>t*</th>
            <th>W+</th>
            <th>W−</th>
            <th>contrast</th>
            <th>studC</th>
          </tr>
        </thead>
        <tbody>
          {a.layers.map((l) => (
            <tr key={l.id} style={{ borderBottom: '1px solid #eee' }}>
              <td>{l.id}</td>
              <td>{l.tStar.toFixed(2)}</td>
              <td>{l.wPlus.toFixed(2)}</td>
              <td>{l.wMinus.toFixed(2)}</td>
              <td>{l.contrast.toFixed(2)}</td>
              <td>{l.studC.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ color: '#888', fontSize: 13, marginTop: 16 }}>{c.validationAnchor}</p>
    </div>
  );
}
