import { Callout, useShellLang } from '@fasl-work/caos-app-shell';

export default function Implementation() {
  const es = useShellLang() === 'es';
  return (
    <article className="page-body prose">
      <h1>{es ? 'Implementación' : 'Implementation'}</h1>
      <p className="lede">{es
        ? 'Un motor TypeScript sin dependencias corre en vivo en el navegador Y en el bake offline; dos contratos de datos en Python + un pipeline numpy-light + un carril torch→ONNX. Cómputo pesado offline → artefacto compacto → cliente delgado.'
        : 'A dependency-free TypeScript engine runs live in the browser AND in the offline bake; two Python data contracts + a numpy-light pipeline + a torch→ONNX lane. Heavy compute offline → compact artifact → thin client.'}</p>

      <h2>{es ? 'El motor (frontend/src/mpm/)' : 'The engine (frontend/src/mpm/)'}</h2>
      <ul>
        <li><code>wofe.ts</code>, {es ? 'W⁺/W⁻/contraste/studC/posterior bajo CI (con la corrección Haldane 0.5 para conteos cero).' : 'W⁺/W⁻/contrast/studC/posterior under CI (with the Haldane 0.5 zero-count correction).'}</li>
        <li><code>binarize.ts</code>, {es ? 'el barrido de umbral de contraste máximo.' : 'the maximizing-contrast threshold sweep.'}</li>
        <li><code>ci.ts</code>, {es ? 'χ² pairwise + el omnibus de Agterberg-Cheng + el CI ratio.' : 'pairwise χ² + the Agterberg-Cheng omnibus + the CI ratio.'}</li>
        <li><code>logreg.ts</code>, {es ? 'regresión logística por IRLS + ridge (la generalización sin CI).' : 'logistic regression by IRLS + ridge (the CI-free generalization).'}</li>
        <li><code>validate.ts</code> + <code>cv.ts</code>, {es ? 'curvas de captura success/prediction + capture@10% + ROC; folds aleatorios vs espaciales-por-bloques + el demostrador de inflación.' : 'success/prediction capture curves + capture@10% + ROC; random vs spatial-block folds + the inflation demonstrator.'}</li>
        <li><code>synth.ts</code>, {es ? 'el generador determinista de áreas sintéticas con pesos plantados (= los oráculos).' : 'the deterministic synthetic-area generator with planted weights (= the oracles).'}</li>
      </ul>

      <Callout variant="note" title={es ? 'El bake de dos lenguajes' : 'The two-language bake'}>
        {es
          ? 'science/bake_cases.mjs importa el MISMO motor TS (vía tsx) y corre analyze.ts sobre cada caso → case-results.json. El pipeline Python (numpy-light) reshapea eso en los traces+manifests por caso (CONTRATO 2). Así los números en vivo y offline son idénticos por construcción.'
          : 'science/bake_cases.mjs imports the SAME TS engine (via tsx) and runs analyze.ts over each case → case-results.json. The Python pipeline (numpy-light) reshapes that into the per-case traces+manifests (CONTRACT 2). So the live and offline numbers are identical by construction.'}
      </Callout>

      <h2>{es ? 'El carril PU-Conformal (offline → ONNX + JSON)' : 'The PU-Conformal lane (offline → ONNX + JSON)'}</h2>
      <p>{es
        ? 'El carril más allá del SOTA se entrena offline en un venv aislado (torch + scikit-learn) sobre el cubo real US MVT y exporta dos artefactos compactos que el navegador consume:'
        : 'The beyond-SOTA lane trains offline in an isolated venv (torch + scikit-learn) on the real US MVT cube and exports two compact artifacts the browser consumes:'}</p>
      <ul>
        <li><code>mpm-puconformal-real.onnx</code>, {es ? 'el score nnPU con la estandarización y la sigmoide precalculadas en el grafo (entrada x[N,6] → p[N,1]); corre en vivo vía onnxruntime-web, la misma ruta que los otros dos ONNX.' : 'the nnPU score with standardization and the sigmoid baked into the graph (input x[N,6] → p[N,1]); runs live via onnxruntime-web, the same path as the other two ONNX.'}</li>
        <li><code>pu-conformal.json</code>, {es ? 'el head-to-head (WofE/LR/RF/GBM/MLP/PU con IC bootstrap), los controles negativos, los cuantiles conformes por α y la sensibilidad al prior de clase pi. El navegador aplica el umbral conforme EN VIVO para dibujar el conjunto prospectivo; sin cómputo pesado en página.' : 'the head-to-head (WofE/LR/RF/GBM/MLP/PU with bootstrap CI), the negative controls, the conformal quantiles per α, and the class-prior pi sensitivity. The browser applies the conformal threshold LIVE to draw the prospective set; no heavy compute in-page.'}</li>
      </ul>
      <p>{es
        ? 'El entrenamiento usa el estimador de riesgo no negativo (nnPU) tratando TODAS las celdas no-depósito como no etiquetadas, particiones espaciales contiguas idénticas para los seis modelos, y split-conformal de clase positiva calibrado sobre bloques separados. pi se barre como parámetro de sensibilidad (SCAR es dudoso por sesgo de exploración). Ningún número "beats SOTA" es inventado: el veredicto de ranking se lee directamente de los IC bootstrap y los controles.'
        : 'Training uses the non-negative risk estimator (nnPU) treating ALL non-deposit cells as unlabeled, identical contiguous spatial folds for all six models, and positive-class split-conformal calibrated on separated blocks. pi is swept as a sensitivity parameter (SCAR is doubtful under exploration bias). No "beats SOTA" number is invented: the ranking verdict is read directly off the bootstrap CIs and the controls.'}</p>

      <h2>{es ? 'Verificación' : 'Verification'}</h2>
      <p>{es
        ? 'node:test (11 oráculos del motor: WofE de forma cerrada, equivalencia WofE↔LR, el omnibus ≈1 vs T>N(D) en la trampa CI, las curvas de captura, el gap de inflación), ruff, pytest (los 2 contratos), python -m pmlab.pipeline all (10 casos), check_artifacts (CONTRATO 2), re-run byte-idéntico, npm run build.'
        : 'node:test (11 engine oracles: closed-form WofE, the WofE↔LR equivalence, the omnibus ≈1 vs T>N(D) on the CI trap, the capture curves, the inflation gap), ruff, pytest (the 2 contracts), python -m pmlab.pipeline all (10 cases), check_artifacts (CONTRACT 2), byte-identical re-run, npm run build.'}</p>
    </article>
  );
}
