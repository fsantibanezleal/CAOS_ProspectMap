import { Callout, Cite, Equation, InlineMath, ReferenceList, Tabs, useShellLang } from '@fasl-work/caos-app-shell';

export default function Methodology() {
  const es = useShellLang() === 'es';
  return (
    <article className="page-body prose">
      <h1>{es ? 'Metodología' : 'Methodology'}</h1>
      <p className="lede">{es
        ? 'Weights of Evidence → el test de independencia condicional → la regresión logística (la generalización sin CI) → la validación honesta por cross-validation espacial.'
        : 'Weights of Evidence → the conditional-independence test → logistic regression (the CI-free generalization) → honest validation by spatial cross-validation.'}</p>

      <Tabs ariaLabel={es ? 'metodología' : 'methodology'} tabs={[
        {
          id: 'wofe', label: 'Weights of Evidence',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'Fija una grilla de celdas unitarias. Para un patrón de evidencia binario ' : 'Fix a grid of unit cells. For a binary evidence pattern '}<InlineMath tex="B" />{es ? ' y el conjunto de depósitos ' : ' and the deposit set '}<InlineMath tex="D" />{es ? ', los pesos son los log-likelihood ratios:' : ', the weights are the log-likelihood ratios:'}</p>
              <Equation tex="W^{+}=\ln\frac{P(B\mid D)}{P(B\mid \bar D)},\qquad W^{-}=\ln\frac{P(\bar B\mid D)}{P(\bar B\mid \bar D)}" />
              <p>{es ? 'El contraste ' : 'The contrast '}<InlineMath tex="C=W^{+}-W^{-}" />{es ? ' mide la asociación espacial; el contraste estudentizado ' : ' measures the spatial association; the studentized contrast '}<InlineMath tex="C/s(C)" />{es ? ' es la guía de significancia, con ' : ' is the significance guide, with '}<InlineMath tex="s^2(W^{+})=\tfrac{1}{n_{B\cap D}}+\tfrac{1}{n_{B\cap \bar D}}" />{es ? '.' : '.'}</p>
              <p>{es ? 'El log-odds posterior de una celda = el logit previo + la suma de los pesos presente/ausente, bajo independencia condicional:' : 'The posterior log-odds of a cell = the prior logit + the sum of the present/absent weights, under conditional independence:'}</p>
              <Equation tex="\operatorname{logit}P(D\mid B_1^{k_1},\dots,B_m^{k_m})=\ln O(D)+\sum_{j} W_j^{k_j}" />
              <Callout variant="note" title={es ? 'El umbral de contraste máximo' : 'The maximizing-contrast threshold'}>
                {es ? 'Una capa continua se binariza en el umbral ' : 'A continuous layer is binarized at the threshold '}<InlineMath tex="t^{*}=\arg\max_t C(t)" />{es ? ' (sujeto a un piso de studC). Binarizar fuera de ' : ' (subject to a studC floor). Binarizing off '}<InlineMath tex="t^{*}" />{es ? ' degrada el contraste. La app hoy binariza siempre en t*; un control de umbral en vivo está planificado.' : ' degrades the contrast. The app currently always binarizes at t*; a live threshold control is planned.'} <Cite id="bonhamcarter1994" paren />
              </Callout>
            </div>
          ),
        },
        {
          id: 'ci', label: es ? 'Independencia condicional' : 'Conditional independence',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'La suma posterior sólo es válida si los patrones son condicionalmente independientes dado ' : 'The posterior sum is only valid if the patterns are conditionally independent given '}<InlineMath tex="D" />{es ? '. Capas favorables correlacionadas (magnetismo y gravedad sobre la misma intrusión) doble-cuentan y SOBRE-ESTIMAN el posterior. El test omnibus de Agterberg-Cheng ' : '. Correlated favourable layers (magnetics and gravity over the same intrusion) double-count and OVER-ESTIMATE the posterior. The Agterberg-Cheng omnibus test '}<Cite id="agterbergcheng2002" paren />{es ? ' usa la identidad:' : ' uses the identity:'}</p>
              <Equation tex="T=\sum_{\text{celdas}} P_{\text{post}}\;\approx\;N(D)\quad\text{bajo CI};\qquad z=\frac{T-N(D)}{s(T)}" />
              <p>{es ? 'El CI ratio ' : 'The CI ratio '}<InlineMath tex="N(D)/T" />{es ? ' es la guía: ≈ 1 está bien; < 0.85 marca una violación problemática (el posterior está inflado).' : ' is the guide: ≈ 1 is fine; < 0.85 flags a problematic violation (the posterior is inflated).'}</p>
            </div>
          ),
        },
        {
          id: 'lr', label: es ? 'Regresión logística' : 'Logistic regression',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'El arreglo cuando CI falla: la regresión logística ajusta las capas EN CONJUNTO, así que no requiere independencia condicional ' : 'The fix when CI fails: logistic regression fits the layers JOINTLY, so it does not require conditional independence '}<Cite id="agterberg1990" paren />{es ? ':' : ':'}</p>
              <Equation tex="\operatorname{logit}P(D\mid x)=\beta_0+\sum_j \beta_j x_j" />
              <p>{es ? 'Sobre patrones binarios condicionalmente independientes, ' : 'On conditionally-independent binary patterns, '}<InlineMath tex="\beta_j\approx C_j" />{es ? ' (la equivalencia WofE↔LR). Con violación de CI, se espera que los ' : ' (the WofE↔LR equivalence). Under CI violation the '}<InlineMath tex="\beta_j" />{es ? ' se encojan en vez de doble-contar (el arreglo teórico; el test omnibus de la app corre sobre el posterior WofE, un readout de calibración de LR aún no se muestra). Se ajusta por IRLS con ridge (el conjunto positivo es pequeño y desbalanceado).' : ' are expected to shrink rather than double-count (the theoretical fix; the app\'s omnibus test runs on the WofE posterior, an LR calibration readout is not yet shown). Fit by IRLS with ridge (the positive set is small and imbalanced).'}</p>
            </div>
          ),
        },
        {
          id: 'val', label: es ? 'Validación' : 'Validation',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'La regla cardinal: un modelo que ajusta sus propios depósitos de entrenamiento NO es evidencia de skill predictivo. Sólo la captura espacialmente held-out lo es ' : 'The cardinal rule: a model fitting its own training deposits is NOT evidence of predictive skill. Only spatially held-out capture is '}<Cite id="chungfabbri2003" paren />{es ? '. La curva de prediction-rate ordena las celdas por prospectividad y grafica % depósitos capturados vs % área.' : '. The prediction-rate curve ranks cells by prospectivity and plots % deposits captured vs % area.'}</p>
              <Callout variant="strong" title={es ? 'Inflación por CV aleatorio' : 'Random-CV inflation'}>
                {es ? 'Las celdas de depósitos y sus vecinas no son independientes (autocorrelación espacial). Un split aleatorio pone una celda de test junto a una de entrenamiento ⇒ el AUC se infla. El arreglo es el CV espacial por bloques + buffered leave-one-deposit-out ' : 'Deposit cells and their neighbours are not independent (spatial autocorrelation). A random split puts a test cell next to a training cell ⇒ the AUC inflates. The fix is spatial block CV + buffered leave-one-deposit-out '}<Cite id="roberts2017" paren />{es ? '. ProspectMap muestra el MISMO modelo colapsar de aleatorio a espacial.' : '. ProspectMap shows the SAME model collapse from random to spatial.'}
              </Callout>
            </div>
          ),
        },
      ]} />
      <ReferenceList />
    </article>
  );
}
