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
              <p>{es ? 'La suma posterior sólo es válida si los patrones son condicionalmente independientes dado ' : 'The posterior sum is only valid if the patterns are conditionally independent given '}<InlineMath tex="D" />{es ? '. Capas favorables correlacionadas (magnetismo y gravedad sobre la misma intrusión) doble-cuentan y sobre-estiman el posterior. El test omnibus de Agterberg-Cheng ' : '. Correlated favourable layers (magnetics and gravity over the same intrusion) double-count and over-estimate the posterior. The Agterberg-Cheng omnibus test '}<Cite id="agterbergcheng2002" paren />{es ? ' usa la identidad:' : ' uses the identity:'}</p>
              <Equation tex="T=\sum_{\text{celdas}} P_{\text{post}}\;\approx\;N(D)\quad\text{bajo CI};\qquad z=\frac{T-N(D)}{s(T)}" />
              <p>{es ? 'El CI ratio ' : 'The CI ratio '}<InlineMath tex="N(D)/T" />{es ? ' es la guía: ≈ 1 está bien; < 0.85 marca una violación problemática (el posterior está inflado).' : ' is the guide: ≈ 1 is fine; < 0.85 flags a problematic violation (the posterior is inflated).'}</p>
            </div>
          ),
        },
        {
          id: 'lr', label: es ? 'Regresión logística' : 'Logistic regression',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'El arreglo cuando CI falla: la regresión logística ajusta las capas en conjunto, así que no requiere independencia condicional ' : 'The fix when CI fails: logistic regression fits the layers jointly, so it does not require conditional independence '}<Cite id="agterberg1990" paren />{es ? ':' : ':'}</p>
              <Equation tex="\operatorname{logit}P(D\mid x)=\beta_0+\sum_j \beta_j x_j" />
              <p>{es ? 'Sobre patrones binarios condicionalmente independientes, ' : 'On conditionally-independent binary patterns, '}<InlineMath tex="\beta_j\approx C_j" />{es ? ' (la equivalencia WofE↔LR). Con violación de CI, se espera que los ' : ' (the WofE↔LR equivalence). Under CI violation the '}<InlineMath tex="\beta_j" />{es ? ' se encojan en vez de doble-contar (el arreglo teórico; el test omnibus de la app se ejecuta sobre el posterior WofE, un readout de calibración de LR aún no se muestra). Se ajusta por IRLS con ridge (el conjunto positivo es pequeño y desbalanceado).' : ' are expected to shrink rather than double-count (the theoretical fix; the app\'s omnibus test runs on the WofE posterior, an LR calibration readout is not yet shown). Fit by IRLS with ridge (the positive set is small and imbalanced).'}</p>
            </div>
          ),
        },
        {
          id: 'val', label: es ? 'Validación' : 'Validation',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'La regla cardinal: un modelo que ajusta sus propios depósitos de entrenamiento no es evidencia de skill predictivo. Sólo la captura espacialmente held-out lo es ' : 'The cardinal rule: a model fitting its own training deposits is not evidence of predictive skill. Only spatially held-out capture is '}<Cite id="chungfabbri2003" paren />{es ? '. La curva de prediction-rate ordena las celdas por prospectividad y grafica % depósitos capturados vs % área.' : '. The prediction-rate curve ranks cells by prospectivity and plots % deposits captured vs % area.'}</p>
              <Callout variant="strong" title={es ? 'Inflación por CV aleatorio' : 'Random-CV inflation'}>
                {es ? 'Las celdas de depósitos y sus vecinas no son independientes (autocorrelación espacial). Un split aleatorio pone una celda de test junto a una de entrenamiento ⇒ el AUC se infla. El arreglo es el CV espacial por bloques + buffered leave-one-deposit-out ' : 'Deposit cells and their neighbours are not independent (spatial autocorrelation). A random split puts a test cell next to a training cell ⇒ the AUC inflates. The fix is spatial block CV + buffered leave-one-deposit-out '}<Cite id="roberts2017" paren />{es ? '. ProspectMap muestra el mismo modelo colapsar de aleatorio a espacial.' : '. ProspectMap shows the same model collapse from random to spatial.'}
              </Callout>
              <p>{es ? 'El Benchmark refuerza esto con particiones espaciales contiguas (k-means sobre coordenadas), más estrictas que el esquema entrelazado blockId % k: entrelazar deja cada bloque held-out adyacente a bloques de entrenamiento, lo que permite a un modelo de grano fino memorizar la firma local autocorrelada e inflar el AUC.' : 'The Benchmark hardens this with contiguous spatial folds (k-means on coordinates), stricter than the interleaved blockId % k scheme: interleaving leaves each held-out block adjacent to training blocks, which lets a fine-grained model memorize the autocorrelated local signature and inflate the AUC.'} <Cite id="roberts2017" paren /></p>
            </div>
          ),
        },
        {
          id: 'tabular', label: es ? 'ML tabular (RF/GBM)' : 'Tabular ML (RF/GBM)',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'El rung SOTA-clásico: modelos tabulares no lineales que aprenden interacciones que WofE (aditivo en log-odds) no puede. En la evaluación canónica de MPM (Au epitermal), el random forest supera a redes, árboles de regresión y SVM ' : 'The SOTA-classical rung: non-linear tabular models that learn interactions the additive-in-log-odds WofE cannot. In the canonical MPM evaluation (epithermal Au), random forest beat neural nets, regression trees and SVM '}<Cite id="rodriguezgaliano2015" paren />{es ? '. El gradient boosting ajusta una suma aditiva por etapas de árboles ' : '. Gradient boosting fits a stagewise additive sum of trees '}<InlineMath tex="F_M(x)=\sum_{m=1}^{M}\nu\,h_m(x)" />{es ? ', cada árbol ' : ', each tree '}<InlineMath tex="h_m" />{es ? ' ajustado al gradiente negativo de la pérdida logística; el random forest promedia árboles bagged decorrelacionados.' : ' fit to the negative gradient of the logistic loss; the random forest averages decorrelated bagged trees.'}</p>
              <Equation tex="\hat p(x)=\sigma\big(F_M(x)\big),\qquad F_m(x)=F_{m-1}(x)+\nu\,\arg\min_{h}\sum_i \ell\big(y_i,\,F_{m-1}(x_i)+h(x_i)\big)" />
              <p>{es ? 'Entrenados con negativos muestreados (buffered) sobre el cubo real, se puntúan en las mismas particiones espaciales que WofE/LR/PU. Cuándo fallan: sin control espacial sobreajustan la autocorrelación (por eso el CV contiguo estricto), y siguen necesitando pseudo-negativos, el sesgo que corrige PU.' : 'Trained with sampled (buffered) negatives on the real cube, they are scored on the same spatial folds as WofE/LR/PU. When they fail: without spatial control they overfit autocorrelation (hence the strict contiguous CV), and they still need pseudo-negatives, the bias PU corrects.'}</p>
            </div>
          ),
        },
        {
          id: 'pu', label: es ? 'Aprendizaje PU' : 'PU learning',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'La falacia del negativo verdadero: en exploración, una celda sin depósito conocido no es un negativo, es no etiquetada (puede alojar un depósito no descubierto). Entrenar con pseudo-negativos sesga el clasificador. El aprendizaje PU trata los depósitos como positivos y todo lo demás como no etiquetado ' : 'The true-negative fallacy: in exploration, a cell with no known deposit is not a negative, it is unlabeled (it may host an undiscovered deposit). Training on pseudo-negatives biases the classifier. PU learning treats deposits as positives and everything else as unlabeled '}<Cite id="xiongzuo2021" paren />{es ? '. Bajo el supuesto SCAR (positivos etiquetados al azar), Elkan-Noto relacionan el score observado con el posterior verdadero por una constante de frecuencia de etiqueta ' : '. Under the SCAR assumption (labeled positives selected at random), Elkan-Noto relate the observed score to the true posterior by a label-frequency constant '}<InlineMath tex="c" /> <Cite id="elkannoto2008" paren />{es ? ':' : ':'}</p>
              <Equation tex="p(s{=}1\mid x)=c\,\cdot\,p(y{=}1\mid x),\qquad c=p(s{=}1\mid y{=}1)" />
              <p>{es ? 'Para un modelo flexible sobre pocos positivos, el estimador de riesgo no negativo (nnPU) evita el sobreajuste ' : 'For a flexible model over few positives, the non-negative risk estimator (nnPU) prevents overfitting '}<Cite id="kiryo2017" paren />{es ? ', usando el prior de clase ' : ', using the class prior '}<InlineMath tex="\pi=p(y{=}1)" />{es ? ':' : ':'}</p>
              <Equation tex="\widetilde R_{\text{nnPU}}=\pi\,\hat R_p^{+}+\max\!\Big(0,\;\hat R_u^{-}-\pi\,\hat R_p^{-}\Big)" />
              <p>{es ? 'El clamp en 0 corrige el término de riesgo negativo cuando se vuelve negativo por sobreajuste. Crítica adversarial: SCAR se viola por sesgo de exploración (los depósitos se sobre-etiquetan en cinturones bien explorados), así que ' : 'The clamp at 0 corrects the negative-risk term when overfitting drives it below zero. Adversarial critique: SCAR is violated by exploration bias (deposits are over-labeled in well-explored belts), so '}<InlineMath tex="\pi" />{es ? ' se trata como un parámetro de sensibilidad y se barre, no se fija.' : ' is treated as a sensitivity parameter and swept, not fixed.'}</p>
            </div>
          ),
        },
        {
          id: 'conformal', label: es ? 'Incertidumbre / conforme' : 'Uncertainty / conformal',
          content: (
            <div className="pf-doc-sec">
              <p>{es ? 'La mayoría de los mapas MPM entregan una sola probabilidad sin banda calibrada. La predicción conforme da conjuntos de predicción libres de distribución con cobertura garantizada en muestra finita ' : 'Most MPM maps ship a single probability with no calibrated band. Conformal prediction yields distribution-free prediction sets with finite-sample coverage '}<Cite id="angelopoulos2021" paren />{es ? '. Con una puntuación de no conformidad de clase positiva ' : '. With a positive-class nonconformity score '}<InlineMath tex="s_i=1-\hat p(x_i)" />{es ? ' sobre depósitos de calibración espacialmente separados, el cuantil conforme ' : ' over spatially-separated calibration deposits, the conformal quantile '}<InlineMath tex="\hat q=\text{Quantile}\big(\{s_i\};\,\lceil(n+1)(1-\alpha)\rceil/n\big)" />{es ? ' define el conjunto prospectivo:' : ' defines the prospective set:'}</p>
              <Equation tex="\mathcal{C}_\alpha(x)=\{\,x:\hat p(x)\ge 1-\hat q\,\},\qquad \Pr\big(\text{un depósito held-out}\in\mathcal{C}_\alpha\big)\ge 1-\alpha" />
              <Callout variant="honest" title={es ? 'Exchangeabilidad rota' : 'Exchangeability broken'}>
                {es ? 'La garantía conforme supone exchangeabilidad, que la autocorrelación espacial rompe. Bajo bloqueo espacial la garantía es marginal sobre bloques y se degrada con el shift de distribución bloque-a-bloque. Sobre MVT muy agrupado los conjuntos pueden ser casi vacíos (marcar casi todo el belt): esa banda ancha ES el hallazgo honesto ' : 'The conformal guarantee assumes exchangeability, which spatial autocorrelation breaks. Under spatial blocking the guarantee is marginal over blocks and degrades under block-to-block distribution shift. On strongly clustered MVT the sets can be near-vacuous (flagging almost the whole belt): that wide band IS the honest finding '}<Cite id="roberts2017" paren />{es ? ', no un defecto a ocultar.' : ', not a defect to hide.'}
              </Callout>
            </div>
          ),
        },
      ]} />
      <ReferenceList />
    </article>
  );
}
