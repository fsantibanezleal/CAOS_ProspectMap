import { Callout, Cite, ReferenceList, useShellLang } from '@fasl-work/caos-app-shell';

export default function Introduction() {
  const es = useShellLang() === 'es';
  return (
    <article className="page-body prose">
      <h1>{es ? 'Introducción' : 'Introduction'}</h1>
      <p className="lede">{es
        ? 'ProspectMap responde "¿dónde es más probable el próximo depósito?" — fusiona capas de evidencia geocientífica abierta en un mapa posterior de prospectividad, y expone honestamente cuándo NO confiar en él.'
        : 'ProspectMap answers "where is the next deposit most likely to be?" — it fuses open geoscience evidence layers into a posterior prospectivity map, and honestly exposes when NOT to trust it.'}</p>

      <h2>{es ? 'El problema' : 'The problem'}</h2>
      <p>{es
        ? 'El mapeo de prospectividad mineral (MPM) combina varias capas de evidencia (geofísica magnética/gravimétrica/radiométrica, geoquímica de sedimentos, proximidad a estructuras, litología) sobre una grilla del área de estudio para estimar P(depósito presente | evidencia) por celda. El método canónico data-driven es Weights of Evidence (WofE) '
        : 'Mineral prospectivity mapping (MPM) combines several evidence layers (magnetic/gravity/radiometric geophysics, stream-sediment geochemistry, structure proximity, lithology) over a study-area grid to estimate P(deposit present | evidence) per cell. The canonical data-driven method is Weights of Evidence (WofE) '}
        <Cite id="bonhamcarter1989" paren />{es ? ', una actualización bayesiana de log-odds.' : ', a Bayesian log-odds update.'}</p>

      <Callout variant="honest" title={es ? 'Lo que separa esto de un heatmap vistoso' : 'What separates this from a colourful heatmap'}>
        {es
          ? 'ProspectMap hace de primera clase las DOS formas en que un mapa de prospectividad miente: (1) la violación de independencia condicional — capas favorables correlacionadas doble-cuentan e inflan el posterior; (2) la inflación de CV — un split aleatorio filtra vecinos de depósitos de entrenamiento y el AUC parece fantástico. El número reportado por defecto es capture@10% bajo cross-validation ESPACIAL.'
          : 'ProspectMap makes first-class the TWO ways a prospectivity map lies: (1) conditional-independence violation — correlated favourable layers double-count and inflate the posterior; (2) CV inflation — a random split leaks neighbours of training deposits and the AUC looks fantastic. The default reported number is capture@10% under SPATIAL cross-validation.'}
      </Callout>

      <h2>{es ? 'Qué es y qué NO es' : 'What it is and is NOT'}</h2>
      <ul>
        <li>{es ? 'ES generación de blancos de exploración (decision support de DÓNDE mirar próximo).' : 'It IS exploration target generation (decision support for WHERE to look next).'}</li>
        <li>{es ? 'NO es una estimación de recursos/reservas (la frontera JORC / NI 43-101), ni una garantía de "perfora aquí".' : 'It is NOT a resource/reserve estimate (the JORC / NI 43-101 boundary), nor a "drill here" guarantee.'}</li>
        <li>{es ? 'Las etiquetas de depósitos son presence-only: no hay "ausencia" confirmada; las negativas se muestrean, nunca se observan.' : 'Deposit labels are presence-only: there is no confirmed "absence"; negatives are sampled, never observed.'}</li>
      </ul>

      <h2>{es ? 'Datos' : 'Data'}</h2>
      <p>{es
        ? 'Las áreas de estudio de esta versión son SINTÉTICAS (campos geoestadísticos suaves + una red de fallas plantada + depósitos por un proceso de Poisson no homogéneo sobre una prospectividad latente conocida), claramente etiquetadas. Son el único caso con ground-truth conocido, así que son los controles exactos. Datasets abiertos reales (Lawley et al. 2022 Zn-Pb de USGS ScienceBase '
        : 'The study areas in this version are SYNTHETIC (smooth geostatistical fields + a planted fault network + deposits by an inhomogeneous-Poisson process on a known latent prospectivity), clearly labelled. They are the only data with known ground truth, so they are the exact controls. Real open datasets (Lawley et al. 2022 Zn-Pb from USGS ScienceBase '}
        <Cite id="lawley2022" paren />{es ? ', Geoscience Australia CC-BY) son el siguiente paso; el pipeline acepta un cubo real de forma idéntica.' : ', Geoscience Australia CC-BY) are the next step; the pipeline accepts a real cube identically.'}</p>

      <ReferenceList />
    </article>
  );
}
