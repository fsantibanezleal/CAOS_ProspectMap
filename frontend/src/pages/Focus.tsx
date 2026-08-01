// ADR-0070 scenario focus view: one selected area, full page, nothing competing with the map.
//
// ADDITIVE. The App (Tool.tsx) keeps every tab and all its explanation; this route is a second way to look
// at the SAME area through the SAME live WofE recomputation. It renders OUTSIDE <AppShell> on purpose: the
// shell header and footer are exactly the chrome a focus view exists to escape.
//
// THE HONEST NEGATIVE IS THE HEADLINE HERE, NOT A FOOTNOTE. This product's finding is that regional
// geophysics alone has little spatial-transfer skill on a clustered MVT belt: the spatially-blocked CV AUC
// sits near chance while the random-CV AUC looks good. A focus view that showed a confident-looking
// posterior map without that number beside it would be the single most misleading screen in the line, so
// the state named on the stage IS the spatial-CV verdict and the inflation gap is a first-class HUD value.

import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useShellLang } from '@fasl-work/caos-app-shell';
import {
  analyzeCube, bestWeights, CASES, caseById, makeSyntheticArea, posterior,
  cubeFromFile, depositSet, REAL_CASES, type Cube,
} from '../mpm/index.ts';
import { loadRealCube } from '../lib/artifacts.ts';
import { MapView } from '../viz/MapView.tsx';

/** What the spatially-blocked CV AUC MEANS, said on the stage.
 *
 *  0.5 is chance. The bands below are the plain reading of a ROC AUC and are labelled as such; they are NOT
 *  a published acceptance threshold for prospectivity models, and the wording avoids implying one. */
function skillState(spatialAuc: number, gap: number, es: boolean): { label: string; text: string } {
  if (!Number.isFinite(spatialAuc)) {
    return { label: es ? 'Sin validacion espacial' : 'No spatial validation',
             text: es ? 'No hay suficientes bloques con ocurrencias para una validacion cruzada espacial en esta area.'
                      : 'There are not enough blocks containing occurrences to run a spatial cross-validation on this area.' };
  }
  if (spatialAuc < 0.6) {
    return {
      label: es ? 'Sin habilidad de transferencia' : 'No transfer skill',
      text: es
        ? `Con validacion cruzada por bloques espaciales el AUC es ${spatialAuc.toFixed(2)}, practicamente azar. El mapa describe donde estan las ocurrencias conocidas, pero no predice bloques no vistos: la brecha de ${gap.toFixed(2)} frente a la CV aleatoria es autocorrelacion espacial, no habilidad.`
        : `Under spatially-blocked cross-validation the AUC is ${spatialAuc.toFixed(2)}, essentially chance. The map describes where the known occurrences are, but it does not predict unseen blocks: the ${gap.toFixed(2)} gap against random CV is spatial autocorrelation, not skill.`,
    };
  }
  if (spatialAuc < 0.7) {
    return {
      label: es ? 'Habilidad marginal' : 'Marginal skill',
      text: es
        ? `AUC espacial ${spatialAuc.toFixed(2)}: algo por encima del azar, pero la brecha de ${gap.toFixed(2)} contra la CV aleatoria muestra cuanto del desempeno aparente proviene de ocurrencias vecinas.`
        : `Spatial AUC ${spatialAuc.toFixed(2)}: above chance, but the ${gap.toFixed(2)} gap against random CV shows how much of the apparent performance comes from neighbouring occurrences.`,
    };
  }
  return {
    label: es ? 'Habilidad real de transferencia' : 'Real transfer skill',
    text: es
      ? `AUC espacial ${spatialAuc.toFixed(2)}: el modelo ordena bloques que no vio, no solo los vecinos de ocurrencias conocidas.`
      : `Spatial AUC ${spatialAuc.toFixed(2)}: the model ranks blocks it did not see, not merely the neighbours of known occurrences.`,
  };
}

export default function Focus() {
  const { caseId } = useParams();
  const es = useShellLang() === 'es';

  // The route id is resolved against BOTH registries: on the real lane the selected scenario is a surveyed
  // belt, not a synthetic case, and resolving only against CASES would open something the user never picked.
  const real = useMemo(() => REAL_CASES.find((c) => c.id === caseId) ?? null, [caseId]);
  const synth = useMemo(() => (real ? null : caseById(caseId ?? CASES[0].id)), [caseId, real]);

  const [realCube, setRealCube] = useState<Cube | null>(null);
  const [realMeta, setRealMeta] = useState<{ citation: string; license: string; honesty: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!real) { setRealCube(null); return; }
    let cancel = false;
    loadRealCube(real.file).then((f) => { if (!cancel) { setRealMeta(f); setRealCube(cubeFromFile(f)); setErr(null); } })
      .catch(() => { if (!cancel) setErr('cube'); });
    return () => { cancel = true; };
  }, [real]);

  const synthCube = useMemo(() => (synth ? makeSyntheticArea(synth.spec).cube : null), [synth]);
  const cube: Cube | null = real ? realCube : synthCube;
  const layerIds = real ? real.layerIds : (synth?.layerIds ?? []);

  const [off, setOff] = useState<Record<string, boolean>>({});
  useEffect(() => { setOff({}); }, [caseId]);
  const activeIds = useMemo(() => layerIds.filter((id) => !off[id]), [layerIds, off]);

  // The SAME live recomputation the App runs: WofE weights, posterior, and a spatially-blocked CV.
  const analysis = useMemo(
    () => (cube && activeIds.length ? analyzeCube(cube, activeIds) : null),
    [cube, activeIds],
  );
  const post = useMemo(() => {
    if (!cube || !activeIds.length) return null;
    const best = activeIds.map((id) => bestWeights(cube, id));
    return posterior(cube, best.map((b) => b.pattern), best.map((b) => b.weights));
  }, [cube, activeIds]);

  const name = real ? real.name : (synth?.name ?? '');
  const id = real ? real.id : (synth?.id ?? '');
  const st = analysis
    ? skillState(analysis.cv.spatialAuc, analysis.cv.inflationGap, es)
    : { label: es ? 'Cargando' : 'Loading', text: es ? 'Recalculando WofE sobre el cubo.' : 'Recomputing WofE over the cube.' };

  const hud = analysis ? [
    { v: analysis.cv.spatialAuc.toFixed(2), l: es ? 'AUC espacial' : 'spatial AUC', tone: 'accent' },
    { v: analysis.cv.randomAuc.toFixed(2), l: es ? 'AUC aleatoria' : 'random AUC' },
    { v: analysis.cv.inflationGap.toFixed(2), l: es ? 'brecha' : 'inflation gap', tone: 'blue' },
    { v: `${analysis.nDeposits}`, l: es ? 'ocurrencias' : 'occurrences' },
    { v: `${(analysis.priorProb * 100).toFixed(1)}%`, l: es ? 'prior' : 'prior' },
    { v: analysis.posteriorSummary.max.toFixed(3), l: es ? 'posterior max' : 'posterior max' },
  ] : [];

  const range: [number, number] = post
    ? [analysis?.posteriorSummary.min ?? 0, analysis?.posteriorSummary.max ?? 1]
    : [0, 1];

  return (
    <div className="pmf">
      <div className="pmf-stage">
        {cube && post
          ? <MapView nx={cube.nx} ny={cube.ny} field={post.prob} range={range}
                     deposits={[...depositSet(cube)]} height={0}
                     lang={es ? 'es' : 'en'} valueLabel="P" />
          : <div className="pmf-empty">{err
              ? (es ? 'No se pudo cargar el cubo real.' : 'The real cube could not be loaded.')
              : (es ? 'Calculando…' : 'Computing…')}</div>}

        <div className="pmf-badge">
          <div className="pmf-badge-t">{st.label}</div>
          <div className="pmf-badge-d">{st.text}</div>
        </div>

        <div className="pmf-hud">
          {hud.map((h) => (
            <div className="pmf-hud-item" key={h.l}>
              <div className={`pmf-hud-v${h.tone ? ' ' + h.tone : ''}`}>{h.v}</div>
              <div className="pmf-hud-l">{h.l}</div>
            </div>
          ))}
        </div>

        <Link className="pmf-exit" to="/">{es ? 'Volver a la app' : 'Back to the app'}</Link>
      </div>

      <aside className="pmf-rail">
        <div className="pmf-rail-h">
          <div>
            <div className="pmf-title">{name}</div>
            <div className="pmf-sub">{id} · {real ? 'real' : 'synthetic'}</div>
          </div>
        </div>

        {real && (
          <div className="pmf-prov">
            {es ? 'Datos reales publicados: ' : 'Real published data: '}{realMeta?.citation ?? real.name}
            {realMeta?.license ? ` · ${realMeta.license}` : ''}
            
          </div>
        )}

        <div className="pmf-lbl">{es ? 'Capas de evidencia' : 'Evidence layers'}</div>
        <div className="pmf-layers">
          {layerIds.map((lid) => (
            <button key={lid} className={off[lid] ? '' : 'on'}
                    onClick={() => setOff((o) => ({ ...o, [lid]: !o[lid] }))}>{lid}</button>
          ))}
        </div>

        <div className="pmf-note">
          {es
            ? 'El posterior se recalcula EN VIVO con Weights of Evidence sobre las capas activas, no es un mapa horneado. La CV espacial por bloques es la cifra que importa: la CV aleatoria reparte celdas vecinas entre entrenamiento y prueba, y en un cinturon con ocurrencias agrupadas eso mide autocorrelacion, no prediccion. Este posterior es NUESTRA recomputacion, no el modelo publicado.'
            : 'The posterior is recomputed LIVE with Weights of Evidence over the active layers; it is not a baked map. The spatially-blocked CV is the number that matters: random CV splits neighbouring cells across train and test, and on a belt with clustered occurrences that measures autocorrelation rather than prediction. This posterior is OUR recomputation, not the published model.'}
        </div>

        <div className="pmf-cases">
          {[...REAL_CASES, ...CASES].slice(0, 12).map((c) => (
            <Link key={c.id} to={`/focus/${c.id}`} className={c.id === id ? 'on' : ''}>{c.id}</Link>
          ))}
        </div>
      </aside>
    </div>
  );
}
