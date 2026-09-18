// The learned-model feature contract (browser side), mirrors data-pipeline/pipeline/model/learned.py :: MPM_FEATURES.
// The per-cell evidence feature vector fed to the mpm-classifier ONNX + the geology-ood autoencoder. The single source
// of truth for the order; a drift vs the Python contract would mis-feed the model.
import type { LearnedFile } from './artifacts.ts';

export const MPM_FEATURES = ['mag', 'rad', 'geochem', 'struct'] as const;
export const N_FEATURES = MPM_FEATURES.length;

// The real-data lane (US Midcontinent MVT cube, pipeline/real_usmvt.py) has its own 6-feature space, so the synthetic
// 4-feature ONNX cannot be applied to it. The real models (pipeline/real_learned.py -> *-real.onnx) are trained on these.
export const REAL_FEATURES = ['mag', 'grav', 'lab', 'satgrav', 'faultprox', 'marginprox'] as const;
export const N_REAL_FEATURES = REAL_FEATURES.length;

export type LearnedLane = 'synthetic' | 'real';

/**
 * The protocol behind a lane's learned-vs-WofE head-to-head, in words. Two AUCs shown side by side must come from one
 * protocol and say which one (GitHub issue #41: the real lane once stored a WofE AUC fitted without cross-validation
 * under `spatial_cv`, beside a cross-validated MLP AUC computed on a different cell set).
 *  - real: declared by pm-learned-real.json (`classifier.protocol`, written by data-pipeline/pipeline/real_learned.py).
 *  - synthetic: pm-learned.json does not declare it; it is fixed by science/gen_train.mjs + science/train_mpm.py.
 */
export function headToHeadProtocol(lane: LearnedLane, learned: LearnedFile | null, es: boolean): string {
  if (lane === 'real') {
    const p = learned?.classifier?.protocol;
    if (!p) {
      return es
        ? 'El archivo de métricas no declara su protocolo, así que sus valores no se comparan.'
        : 'The metrics file does not declare its protocol, so its values are not compared.';
    }
    return es
      ? `Ambos modelos con un único protocolo: las particiones del motor (bloques de ${p.block_cells}x${p.block_cells} celdas, ${p.k} folds), cada modelo ajustado solo con las celdas de los folds de entrenamiento (en WofE: umbrales, pesos y prior), cada una de las ${p.n_cells} celdas del mapa (${p.n_deposit_cells} con depósito) puntuada una vez mientras su fold está retenido, y los scores retenidos agrupados en un único ROC AUC.`
      : `Both models under one protocol: the engine's folds (${p.block_cells}x${p.block_cells}-cell blocks, ${p.k} folds), each model fitted on the training folds' cells only (for WofE: thresholds, weights and prior), each of the ${p.n_cells} map cells (${p.n_deposit_cells} deposit cells) scored once while its fold is held out, the held-out scores pooled into one ROC AUC.`;
  }
  return es
    ? 'Ambos modelos con un único protocolo: las filas etiquetadas (celdas con depósito y negativos muestreados lejos de los depósitos) de los cinco casos sintéticos de entrenamiento, las particiones espaciales del motor (bloques de 20x20 celdas, 5 folds), cada modelo ajustado con los folds de entrenamiento, y los scores retenidos agrupados en un único ROC AUC.'
    : "Both models under one protocol: the labelled rows (deposit cells and negatives sampled away from the deposits) of the five synthetic training cases, the engine's spatial folds (20x20-cell blocks, 5 folds), each model fitted on the training folds, the held-out scores pooled into one ROC AUC.";
}

/** Whether a lane's `spatial_cv` values form a like-for-like pair: the synthetic protocol is fixed by its pipeline; a
 * real-lane file must declare its protocol (an older file stored the no-CV WofE AUC there, GitHub issue #41). */
export function headToHeadDeclared(lane: LearnedLane, learned: LearnedFile | null): boolean {
  return lane === 'synthetic' ? learned != null : learned?.classifier?.protocol != null;
}

/** The verdict of `spatial_cv`, the one like-for-like pair in a learned-metrics file. null when there is none. */
export function headToHeadVerdict(lane: LearnedLane, learned: LearnedFile | null, es: boolean): string | null {
  if (!headToHeadDeclared(lane, learned)) return null;
  const s = learned?.classifier?.spatial_cv;
  if (!s || typeof s.mlp_roc_auc !== 'number' || typeof s.wofe_roc_auc !== 'number' || !s.winner) return null;
  const m = s.mlp_roc_auc.toFixed(3);
  const w = s.wofe_roc_auc.toFixed(3);
  if (s.winner === 'wofe') {
    return es
      ? `Con este protocolo WofE ordena mejor las celdas retenidas (AUC ${w} frente a ${m} del MLP).`
      : `Under this protocol WofE ranks the held-out cells better (AUC ${w} against ${m} for the MLP).`;
  }
  if (s.winner === 'mlp') {
    return es
      ? `Con este protocolo el MLP ordena mejor las celdas retenidas (AUC ${m} frente a ${w} de WofE).`
      : `Under this protocol the MLP ranks the held-out cells better (AUC ${m} against ${w} for WofE).`;
  }
  return es
    ? `Con este protocolo ambos quedan a menos de 0.005 de AUC (${m} y ${w}): un empate.`
    : `Under this protocol the two are within 0.005 AUC of each other (${m} and ${w}): a tie.`;
}

/**
 * What a held-out AUC under the engine's folds does and does not show. The engine's spatial folds interleave blocks
 * (fold = blockId % k), so every held-out block borders training blocks; when the file carries the distance-to-known-
 * deposit baseline scored under the same protocol, say where the MLP stands against it (a model must beat it to claim
 * it learned geology rather than proximity).
 */
export function headToHeadContext(lane: LearnedLane, learned: LearnedFile | null, es: boolean): string {
  const block = learned?.classifier?.protocol?.block_cells ?? 20;
  const parts: string[] = [];
  const s = learned?.classifier?.spatial_cv;
  const n = s?.distance_null_roc_auc;
  if (headToHeadDeclared(lane, learned) && typeof n === 'number' && typeof s?.mlp_roc_auc === 'number') {
    const d = Math.abs(s.mlp_roc_auc - n).toFixed(3);
    const above = s.mlp_roc_auc > n;
    parts.push(es
      ? `Una línea base de distancia al depósito conocido más cercano, que no aprende geología, evaluada igual, alcanza ${n.toFixed(3)}${above ? `, y el MLP la supera por ${d}` : `, ${d} por encima del MLP`}: un modelo debe superarla para afirmar que aprendió geología y no proximidad.`
      : `A distance-to-known-deposit baseline that learns no geology, scored the same way, reaches ${n.toFixed(3)}${above ? `, and the MLP exceeds it by ${d}` : `, ${d} above the MLP`}: a model must beat it to claim it learned geology rather than proximity.`);
  }
  parts.push(es
    ? `Las particiones del motor intercalan bloques de ${block}x${block} celdas, así que cada bloque retenido limita con bloques de entrenamiento.`
    : `The engine's folds interleave ${block}x${block}-cell blocks, so every held-out block borders training blocks.`);
  if (lane === 'real') {
    parts.push(es
      ? 'El head-to-head con particiones contiguas de la página Benchmark es la prueba más estricta de transferencia a terreno no visto.'
      : 'The contiguous-fold head-to-head on the Benchmark page is the stricter test of transfer to unseen ground.');
  }
  return parts.join(' ');
}
