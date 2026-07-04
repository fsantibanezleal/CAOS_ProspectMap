// ProspectMap cases by CATEGORY (the prospectivity problem-type taxonomy). The App shows ONE selected case;
// Experiments/Benchmark show cross-case summaries. Every case is a SYNTHETIC study area (clearly labelled) generated
// deterministically from its SPEC — the only data with KNOWN ground truth, so the controls are exact: C-RECOVER
// (WofE must recover the planted weight ordering), C-NEGATIVE (uninformative ⇒ AUC≈0.5), C-CIVIOLATE (a correlated
// duplicate ⇒ the omnibus test fails on purpose), C-SATURATE (the analytic W⁺→∞ limit). Real open datasets
// (Lawley-2022 USGS Zn-Pb, GA CC-BY) are a documented next step; the pipeline accepts a real cube identically.
//
// This file is the SOURCE OF TRUTH for the cases; the Node bake (science/bake_cases.mjs) runs the SAME engine over it,
// and data-pipeline/pmlab/cases/mpm_cases.py mirrors it (a test cross-checks the ids against the baked case-results).

import type { SynthSpec } from './synth.ts';

export const CAT_TERRANE = 'deposit-type terrane (the geological setting)';
export const CAT_DATA = 'data / validation regime (evidence richness)';
export const CAT_CONTROL = 'control (oracle / negative control)';

export interface MPMCase {
  id: string;
  name: string;
  category: string;
  spec: SynthSpec;
  /** the evidence layers fed to WofE for this case (a subset/superset of the spec's layers). */
  layerIds: string[];
  expectedBand: string;
  validationAnchor: string;
  realOrSynthetic: 'synthetic' | 'analytic control' | 'real (open dataset)';
}

const NX = 100;
const NY = 100;
const ND = 85;
const GAIN = 7;
const ALL = ['mag', 'rad', 'geochem', 'struct'];

function L(id: string, weight: number, favHigh = true, coarse = 13) {
  return { id, weight, favHigh, coarse };
}

export const CASES: MPMCase[] = [
  {
    id: 'K-PORPHYRY',
    name: 'Porphyry-Cu-like terrane',
    category: CAT_TERRANE,
    spec: { nx: NX, ny: NY, seed: 101, nDeposits: ND, gain: GAIN, layers: [L('mag', 1.8), L('rad', 0.0), L('geochem', 1.6), L('struct', 1.0)] },
    layerIds: ALL,
    expectedBand: 'magnetic high + geochem anomaly + proximity to structure are favourable; radiometrics uninformative',
    validationAnchor: 'mag/geochem/struct have positive contrast and clear studentized-C; ROC AUC well above 0.5',
    realOrSynthetic: 'synthetic',
  },
  {
    id: 'K-OROGENIC',
    name: 'Orogenic-Au-like terrane',
    category: CAT_TERRANE,
    spec: { nx: NX, ny: NY, seed: 102, nDeposits: ND, gain: GAIN, layers: [L('mag', 0.0), L('rad', 0.4), L('geochem', 1.4), L('struct', 2.0)] },
    layerIds: ALL,
    expectedBand: 'structure (shear-zone proximity) dominates; geochem secondary; magnetics uninformative',
    validationAnchor: 'struct contrast > geochem contrast; mag near zero; honest per-layer ranking',
    realOrSynthetic: 'synthetic',
  },
  {
    id: 'K-VMS',
    name: 'VMS-like terrane',
    category: CAT_TERRANE,
    spec: { nx: NX, ny: NY, seed: 103, nDeposits: ND, gain: GAIN, layers: [L('mag', 2.0), L('rad', 0.8), L('geochem', 1.5), L('struct', 0.0)] },
    layerIds: ALL,
    expectedBand: 'magnetic + geochem co-located; radiometrics weakly favourable; structure uninformative',
    validationAnchor: 'mag the strongest contrast; struct near zero',
    realOrSynthetic: 'synthetic',
  },
  {
    id: 'K-IOCG',
    name: 'IOCG-like terrane',
    category: CAT_TERRANE,
    spec: { nx: NX, ny: NY, seed: 104, nDeposits: ND, gain: GAIN, layers: [L('mag', 2.2), L('rad', 0.6), L('geochem', 1.2), L('struct', 0.8)] },
    layerIds: ALL,
    expectedBand: 'a strong magnetic signature with multi-layer support (the IOCG fingerprint)',
    validationAnchor: 'mag dominant; all four layers contribute positively',
    realOrSynthetic: 'synthetic',
  },
  {
    id: 'D-RICH',
    name: 'Evidence-rich area',
    category: CAT_DATA,
    spec: { nx: NX, ny: NY, seed: 201, nDeposits: ND, gain: GAIN, layers: [L('mag', 1.6), L('rad', 1.2), L('geochem', 1.6), L('struct', 1.4)] },
    layerIds: ALL,
    expectedBand: 'all four layers informative ⇒ a high-skill posterior',
    validationAnchor: 'ROC AUC high; capture@10% large; but watch the spatial-CV gap',
    realOrSynthetic: 'synthetic',
  },
  {
    id: 'D-SPARSE',
    name: 'Evidence-poor area',
    category: CAT_DATA,
    spec: { nx: NX, ny: NY, seed: 202, nDeposits: ND, gain: GAIN, layers: [L('mag', 0.0), L('rad', 0.0), L('geochem', 0.7), L('struct', 0.0)] },
    layerIds: ALL,
    expectedBand: 'only a weak geochem signal ⇒ little real skill (honest low AUC)',
    validationAnchor: 'ROC AUC only modestly above 0.5; the product does NOT manufacture confidence',
    realOrSynthetic: 'synthetic',
  },
  {
    id: 'C-NEGATIVE',
    name: 'Negative control — uninformative layers',
    category: CAT_CONTROL,
    spec: { nx: NX, ny: NY, seed: 301, nDeposits: ND, gain: GAIN, layers: [L('mag', 0.0), L('rad', 0.0), L('geochem', 0.0), L('struct', 0.0)] },
    layerIds: ALL,
    expectedBand: 'no layer is associated with the (randomly-placed) deposits',
    validationAnchor: 'all contrasts ≈ 0, |studentized-C| < 1.96, ROC AUC ≈ 0.5 — no skill from noise',
    realOrSynthetic: 'analytic control',
  },
  {
    id: 'C-CIVIOLATE',
    name: 'CI-violation oracle — a correlated duplicate',
    category: CAT_CONTROL,
    spec: {
      nx: NX, ny: NY, seed: 302, nDeposits: ND, gain: GAIN,
      layers: [L('mag', 1.8), L('rad', 0.0), L('geochem', 1.2), L('struct', 0.0)],
      duplicate: { srcId: 'mag', id: 'magdup', noise: 0.05 },
    },
    layerIds: ['mag', 'magdup', 'geochem'],
    expectedBand: 'mag and its near-duplicate double-count the same signal ⇒ the WofE posterior is inflated',
    validationAnchor: 'the omnibus test fails (T > N(D), CI ratio < 1, z > 0); logistic regression is the CI-free alternative (calibration readout not yet in-app)',
    realOrSynthetic: 'analytic control',
  },
  {
    id: 'C-RECOVER',
    name: 'Positive control — planted weight recovery',
    category: CAT_CONTROL,
    spec: { nx: NX, ny: NY, seed: 303, nDeposits: ND, gain: GAIN, layers: [L('mag', 2.4), L('rad', 0.0), L('geochem', 1.4), L('struct', 0.7)] },
    layerIds: ALL,
    expectedBand: 'well-separated planted weights mag > geochem > struct',
    validationAnchor: 'WofE recovers the planted ORDERING: contrast(mag) > contrast(geochem) > contrast(struct) > ~contrast(noise)',
    realOrSynthetic: 'analytic control',
  },
  {
    id: 'C-SATURATE',
    name: 'Analytic limit — a near-perfect single layer',
    category: CAT_CONTROL,
    spec: { nx: NX, ny: NY, seed: 304, nDeposits: ND, gain: 12, layers: [L('mag', 4.0), L('rad', 0.0), L('geochem', 0.0), L('struct', 0.0)] },
    layerIds: ['mag'],
    expectedBand: 'a single dominant layer drives a near-saturated posterior',
    validationAnchor: 'high W⁺ + posterior near 1 inside the pattern, no numerical blow-up (Haldane guard); AUC high',
    realOrSynthetic: 'analytic control',
  },
];

const _byId = new Map(CASES.map((c) => [c.id, c]));
export function caseById(id: string): MPMCase {
  const c = _byId.get(id);
  if (!c) throw new Error(`unknown case: ${id}. known: ${[...(_byId.keys())].join(', ')}`);
  return c;
}
