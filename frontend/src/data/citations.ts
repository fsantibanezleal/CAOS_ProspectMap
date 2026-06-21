import type { Citation } from '@fasl-work/caos-app-shell';

// The references ProspectMap's methodology rests on — Weights-of-Evidence, the conditional-independence test, the
// WofE->logistic-regression bridge, the honest spatial-CV / capture-curve validation, and the real Zn-Pb dataset.
export const CITATIONS: Citation[] = [
  {
    id: 'bonhamcarter1989',
    label: 'Bonham-Carter et al. 1989',
    citation: 'Bonham-Carter, G.F., Agterberg, F.P. & Wright, D.F. (1989). Weights of evidence modelling: a new approach to mapping mineral potential. Geol. Survey Canada Paper 89-9, 171–183.',
  },
  {
    id: 'bonhamcarter1994',
    label: 'Bonham-Carter 1994',
    citation: 'Bonham-Carter, G.F. (1994). Geographic Information Systems for Geoscientists: Modelling with GIS. Pergamon (Computer Methods in the Geosciences 13), ch. 9.',
  },
  {
    id: 'agterbergcheng2002',
    label: 'Agterberg & Cheng 2002',
    citation: 'Agterberg, F.P. & Cheng, Q. (2002). Conditional Independence Test for Weights-of-Evidence Modeling. Natural Resources Research, 11(4), 249–255.',
  },
  {
    id: 'agterberg1990',
    label: 'Agterberg & Bonham-Carter 1990',
    citation: 'Agterberg, F.P. & Bonham-Carter, G.F. (1990). Deriving weights of evidence from geoscience contour maps for the prediction of discrete events. 22nd APCOM Symposium, 381–395.',
  },
  {
    id: 'chungfabbri2003',
    label: 'Chung & Fabbri 2003',
    citation: 'Chung, C.-J. & Fabbri, A.G. (2003). Validation of spatial prediction models for landslide hazard mapping. Natural Hazards, 30(3), 451–472.',
  },
  {
    id: 'roberts2017',
    label: 'Roberts et al. 2017',
    citation: 'Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929.',
  },
  {
    id: 'carranza2009',
    label: 'Carranza 2009',
    citation: 'Carranza, E.J.M. (2009). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS. Handbook of Exploration & Environmental Geochemistry, 11. Elsevier.',
  },
  {
    id: 'lawley2022',
    label: 'Lawley et al. 2022',
    citation: 'Lawley, C.J.M. et al. (2022). Data-driven prospectivity modelling of sediment-hosted Zn-Pb mineral systems and their critical raw materials. Ore Geology Reviews, 141, 104635.',
  },
];
