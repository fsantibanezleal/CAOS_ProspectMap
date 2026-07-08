import type { Citation } from '@fasl-work/caos-app-shell';

// The references ProspectMap's methodology rests on, Weights-of-Evidence, the conditional-independence test, the
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
    citation: 'Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929. doi:10.1111/ecog.02881.',
  },
  {
    id: 'rodriguezgaliano2015',
    label: 'Rodriguez-Galiano et al. 2015',
    citation: 'Rodriguez-Galiano, V., Sanchez-Castillo, M., Chica-Olmo, M. & Chica-Rivas, M. (2015). Machine learning predictive models for mineral prospectivity: an evaluation of neural networks, random forest, regression trees and support vector machines. Ore Geology Reviews, 71, 804–818. doi:10.1016/j.oregeorev.2015.01.001.',
  },
  {
    id: 'xiongzuo2021',
    label: 'Xiong & Zuo 2021',
    citation: 'Xiong, Y. & Zuo, R. (2021). A positive and unlabeled learning algorithm for mineral prospectivity mapping. Computers & Geosciences, 147, 104667. doi:10.1016/j.cageo.2020.104667.',
  },
  {
    id: 'elkannoto2008',
    label: 'Elkan & Noto 2008',
    citation: 'Elkan, C. & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. Proc. 14th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD), 213–220. doi:10.1145/1401890.1401920.',
  },
  {
    id: 'kiryo2017',
    label: 'Kiryo et al. 2017',
    citation: 'Kiryo, R., Niu, G., du Plessis, M.C. & Sugiyama, M. (2017). Positive-Unlabeled learning with non-negative risk estimator. Advances in Neural Information Processing Systems 30 (NeurIPS). arXiv:1703.00593.',
  },
  {
    id: 'angelopoulos2021',
    label: 'Angelopoulos & Bates 2021',
    citation: 'Angelopoulos, A.N. & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.',
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
