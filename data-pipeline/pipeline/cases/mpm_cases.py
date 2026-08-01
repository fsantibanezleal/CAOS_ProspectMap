"""ProspectMap cases spanning CATEGORIES (the prospectivity problem-type taxonomy). The App shows ONE selected case;
Experiments/Benchmark show cross-case summaries by category. The cases MIRROR the SPA's frontend/src/mpm/cases.ts
(ids + metadata kept in lock-step; a test cross-checks them against the baked case-results.json). Every case is a
SYNTHETIC study area (clearly labelled), generated deterministically from its SPEC by the TypeScript engine, the only
data with KNOWN ground truth, so the controls are exact (C-RECOVER weight recovery, C-NEGATIVE noise, C-CIVIOLATE the
conditional-dependence trap, C-SATURATE the analytic limit). Real open datasets are a documented next step."""
from __future__ import annotations

from dataclasses import dataclass

CAT_TERRANE = "deposit-type terrane (the geological setting)"
CAT_DATA = "data / validation regime (evidence richness)"
CAT_CONTROL = "control (oracle / negative control)"
CAT_REAL = "real open dataset (a published mineral system)"

_NX = 100
_NY = 100
_ND = 85


@dataclass(frozen=True)
class Case:
    id: str                     # matches frontend/src/mpm/cases.ts
    name: str
    category: str
    n_layers: int               # number of evidence layers fed to WofE
    expected_band: str = ""
    validation_anchor: str = ""
    real_or_synthetic: str = "synthetic"
    deposit_type: str = ""
    nx: int = _NX
    ny: int = _NY
    cell_km: float = 1.0
    n_deposits: int = _ND


CASES: list[Case] = [
    Case("K-PORPHYRY", "Porphyry-Cu-like terrane", CAT_TERRANE, 4,
         "magnetic high + geochem anomaly + proximity to structure favourable; radiometrics uninformative",
         "mag/geochem/struct positive contrast + clear studentized-C; ROC AUC well above 0.5",
         deposit_type="porphyry Cu (synthetic)"),
    Case("K-OROGENIC", "Orogenic-Au-like terrane", CAT_TERRANE, 4,
         "structure (shear-zone proximity) dominates; geochem secondary; magnetics uninformative",
         "struct contrast > geochem contrast; mag near zero", deposit_type="orogenic Au (synthetic)"),
    Case("K-VMS", "VMS-like terrane", CAT_TERRANE, 4,
         "magnetic + geochem co-located; radiometrics weak; structure uninformative",
         "mag the strongest contrast; struct near zero", deposit_type="VMS (synthetic)"),
    Case("K-IOCG", "IOCG-like terrane", CAT_TERRANE, 4,
         "a strong magnetic signature with multi-layer support (the IOCG fingerprint)",
         "mag dominant; all four layers contribute positively", deposit_type="IOCG (synthetic)"),
    Case("D-RICH", "Evidence-rich area", CAT_DATA, 4,
         "all four layers informative -> a high-skill posterior (watch the CI + spatial-CV)",
         "ROC AUC high; capture@10% large; stacking strong layers can violate CI (omnibus flags it)"),
    Case("D-SPARSE", "Evidence-poor area", CAT_DATA, 4,
         "only a weak geochem signal -> little real skill (honest low AUC)",
         "ROC AUC only modestly above 0.5; the product does not manufacture confidence"),
    Case("C-NEGATIVE", "Negative control - uninformative layers", CAT_CONTROL, 4,
         "no layer is associated with the (randomly-placed) deposits",
         "all contrasts ~ 0, |studentized-C| < 1.96, ROC AUC ~ 0.5 - no skill from noise",
         real_or_synthetic="analytic control"),
    Case("C-CIVIOLATE", "CI-violation oracle - a correlated duplicate", CAT_CONTROL, 3,
         "mag and its near-duplicate double-count the same signal -> the WofE posterior is inflated",
         "the omnibus test fails (T > N(D), CI ratio < 1, z > 0); logistic regression is the CI-free alternative (calibration readout not yet in-app)",
         real_or_synthetic="analytic control"),
    Case("C-RECOVER", "Positive control - planted weight recovery", CAT_CONTROL, 4,
         "well-separated planted weights mag > geochem > struct",
         "WofE recovers the ORDERING: contrast(mag) > contrast(geochem) > contrast(struct) > ~contrast(noise)",
         real_or_synthetic="analytic control"),
    Case("C-SATURATE", "Analytic limit - a near-perfect single layer", CAT_CONTROL, 1,
         "a single dominant layer drives a near-saturated posterior",
         "high W+ + posterior near 1 inside the pattern, no numerical blow-up (Haldane guard); AUC high",
         real_or_synthetic="analytic control"),
    Case("REAL-USMVT", "US Midcontinent MVT Zn-Pb belt (Lawley 2022, CMMI)", CAT_REAL, 6,
         "real published data: 4 measured geophysics layers (mag, grav, LAB tomography, satellite gravity) + 2 derived "
         "proximity layers (fault, passive-margin), over the US Midcontinent MVT belt with 858 real Pb-Zn occurrence cells",
         "our browser WofE recomputation (NOT the published H3 + gradient-boosting model); naive AUC is inflated by the "
         "strong deposit clustering (Tri-State district), so the reported skill is the SPATIAL-CV AUC (near chance), an honest result",
         real_or_synthetic="real (open dataset)",
         deposit_type="sediment-hosted Zn-Pb (MVT)",
         nx=144, ny=176, cell_km=5.364, n_deposits=858),
]


def descriptor_row(c: Case) -> dict:
    """The CONTRACT-1 case-bundle row for a case (used by the pipeline's contract check)."""
    return {"case_id": c.id, "nx": c.nx, "ny": c.ny, "cell_km": c.cell_km, "n_layers": c.n_layers,
            "n_deposits": c.n_deposits, "real_or_synthetic": c.real_or_synthetic, "deposit_type": c.deposit_type}
