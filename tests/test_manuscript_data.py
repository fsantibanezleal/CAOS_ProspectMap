"""The technical report's data file (manuscripts/prospectivity/data/pm.json) and its builder.

Version 1.1 of the report plotted each layer's binarization threshold tStar under the label "studentized C/s(C)" and
read a no-CV AUC stored under a cross-validation key (GitHub issues #41 and #44). pm.json is now built from the
artifacts with a key per protocol, and these tests keep it that way: the committed file must equal a fresh build (so
it cannot drift from a re-bake), the per-layer statistics are the engine's, and the exact small-count statistics
match scipy's (values recorded from scipy 1.18: fisher_exact, contingency.odds_ratio(kind="conditional"))."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "manuscripts" / "prospectivity" / "data" / "build_pm.py"


@pytest.fixture(scope="module")
def pm():
    spec = importlib.util.spec_from_file_location("build_pm", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_pm_json_is_a_fresh_build(pm):
    committed = json.loads((BUILDER.parent / "pm.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(pm.build())), "pm.json is stale: run data/build_pm.py"


def test_layers_carry_the_studentized_contrast_not_the_threshold(pm):
    real = pm.build()["real"]
    for layer in real["woe_layers"]:
        assert layer["studentized_C"] == pytest.approx(layer["contrast_C"] / layer["s_C"], abs=1e-12)
        assert layer["studentized_C"] != layer["tStar_threshold"]
    # the six real layers are all above both 5% levels as baked (v1.1 claimed all were below 1.645)
    assert min(layer["studentized_C"] for layer in real["woe_layers"]) > 1.96


# (counts n_B_D, n_B_notD, n_notB_D, n_notB_notD) -> scipy's one-sided Fisher p and conditional 95% interval for ln OR
SCIPY = {
    "mag": ((855, 24134, 3, 352), 1.9032e-03, (0.3407, 3.0103)),
    "grav": ((858, 23655, 0, 831), 2.2839e-13, (2.0961, None)),
    "marginprox": ((857, 17649, 1, 6837), 9.1250e-118, (4.0842, 9.4827)),
}


@pytest.mark.parametrize("layer", sorted(SCIPY))
def test_exact_small_count_statistics_match_scipy(pm, layer):
    counts, p_ref, (lo_ref, hi_ref) = SCIPY[layer]
    assert pm.fisher_exact_p_greater(*counts) == pytest.approx(p_ref, rel=1e-4)
    lo, hi = pm.exact_contrast_interval(*counts)
    assert lo == pytest.approx(lo_ref, abs=1e-4)
    assert (hi is None) if hi_ref is None else hi == pytest.approx(hi_ref, abs=1e-4)


def test_zero_count_studentized_contrast_depends_on_the_correction(pm):
    grav = SCIPY["grav"][0]
    baked = pm.studentized_contrast(*grav, k=0.5)
    assert baked == pytest.approx(2.898, abs=1e-3)
    assert pm.studentized_contrast(*grav, k=0.1) == pytest.approx(1.805, abs=1e-3)
    # a table without a zero count takes no correction at all
    mag = SCIPY["mag"][0]
    assert pm.studentized_contrast(*mag, k=0.5) == pm.studentized_contrast(*mag, k=0.1)
