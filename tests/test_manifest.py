"""CONTRACT 2 (artifact) tests: the manifest points to a real artifact with the recorded byte size, and the lane
verdict is consistent with the gate."""
from pipeline import pipeline


def test_manifest_matches_artifact_and_gate():
    m = pipeline.precompute("K-PORPHYRY", seed=7)
    artifact = pipeline.DERIVED / m["artifact"]["path"]
    assert artifact.exists(), "manifest points to a non-existent artifact"
    assert artifact.stat().st_size == m["artifact"]["bytes"], "manifest byte size drifted from the artifact"
    assert m["schema"].startswith("prospectmap.manifest/")
    assert m["lane"] in ("live", "precompute")
    assert m["gate"]["lane"] == m["lane"], "manifest lane disagrees with the gate verdict"
    # the WofE recompute is client-side TS + onnxruntime-web + small => must be classified LIVE
    assert m["lane"] == "live", f"expected live lane, got {m['lane']} ({m['gate']['reasons']})"
    # the metrics carry the honest MPM numbers
    assert "roc_auc" in m["metrics"] and "ci_ratio" in m["metrics"] and "cv_inflation_gap" in m["metrics"]


def test_manifest_scope_is_case_specific_and_neutral():
    """Every manifest once carried one global text, "The study areas are SYNTHETIC ... No fabricated win", the real
    case included (issue #44). The scope is now per case, correct for the real case, and uses neutral wording."""
    import json
    import re

    from pipeline import registry
    from pipeline.core.manifest import MANIFEST_SCHEMA, case_scope

    for case in registry.list_cases():
        m = json.loads((pipeline.MANIFESTS / f"{case.id}.json").read_text(encoding="utf-8"))
        assert m["schema"] == MANIFEST_SCHEMA
        assert "honesty" not in m, f"{case.id}: the global honesty text is back"
        assert m["scope"] == case_scope(case)
        assert not re.search(r"honest|fabricat", m["scope"], re.IGNORECASE)
        if case.real_or_synthetic.startswith("real"):
            assert m["scope"].startswith("Real open data") and "synthetic" not in m["scope"].lower()
        else:
            assert m["scope"].startswith("Synthetic")
