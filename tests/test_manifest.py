"""CONTRACT 2 (artifact) tests: the manifest points to a real artifact with the recorded byte size, and the lane
verdict is consistent with the gate."""
from pmlab import pipeline


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
