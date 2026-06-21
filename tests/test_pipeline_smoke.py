"""Pipeline smoke + determinism: a case regenerates deterministically (same seed -> byte-identical artifact), the
case set matches the committed bake (no drift between the Python registry and the TS-baked case-results.json), and
run_all writes the flat index."""
import json

from pmlab import pipeline, registry


def test_case_deterministic_same_seed():
    a = pipeline.precompute("K-PORPHYRY", seed=7)
    b = pipeline.precompute("K-PORPHYRY", seed=7)
    assert a["artifact"]["bytes"] == b["artifact"]["bytes"]
    trace = json.loads((pipeline.DERIVED / a["artifact"]["path"]).read_text(encoding="utf-8"))
    assert trace["roc_auc"] > 0.5  # the porphyry case has real (synthetic) skill
    assert trace["schema"].startswith("prospectmap.trace/")


def test_control_case_runs():
    m = pipeline.precompute("C-NEGATIVE", seed=1)  # uninformative layers -> AUC ~ 0.5, must not crash
    trace = json.loads((pipeline.DERIVED / m["artifact"]["path"]).read_text(encoding="utf-8"))
    assert abs(trace["roc_auc"] - 0.5) < 0.12, "negative control must be ~random"


def test_registry_matches_committed_bake():
    case_results, _ = pipeline._load_artifacts()
    baked = set(case_results["cases"].keys())
    declared = {c.id for c in registry.list_cases()}
    assert declared == baked, f"registry vs bake drift: only-registry={declared - baked}, only-bake={baked - declared}"


def test_run_all_writes_index():
    entries = pipeline.run_all(seed=42)
    assert len(entries) == len(registry.list_cases()) >= 4
    idx = json.loads((pipeline.MANIFESTS / "index.json").read_text(encoding="utf-8"))
    assert idx["n_cases"] == len(entries)
    assert idx["schema"].startswith("prospectmap.index/")
