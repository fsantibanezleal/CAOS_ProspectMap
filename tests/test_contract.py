"""CONTRACT 1 (ingestion) tests: good case-bundle descriptors validate; ill-formed ones are rejected with a reason;
a tiny presence-only set / a single-layer model / a synthetic study area is flagged (accepted); the committed
example passes."""
from pathlib import Path

from pmlab.io.contract import validate_case, validate_records
from pmlab.io.formats import read_csv_rows


def _row(**over):
    base = {"case_id": "c", "nx": 100, "ny": 100, "cell_km": 1.0, "n_layers": 4, "n_deposits": 85,
            "real_or_synthetic": "real", "deposit_type": "porphyry Cu"}
    base.update(over)
    return base


def test_good_descriptor_accepted():
    rep = validate_records([_row()])
    assert rep.ok and len(rep.accepted) == 1 and not rep.rejected
    assert rep.accepted[0].n_cells == 100 * 100


def test_bad_descriptors_rejected_not_coerced():
    rows = [
        _row(nx=0),                       # non-positive grid
        _row(ny=-5),                      # negative
        _row(cell_km=0),                  # zero cell size
        _row(n_layers=0),                 # no evidence to fuse
        _row(n_deposits=0),               # WofE needs >= 1 deposit
        _row(nx="lots"),                  # non-numeric
        {"case_id": "m", "nx": 100},      # missing columns
    ]
    rep = validate_records(rows)
    assert len(rep.accepted) == 0 and len(rep.rejected) == len(rows)
    assert all("reason" in r for r in rep.rejected)


def test_honesty_relevant_cases_flagged():
    tiny = validate_records([_row(n_deposits=6)])         # presence-only-tiny
    assert tiny.ok and tiny.flagged and "presence-only-tiny" in " ".join(tiny.flagged[0]["flags"])
    single = validate_records([_row(n_layers=1)])          # single-layer model
    assert single.ok and single.flagged and "single-layer" in " ".join(single.flagged[0]["flags"])
    synth = validate_records([_row(real_or_synthetic="synthetic")])
    assert synth.ok and synth.flagged and "SYNTHETIC" in " ".join(synth.flagged[0]["flags"])


def test_validate_case_gate():
    good = validate_case(_row())
    assert good.ok
    bad = validate_case(_row(n_deposits=0))
    assert not bad.ok and bad.rejected


def test_committed_example_passes_contract():
    csv = Path(__file__).resolve().parents[1] / "data" / "examples" / "cases.csv"
    rep = validate_records(read_csv_rows(csv))
    assert rep.ok and not rep.rejected, f"cases.csv should pass Contract 1: {rep.summary()}"
    assert len(rep.accepted) == 10
