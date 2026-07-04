"""CONTRACT 1, ingestion (a raw evidence-stack bundle -> pipeline). The *bring-your-own-evidence* gate.

* ``validate_records`` validates MPM case-bundle descriptor rows (one per study area). This is what the pipeline runs
  over the case set; it proves the gate and carries flags into the manifest.
* ``validate_case`` validates a single dropped descriptor (a dict), the same policy.

A record is ACCEPTED iff it passes; ill-formed records are REJECTED with a reason (never silently coerced);
plausible-but-honesty-relevant records are FLAGGED (accepted; the flag travels into the manifest). The key MPM honesty
flags: a tiny presence-only positive set (a black-box classifier will overfit; trust WofE + the OOD mask), a
single-layer model (no fusion, no conditional-independence to test), and a synthetic study area. Documented in
data/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import CaseDescriptor

REQUIRED_COLUMNS: tuple[str, ...] = ("case_id", "nx", "ny", "cell_km", "n_layers", "n_deposits")
N_DEPOSITS_TINY = 10        # < this many known deposits => presence-only-tiny FLAG (DL/black-box overfits)
MAX_CELLS = 4_000_000       # an absurdly large committed grid => FLAG (reduce offline)


@dataclass
class ContractReport:
    accepted: list[CaseDescriptor]
    rejected: list[dict[str, Any]]
    flagged: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return len(self.accepted) > 0

    def summary(self) -> str:
        return f"{len(self.accepted)} accepted, {len(self.rejected)} rejected, {len(self.flagged)} flagged"


def validate_records(raw_rows: list[dict[str, Any]]) -> ContractReport:
    """Apply CONTRACT 1 to raw case-bundle rows (e.g. from a CSV). Pure; deterministic; no I/O."""
    accepted: list[CaseDescriptor] = []
    rejected: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []

    for i, row in enumerate(raw_rows):
        cid = str(row.get("case_id", f"row{i}"))
        missing = [c for c in REQUIRED_COLUMNS if c not in row or row[c] in (None, "")]
        if missing:
            rejected.append({"row": i, "case_id": cid, "reason": f"missing/empty columns: {missing}"})
            continue
        try:
            nx = int(float(row["nx"]))
            ny = int(float(row["ny"]))
            cell_km = float(row["cell_km"])
            n_layers = int(float(row["n_layers"]))
            n_deposits = int(float(row["n_deposits"]))
        except (TypeError, ValueError):
            rejected.append({"row": i, "case_id": cid, "reason": "non-numeric grid/layer/deposit field"})
            continue

        bad: list[str] = []
        if nx <= 0 or ny <= 0:
            bad.append(f"grid {nx}x{ny} must be positive")
        if not (cell_km > 0):
            bad.append(f"cell_km={cell_km:g} must be > 0")
        if n_layers < 1:
            bad.append(f"n_layers={n_layers} must be >= 1 (no evidence to fuse)")
        if n_deposits < 1:
            bad.append(f"n_deposits={n_deposits} must be >= 1 (WofE needs at least one known deposit)")
        if bad:
            rejected.append({"row": i, "case_id": cid, "reason": "; ".join(bad)})
            continue

        rec_flags: list[str] = []
        if n_deposits < N_DEPOSITS_TINY:
            rec_flags.append(f"presence-only-tiny: only {n_deposits} known deposits (< {N_DEPOSITS_TINY}) "
                             "- a black-box classifier will overfit; trust the white-box WofE + the OOD mask")
        if n_layers == 1:
            rec_flags.append("single-layer model: no evidence fusion and no conditional-independence to test")
        if nx * ny > MAX_CELLS:
            rec_flags.append(f"grid {nx * ny} cells is very large (> {MAX_CELLS}) - reduce offline to a coarse cube")
        if str(row.get("real_or_synthetic", "synthetic")).startswith("synth"):
            rec_flags.append("SYNTHETIC study area (clearly labelled): known ground truth, used as a control")
        if rec_flags:
            flagged.append({"case_id": cid, "flags": rec_flags})

        accepted.append(CaseDescriptor(
            case_id=cid, nx=nx, ny=ny, cell_km=cell_km, n_layers=n_layers, n_deposits=n_deposits,
            real_or_synthetic=str(row.get("real_or_synthetic", "synthetic")),
            deposit_type=str(row.get("deposit_type", "")), flags=tuple(rec_flags)))
    return ContractReport(accepted=accepted, rejected=rejected, flagged=flagged)


def validate_case(meta: dict[str, Any]) -> ContractReport:
    """Apply CONTRACT 1 to a single dropped case-bundle descriptor."""
    return validate_records([meta])
