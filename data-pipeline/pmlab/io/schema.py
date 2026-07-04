"""Typed objects passed between pipeline stages, the inter-stage contract. Plain dataclasses (no heavy deps)."""
from __future__ import annotations

from dataclasses import dataclass

# the case CATEGORIES (mirrors frontend/src/mpm/cases.ts)
CATEGORIES = (
    "deposit-type terrane (the geological setting)",
    "data / validation regime (evidence richness)",
    "control (oracle / negative control)",
)


@dataclass(frozen=True)
class CaseDescriptor:
    """One validated MPM case bundle descriptor (CONTRACT 1 output). A case is a co-registered evidence cube
    (nx*ny unit cells, n_layers bands) + a presence-only deposit point pattern (n_deposits occupied cells) over a
    study-area mask. For the synthetic cases the cube + deposits are regenerated from the case SPEC by the TypeScript
    engine (frontend/src/mpm/), only the SPEC + the baked summary are committed (no raster blobs)."""

    case_id: str
    nx: int
    ny: int
    cell_km: float
    n_layers: int
    n_deposits: int
    real_or_synthetic: str = "synthetic"
    deposit_type: str = ""
    flags: tuple[str, ...] = ()

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny
