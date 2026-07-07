"""Real-data Cube builder for ProspectMap's "Real sample" lane: the US Midcontinent
Mississippi-Valley-type (MVT) Zn-Pb belt, from the Lawley et al. (2022) Tri-National Critical
Minerals Mapping Initiative (CMMI) open data release (USGS, public domain).

WHAT THIS DOES (offline, one-shot precompute; the browser only loads the baked arrays):
  1. Reads the openly-licensed CMMI files already fetched into data/raw/REAL-USMVT (see provenance).
  2. Clips + rasterizes the evidential layers onto a shared regular grid over the Midcontinent box:
       REAL measured geophysics : mag (RTP magnetic), grav (Bouguer gravity), lab (depth to the
                                  lithosphere-asthenosphere boundary, a tomography-derived layer),
                                  satgrav (satellite-gravity shape index).
       DERIVED (by us, from real vectors) : faultprox (distance to nearest mapped fault),
                                  marginprox (distance to nearest ancient passive margin).
  3. Sets depositIdx = grid cells containing >= 1 real Pb-Zn (MVT/CD) occurrence.
  4. Picks each continuous layer's FAVOURABLE cumulative direction data-drivenly (standard WofE
     practice, Bonham-Carter 1994), so the browser sweep reproduces the same contrast.
  5. Bakes data/derived/REAL-USMVT/cube.json (per-cell arrays + grid meta + provenance) and
     provenance.json. The SAME live TypeScript WofE engine then runs on it in the browser, and the
     Node bake (science/bake_real.mjs) emits the matching trace.json + case-results entry.

HONESTY: our browser posterior is OUR Weights-of-Evidence recomputation on this rasterized
sub-region. It is NOT the published Lawley H3 + gradient-boosting model output. Real geophysical
layers are physically correlated, so the conditional-independence omnibus test is expected to fire
here (route to logistic regression). Both facts are surfaced in the app.

Run (isolated venv, never global):
    .venv-precompute/Scripts/python.exe -m pmlab.real_usmvt
Dependencies (installed into .venv-precompute only): numpy, rasterio, pyshp, pyproj, scipy.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, rowcol
from rasterio.warp import reproject
from rasterio.windows import from_bounds as window_from_bounds
from scipy.ndimage import distance_transform_edt

# ----------------------------------------------------------------------------------------------
# Study area + grid. US Midcontinent MVT belt: the Tri-State (SW Missouri / Kansas / Oklahoma),
# the SE Missouri Viburnum Trend, and the Upper Mississippi Valley district all fall inside this box.
WEST, EAST, SOUTH, NORTH = -97.0, -88.0, 35.0, 43.5
NX, NY = 144, 176  # ~5.4 km cells (near-square in km at this latitude), 25344 cells (<40k budget)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "REAL-USMVT"
OUT = ROOT / "data" / "derived" / "REAL-USMVT"

CASE_ID = "REAL-USMVT"

# Raster sources: (path, layer id, resampling, kind-note). Continuous REAL geophysics.
RASTERS = [
    ("unz_magnetic_rtp_uscanada/GeophysicsMag_RTP_USCanada.tif", "mag", Resampling.average),
    ("unz_gravity_uscanada/GeophysicsGravity_USCanada.tif", "grav", Resampling.average),
    ("unz_lab_uscanada/GeophysicsLAB_USCanada.tif", "lab", Resampling.bilinear),
    (
        "unz_satgrav_shapeindex_uscanada/GeophysicsSatelliteGravity_ShapeIndex_USCanada/"
        "GeophysicsSatelliteGravity_ShapeIndex_USCanada.tif",
        "satgrav",
        Resampling.bilinear,
    ),
]
FAULTS_SHP = "unz_faults_uscanada/GeologyFaults_USCanada.shp"
MARGINS_SHP = "unz_passivemargins_ancient/GeologyAncientPassiveMargins_World.shp"
OCC_CSV = "occurrences.csv"

LAYER_NAMES = {
    "mag": "RTP magnetic anomaly (REAL)",
    "grav": "Bouguer gravity anomaly (REAL)",
    "lab": "Depth to lithosphere-asthenosphere boundary (REAL, tomography)",
    "satgrav": "Satellite-gravity shape index (REAL)",
    "faultprox": "Distance to nearest mapped fault (DERIVED)",
    "marginprox": "Distance to nearest ancient passive margin (DERIVED)",
}
DERIVED_IDS = {"faultprox", "marginprox"}
LAYER_ORDER = ["mag", "grav", "lab", "satgrav", "faultprox", "marginprox"]

NODATA_FLOOR = -1e30  # CMMI grids use -3.4e38 for no-data


def _dst_transform():
    return from_bounds(WEST, SOUTH, EAST, NORTH, NX, NY)


def read_raster_to_grid(path: Path, resampling: Resampling) -> np.ndarray:
    """Clip a CMMI GeoTIFF to the box and resample onto the (NY, NX) grid. NaN = no-data."""
    with rasterio.open(path) as ds:
        win = window_from_bounds(WEST, SOUTH, EAST, NORTH, transform=ds.transform)
        # pad the window by 1 px so bilinear near the edges has neighbours
        win = win.round_offsets().round_lengths()
        src = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype("float64")
        src_transform = ds.window_transform(win)
        src[src <= NODATA_FLOOR] = np.nan
        dst = np.full((NY, NX), np.nan, dtype="float64")
        reproject(
            source=src,
            destination=dst,
            src_transform=src_transform,
            src_crs=ds.crs,
            dst_transform=_dst_transform(),
            dst_crs="EPSG:4326",
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return dst


def rasterize_lines(shp_path: Path, id_field_bbox_only: bool = False) -> np.ndarray:
    """Burn polyline vertices/segments that fall in (or near) the box onto a (NY, NX) 0/1 mask."""
    mask = np.zeros((NY, NX), dtype=bool)
    tr = _dst_transform()
    # cell size in degrees, for a densification step of ~1/3 cell along each segment
    dlon = (EAST - WEST) / NX
    dlat = (NORTH - SOUTH) / NY
    step = min(dlon, dlat) / 3.0
    pad = 0.5  # degrees; keep segments just outside the box so edge distances are honest
    r = shapefile.Reader(str(shp_path))
    for shp in r.iterShapes():
        bx = shp.bbox  # [minx, miny, maxx, maxy]
        if bx[2] < WEST - pad or bx[0] > EAST + pad or bx[3] < SOUTH - pad or bx[1] > NORTH + pad:
            continue
        pts = shp.points
        parts = list(shp.parts) + [len(pts)]
        for pi in range(len(parts) - 1):
            seg = pts[parts[pi]:parts[pi + 1]]
            for a, b in zip(seg[:-1], seg[1:]):
                (x0, y0), (x1, y1) = a, b
                dist = math.hypot(x1 - x0, y1 - y0)
                n = max(1, int(dist / step))
                for k in range(n + 1):
                    t = k / n
                    lon = x0 + (x1 - x0) * t
                    lat = y0 + (y1 - y0) * t
                    if not (WEST <= lon <= EAST and SOUTH <= lat <= NORTH):
                        continue
                    row, col = rowcol(tr, lon, lat)
                    if 0 <= row < NY and 0 <= col < NX:
                        mask[row, col] = True
    return mask


def proximity_km(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (km) from every cell to the nearest burned line cell."""
    if not mask.any():
        return np.full((NY, NX), np.nan)
    # cell size in km (mean latitude), used to scale the pixel-distance transform
    mid = math.radians((SOUTH + NORTH) / 2.0)
    dx_km = (EAST - WEST) / NX * 111.320 * math.cos(mid)
    dy_km = (NORTH - SOUTH) / NY * 110.574
    dt = distance_transform_edt(~mask, sampling=(dy_km, dx_km))
    return dt.astype("float64")


def proximity_to_shapes_km(shp_path: Path, region=(-140.0, -50.0, 20.0, 75.0)) -> np.ndarray:
    """Distance (km) from every cell centre to the nearest vertex of any polyline in `region`, even
    when that feature lies OUTSIDE the study box (the ancient passive margins are continental edges
    that bound, but do not cross, the cratonic Midcontinent; the evidence is the craton-interior
    distance gradient). Local planar km projection about the box centre + a KD-tree nearest search."""
    lon0 = (WEST + EAST) / 2.0
    lat0 = (SOUTH + NORTH) / 2.0
    kx = 111.320 * math.cos(math.radians(lat0))
    ky = 110.574
    verts = []
    r = shapefile.Reader(str(shp_path))
    w, e, s, n = region
    for shp in r.iterShapes():
        for lon, lat in shp.points:
            if w <= lon <= e and s <= lat <= n:
                verts.append(((lon - lon0) * kx, (lat - lat0) * ky))
    if not verts:
        return np.full((NY, NX), np.nan)
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(verts))
    # cell-centre coordinates
    dlon = (EAST - WEST) / NX
    dlat = (NORTH - SOUTH) / NY
    cols = (WEST + (np.arange(NX) + 0.5) * dlon - lon0) * kx
    rows = (NORTH - (np.arange(NY) + 0.5) * dlat - lat0) * ky  # row 0 = north
    gx, gy = np.meshgrid(cols, rows)
    pts = np.column_stack([gx.reshape(-1), gy.reshape(-1)])
    d, _ = tree.query(pts, k=1)
    return d.reshape(NY, NX).astype("float64")


def load_occurrences() -> np.ndarray:
    """US MVT/CD Pb-Zn occurrences inside the box -> a (NY, NX) 0/1 deposit mask."""
    mask = np.zeros((NY, NX), dtype=bool)
    tr = _dst_transform()
    n_in = 0
    with open(RAW / OCC_CSV, encoding="latin-1", newline="") as fh:
        for rec in csv.DictReader(fh):
            if "United States" not in rec.get("Admin", ""):
                continue
            try:
                lon = float(rec["Longitude"])
                lat = float(rec["Latitude"])
            except (KeyError, ValueError):
                continue
            if not (WEST <= lon <= EAST and SOUTH <= lat <= NORTH):
                continue
            row, col = rowcol(tr, lon, lat)
            if 0 <= row < NY and 0 <= col < NX:
                mask[row, col] = True
                n_in += 1
    print(f"  occurrences inside box: {n_in} -> {int(mask.sum())} unique deposit cells")
    return mask


def normalize01(a: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] over finite cells (WofE only uses per-layer ordering; this keeps JSON small)."""
    finite = np.isfinite(a)
    if not finite.any():
        return a
    lo = np.nanmin(a[finite])
    hi = np.nanmax(a[finite])
    if hi <= lo:
        out = np.where(finite, 0.5, np.nan)
        return out
    return (a - lo) / (hi - lo)


def favourable_direction(values01: np.ndarray, deposit_mask: np.ndarray) -> bool:
    """Data-driven: is a HIGH value favourable? Compare the mean layer value at deposit cells vs the
    background. If deposits sit at high values -> highIsFavourable True, else False. This mirrors the
    WofE practitioner choosing the cumulative sweep direction (Bonham-Carter 1994, ch.9)."""
    dep = deposit_mask & np.isfinite(values01)
    bg = (~deposit_mask) & np.isfinite(values01)
    if dep.sum() == 0 or bg.sum() == 0:
        return True
    return bool(values01[dep].mean() >= values01[bg].mean())


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Building {CASE_ID}: box W{WEST} E{EAST} S{SOUTH} N{NORTH}, grid {NX}x{NY}")

    deposit_mask = load_occurrences()

    layers = {}
    directions = {}
    for rel, lid, resamp in RASTERS:
        print(f"  raster {lid} <- {rel.split('/')[-1]}")
        grid = read_raster_to_grid(RAW / rel, resamp)
        grid01 = normalize01(grid)
        layers[lid] = grid01
        directions[lid] = favourable_direction(grid01, deposit_mask)

    print("  faultprox <- GeologyFaults_USCanada.shp")
    fault_km = proximity_km(rasterize_lines(RAW / FAULTS_SHP))
    layers["faultprox"] = normalize01(fault_km)
    directions["faultprox"] = False  # near a fault (low distance) is favourable

    print("  marginprox <- GeologyAncientPassiveMargins_World.shp (regional distance gradient)")
    margin_km = proximity_to_shapes_km(RAW / MARGINS_SHP)
    layers["marginprox"] = normalize01(margin_km)
    directions["marginprox"] = False  # near a paleo passive margin is favourable

    # deposit indices (row-major, row 0 = north, matching the browser Cube + MapView)
    deposit_idx = np.flatnonzero(deposit_mask.reshape(-1)).astype(int).tolist()

    # cell size (km) for metadata/display
    mid = math.radians((SOUTH + NORTH) / 2.0)
    dx_km = (EAST - WEST) / NX * 111.320 * math.cos(mid)
    dy_km = (NORTH - SOUTH) / NY * 110.574
    cell_km = round((dx_km + dy_km) / 2.0, 3)

    def enc(a: np.ndarray) -> list:
        # round to 4 dp; None for NaN so JSON stays valid and the browser maps it to NaN
        flat = a.reshape(-1)
        return [None if not math.isfinite(v) else round(float(v), 4) for v in flat]

    cube = {
        "schema": "prospectmap.realcube/v1",
        "case_id": CASE_ID,
        "name": "US Midcontinent MVT Zn-Pb belt (Lawley 2022, CMMI)",
        "real_or_synthetic": "real (open dataset)",
        "nx": NX,
        "ny": NY,
        "cellKm": cell_km,
        "box": {"west": WEST, "east": EAST, "south": SOUTH, "north": NORTH, "crs": "EPSG:4326"},
        "layer_ids": LAYER_ORDER,
        "layers": [
            {
                "id": lid,
                "name": LAYER_NAMES[lid],
                "kind": "continuous",
                "highIsFavourable": bool(directions[lid]),
                "provenance": "DERIVED" if lid in DERIVED_IDS else "REAL",
                "values": enc(layers[lid]),
            }
            for lid in LAYER_ORDER
        ],
        "depositIdx": deposit_idx,
        "n_deposits": len(deposit_idx),
        "citation": (
            "Lawley, C.J.M., McCafferty, A.E., Graham, G.E., Huston, D.L., Kelley, K.D., Czarnota, K., "
            "Paradis, S., Peter, J.M., and others (2022). Data-driven prospectivity modelling of "
            "sediment-hosted Zn-Pb mineral systems and their critical raw materials. Ore Geology "
            "Reviews 141, 104635. DOI 10.1016/j.oregeorev.2021.104635. Data release: USGS ScienceBase "
            "item 6193e9f3d34eb622f68f13a5, DOI 10.5066/P970GDD5 (US public domain)."
        ),
        "license": "US public domain (usa.gov/publicdomain/label/1.0); GA-sourced parts CC-BY, attributed.",
        "honesty": (
            "Our posterior is OUR browser Weights-of-Evidence recomputation on this rasterized "
            "sub-region, NOT the published Lawley H3 + gradient-boosting model. mag/grav/lab/satgrav "
            "are REAL measured geophysics; faultprox/marginprox are DERIVED by us from real vector "
            "geology. Real geophysics is physically correlated, so the conditional-independence "
            "omnibus test is expected to fire (route to logistic regression)."
        ),
    }

    (OUT / "cube.json").write_text(json.dumps(cube), encoding="utf-8")
    print(f"  wrote {OUT / 'cube.json'} ({(OUT / 'cube.json').stat().st_size / 1e6:.2f} MB)")

    provenance = {
        "case_id": CASE_ID,
        "built": "2026-07-07",
        "release": {
            "title": "National-Scale Geophysical, Geologic, and Mineral Resource Data and Grids for "
            "the US, Canada, and Australia (CMMI, ver 1.1, March 2025)",
            "sciencebase_item": "6193e9f3d34eb622f68f13a5",
            "release_doi": "10.5066/P970GDD5",
            "paper_doi": "10.1016/j.oregeorev.2021.104635",
            "license": "US public domain (usa.gov/publicdomain/label/1.0); GA parts CC-BY.",
        },
        "box": {"west": WEST, "east": EAST, "south": SOUTH, "north": NORTH, "crs": "EPSG:4326"},
        "grid": {"nx": NX, "ny": NY, "cellKm": cell_km},
        "files": [
            {"role": "target (Pb-Zn occurrences)", "name": "GeologyMineralOccurrences_USCanada_Australia.csv",
             "item": "61955280d34eb622f690699b"},
            {"role": "mag (REAL)", "name": "GeophysicsMag_RTP_USCanada.tif", "item": "619a9a3ad34eb622f692f961"},
            {"role": "grav (REAL)", "name": "GeophysicsGravity_USCanada.tif", "item": "619a9f02d34eb622f692f96c"},
            {"role": "lab (REAL, tomography)", "name": "GeophysicsLAB_USCanada.tif", "item": "61da068cd34ed7929400b5a2"},
            {"role": "satgrav (REAL)", "name": "GeophysicsSatelliteGravity_ShapeIndex_USCanada.tif",
             "item": "61db4e05d34ed7929400ba11"},
            {"role": "faultprox (DERIVED)", "name": "GeologyFaults_USCanada.shp", "item": "61e06729d34e8911d9fe9da9"},
            {"role": "marginprox (DERIVED)", "name": "GeologyAncientPassiveMargins_World.shp",
             "item": "619550d9d34eb622f69061b7"},
        ],
        "directions": {k: ("high-favourable" if v else "low-favourable") for k, v in directions.items()},
        "n_deposits": len(deposit_idx),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"  wrote {OUT / 'provenance.json'}")
    print(f"  directions: {provenance['directions']}")
    print(f"Done. {len(deposit_idx)} deposit cells, prior ~ {len(deposit_idx)/(NX*NY):.4f}")


if __name__ == "__main__":
    build()
