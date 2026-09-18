"""pu-conformal.json must describe the folds its code uses (issue #44): pipeline/pu_conformal.py::block_folds assigns
five k-means regions on the cell coordinates, while the file claimed "contiguous spatial blocks (blockId % k),
identical to the live TS engine" and recorded an unused block size. Reads the committed artifact only."""
from __future__ import annotations

import json
import re

from pipeline import pipeline


def test_pu_conformal_protocol_names_the_k_means_regions():
    d = json.loads((pipeline.DERIVED / "pu-conformal.json").read_text(encoding="utf-8"))
    assert d["schema"] == "prospectmap.puconformal/v2"
    protocol = d["protocol"]
    assert protocol["regions"] == "k-means" and "k-means" in protocol["scheme"]
    assert "identical to the live TS engine" not in protocol["scheme"]
    assert "block_cells" not in protocol, "k-means regions have no block size"
    assert protocol["folds"] == 5


def test_pu_conformal_texts_are_neutral():
    d = json.loads((pipeline.DERIVED / "pu-conformal.json").read_text(encoding="utf-8"))
    assert "honesty" not in d and d["scope"]
    assert not re.search(r"honest|fabricat", d["scope"] + " " + d["verdict"]["text"], re.IGNORECASE)
