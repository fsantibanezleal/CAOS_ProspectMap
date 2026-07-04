# Guide, the GPU lane (optional)

Only for products whose offline engine/sweep/training genuinely needs CUDA (large DEM, big Monte-Carlo, heavy
model training). Never required for the live/replay path.

1. Pin the CUDA build in `requirements-gpu.txt` (e.g. `cupy-cuda12x`, `torch==…+cu12x`, `numba`, `taichi`).
2. On a CUDA box: `.venv-pipeline/bin/python -m pip install -r requirements-gpu.txt`.
3. Document the engine in `docs/frameworks/<tool>/`.

The committed artifacts are produced offline regardless of lane, so a GPU-only product still deploys as a static
replay (the browser never needs the GPU). ProspectMap's heavy lane (torch → ONNX, pinned CPU-only in
`data-pipeline/requirements-precompute.txt`) needs no CUDA; there is no `requirements-gpu.txt` in this repo, add
one only if a future heavy case genuinely needs it.
