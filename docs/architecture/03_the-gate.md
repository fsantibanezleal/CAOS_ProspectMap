# The live-vs-precompute gate

`data-pipeline/pmlab/core/gate.py :: classify_lane()`. A case runs **live** in the browser iff, by measurement,
never by hand-wave:

- it is **client-side** (no server needed), and
- its runtimes are a subset of the deployed client set (`LIVE_RUNTIMES = {ts-mpm, onnxruntime-web}`), and
- `run_ms <= RUN_MS_GATE` (interaction budget), and
- `trace_bytes <= TRACE_BYTES_GATE` (small artifact).

ProspectMap's live lane is the dependency-free TypeScript WofE engine (`frontend/src/mpm/`) plus the learned
classifier + the geology OOD autoencoder via onnxruntime-web. A teaching-scale WofE recompute is milliseconds and the
traces are small, so every case passes and is classified `live`. Otherwise a case would be **precompute**: the offline
pipeline bakes the artifact and the SPA replays it. Either way a committed artifact always exists, so the site replays
instantly on first paint (ADR-0054).

The verdict + the measured budgets are written into the manifest (`gate` field) and CI fails if `manifest.lane`
disagrees with the gate, so a case can never be mislabeled "live".
