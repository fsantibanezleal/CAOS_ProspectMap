# Guide, run the precompute / retrain pipeline

```bash
# light lane (numpy only), rebuild the replay artifacts from the committed bake
python -m venv .venv-pipeline
.venv-pipeline/Scripts/pip install -r data-pipeline/requirements.txt -r requirements-dev.txt -e .
.venv-pipeline/Scripts/python -m pmlab.pipeline all            # all cases (or:  ... pmlab.pipeline K-PORPHYRY)
.venv-pipeline/Scripts/python -m pytest                        # the 3 contract/pipeline tests
.venv-pipeline/Scripts/python scripts/check_artifacts.py       # Contract 2: index <-> manifests <-> artifacts

# heavy lane (local only), re-bake the cases (Node, the same TS engine) + train the 2 models (torch -> ONNX)
python -m venv .venv-precompute
.venv-precompute/Scripts/pip install -r data-pipeline/requirements-precompute.txt
.venv-pipeline/Scripts/python -m pmlab.pipeline all --retrain
```

Outputs land in `data/derived/<case>/trace.json` + `data/derived/manifests/<case>.json` + `index.json`, plus (after
`--retrain`) `mpm-classifier.onnx` + `geology-ood.onnx` + `pm-learned.json`. The run is deterministic, same inputs +
seed => byte-identical artifact. Stages + their roles:
[../architecture/05_precompute-pipeline.md](../architecture/05_precompute-pipeline.md).
