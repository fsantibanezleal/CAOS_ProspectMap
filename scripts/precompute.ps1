# Run the offline pipeline (pass-through args). E.g.:  ./scripts/precompute.ps1 K-PORPHYRY --seed 42
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$vp = Join-Path ".venv-pipeline" "Scripts\python.exe"
if (-not (Test-Path $vp)) { $vp = Join-Path ".venv-pipeline" "bin/python" }
& $vp -m pmlab.pipeline @args
