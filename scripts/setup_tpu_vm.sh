#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-tpu.txt

python - <<'PY'
import jax
print("JAX", jax.__version__)
print("Devices:", jax.devices())
PY
