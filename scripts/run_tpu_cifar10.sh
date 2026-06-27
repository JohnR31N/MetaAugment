#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/cifar10_wrn28_10_metaaugment.yaml}"
DATA_DIR="${DATA_DIR:-/tmp/metaaugment_data}"
WORKDIR="${WORKDIR:-/tmp/metaaugment_runs/cifar10}"

python -m meta_augment.cli.train \
  --config "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --workdir "${WORKDIR}" \
  "$@"
