#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-cifar10 cifar100}"
METHODS="${METHODS:-baseline metaaugment}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/multiseed}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-50}"

mkdir -p "${OUTPUT_ROOT}/logs"

has_final_result() {
  local log_path="$1"
  test -f "${log_path}" && grep -q "test_top1=" "${log_path}"
}

config_for() {
  local dataset="$1"
  local method="$2"
  case "${dataset}:${method}" in
    cifar10:baseline)
      echo "configs/cifar10_preact_resnet18_baseline_competitor.yaml"
      ;;
    cifar10:metaaugment)
      echo "configs/cifar10_preact_resnet18_metaaugment_competitor.yaml"
      ;;
    cifar100:baseline)
      echo "configs/cifar100_preact_resnet18_baseline_competitor.yaml"
      ;;
    cifar100:metaaugment)
      echo "configs/cifar100_preact_resnet18_metaaugment_competitor.yaml"
      ;;
    *)
      echo "Unsupported dataset/method pair: ${dataset}/${method}" >&2
      return 1
      ;;
  esac
}

for seed in ${SEEDS}; do
  for dataset in ${DATASETS}; do
    for method in ${METHODS}; do
      config="$(config_for "${dataset}" "${method}")"
      run_name="${dataset}_preact_resnet18_${method}_seed${seed}"
      workdir="${OUTPUT_ROOT}/${run_name}"
      log_path="${OUTPUT_ROOT}/logs/${run_name}.log"

      if has_final_result "${log_path}"; then
        echo "[$(date --iso-8601=seconds)] SKIP ${run_name} (final result already exists)"
        continue
      fi

      echo "[$(date --iso-8601=seconds)] START ${run_name}"
      python -m meta_augment.cli.train \
        --config "${config}" \
        --data-dir "${DATA_DIR}" \
        --workdir "${workdir}" \
        --seed "${seed}" \
        --keep-checkpoints "${KEEP_CHECKPOINTS}" \
        --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
        2>&1 | tee "${log_path}"
      echo "[$(date --iso-8601=seconds)] DONE ${run_name}"
    done
  done
done
