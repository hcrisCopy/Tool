#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CODE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-when2tool_action/configs/qwen3_4b_instruct_2507.yaml}"
MODEL="qwen3-4b-instruct-2507"
RUN_ROOT="${RUN_ROOT:-../CallTool_data/when2tool_precise_shield/${MODEL}}"
DATA_DIR="${RUN_ROOT}/data"
LABELS_DIR="${RUN_ROOT}/labels/${MODEL}"
ADAPTER_ROOT="${RUN_ROOT}/stages/07_training/adapters"
STAGE_ROOT="${RUN_ROOT}/stages/08_evaluation"
OUTPUT_ROOT="${STAGE_ROOT}/outputs"
COMPARISON_DIR="${STAGE_ROOT}/comparison"
MANIFEST_DIR="${STAGE_ROOT}/manifests"
LOG_DIR="${STAGE_ROOT}/logs"
RUNTIME_PROVENANCE="${MANIFEST_DIR}/runtime_provenance.json"
STAGE_START="${STAGE_START:-all}"
EVAL_GENERATION_SEEDS="${EVAL_GENERATION_SEEDS:-0 1 2}"

case "${STAGE_START}" in
  all|evaluation|summary)
    ;;
  *)
    echo "STAGE_START must be all, evaluation, or summary" >&2
    exit 2
    ;;
esac

read -r -a generation_seeds <<< "${EVAL_GENERATION_SEEDS}"
if [[ "${#generation_seeds[@]}" -lt 1 ]]; then
  echo "EVAL_GENERATION_SEEDS must contain at least one seed" >&2
  exit 2
fi
for seed in "${generation_seeds[@]}"; do
  case "${seed}" in
    0|1|2)
      ;;
    *)
      echo "Evaluation generation seeds must be a subset of: 0 1 2" >&2
      exit 2
      ;;
  esac
done

required_files=(
  "${DATA_DIR}/tasks_v1_test_category.json"
  "${DATA_DIR}/tasks_v1_test_fulltools_category.json"
  "${LABELS_DIR}/test_labels_no_reasoning_fulltools.json"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required Stage 8 input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${MANIFEST_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/stage8_evaluation.log") 2>&1

"${PYTHON_BIN}" -m when2tool_action.scripts.audit_provenance \
  --config "${CONFIG}" \
  --output "${RUNTIME_PROVENANCE}"

if [[ "${STAGE_START}" == all || "${STAGE_START}" == evaluation ]]; then
  conditions=(
    base_model
    target_neuron_lora
    dense_mlp_lora
    random_neuron_lora_seed0
    random_neuron_lora_seed1
    random_neuron_lora_seed2
  )
  for scope in scoped full; do
    if [[ "${scope}" == scoped ]]; then
      data_path="${DATA_DIR}/tasks_v1_test_category.json"
    else
      data_path="${DATA_DIR}/tasks_v1_test_fulltools_category.json"
    fi
    for condition in "${conditions[@]}"; do
      adapter_args=()
      if [[ "${condition}" != base_model ]]; then
        adapter_path="${ADAPTER_ROOT}/${condition}"
        if [[ ! -d "${adapter_path}" ]]; then
          echo "Missing trained adapter: ${adapter_path}" >&2
          exit 1
        fi
        adapter_args=(--adapter-dir "${adapter_path}")
      fi
      "${PYTHON_BIN}" -m when2tool_action.scripts.run_trained_eval \
        --config "${CONFIG}" \
        --runtime-provenance "${RUNTIME_PROVENANCE}" \
        --data "${data_path}" \
        --labels "${LABELS_DIR}/test_labels_no_reasoning_fulltools.json" \
        --output-root "${OUTPUT_ROOT}" \
        --tool-scope "${scope}" \
        --condition-id "${condition}" \
        "${adapter_args[@]}" \
        --generation-seeds "${generation_seeds[@]}" \
        --resume
    done
  done
fi

if [[ "${STAGE_START}" == all || "${STAGE_START}" == summary ]]; then
  "${PYTHON_BIN}" -m when2tool_action.scripts.summarize_trained_evals \
    --output-root "${OUTPUT_ROOT}" \
    --output-dir "${COMPARISON_DIR}" \
    --require-complete
fi

echo "Stage 8 completed for STAGE_START=${STAGE_START}: ${STAGE_ROOT}"
