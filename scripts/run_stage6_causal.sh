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
PRIMARY_MASK="${PRIMARY_MASK:-${RUN_ROOT}/stages/05_probing/discovery/rho0.003_signed/tool_action_neurons.json}"
STAGE_ROOT="${RUN_ROOT}/stages/06_causal"
OUTPUT_DIR="${STAGE_ROOT}/conditions"
MANIFEST_DIR="${STAGE_ROOT}/manifests"
LOG_DIR="${STAGE_ROOT}/logs"
RUNTIME_PROVENANCE="${MANIFEST_DIR}/runtime_provenance.json"
CAUSAL_GENERATION_SEEDS="${CAUSAL_GENERATION_SEEDS:-0}"

read -r -a generation_seeds <<< "${CAUSAL_GENERATION_SEEDS}"
if [[ "${#generation_seeds[@]}" -lt 1 ]]; then
  echo "CAUSAL_GENERATION_SEEDS must contain at least one seed" >&2
  exit 2
fi
for seed in "${generation_seeds[@]}"; do
  case "${seed}" in
    0|1|2)
      ;;
    *)
      echo "Causal generation seeds must be a subset of: 0 1 2" >&2
      exit 2
      ;;
  esac
done

required_files=(
  "${DATA_DIR}/tasks_v1_test_fulltools_category.json"
  "${LABELS_DIR}/test_labels_no_reasoning_fulltools.json"
  "${PRIMARY_MASK}"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required Stage 6 input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${MANIFEST_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/stage6_causal.log") 2>&1

"${PYTHON_BIN}" -m when2tool_action.scripts.audit_provenance \
  --config "${CONFIG}" \
  --output "${RUNTIME_PROVENANCE}"

"${PYTHON_BIN}" -m when2tool_action.scripts.run_neuron_ablation \
  --config "${CONFIG}" \
  --runtime-provenance "${RUNTIME_PROVENANCE}" \
  --data "${DATA_DIR}/tasks_v1_test_fulltools_category.json" \
  --labels "${LABELS_DIR}/test_labels_no_reasoning_fulltools.json" \
  --mask "${PRIMARY_MASK}" \
  --output-dir "${OUTPUT_DIR}" \
  --generation-seeds "${generation_seeds[@]}" \
  --resume

echo "Stage 6 completed/resumed: ${OUTPUT_DIR}/summary.json"
