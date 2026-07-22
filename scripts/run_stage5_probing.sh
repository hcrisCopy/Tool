#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Large artifacts remain in the sibling data tree.
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CODE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-when2tool_action/configs/qwen3_4b_instruct_2507.yaml}"
MODEL="qwen3-4b-instruct-2507"
RUN_ROOT="${RUN_ROOT:-../CallTool_data/when2tool_precise_shield/${MODEL}}"
DATA_DIR="${RUN_ROOT}/data"
LABELS_DIR="${RUN_ROOT}/labels/${MODEL}"
STAGE_ROOT="${RUN_ROOT}/stages/05_probing"
ACTIVATIONS_DIR="${STAGE_ROOT}/activations"
DISCOVERY_DIR="${STAGE_ROOT}/discovery"
MANIFEST_DIR="${STAGE_ROOT}/manifests"
LOG_DIR="${STAGE_ROOT}/logs"
RUNTIME_PROVENANCE="${MANIFEST_DIR}/runtime_provenance.json"
STAGE_START="${STAGE_START:-fresh}"
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-1}"

case "${STAGE_START}" in
  fresh|activations|discovery)
    ;;
  *)
    echo "STAGE_START must be fresh, activations, or discovery" >&2
    exit 2
    ;;
esac

if ! [[ "${EXTRACTION_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXTRACTION_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

required_files=(
  "${DATA_DIR}/tasks_v1_train_fulltools_category.json"
  "${DATA_DIR}/tasks_v1_test_fulltools_category.json"
  "${LABELS_DIR}/train_labels_no_reasoning_fulltools.json"
  "${LABELS_DIR}/test_labels_no_reasoning_fulltools.json"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required Stage 5 input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${MANIFEST_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/stage5_probing.log") 2>&1

"${PYTHON_BIN}" -m when2tool_action.scripts.audit_provenance \
  --config "${CONFIG}" \
  --output "${RUNTIME_PROVENANCE}"

if [[ "${STAGE_START}" == fresh || "${STAGE_START}" == activations ]]; then
  "${PYTHON_BIN}" -m when2tool_action.scripts.extract_mlp_activations \
    --config "${CONFIG}" \
    --runtime-provenance "${RUNTIME_PROVENANCE}" \
    --data-dir "${DATA_DIR}" \
    --labels-dir "${LABELS_DIR}" \
    --output-dir "${ACTIVATIONS_DIR}" \
    --batch-size "${EXTRACTION_BATCH_SIZE}" \
    --device cuda:0
fi

if [[ "${STAGE_START}" == fresh || "${STAGE_START}" == discovery ]]; then
  "${PYTHON_BIN}" -m when2tool_action.scripts.probe_tool_action_neurons \
    --config "${CONFIG}" \
    --runtime-provenance "${RUNTIME_PROVENANCE}" \
    --activations-dir "${ACTIVATIONS_DIR}" \
    --train-labels "${LABELS_DIR}/train_labels_no_reasoning_fulltools.json" \
    --test-labels "${LABELS_DIR}/test_labels_no_reasoning_fulltools.json" \
    --output-dir "${DISCOVERY_DIR}" \
    --rhos 0.001 0.003 0.005 \
    --variants signed positive abs \
    --control-seed 42 \
    --probe-c 0.0001 \
    --mean-chunk-size 64
fi

echo "Stage 5 completed. Primary mask: ${DISCOVERY_DIR}/rho0.003_signed/tool_action_neurons.json"
