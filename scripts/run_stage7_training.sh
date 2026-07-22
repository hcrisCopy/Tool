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
CAUSAL_SUMMARY="${CAUSAL_SUMMARY:-${RUN_ROOT}/stages/06_causal/conditions/summary.json}"
STAGE_ROOT="${RUN_ROOT}/stages/07_training"
SFT_DIR="${STAGE_ROOT}/sft"
ADAPTER_DIR="${STAGE_ROOT}/adapters"
MANIFEST_DIR="${STAGE_ROOT}/manifests"
LOG_DIR="${STAGE_ROOT}/logs"
RUNTIME_PROVENANCE="${MANIFEST_DIR}/runtime_provenance.json"
SCOPED_SOURCE="${SFT_DIR}/train_scoped_current_seed0_fullrecord.json"
SFT_JSONL="${SFT_DIR}/train_action_trajectories.jsonl"
SFT_MANIFEST="${SFT_DIR}/train_action_trajectories.manifest.json"
STAGE_START="${STAGE_START:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TRAIN_CONDITIONS="${TRAIN_CONDITIONS:-target dense random0 random1 random2}"

case "${STAGE_START}" in
  all|source|training)
    ;;
  *)
    echo "STAGE_START must be all, source, or training" >&2
    exit 2
    ;;
esac
case "${NPROC_PER_NODE}" in
  1|2|4|8)
    ;;
  *)
    echo "NPROC_PER_NODE must be one of 1, 2, 4, 8 (global batch is fixed at 8)" >&2
    exit 2
    ;;
esac
read -r -a train_conditions <<< "${TRAIN_CONDITIONS}"
if [[ "${#train_conditions[@]}" -lt 1 ]]; then
  echo "TRAIN_CONDITIONS must contain at least one condition" >&2
  exit 2
fi
declare -A seen_conditions=()
for condition in "${train_conditions[@]}"; do
  case "${condition}" in
    target|dense|random0|random1|random2)
      ;;
    *)
      echo "TRAIN_CONDITIONS entries must be target, dense, random0, random1, or random2" >&2
      exit 2
      ;;
  esac
  if [[ -n "${seen_conditions[${condition}]:-}" ]]; then
    echo "TRAIN_CONDITIONS contains duplicate condition: ${condition}" >&2
    exit 2
  fi
  seen_conditions["${condition}"]=1
done

required_files=(
  "${DATA_DIR}/tasks_v1_train_category.json"
  "${DATA_DIR}/tasks_v1_train_fulltools_category.json"
  "${LABELS_DIR}/train_labels_no_reasoning_fulltools.json"
  "${LABELS_DIR}/train_hard_no_tool_generations_fulltools.json"
  "${PRIMARY_MASK}"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required Stage 7 input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${SFT_DIR}" "${ADAPTER_DIR}" "${MANIFEST_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/stage7_training.log") 2>&1

"${PYTHON_BIN}" -m when2tool_action.scripts.audit_provenance \
  --config "${CONFIG}" \
  --output "${RUNTIME_PROVENANCE}"

if [[ "${STAGE_START}" == all || "${STAGE_START}" == source ]]; then
  "${PYTHON_BIN}" -m when2tool_action.scripts.run_eval \
    --config "${CONFIG}" \
    --runtime-provenance "${RUNTIME_PROVENANCE}" \
    --data "${DATA_DIR}/tasks_v1_train_category.json" \
    --labels "${LABELS_DIR}/train_labels_no_reasoning_fulltools.json" \
    --output "${SCOPED_SOURCE}" \
    --setting-name train_sft_current_no_reasoning_scoped \
    --tool-scope scoped \
    --prompt-mode current \
    --reasoning-mode no_reasoning \
    --record-mode full \
    --seeds 0 \
    --resume

  "${PYTHON_BIN}" -m when2tool_action.scripts.prepare_sft_trajectories \
    --config "${CONFIG}" \
    --runtime-provenance "${RUNTIME_PROVENANCE}" \
    --data "${DATA_DIR}/tasks_v1_train_fulltools_category.json" \
    --scoped-data "${DATA_DIR}/tasks_v1_train_category.json" \
    --labels "${LABELS_DIR}/train_labels_no_reasoning_fulltools.json" \
    --no-tool-generations "${LABELS_DIR}/train_hard_no_tool_generations_fulltools.json" \
    --scoped-trajectories "${SCOPED_SOURCE}" \
    --output "${SFT_JSONL}" \
    --manifest "${SFT_MANIFEST}"
fi

if [[ "${STAGE_START}" == all || "${STAGE_START}" == training ]]; then
  if [[ "${CAUSAL_GATE_PASSED:-0}" != 1 ]]; then
    echo "Training is gated. Inspect ${CAUSAL_SUMMARY}, then rerun with CAUSAL_GATE_PASSED=1." >&2
    exit 3
  fi
  if [[ ! -f "${CAUSAL_SUMMARY}" || ! -f "${SFT_JSONL}" || ! -f "${SFT_MANIFEST}" ]]; then
    echo "Missing causal summary or prepared SFT artifacts; training will not start." >&2
    exit 1
  fi

  train_one() {
    local mode="$1"
    local output="$2"
    shift 2
    local module_args=(
      -m when2tool_action.scripts.train_masked_lora
      --config "${CONFIG}"
      --runtime-provenance "${RUNTIME_PROVENANCE}"
      --train-data "${SFT_JSONL}"
      --train-manifest "${SFT_MANIFEST}"
      --mask "${PRIMARY_MASK}"
      --output-dir "${output}"
      --mode "${mode}"
      "$@"
    )
    if [[ "${NPROC_PER_NODE}" == 1 ]]; then
      "${PYTHON_BIN}" "${module_args[@]}"
    else
      "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${NPROC_PER_NODE}" \
        "${module_args[@]}"
    fi
  }

  for condition in "${train_conditions[@]}"; do
    case "${condition}" in
      target)
        train_one target "${ADAPTER_DIR}/target_neuron_lora"
        ;;
      dense)
        train_one dense "${ADAPTER_DIR}/dense_mlp_lora"
        ;;
      random*)
        random_seed="${condition#random}"
        train_one random \
          "${ADAPTER_DIR}/random_neuron_lora_seed${random_seed}" \
          --random-seed "${random_seed}"
        ;;
    esac
  done
fi

echo "Stage 7 completed for STAGE_START=${STAGE_START}: ${STAGE_ROOT}"
