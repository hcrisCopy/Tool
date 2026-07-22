#!/usr/bin/env bash
set -euo pipefail

CONFIG="when2tool_action/configs/qwen3_4b_instruct_2507.yaml"
MODEL="qwen3-4b-instruct-2507"
ROOT="../CallTool_data/when2tool_precise_shield/${MODEL}"
DATA="${ROOT}/data"
LABELS="${ROOT}/labels/${MODEL}"
FULL_ANALYSIS="${ROOT}/analysis/fulltools"
SCOPED_ADAPTED_ANALYSIS="${ROOT}/analysis/scoped_adapted"
SCOPED_ORIGINAL_ANALYSIS="${ROOT}/analysis/scoped_original_w2t"

FULL_OUTPUTS=(
  "${ROOT}/outputs/fulltools/current_no_reasoning_fulltools.json"
  "${ROOT}/outputs/fulltools/necessary_tool_no_reasoning_fulltools.json"
  "${ROOT}/outputs/fulltools/sparse_tool_no_reasoning_fulltools.json"
  "${ROOT}/outputs/fulltools/probe_prefill_t0.1_fulltools.json"
  "${ROOT}/outputs/fulltools/probe_prefill_t0.3_fulltools.json"
  "${ROOT}/outputs/fulltools/probe_prefill_t0.5_fulltools.json"
  "${ROOT}/outputs/fulltools/probe_prefill_t0.7_fulltools.json"
  "${ROOT}/outputs/fulltools/probe_prefill_t0.9_fulltools.json"
)

SCOPED_PROMPT_NAMES=(
  force_tool_no_reasoning_scoped.json
  current_no_reasoning_scoped.json
  necessary_tool_no_reasoning_scoped.json
  sparse_tool_no_reasoning_scoped.json
  no_tool_no_reasoning_scoped.json
  force_tool_reasoning_scoped.json
  current_reasoning_scoped.json
  necessary_tool_reasoning_scoped.json
  sparse_tool_reasoning_scoped.json
  no_tool_reasoning_scoped.json
)

SCOPED_ADAPTED_OUTPUTS=()
SCOPED_ORIGINAL_OUTPUTS=()
for name in "${SCOPED_PROMPT_NAMES[@]}"; do
  SCOPED_ADAPTED_OUTPUTS+=("${ROOT}/outputs/scoped_adapted/${name}")
  SCOPED_ORIGINAL_OUTPUTS+=("${ROOT}/outputs/scoped_original_w2t/${name}")
done
for threshold in 0.1 0.3 0.5 0.7 0.9; do
  SCOPED_ADAPTED_OUTPUTS+=(
    "${ROOT}/outputs/scoped_adapted/probe_prefill_t${threshold}_scoped.json"
  )
  SCOPED_ORIGINAL_OUTPUTS+=(
    "${ROOT}/outputs/scoped_original_w2t/probe_prefill_t${threshold}_scoped_original_w2t.json"
  )
done

ANALYSIS_ARGS=()
case "${OVERWRITE_ANALYSIS:-0}" in
  0)
    for analysis_dir in \
      "${FULL_ANALYSIS}" \
      "${SCOPED_ADAPTED_ANALYSIS}" \
      "${SCOPED_ORIGINAL_ANALYSIS}"; do
      if [[ -e "${analysis_dir}" ]]; then
        echo "Refusing to start: analysis target already exists: ${analysis_dir}" >&2
        echo "Remove it deliberately or rerun with OVERWRITE_ANALYSIS=1." >&2
        exit 1
      fi
    done
    ;;
  1)
    ANALYSIS_ARGS=(--overwrite)
    ;;
  *)
    echo "OVERWRITE_ANALYSIS must be 0 or 1" >&2
    exit 2
    ;;
esac

python -m when2tool_action.scripts.prepare_category_fulltools --config "${CONFIG}"
python -m when2tool_action.scripts.audit_provenance --config "${CONFIG}"

python -m when2tool_action.scripts.extract_tool_labels \
  --config "${CONFIG}" --tool-scope full
python -m when2tool_action.scripts.extract_tool_labels \
  --config "${CONFIG}" --tool-scope scoped

python -m when2tool_action.scripts.extract_hidden \
  --config "${CONFIG}" --tool-scope full
python -m when2tool_action.scripts.extract_hidden \
  --config "${CONFIG}" --tool-scope scoped

python -m when2tool_action.scripts.train_probes \
  --config "${CONFIG}" --probe-dir "${ROOT}/probes/fulltools"
python -m when2tool_action.scripts.train_probes \
  --config "${CONFIG}" --probe-dir "${ROOT}/probes/scoped" --binary-only

if [[ ! -f "${ROOT}/probes/scoped_original_w2t/migration_receipt.json" ]]; then
  LEGACY_SCOPED_ROOT="${LEGACY_SCOPED_ROOT:-../CallTool_data/experiments_2d/${MODEL}}"
  if [[ ! -d "${LEGACY_SCOPED_ROOT}" ]]; then
    echo "Missing original scoped baseline and legacy source: ${LEGACY_SCOPED_ROOT}" >&2
    exit 1
  fi
  python -m when2tool_action.scripts.import_legacy_scoped \
    --config "${CONFIG}" \
    --legacy-root "${LEGACY_SCOPED_ROOT}" \
    --output-root "${ROOT}" \
    --transfer-mode hardlink
fi

python -m when2tool_action.scripts.run_scoped_baseline \
  --config "${CONFIG}" \
  --data "${DATA}/tasks_v1_test_category.json" \
  --labels "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --output-dir "${ROOT}/outputs/scoped_adapted" \
  --resume

python -m when2tool_action.scripts.run_probe_prefill \
  --config "${CONFIG}" --tool-scope scoped \
  --data "${DATA}/tasks_v1_test_category.json" \
  --labels "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --probe-dir "${ROOT}/probes/scoped" \
  --output-dir "${ROOT}/outputs/scoped_adapted" \
  --resume

python -m when2tool_action.scripts.relabel_scoped_outputs \
  --config "${CONFIG}" \
  --inputs "${SCOPED_ADAPTED_OUTPUTS[@]:0:${#SCOPED_PROMPT_NAMES[@]}}" \
  --source-labels "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --target-labels "${LABELS}/test_labels_no_reasoning_scoped_original_w2t.json" \
  --output-dir "${ROOT}/outputs/scoped_original_w2t" \
  --protocol-id scoped_original_w2t \
  --overwrite

python -m when2tool_action.scripts.run_probe_prefill \
  --config "${CONFIG}" --tool-scope scoped \
  --probe-protocol scoped_original_w2t \
  --data "${DATA}/tasks_v1_test_category.json" \
  --labels "${LABELS}/test_labels_no_reasoning_scoped_original_w2t.json" \
  --probe-dir "${ROOT}/probes/scoped_original_w2t" \
  --output-dir "${ROOT}/outputs/scoped_original_w2t" \
  --resume

for prompt in current necessary_tool sparse_tool; do
  python -m when2tool_action.scripts.run_eval \
    --config "${CONFIG}" \
    --data "${DATA}/tasks_v1_test_fulltools_category.json" \
    --labels "${LABELS}/test_labels_no_reasoning_fulltools.json" \
    --output "${ROOT}/outputs/fulltools/${prompt}_no_reasoning_fulltools.json" \
    --setting-name "${prompt}_no_reasoning_fulltools" \
    --tool-scope full --prompt-mode "${prompt}" \
    --reasoning-mode no_reasoning --record-mode lite \
    --resume
done

python -m when2tool_action.scripts.run_probe_prefill \
  --config "${CONFIG}" --tool-scope full \
  --data "${DATA}/tasks_v1_test_fulltools_category.json" \
  --labels "${LABELS}/test_labels_no_reasoning_fulltools.json" \
  --probe-dir "${ROOT}/probes/fulltools" \
  --output-dir "${ROOT}/outputs/fulltools" \
  --resume

python -m when2tool_action.scripts.collect_action_stats \
  --outputs "${FULL_OUTPUTS[@]}" \
  --labels \
    "${LABELS}/train_labels_no_reasoning_fulltools.json" \
    "${LABELS}/test_labels_no_reasoning_fulltools.json" \
  --output-dir "${FULL_ANALYSIS}" \
  --expected-seeds 0 1 2 \
  "${ANALYSIS_ARGS[@]}"

python -m when2tool_action.scripts.collect_action_stats \
  --outputs "${SCOPED_ADAPTED_OUTPUTS[@]}" \
  --labels \
    "${LABELS}/train_labels_no_reasoning_scoped.json" \
    "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --output-dir "${SCOPED_ADAPTED_ANALYSIS}" \
  --expected-seeds 0 1 2 \
  "${ANALYSIS_ARGS[@]}"

python -m when2tool_action.scripts.collect_action_stats \
  --outputs "${SCOPED_ORIGINAL_OUTPUTS[@]}" \
  --labels \
    "${LABELS}/train_labels_no_reasoning_scoped_original_w2t.json" \
    "${LABELS}/test_labels_no_reasoning_scoped_original_w2t.json" \
  --output-dir "${SCOPED_ORIGINAL_ANALYSIS}" \
  --expected-seeds 0 1 2 \
  "${ANALYSIS_ARGS[@]}"
