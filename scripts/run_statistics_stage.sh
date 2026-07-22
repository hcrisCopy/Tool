#!/usr/bin/env bash
set -euo pipefail

CONFIG="when2tool_action/configs/qwen3_4b_instruct_2507.yaml"
MODEL="qwen3-4b-instruct-2507"
ROOT="../CallTool_data/when2tool_precise_shield/${MODEL}"
DATA="${ROOT}/data"
LABELS="${ROOT}/labels/${MODEL}"
FULL_ANALYSIS="${ROOT}/analysis/fulltools"
SCOPED_ANALYSIS="${ROOT}/analysis/scoped"

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

SCOPED_OUTPUTS=(
  "${ROOT}/outputs/scoped/force_tool_no_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/current_no_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/necessary_tool_no_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/sparse_tool_no_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/no_tool_no_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/force_tool_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/current_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/necessary_tool_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/sparse_tool_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/no_tool_reasoning_scoped.json"
  "${ROOT}/outputs/scoped/probe_prefill_t0.1_scoped.json"
  "${ROOT}/outputs/scoped/probe_prefill_t0.3_scoped.json"
  "${ROOT}/outputs/scoped/probe_prefill_t0.5_scoped.json"
  "${ROOT}/outputs/scoped/probe_prefill_t0.7_scoped.json"
  "${ROOT}/outputs/scoped/probe_prefill_t0.9_scoped.json"
)

ANALYSIS_ARGS=()
case "${OVERWRITE_ANALYSIS:-0}" in
  0)
    for analysis_dir in "${FULL_ANALYSIS}" "${SCOPED_ANALYSIS}"; do
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

python -m when2tool_action.scripts.run_scoped_baseline \
  --config "${CONFIG}" \
  --data "${DATA}/tasks_v1_test_category.json" \
  --labels "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --output-dir "${ROOT}/outputs/scoped"

python -m when2tool_action.scripts.run_probe_prefill \
  --config "${CONFIG}" --tool-scope scoped \
  --data "${DATA}/tasks_v1_test_category.json" \
  --labels "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --probe-dir "${ROOT}/probes/scoped" \
  --output-dir "${ROOT}/outputs/scoped"

for prompt in current necessary_tool sparse_tool; do
  python -m when2tool_action.scripts.run_eval \
    --config "${CONFIG}" \
    --data "${DATA}/tasks_v1_test_fulltools_category.json" \
    --labels "${LABELS}/test_labels_no_reasoning_fulltools.json" \
    --output "${ROOT}/outputs/fulltools/${prompt}_no_reasoning_fulltools.json" \
    --setting-name "${prompt}_no_reasoning_fulltools" \
    --tool-scope full --prompt-mode "${prompt}" \
    --reasoning-mode no_reasoning --record-mode lite
done

python -m when2tool_action.scripts.run_probe_prefill \
  --config "${CONFIG}" --tool-scope full \
  --data "${DATA}/tasks_v1_test_fulltools_category.json" \
  --labels "${LABELS}/test_labels_no_reasoning_fulltools.json" \
  --probe-dir "${ROOT}/probes/fulltools" \
  --output-dir "${ROOT}/outputs/fulltools"

python -m when2tool_action.scripts.collect_action_stats \
  --outputs "${FULL_OUTPUTS[@]}" \
  --labels "${LABELS}/test_labels_no_reasoning_fulltools.json" \
  --output-dir "${FULL_ANALYSIS}" \
  --expected-seeds 0 1 2 \
  "${ANALYSIS_ARGS[@]}"

python -m when2tool_action.scripts.collect_action_stats \
  --outputs "${SCOPED_OUTPUTS[@]}" \
  --labels "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --output-dir "${SCOPED_ANALYSIS}" \
  --expected-seeds 0 1 2 \
  "${ANALYSIS_ARGS[@]}"
