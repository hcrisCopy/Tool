#!/usr/bin/env bash
set -euo pipefail

CONFIG="when2tool_action/configs/qwen3_4b_instruct_2507.yaml"
MODEL="qwen3-4b-instruct-2507"
ROOT="../CallTool_data/when2tool_precise_shield/${MODEL}"
DATA="${ROOT}/data"
LABELS="${ROOT}/labels/${MODEL}"

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
  --outputs "${ROOT}"/outputs/fulltools/*.json \
  --labels "${LABELS}/test_labels_no_reasoning_fulltools.json" \
  --output-dir "${ROOT}/analysis/fulltools"

python -m when2tool_action.scripts.collect_action_stats \
  --outputs "${ROOT}"/outputs/scoped/*.json \
  --labels "${LABELS}/test_labels_no_reasoning_scoped.json" \
  --output-dir "${ROOT}/analysis/scoped"
