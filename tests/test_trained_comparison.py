from __future__ import annotations

import json

import pytest

from when2tool_action.adapter_evaluation import (
    ADAPTER_EVAL_SCHEMA_VERSION,
    CONDITION_IDS,
    collect_trained_evaluation_grid,
    compute_adapter_metrics,
    publish_trained_comparison,
    summarize_adapter_metrics,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _behavior_rows():
    return [
        {
            "gold_action": action,
            "pred_action": action,
            "final_correct": True,
            "total_tool_calls": int(action != "NONE"),
            "invalid_tool_calls": 0,
            "tool_call_categories": [] if action == "NONE" else [action],
            "mixed_category_calls": False,
            "tool_parse_failures": 0,
            "first_env_correct": bool(action != "NONE"),
            "exact_tool_allowed": bool(action != "NONE"),
            "first_arguments_valid": bool(action != "NONE"),
        }
        for action in ("NONE", "A", "B", "C")
    ]


def _metric_row(condition_id, scope, seed):
    return {
        "condition_id": condition_id,
        "tool_scope": scope,
        "generation_seed": seed,
        **compute_adapter_metrics(_behavior_rows()),
    }


def _write_cell(
    root,
    condition_id,
    scope,
    *,
    labels_sha="l" * 64,
    bundle_sha_override=None,
    training_protocol_marker="p",
):
    cell = root / scope / condition_id
    seeds = [0, 1, 2]
    adapter_sha = None if condition_id == "base_model" else condition_id[0] * 64
    if bundle_sha_override is not None:
        adapter_sha = bundle_sha_override
    training_control = (
        None
        if condition_id == "base_model"
        else {"condition_id": condition_id, "row_selection": condition_id}
    )
    training_protocol = (
        None
        if condition_id == "base_model"
        else {
            "protocol_fingerprint": training_protocol_marker * 64,
            "shared": True,
        }
    )
    config = {
        "model": "model-slug",
        "backend": "transformers-hf",
        "adapter": {
            "condition_id": condition_id,
            "bundle_sha256": adapter_sha,
            "training_control": training_control,
            "training_protocol": training_protocol,
        },
        "config_sha256": "c" * 64,
        "data_sha256": ("s" if scope == "scoped" else "f") * 64,
        "labels_sha256": labels_sha,
        "runtime_provenance_sha256": "r" * 64,
        "project_git_commit": "commit",
        "task_ids_sha256": "t" * 64,
        "tool_scope": scope,
    }
    manifest = {
        "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
        "manifest_type": "trained-hf-evaluation",
        "condition_id": condition_id,
        "tool_scope": scope,
        "generation_seeds": seeds,
        "config": config,
    }
    rows = [
        _metric_row(condition_id, scope, seed)
        for seed in seeds
    ]
    summary = {
        "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
        "metadata": {
            "condition_id": condition_id,
            "tool_scope": scope,
            "backend": "transformers-hf",
            "adapter_bundle_sha256": adapter_sha,
            "data_sha256": config["data_sha256"],
            "labels_sha256": labels_sha,
            "generation_seeds": seeds,
            "vllm_probe_prefill_is_context_only": True,
        },
        "per_seed_metrics": rows,
        "aggregate": summarize_adapter_metrics(rows),
    }
    _write_json(cell / "evaluation_manifest.json", manifest)
    _write_json(cell / "summary.json", summary)
    for row in rows:
        seed = row["generation_seed"]
        metrics = {
            key: value
            for key, value in row.items()
            if key not in {"condition_id", "tool_scope", "generation_seed"}
        }
        _write_json(
            cell / "trajectories" / f"seed_{seed}.json",
            {
                "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
                "config": config,
                "condition_id": condition_id,
                "tool_scope": scope,
                "generation_seed": seed,
                "run_id": f"hf_{condition_id}_{scope}_seed_{seed}",
                "metrics": metrics,
                "rows": _behavior_rows(),
            },
        )


def _write_grid(root):
    for scope in ("scoped", "full"):
        for condition_id in CONDITION_IDS:
            _write_cell(root, condition_id, scope)


def test_complete_grid_publishes_two_tables_json_and_registered_plots(tmp_path):
    root = tmp_path / "evaluations"
    _write_grid(root)
    destination = tmp_path / "comparison"
    payload = publish_trained_comparison(
        root, destination, require_complete=True
    )
    assert payload["panel_status"] == "complete"
    assert payload["completed_cells"] == 12
    assert payload["vllm_probe_prefill_context"]["included_in_numeric_comparison"] is False
    assert payload["hypothesis_3"]["statistical_significance_tested"] is False
    scoped_rows = [
        row
        for row in payload["condition_summaries"]
        if row["tool_scope"] == "scoped"
    ]
    assert all(type(row["scoped_pareto_front"]) is bool for row in scoped_rows)
    base = next(row for row in scoped_rows if row["condition_id"] == "base_model")
    assert base["TC_reduction_vs_same_backend_base"] == pytest.approx(0.0)
    assert base["Accuracy_loss_vs_same_backend_base"] == pytest.approx(0.0)
    for filename in (
        "comparison_per_seed.csv",
        "comparison_summary.csv",
        "comparison_summary.json",
        "target_vs_controls.csv",
        "target_vs_controls.json",
        "scoped_accuracy_vs_totaltc.png",
        "full_action_metrics.png",
    ):
        assert (destination / filename).is_file()
    target_payload = json.loads(
        (destination / "target_vs_controls.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value}")
        ),
    )
    assert target_payload["statistical_significance_tested"] is False
    assert len(target_payload["rows"]) == 2 * 16
    assert all(
        row["delta_target_minus_random3_mean"] == pytest.approx(0.0)
        for row in target_payload["rows"]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        publish_trained_comparison(root, destination, require_complete=True)


def test_require_complete_rejects_missing_cell(tmp_path):
    root = tmp_path / "partial"
    _write_cell(root, "base_model", "full")
    with pytest.raises(FileNotFoundError, match="Incomplete 12-cell"):
        collect_trained_evaluation_grid(root, require_complete=True)
    cells, missing = collect_trained_evaluation_grid(root, require_complete=False)
    assert len(cells) == 1
    assert len(missing) == 11


def test_cross_cell_label_hash_mismatch_is_rejected(tmp_path):
    root = tmp_path / "mismatch"
    _write_grid(root)
    # Rewrite one internally consistent cell with a different frozen label SHA.
    _write_cell(root, "target_neuron_lora", "full", labels_sha="x" * 64)
    with pytest.raises(ValueError, match="frozen labels"):
        collect_trained_evaluation_grid(root, require_complete=True)


def test_scoped_and_full_must_use_same_adapter_bundle(tmp_path):
    root = tmp_path / "bundle_mismatch"
    _write_grid(root)
    _write_cell(
        root,
        "target_neuron_lora",
        "full",
        bundle_sha_override="z" * 64,
    )
    with pytest.raises(ValueError, match="different adapter artifact"):
        collect_trained_evaluation_grid(root, require_complete=True)


def test_all_nonbase_adapters_must_share_training_protocol(tmp_path):
    root = tmp_path / "protocol_mismatch"
    _write_grid(root)
    for scope in ("scoped", "full"):
        _write_cell(
            root,
            "dense_mlp_lora",
            scope,
            training_protocol_marker="q",
        )
    with pytest.raises(ValueError, match="one frozen training protocol"):
        collect_trained_evaluation_grid(root, require_complete=True)
