from __future__ import annotations

import copy
import json

import pytest
import torch

from when2tool_action.constants import ACTIONS
from when2tool_action.neuron_ablation import (
    NEURON_MASK_SCHEMA_VERSION,
    RANDOM_MASK_SEEDS,
    NeuronAblationHooks,
    build_ablation_conditions,
    build_causal_criteria,
    compute_ablation_metrics,
    load_neuron_mask,
    parse_neuron_mask,
    validate_primary_mask_contract,
    write_ablation_reports,
)
from when2tool_action.scripts import run_neuron_ablation
from when2tool_action.io_utils import canonical_json_sha256


def _mask_payload() -> dict:
    return {
        "schema_version": NEURON_MASK_SCHEMA_VERSION,
        "classes": {
            action: {
                "total_neurons": 3,
                "layers": {
                    "0": [
                        {"neuron_idx": 1, "saliency_class": 3.0},
                        {"neuron_idx": 3, "saliency_class": 2.0},
                    ],
                    "1": [{"neuron_idx": 5, "saliency_class": 1.0}],
                    "2": [],
                },
            }
            for action in ACTIONS
        },
    }


def _primary_mask_payload() -> dict:
    payload = _mask_payload()
    payload.update(
        {
            "rho": 0.25,
            "activation_variant": "signed",
            "runtime_provenance_sha256": "6" * 64,
            "project_git_commit": "commit",
            "config": {
                "selection_split": "train",
                "test_used_for_selection": False,
                "control_seed": 42,
                "project_git_commit": "commit",
                "topk_rule": "floor(rho * intermediate_size) independently per layer",
                "topk_per_layer": 2,
            },
            "hashes": {
                "config_sha256": "1" * 64,
                "train_activation_sha256": "2" * 64,
                "train_activation_manifest_sha256": "3" * 64,
                "train_labels_sha256": "4" * 64,
                "down_proj_column_norms_sha256": "5" * 64,
                "runtime_provenance_sha256": "6" * 64,
            },
            "class_order": list(ACTIONS),
        }
    )
    root_counts = {}
    all_features = set()
    total_assignments = 0
    control_classes = {}
    for action_index, action in enumerate(ACTIONS):
        class_payload = payload["classes"][action]
        counts = {
            layer: len(entries) for layer, entries in class_payload["layers"].items()
        }
        class_payload.update(
            {
                "target_examples": 2,
                "control_examples": 2,
                "topk_per_layer_before_set_difference": 2,
                "total_neurons": sum(counts.values()),
                "layer_counts": counts,
            }
        )
        root_counts[action] = counts
        features = {
            (int(layer), entry["neuron_idx"])
            for layer, entries in class_payload["layers"].items()
            for entry in entries
        }
        all_features.update(features)
        total_assignments += len(features)
        target_ids = [action_index * 10 + 1, action_index * 10 + 2]
        control_ids = [action_index * 10 + 3, action_index * 10 + 4]
        control_classes[action] = {
            "actual_n_each_side": 2,
            "target_ids": target_ids,
            "target_ids_sha256": canonical_json_sha256(target_ids),
            "control_ids": control_ids,
            "control_ids_sha256": canonical_json_sha256(control_ids),
            "target_difficulty_counts": {"easy": 2},
            "control_difficulty_counts": {"easy": 2},
        }
    payload["layer_counts"] = root_counts
    payload["total_neurons"] = len(all_features)
    payload["total_class_assignments"] = total_assignments
    payload["union_features_sha256"] = canonical_json_sha256(
        [[layer, neuron] for layer, neuron in sorted(all_features)]
    )
    payload["control"] = {"base_seed": 42, "classes": control_classes}
    return payload


def test_mask_schema_and_25_condition_random_complements() -> None:
    payload = _mask_payload()
    payload["classes"]["NONE"]["layers"]["0"] = [{"neuron_idx": 6}]
    payload["classes"]["B"]["layers"]["0"] = [{"neuron_idx": 2}]
    payload["classes"]["C"]["layers"]["0"] = [{"neuron_idx": 4}]
    mask = parse_neuron_mask(
        payload, num_hidden_layers=3, intermediate_size=8
    )
    conditions = build_ablation_conditions(mask, intermediate_size=8)
    assert len(conditions) == 25
    assert conditions[0].condition_id == "no_mask"
    assert {condition.kind for condition in conditions} == {
        "no_mask",
        "target_mask",
        "random_mask",
    }
    target = next(item for item in conditions if item.condition_id == "target_A")
    target_union = {
        layer: {
            index
            for action in ACTIONS
            for index in mask.indices(action).get(layer, ())
        }
        for layer in range(3)
    }
    for seed in RANDOM_MASK_SEEDS:
        random_condition = next(
            item for item in conditions if item.condition_id == f"random_A_seed{seed}"
        )
        assert random_condition.mask_seed == seed
        for layer, target_indices in target.layers.items():
            random_indices = random_condition.layers[layer]
            assert len(random_indices) == len(target_indices)
            assert set(random_indices).isdisjoint(target_union[layer])
            manifest = random_condition.random_sampling["layers"][str(layer)]
            assert manifest["eligible_size"] == 8 - len(target_union[layer])
            assert manifest["sampled_indices"] == list(random_indices)
            assert len(manifest["eligible_indices_sha256"]) == 64
        assert len(random_condition.indices_sha256) == 64
    repeated = build_ablation_conditions(mask, intermediate_size=8)
    assert [item.to_dict() for item in repeated] == [
        item.to_dict() for item in conditions
    ]


def test_mask_schema_rejects_duplicates_and_out_of_range() -> None:
    duplicate = _mask_payload()
    duplicate["classes"]["A"]["layers"]["0"].append({"neuron_idx": 1})
    with pytest.raises(ValueError, match="Duplicate neurons"):
        parse_neuron_mask(duplicate, num_hidden_layers=3, intermediate_size=8)

    outside = _mask_payload()
    outside["classes"]["B"]["layers"]["1"] = [{"neuron_idx": 8}]
    with pytest.raises(ValueError, match="outside"):
        parse_neuron_mask(outside, num_hidden_layers=3, intermediate_size=8)

    wrong_schema = _mask_payload()
    wrong_schema["schema_version"] = "old"
    with pytest.raises(ValueError, match="schema_version"):
        parse_neuron_mask(wrong_schema, num_hidden_layers=3, intermediate_size=8)

    wrong_model = _mask_payload()
    wrong_model["model"] = {"architecture": "OtherForCausalLM"}
    with pytest.raises(ValueError, match="model.architecture"):
        parse_neuron_mask(
            wrong_model,
            num_hidden_layers=3,
            intermediate_size=8,
            expected_model={"architecture": "Qwen3ForCausalLM"},
        )


def test_primary_mask_contract_rejects_test_selection_totals_and_overlap(
    tmp_path,
) -> None:
    payload = _primary_mask_payload()
    validate_primary_mask_contract(
        payload,
        num_hidden_layers=3,
        intermediate_size=8,
        expected_rho=0.25,
        expected_variant="signed",
        require_train_selection=True,
    )
    path = tmp_path / "mask.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_neuron_mask(
        path,
        num_hidden_layers=3,
        intermediate_size=8,
        expected_rho=0.25,
        expected_variant="signed",
        require_train_selection=True,
    )
    assert loaded.indices("A")[0] == (1, 3)

    test_selected = copy.deepcopy(payload)
    test_selected["config"]["selection_split"] = "test"
    with pytest.raises(ValueError, match="selection_split"):
        validate_primary_mask_contract(
            test_selected,
            num_hidden_layers=3,
            intermediate_size=8,
            expected_rho=0.25,
            expected_variant="signed",
        )
    bad_total = copy.deepcopy(payload)
    bad_total["classes"]["A"]["total_neurons"] += 1
    with pytest.raises(ValueError, match="total_neurons"):
        validate_primary_mask_contract(
            bad_total,
            num_hidden_layers=3,
            intermediate_size=8,
            expected_rho=0.25,
            expected_variant="signed",
        )
    overlap = copy.deepcopy(payload)
    overlap["control"]["classes"]["B"]["control_ids"][0] = overlap["control"][
        "classes"
    ]["B"]["target_ids"][0]
    overlap["control"]["classes"]["B"]["control_ids_sha256"] = canonical_json_sha256(
        overlap["control"]["classes"]["B"]["control_ids"]
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_primary_mask_contract(
            overlap,
            num_hidden_layers=3,
            intermediate_size=8,
            expected_rho=0.25,
            expected_variant="signed",
        )


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(6, 1, bias=False)
        self.mlp.down_proj.weight.data.fill_(1)


def test_forward_pre_hook_masks_last_dimension_and_is_removed() -> None:
    layer = _Layer()
    values = torch.ones((2, 3, 6))
    hooks = NeuronAblationHooks(
        [layer], {0: (1, 4)}, intermediate_size=6
    )
    with hooks:
        masked_output = layer.mlp.down_proj(values)
        assert torch.all(masked_output == 4)
    assert hooks.call_counts == {0: 1}
    unmasked_output = layer.mlp.down_proj(values)
    assert torch.all(unmasked_output == 6)


def test_hook_removal_is_reliable_when_generation_raises() -> None:
    layer = _Layer()
    hooks = NeuronAblationHooks(
        [layer], {0: (0,)}, intermediate_size=6
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        with hooks:
            raise RuntimeError("generation failed")
    assert torch.all(layer.mlp.down_proj(torch.ones((1, 6))) == 6)


def _behavior_row(gold: str, pred: str, correct: bool, calls: int) -> dict:
    return {
        "gold_action": gold,
        "pred_action": pred,
        "final_correct": correct,
        "total_tool_calls": calls,
        "invalid_tool_calls": int(pred == "INVALID"),
        "tool_call_categories": [pred] if calls else [],
        "mixed_category_calls": False,
        "tool_parse_failures": 0,
        "first_env_correct": bool(calls and pred == gold),
        "exact_tool_allowed": bool(calls and pred == gold),
        "first_arguments_valid": bool(calls and pred != "INVALID"),
    }


def test_metrics_cover_action_recall_calls_overcall_invalid_and_offtarget() -> None:
    rows = [
        _behavior_row("NONE", "A", False, 1),
        _behavior_row("A", "A", True, 1),
        _behavior_row("B", "INVALID", False, 1),
        _behavior_row("C", "NONE", False, 0),
    ]
    rows[2]["tool_parse_failures"] = 2
    metrics = compute_ablation_metrics(rows, masked_class="B")
    assert metrics["ActionAcc"] == pytest.approx(0.25)
    assert metrics["Recall_NONE"] == pytest.approx(0.0)
    assert metrics["Recall_A"] == pytest.approx(1.0)
    assert metrics["Recall_B"] == pytest.approx(0.0)
    assert metrics["Recall_C"] == pytest.approx(0.0)
    assert metrics["FinalAcc"] == pytest.approx(0.25)
    assert metrics["TotalTC"] == 3
    assert metrics["OverCall"] == pytest.approx(1.0)
    assert metrics["InvalidRate"] == pytest.approx(0.25)
    assert metrics["OffTargetError"] == pytest.approx(2 / 3)
    assert metrics["UnderCall"] == pytest.approx(1 / 3)
    assert metrics["WrongCat"] == pytest.approx(0.0)
    assert metrics["UnderCall_C"] == pytest.approx(1.0)
    assert metrics["WrongCat_B"] == pytest.approx(0.0)
    assert metrics["MixedCategory"] == pytest.approx(0.0)
    assert metrics["FirstCallN"] == 3
    assert metrics["FirstEnvCorrect"] == pytest.approx(1 / 3)
    assert metrics["ExactToolAllowed"] == pytest.approx(1 / 3)
    assert metrics["FirstArgumentsValid"] == pytest.approx(2 / 3)
    assert metrics["ToolParseFailures"] == 2
    assert metrics["ToolParseFailureRate"] == pytest.approx(0.25)
    assert compute_ablation_metrics(rows, masked_class=None)["OffTargetError"] is None


def test_condition_summary_uses_json_null_for_undefined_metric(tmp_path) -> None:
    mask = parse_neuron_mask(
        _mask_payload(), num_hidden_layers=3, intermediate_size=8
    )
    condition = build_ablation_conditions(mask, intermediate_size=8)[0]
    rows = [
        _behavior_row(action, action, True, int(action != "NONE"))
        for action in ACTIONS
    ]
    checkpoint = {
        "schema_version": "when2tool-neuron-ablation-v1",
        "condition": condition.to_dict(),
        "generation_seed": 0,
        "metrics": compute_ablation_metrics(rows, masked_class=None),
        "rows": rows,
    }
    summary = write_ablation_reports(
        [checkpoint],
        tmp_path,
        expected_condition_ids=["no_mask"],
        expected_generation_seeds=[0],
    )
    row = summary["condition_summaries"][0]
    assert row["OffTargetError_mean"] is None
    json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value}")
        ),
    )


def test_causal_manifest_resume_only_allows_validated_seed_append(
    tmp_path, monkeypatch
) -> None:
    mask = parse_neuron_mask(
        _mask_payload(), num_hidden_layers=3, intermediate_size=8
    )
    condition = build_ablation_conditions(mask, intermediate_size=8)[0]
    common = {"protocol": "fixed"}
    old = {
        "schema_version": "when2tool-neuron-ablation-v1",
        "manifest_type": "hf-neuron-ablation-condition-matrix",
        "config": common,
        "generation_seeds": [0],
        "condition_count": 1,
        "seed_condition_count": 1,
        "conditions": [condition.to_dict()],
    }
    manifest_path = tmp_path / "condition_manifest.json"
    manifest_path.write_text(json.dumps(old), encoding="utf-8")
    checkpoint = run_neuron_ablation._checkpoint_path(tmp_path, condition, 0)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}", encoding="utf-8")
    validated = []

    def fake_read(path, *, expected, task_ids):
        validated.append((path, expected["generation_seed"], task_ids))
        return {}

    monkeypatch.setattr(run_neuron_ablation, "_read_checkpoint", fake_read)
    run_neuron_ablation._prepare_condition_manifest(
        tmp_path,
        common=common,
        conditions=[condition],
        generation_seeds=(0, 1),
        task_ids=[10],
        overwrite=False,
        resume=True,
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        "generation_seeds"
    ] == [0, 1]
    assert validated == [(checkpoint, 0, [10])]
    with pytest.raises(ValueError, match="append-only"):
        run_neuron_ablation._prepare_condition_manifest(
            tmp_path,
            common=common,
            conditions=[condition],
            generation_seeds=(0,),
            task_ids=[10],
            overwrite=False,
            resume=True,
        )


def test_recall_drop_criterion_is_explicitly_not_significance() -> None:
    metric_rows = []
    for action in ACTIONS:
        metric = f"Recall_{action}"
        baseline = {
            "condition_id": "no_mask",
            "generation_seed": 0,
            metric: 0.9,
        }
        # Shared baseline is added only once, with all four recall fields.
        if not metric_rows:
            baseline.update({f"Recall_{other}": 0.9 for other in ACTIONS})
            metric_rows.append(baseline)
        metric_rows.append(
            {
                "condition_id": f"target_{action}",
                "generation_seed": 0,
                metric: 0.4,
            }
        )
        for seed, drop in zip(RANDOM_MASK_SEEDS, (0.05, 0.08, 0.10, 0.06, 0.09)):
            metric_rows.append(
                {
                    "condition_id": f"random_{action}_seed{seed}",
                    "generation_seed": 0,
                    metric: 0.9 - drop,
                }
            )
    criteria = build_causal_criteria(metric_rows)
    assert len(criteria) == 4
    assert all(row["status"] == "complete" for row in criteria)
    assert all(row["criterion_met"] is True for row in criteria)
    assert all("not statistical significance" in row["interpretation"] for row in criteria)


def test_complete_matrix_writes_metrics_summary_and_three_registered_plots(
    tmp_path,
) -> None:
    mask = parse_neuron_mask(
        _mask_payload(), num_hidden_layers=3, intermediate_size=8
    )
    conditions = build_ablation_conditions(mask, intermediate_size=8)
    checkpoints = []
    for condition in conditions:
        rows = [
            _behavior_row(action, action, True, int(action != "NONE"))
            for action in ACTIONS
        ]
        if condition.kind == "target_mask":
            target_index = ACTIONS.index(condition.masked_class)
            replacement = "A" if condition.masked_class == "NONE" else "NONE"
            rows[target_index] = _behavior_row(
                condition.masked_class, replacement, False, int(replacement != "NONE")
            )
        checkpoints.append(
            {
                "schema_version": "when2tool-neuron-ablation-v1",
                "condition": condition.to_dict(),
                "generation_seed": 0,
                "metrics": compute_ablation_metrics(
                    rows, masked_class=condition.masked_class
                ),
                "rows": rows,
            }
        )
    summary = write_ablation_reports(
        checkpoints,
        tmp_path,
        expected_condition_ids=[condition.condition_id for condition in conditions],
        expected_generation_seeds=[0],
    )
    assert summary["panel_status"] == "complete"
    assert summary["criteria_are_statistical_significance_tests"] is False
    for filename in (
        "per_condition_metrics.csv",
        "summary.csv",
        "summary.json",
        "recall_drop.png",
        "target_vs_random.png",
        "confusion_before_after.png",
    ):
        assert (tmp_path / filename).is_file()


def test_causal_runner_passes_explicit_stage_runtime_receipt(
    tmp_path, monkeypatch
) -> None:
    receipt = tmp_path / "stages" / "06_causal" / "runtime_provenance.json"
    captured = []

    def fake_validate(config, path):
        captured.append((config, path))
        return {"path": path, "sha256": "a" * 64, "git_commit": "commit"}

    monkeypatch.setattr(
        run_neuron_ablation, "validate_runtime_provenance", fake_validate
    )
    config = object()
    result = run_neuron_ablation._load_stage_runtime_provenance(config, receipt)
    assert captured == [(config, receipt.resolve())]
    assert result["sha256"] == "a" * 64
