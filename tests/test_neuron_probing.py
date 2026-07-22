from __future__ import annotations

import json

import pytest
import torch

from when2tool_action.constants import ACTIONS
from when2tool_action.masked_lora import build_row_selection
from when2tool_action.mlp_activations import (
    ACTIVATION_SCHEMA_VERSION,
    DOWN_NORMS_FILENAME,
    PINNED_TRANSFORMERS_VERSION,
    activation_filename,
    activation_manifest_filename,
)
from when2tool_action.neuron_ablation import parse_neuron_mask
from when2tool_action.neuron_probing import (
    DEFAULT_CONTROL_SEED,
    MASK_SCHEMA_VERSION,
    activation_variant_mean,
    build_mask_artifact,
    build_stratified_controls,
    fit_selected_feature_probes,
    group_name,
    load_activation_split,
    preflight_probe_outputs,
)
from when2tool_action.scripts.probe_tool_action_neurons import build_parser


def _imbalanced_rows() -> list[dict]:
    specifications = [
        ("NONE", "easy"),
        ("NONE", "easy"),
        ("NONE", "easy"),
        ("NONE", "easy"),
        ("NONE", "hard"),
        ("NONE", "hard"),
        ("A", "easy"),
        ("A", "hard"),
        ("B", "easy"),
        ("B", "hard"),
        ("C", "easy"),
        ("C", "hard"),
    ]
    return [
        {
            "id": 100 + index,
            "gold_action": action,
            "difficulty": difficulty,
            "tool_necessary": int(action != "NONE"),
        }
        for index, (action, difficulty) in enumerate(specifications)
    ]


def test_controls_are_deterministic_no_replacement_and_maximally_stratified() -> None:
    rows = _imbalanced_rows()
    first = build_stratified_controls(rows, seed=17)
    second = build_stratified_controls(rows, seed=17)
    assert first == second
    none = first["NONE"]
    assert none.target_pool_size == 6
    assert none.rest_pool_size == 6
    assert none.nominal_n == 6
    # easy has 4 target vs 3 rest and hard has 2 target vs 3 rest.
    assert len(none.target_indices) == len(none.control_indices) == 5
    assert none.difficulty_allocation == {
        "easy": {
            "target_available": 4,
            "control_available": 3,
            "used_each_side": 3,
        },
        "hard": {
            "target_available": 2,
            "control_available": 3,
            "used_each_side": 2,
        },
    }
    assert len(none.target_ids) == len(set(none.target_ids))
    assert len(none.control_ids) == len(set(none.control_ids))
    assert none.target_difficulty_counts == none.control_difficulty_counts
    assert all(rows[index]["gold_action"] == "NONE" for index in none.target_indices)
    assert all(rows[index]["gold_action"] != "NONE" for index in none.control_indices)
    for action in ("A", "B", "C"):
        assert len(first[action].target_indices) == 2


def test_cli_control_seed_default_is_frozen_to_42() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--activations-dir",
                "activations",
                "--train-labels",
                "train.json",
                "--test-labels",
                "test.json",
                "--output-dir",
                "discovery",
            ]
        )
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--activations-dir",
            "activations",
            "--train-labels",
            "train.json",
            "--test-labels",
            "test.json",
            "--output-dir",
            "discovery",
            "--runtime-provenance",
            "runtime_provenance.json",
        ]
    )
    assert DEFAULT_CONTROL_SEED == 42
    assert args.control_seed == 42


def test_nine_group_names_are_unique_and_any_group_collision_is_rejected(
    tmp_path,
) -> None:
    rhos = (0.001, 0.003, 0.005)
    variants = ("signed", "positive", "abs")
    names = {
        group_name(rho, variant) for rho in rhos for variant in variants
    }
    assert len(names) == 9
    preflight_probe_outputs(tmp_path, rhos, variants)
    collision = tmp_path / "rho0.003_signed"
    collision.mkdir()
    with pytest.raises(FileExistsError, match="rho0.003_signed"):
        preflight_probe_outputs(tmp_path, rhos, variants)


def test_activation_loader_rejects_a_different_stage5_runtime_receipt(
    tmp_path,
) -> None:
    tensor_path = tmp_path / activation_filename("train")
    tensor_path.write_bytes(b"not reached")
    manifest = {
        "schema_version": ACTIVATION_SCHEMA_VERSION,
        "split": "train",
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
        "right_padding": True,
        "attention_implementation": "sdpa",
        "transformers_version": PINNED_TRANSFORMERS_VERSION,
        "tensor_file": tensor_path.name,
        "dtype": "torch.float16",
        "down_proj_column_norms_file": DOWN_NORMS_FILENAME,
        "runtime_provenance_sha256": "a" * 64,
        "project_git_commit": "commit-a",
    }
    (tmp_path / activation_manifest_filename("train")).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different Stage-5"):
        load_activation_split(
            tmp_path,
            "train",
            runtime_provenance={
                "sha256": "b" * 64,
                "git_commit": "commit-b",
            },
        )


def test_activation_variant_means_preserve_signed_cancellation() -> None:
    activations = torch.tensor(
        [[[1.0, -2.0, 3.0]], [[-1.0, 4.0, -3.0]]],
        dtype=torch.float16,
    )
    indices = (0, 1)
    signed = activation_variant_mean(
        activations, indices, variant="signed", chunk_size=1
    )
    positive = activation_variant_mean(
        activations, indices, variant="positive", chunk_size=2
    )
    absolute = activation_variant_mean(
        activations, indices, variant="abs", chunk_size=2
    )
    assert torch.equal(signed, torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.equal(positive, torch.tensor([[0.5, 2.0, 1.5]]))
    assert torch.equal(absolute, torch.tensor([[1.0, 3.0, 3.0]]))


def test_mask_uses_exact_topk_set_difference_without_refill() -> None:
    controls = build_stratified_controls(_imbalanced_rows(), seed=17)
    class_saliency = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.4, 0.3]],
        dtype=torch.float32,
    )
    control_saliency = torch.tensor(
        [[0.4, 0.1, 0.3, 0.2], [0.1, 0.3, 0.4, 0.2]],
        dtype=torch.float32,
    )
    scores = {
        action: {
            "mean_class": torch.full((2, 4), 2.0),
            "mean_control": torch.full((2, 4), 1.0),
            "saliency_class": class_saliency,
            "saliency_control": control_saliency,
            "direction": torch.ones((2, 4), dtype=torch.int8),
        }
        for action in ACTIONS
    }
    mask = build_mask_artifact(
        scores,
        torch.ones((2, 4), dtype=torch.float32),
        controls,
        rho=0.5,
        variant="signed",
        model_metadata={
            "slug": "fake-qwen",
            "architecture": "Qwen3ForCausalLM",
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "intermediate_size": 4,
        },
        config_metadata={
            "control_seed": 17,
            "mean_chunk_size": 2,
            "probe_C": 0.0001,
            "project_git_commit": "commit",
        },
        selection_hashes={
            "train_activation_sha256": "a" * 64,
            "runtime_provenance_sha256": "b" * 64,
        },
    )
    assert mask["schema_version"] == MASK_SCHEMA_VERSION
    assert mask["config"]["test_used_for_selection"] is False
    assert mask["total_neurons"] == 2
    for action in ACTIONS:
        selected = mask["classes"][action]
        assert selected["total_neurons"] == 2
        assert [row["neuron_idx"] for row in selected["layers"]["0"]] == [1]
        assert [row["neuron_idx"] for row in selected["layers"]["1"]] == [3]
        assert selected["layer_counts"] == {"0": 1, "1": 1}
        assert selected["layers"]["0"][0]["down_norm"] == 1.0
    none_control = mask["control"]["classes"]["NONE"]
    assert none_control["actual_n_each_side"] == 5
    assert len(none_control["target_ids"]) == len(none_control["control_ids"]) == 5

    parsed = parse_neuron_mask(
        mask,
        num_hidden_layers=2,
        intermediate_size=4,
        expected_model={
            "slug": "fake-qwen",
            "architecture": "Qwen3ForCausalLM",
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "intermediate_size": 4,
        },
    )
    row_selection = build_row_selection(
        parsed,
        mode="target",
        num_hidden_layers=2,
        intermediate_size=4,
    )
    assert row_selection.layers == {0: (1,), 1: (3,)}

    with pytest.raises(ValueError, match="selects k=0"):
        build_mask_artifact(
            scores,
            torch.ones((2, 4), dtype=torch.float32),
            controls,
            rho=0.001,
            variant="signed",
            model_metadata=mask["model"],
            config_metadata={
                "control_seed": 17,
                "mean_chunk_size": 2,
                "probe_C": 0.0001,
                "project_git_commit": "commit",
            },
            selection_hashes={
                "train_activation_sha256": "a" * 64,
                "runtime_provenance_sha256": "b" * 64,
            },
        )


def _probe_mask() -> dict:
    empty_layers = {"0": []}
    return {
        "schema_version": MASK_SCHEMA_VERSION,
        "classes": {
            "NONE": {
                "layers": {
                    "0": [
                        {"neuron_idx": 0},
                        {"neuron_idx": 1},
                    ]
                }
            },
            "A": {"layers": dict(empty_layers)},
            "B": {"layers": dict(empty_layers)},
            "C": {"layers": dict(empty_layers)},
        },
    }


def test_selected_union_binary_and_four_class_probes_fit_train_evaluate_test() -> None:
    points = {
        "NONE": (-2.0, -2.0),
        "A": (2.0, -2.0),
        "B": (-2.0, 2.0),
        "C": (2.0, 2.0),
    }
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    train_values: list[list[list[float]]] = []
    test_values: list[list[list[float]]] = []
    for action in ACTIONS:
        for repeat in range(4):
            train_rows.append({"gold_action": action})
            train_values.append([list(points[action])])
        for repeat in range(2):
            test_rows.append({"gold_action": action})
            test_values.append([list(points[action])])
    report, model = fit_selected_feature_probes(
        torch.tensor(train_values, dtype=torch.float16),
        torch.tensor(test_values, dtype=torch.float16),
        train_rows,
        test_rows,
        _probe_mask(),
        c=100.0,
        seed=23,
    )
    assert report["n_union_features"] == 2
    for key in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "majority_baseline",
    ):
        assert key in report["binary_tool_needed"]
        assert key in report["four_class_action"]
    assert report["binary_tool_needed"]["accuracy"] == 1.0
    assert report["four_class_action"]["accuracy"] == 1.0
    assert tuple(model["feature_pairs"].shape) == (2, 2)
    assert model["four_class"]["classes"] == list(ACTIONS)
    with pytest.raises(ValueError, match="four-class test is missing"):
        fit_selected_feature_probes(
            torch.tensor(train_values, dtype=torch.float16),
            torch.tensor(test_values[:-2], dtype=torch.float16),
            train_rows,
            test_rows[:-2],
            _probe_mask(),
            c=100.0,
            seed=23,
        )
