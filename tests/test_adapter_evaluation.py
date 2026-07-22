from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from when2tool_action.adapter_evaluation import (
    BASE_CONDITION_ID,
    compute_adapter_metrics,
    inspect_adapter,
    load_adapter_into_agent,
    summarize_adapter_metrics,
    validate_condition_id,
    write_adapter_summary,
)
from when2tool_action.scripts import run_trained_eval
from when2tool_action.io_utils import canonical_json_sha256


def _write_adapter(
    path,
    *,
    control_id="target_neuron_lora",
    random_mask_seed=None,
    condition_id=None,
):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "model-slug",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "bias": "none",
                "inference_mode": True,
                "target_modules": ["gate_proj", "up_proj"],
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"fake-weights")
    if condition_id is None:
        condition_id = (
            f"random_neuron_lora_seed{random_mask_seed}"
            if control_id == "random_neuron_lora"
            else control_id
        )
    hyperparameters = {
        "epochs": 1.0,
        "learning_rate": 1e-5,
        "global_batch_size": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["gate_proj", "up_proj"],
        "precision": "bf16",
        "gradient_checkpointing": True,
        "max_length": 8192,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "seed": 42,
        "data_seed": 42,
        "loss": "assistant_tokens_only_prefix_difference",
        "report_to": "none",
    }
    fingerprint_payload = {
        "contract": "when2tool-stage7-shared-training-protocol-v1",
        "sft_jsonl_sha256": "1" * 64,
        "sft_manifest_sha256": "2" * 64,
        "primary_mask_sha256": "3" * 64,
        "runtime_provenance_sha256": "4" * 64,
        "project_git_commit": "commit",
        "base_model_slug": "model-slug",
        "config_sha256": "5" * 64,
        "full_menu_sha256": "6" * 64,
        "hyperparameters": hyperparameters,
        "world_size": 1,
        "condition_specific_row_selection": "excluded_by_design",
    }
    (path / "train_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "when2tool-masked-lora-v1",
                "control_id": control_id,
                "condition_id": condition_id,
                "mode": "target",
                "random_mask_seed": random_mask_seed,
                "training_seed": 42,
                "base_model": "model-slug",
                "config_sha256": "5" * 64,
                "full_menu_sha256": "6" * 64,
                "inputs": {
                    "sft_jsonl": {
                        "sha256": "1" * 64,
                        "n_records": 900,
                    },
                    "sft_manifest": {
                        "sha256": "2" * 64,
                        "input_bundle_sha256": "7" * 64,
                    },
                    "neuron_mask": {
                        "sha256": "3" * 64,
                        "schema_version": "when2tool-neuron-mask-v1",
                    },
                },
                "runtime_provenance": {
                    "sha256": "4" * 64,
                    "project_git_commit": "commit",
                },
                "hyperparameters": hyperparameters,
                "dependency_versions": {"torch": "2.6.0"},
                "distributed": {"world_size": 1},
                "protocol_fingerprint": canonical_json_sha256(fingerprint_payload),
                "row_selection": {"indices_sha256": "8" * 64},
            }
        ),
        encoding="utf-8",
    )


def test_adapter_identity_hashes_every_file_and_base_is_explicit(tmp_path) -> None:
    base = inspect_adapter(BASE_CONDITION_ID, None, base_model_slug="model-slug")
    assert base.kind == "base"
    assert base.bundle_sha256 is None
    with pytest.raises(ValueError, match="forbids"):
        inspect_adapter(BASE_CONDITION_ID, tmp_path, base_model_slug="model-slug")

    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    identity = inspect_adapter(
        "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
    )
    assert identity.kind == "peft_adapter"
    assert len(identity.bundle_sha256) == 64
    assert {item["path"] for item in identity.files} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "train_manifest.json",
    }
    original = identity.bundle_sha256
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"changed")
    changed = inspect_adapter(
        "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
    )
    assert changed.bundle_sha256 != original


def test_adapter_validation_rejects_control_id_or_protocol_mismatch(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir, control_id="dense_mlp_lora")
    with pytest.raises(ValueError, match="control_id"):
        inspect_adapter(
            "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
        )
    with pytest.raises(ValueError, match="condition_id"):
        validate_condition_id("../escape")


def test_adapter_validation_rejects_rewritten_training_protocol(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    manifest_path = adapter_dir / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_provenance"]["project_git_commit"] = "other-commit"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_fingerprint does not recompute"):
        inspect_adapter(
            "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
        )


def test_adapter_validation_rejects_unregistered_hyperparameters(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    manifest_path = adapter_dir / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hyperparameters"]["learning_rate"] = 2e-5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="registered Stage-7 contract"):
        inspect_adapter(
            "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
        )


def test_adapter_validation_rejects_nonportable_base_identity(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "/" + "internal/model/path"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        inspect_adapter(
            "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
        )


def test_adapter_validation_rejects_unregistered_lora_config(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inference_mode"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="inference_mode"):
        inspect_adapter(
            "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
        )


def test_random_condition_id_separates_eval_identity_from_training_control(tmp_path) -> None:
    adapter_dir = tmp_path / "random_adapter"
    _write_adapter(
        adapter_dir, control_id="random_neuron_lora", random_mask_seed=1
    )
    identity = inspect_adapter(
        "random_neuron_lora_seed1", adapter_dir, base_model_slug="model-slug"
    )
    assert identity.condition_id == "random_neuron_lora_seed1"
    assert identity.training_control["control_id"] == "random_neuron_lora"
    assert identity.training_control["random_mask_seed"] == 1
    manifest_path = adapter_dir / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["condition_id"] = "random_neuron_lora_seed2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="random_mask_seed"):
        inspect_adapter(
            "random_neuron_lora_seed2", adapter_dir, base_model_slug="model-slug"
        )


def test_adapter_directory_cannot_impersonate_another_condition(tmp_path) -> None:
    adapter_dir = tmp_path / "renamed_adapter"
    _write_adapter(
        adapter_dir,
        control_id="target_neuron_lora",
        condition_id="dense_mlp_lora",
    )
    with pytest.raises(ValueError, match="Training condition_id"):
        inspect_adapter(
            "target_neuron_lora", adapter_dir, base_model_slug="model-slug"
        )


class _FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)


def test_injected_fake_loader_attaches_one_adapter_without_importing_peft(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    base_model = _FrozenModel()
    agent = SimpleNamespace(model=base_model, device=torch.device("cpu"))
    calls = []

    def fake_loader(model, path, adapter_id):
        calls.append((model, path, adapter_id))
        return _FrozenModel()

    returned = load_adapter_into_agent(
        agent,
        condition_id="target_neuron_lora",
        adapter_dir=adapter_dir,
        loader=fake_loader,
    )
    assert returned is agent
    assert calls == [(base_model, adapter_dir.resolve(), "target_neuron_lora")]
    assert returned.model.training is False


def _row(gold, pred, final, calls, *, mixed=False):
    categories = [pred] if calls else []
    if mixed:
        categories = ["A", pred]
    return {
        "gold_action": gold,
        "pred_action": pred,
        "final_correct": final,
        "total_tool_calls": calls,
        "invalid_tool_calls": int(pred == "INVALID"),
        "tool_call_categories": categories,
        "mixed_category_calls": mixed,
        "tool_parse_failures": 0,
        "first_env_correct": bool(calls and pred == gold),
        "exact_tool_allowed": bool(calls and pred == gold),
        "first_arguments_valid": bool(calls and pred != "INVALID"),
    }


def test_adapter_metrics_include_registered_action_and_error_panel() -> None:
    rows = [
        _row("NONE", "A", False, 1),
        _row("A", "A", True, 1),
        _row("B", "NONE", False, 0),
        _row("C", "B", False, 2, mixed=True),
    ]
    metrics = compute_adapter_metrics(rows)
    assert metrics["Accuracy"] == pytest.approx(0.25)
    assert metrics["FinalAcc"] == pytest.approx(0.25)
    assert metrics["TotalTC"] == 4
    assert metrics["TCR"] == pytest.approx(1.0)
    assert metrics["TCR"] == metrics["AvgTC"]
    assert metrics["ToolNeed_F1"] == pytest.approx(2 / 3)
    assert metrics["ActionAcc"] == pytest.approx(0.25)
    assert metrics["Recall_NONE"] == pytest.approx(0.0)
    assert metrics["Recall_A"] == pytest.approx(1.0)
    assert metrics["OverCall"] == pytest.approx(1.0)
    assert metrics["UnderCall"] == pytest.approx(1 / 3)
    assert metrics["WrongCat"] == pytest.approx(1 / 3)
    assert metrics["Invalid"] == pytest.approx(0.0)
    assert metrics["Mixed"] == pytest.approx(0.25)


def test_eval_manifest_resume_only_allows_validated_seed_append(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "evaluation"
    checkpoint = output_dir / "trajectories" / "seed_0.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}", encoding="utf-8")
    common = {
        "adapter": {"condition_id": "base_model"},
        "tool_scope": "full",
    }
    old = {
        "schema_version": "when2tool-adapter-eval-v1",
        "manifest_type": "trained-hf-evaluation",
        "condition_id": "base_model",
        "tool_scope": "full",
        "generation_seeds": [0],
        "config": common,
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(old), encoding="utf-8")
    validated = []

    def fake_read(path, *, expected, task_ids):
        validated.append((path, expected["generation_seed"], task_ids))
        return {}

    monkeypatch.setattr(run_trained_eval, "_read_checkpoint", fake_read)
    extended = {**old, "generation_seeds": [0, 1]}
    run_trained_eval._prepare_evaluation_manifest(
        manifest_path,
        extended,
        output_dir=output_dir,
        common=common,
        task_ids=[10],
        overwrite=False,
        resume=True,
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        "generation_seeds"
    ] == [0, 1]
    assert validated == [(checkpoint, 0, [10])]
    with pytest.raises(ValueError, match="append-only"):
        run_trained_eval._prepare_evaluation_manifest(
            manifest_path,
            old,
            output_dir=output_dir,
            common=common,
            task_ids=[10],
            overwrite=False,
            resume=True,
        )


def test_summary_aggregates_inside_adapter_condition_across_generation_seeds(
    tmp_path,
) -> None:
    rows = [
        {
            "condition_id": "base_model",
            "tool_scope": "full",
            "generation_seed": seed,
            "Accuracy": value,
            "FinalAcc": value,
            "TotalTC": 10 + seed,
        }
        for seed, value in ((0, 0.5), (1, 0.7), (2, 0.6))
    ]
    aggregate = summarize_adapter_metrics(rows)
    assert aggregate["n_generation_seeds"] == 3
    assert aggregate["Accuracy_mean"] == pytest.approx(0.6)
    assert aggregate["Accuracy_population_sd"] == pytest.approx(
        (2 / 300) ** 0.5
    )
    summary = write_adapter_summary(
        tmp_path,
        metadata={"condition_id": "base_model", "tool_scope": "full"},
        per_seed_rows=rows,
    )
    assert summary["metric_contract"]["vllm_probe_prefill"].startswith(
        "context-only"
    )
    for name in ("per_seed_metrics.csv", "summary.csv", "summary.json"):
        assert (tmp_path / name).is_file()


def test_trained_runner_passes_explicit_stage_runtime_receipt(
    tmp_path, monkeypatch
) -> None:
    receipt = tmp_path / "stages" / "08_evaluation" / "runtime_provenance.json"
    captured = []

    def fake_validate(config, path):
        captured.append((config, path))
        return {"path": path, "sha256": "b" * 64, "git_commit": "commit"}

    monkeypatch.setattr(run_trained_eval, "validate_runtime_provenance", fake_validate)
    config = object()
    result = run_trained_eval._load_stage_runtime_provenance(config, receipt)
    assert captured == [(config, receipt.resolve())]
    assert result["sha256"] == "b" * 64
