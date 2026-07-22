"""Train target/random/dense gate/up LoRA under one frozen protocol."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from when2tool_action.config import REPO_ROOT, load_config, require_inputs
from when2tool_action.hf_agent import (
    QWEN3_4B_INTERMEDIATE_SIZE,
    validate_qwen3_model,
)
from when2tool_action.io_utils import (
    atomic_write_json,
    canonical_json_sha256,
    read_json,
    sha256_file,
)
from when2tool_action.masked_lora import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MASKED_LORA_SCHEMA_VERSION,
    MAX_LENGTH,
    RANDOM_LORA_SEEDS,
    TRAINING_SEED,
    AssistantOnlyCollator,
    AssistantOnlyDataset,
    audit_portable_adapter_metadata,
    attach_masked_lora,
    build_row_selection,
    load_offline_tokenizer,
    load_sft_records,
    normalize_peft_base_model_identity,
    normalize_tokenizer_model_identity,
    validate_training_dependency_versions,
    validate_probe_mask_metadata,
)
from when2tool_action.neuron_ablation import (
    NEURON_MASK_SCHEMA_VERSION,
    load_neuron_mask,
)
from when2tool_action.provenance import validate_runtime_provenance
from when2tool_action.upstream import frozen_full_menu


EPOCHS = 1.0
LEARNING_RATE = 1e-5
GLOBAL_BATCH_SIZE = 8
PER_DEVICE_BATCH_SIZE = 1
LR_SCHEDULER = "constant"
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0
CONTROL_IDS = {
    "target": "target_neuron_lora",
    "dense": "dense_mlp_lora",
}


def _control_id(mode: str, random_seed: int | None) -> str:
    if mode == "random":
        if random_seed not in RANDOM_LORA_SEEDS:
            raise ValueError(f"random mode seed must be one of {RANDOM_LORA_SEEDS}")
        return "random_neuron_lora"
    if mode not in CONTROL_IDS or random_seed is not None:
        raise ValueError("Invalid non-random control specification")
    return CONTROL_IDS[mode]


def _condition_id(mode: str, random_seed: int | None) -> str:
    """Return the seed-specific evaluation identity for one trained adapter."""

    control_id = _control_id(mode, random_seed)
    if mode == "random":
        return f"{control_id}_seed{random_seed}"
    return control_id


def _load_stage_runtime_provenance(config: object, path: Path) -> dict[str, object]:
    return validate_runtime_provenance(config, path.resolve())


def _validate_sft_runtime_binding(
    manifest: dict[str, Any], runtime_receipt: dict[str, object]
) -> None:
    """Require trajectory construction and training to use one Stage-07 receipt."""

    sft_runtime = manifest.get("runtime_provenance")
    if not isinstance(sft_runtime, dict):
        raise TypeError("SFT manifest lacks runtime_provenance")
    expected = {
        "file": Path(runtime_receipt["path"]).name,
        "sha256": runtime_receipt["sha256"],
        "project_git_commit": runtime_receipt["git_commit"],
    }
    if sft_runtime != expected:
        raise ValueError(
            "SFT trajectories and masked-LoRA training use different "
            "Stage-07 runtime provenance receipts"
        )


def _load_primary_training_mask(config: object, path: Path) -> object:
    """Load only the preregistered train-selected rho=.003 signed mask."""

    model = config.model
    return load_neuron_mask(
        path,
        num_hidden_layers=model.num_hidden_layers,
        intermediate_size=QWEN3_4B_INTERMEDIATE_SIZE,
        expected_model={
            "slug": model.slug,
            "architecture": model.architecture,
            "num_hidden_layers": model.num_hidden_layers,
            "hidden_size": model.hidden_size,
            "intermediate_size": QWEN3_4B_INTERMEDIATE_SIZE,
        },
        expected_rho=0.003,
        expected_variant="signed",
        require_train_selection=True,
    )


def _distributed_context() -> dict[str, int]:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE/RANK/LOCAL_RANK must be integers") from error
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid WORLD_SIZE/RANK combination")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    denominator = world_size * PER_DEVICE_BATCH_SIZE
    if GLOBAL_BATCH_SIZE % denominator != 0:
        raise ValueError(
            f"global batch {GLOBAL_BATCH_SIZE} is not divisible by "
            f"world_size*per_device_batch={denominator}"
        )
    accumulation = GLOBAL_BATCH_SIZE // denominator
    if accumulation < 1:
        raise ValueError("Gradient accumulation must be at least one")
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "gradient_accumulation_steps": accumulation,
    }


def _preflight_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(path)
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(
            f"Training output directory is non-empty; choose a new directory: {path}"
        )


def _hyperparameters(distributed: dict[str, int]) -> dict[str, Any]:
    return {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": distributed["gradient_accumulation_steps"],
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": list(LORA_TARGET_MODULES),
        "precision": "bf16",
        "gradient_checkpointing": True,
        "max_length": MAX_LENGTH,
        "lr_scheduler_type": LR_SCHEDULER,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "seed": TRAINING_SEED,
        "data_seed": TRAINING_SEED,
        "loss": "assistant_tokens_only_prefix_difference",
        "report_to": "none",
    }


def _training_protocol_fingerprint(
    *,
    sft_jsonl_sha256: str,
    sft_manifest_sha256: str,
    primary_mask_sha256: str,
    runtime_provenance_sha256: str,
    project_git_commit: str,
    base_model_slug: str,
    config_sha256: str,
    full_menu_sha256: str,
    hyperparameters: dict[str, Any],
    world_size: int,
) -> str:
    """Hash all shared fairness inputs, excluding condition-specific row choices."""

    return canonical_json_sha256(
        {
            "contract": "when2tool-stage7-shared-training-protocol-v1",
            "sft_jsonl_sha256": sft_jsonl_sha256,
            "sft_manifest_sha256": sft_manifest_sha256,
            "primary_mask_sha256": primary_mask_sha256,
            "runtime_provenance_sha256": runtime_provenance_sha256,
            "project_git_commit": project_git_commit,
            "base_model_slug": base_model_slug,
            "config_sha256": config_sha256,
            "full_menu_sha256": full_menu_sha256,
            "hyperparameters": hyperparameters,
            "world_size": world_size,
            "condition_specific_row_selection": "excluded_by_design",
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one fixed action-SFT control. Launch directly for one GPU or "
            "with torchrun; global batch remains exactly 8."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        required=True,
        help="Stage-07 runtime provenance receipt; the older global receipt is forbidden.",
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Default: replace train-data .jsonl with .manifest.json",
    )
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("target", "random", "dense"), required=True)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help=f"Required only for random mode; choices {RANDOM_LORA_SEEDS}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    distributed = _distributed_context()
    if args.mode == "random" and args.random_seed not in RANDOM_LORA_SEEDS:
        raise ValueError(f"random mode requires --random-seed in {RANDOM_LORA_SEEDS}")
    if args.mode != "random" and args.random_seed is not None:
        raise ValueError("--random-seed is valid only for random mode")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bf16 masked-LoRA training requires CUDA; CPU fallback is forbidden"
        )

    config = load_config(args.config)
    require_inputs(config)
    data_path = args.train_data.resolve()
    manifest_path = (
        args.train_manifest.resolve()
        if args.train_manifest
        else data_path.with_suffix(".manifest.json")
    )
    mask_path = args.mask.resolve()
    output_dir = args.output_dir.resolve()
    runtime_provenance_path = args.runtime_provenance.resolve()
    if runtime_provenance_path in {data_path, manifest_path, mask_path}:
        raise ValueError("Runtime provenance must be distinct from training inputs")
    if output_dir in {
        data_path.parent,
        manifest_path.parent,
        mask_path.parent,
        runtime_provenance_path.parent,
    }:
        raise ValueError("Adapter output directory must not be an input directory")
    _preflight_output(output_dir)
    runtime_receipt = (
        _load_stage_runtime_provenance(config, runtime_provenance_path)
        if distributed["rank"] == 0
        else None
    )
    dependency_versions = validate_training_dependency_versions()

    schemas, _routes, menu_sha256 = frozen_full_menu()
    tools = list(schemas)
    if len(tools) != 33:
        raise AssertionError("Canonical full menu must contain exactly 33 tools")
    records, sft_manifest = load_sft_records(
        data_path,
        manifest_path,
        expected_full_menu_sha256=menu_sha256,
    )
    if distributed["rank"] == 0:
        assert runtime_receipt is not None
        _validate_sft_runtime_binding(sft_manifest, runtime_receipt)
    mask = _load_primary_training_mask(config, mask_path)
    selection = build_row_selection(
        mask,
        mode=args.mode,
        num_hidden_layers=config.model.num_hidden_layers,
        intermediate_size=QWEN3_4B_INTERMEDIATE_SIZE,
        random_seed=args.random_seed,
    )
    probe_mask_receipt = validate_probe_mask_metadata(read_json(mask_path), selection)
    tokenizer = load_offline_tokenizer(config.paths.model)
    dataset = AssistantOnlyDataset(records, tokenizer, tools, max_length=MAX_LENGTH)
    collator = AssistantOnlyCollator(tokenizer.pad_token_id)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import (
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(TRAINING_SEED)
    model = AutoModelForCausalLM.from_pretrained(
        str(config.paths.model),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    validate_qwen3_model(
        model,
        expected_architecture=config.model.architecture,
        expected_num_hidden_layers=config.model.num_hidden_layers,
        expected_hidden_size=config.model.hidden_size,
        expected_intermediate_size=QWEN3_4B_INTERMEDIATE_SIZE,
    )
    model.config.use_cache = False
    model, controller, parameter_counts = attach_masked_lora(model, selection)
    normalized_adapters = normalize_peft_base_model_identity(model, config.model.slug)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    hparams = _hyperparameters(distributed)
    control_id = _control_id(args.mode, args.random_seed)
    condition_id = _condition_id(args.mode, args.random_seed)
    sft_jsonl_sha256 = sft_manifest["output"]["sha256"]
    sft_manifest_sha256 = sha256_file(manifest_path)
    mask_sha256 = mask.source_sha256
    config_sha256 = sha256_file(config.source)
    protocol_fingerprint = None
    if distributed["rank"] == 0:
        assert runtime_receipt is not None
        protocol_fingerprint = _training_protocol_fingerprint(
            sft_jsonl_sha256=sft_jsonl_sha256,
            sft_manifest_sha256=sft_manifest_sha256,
            primary_mask_sha256=mask_sha256,
            runtime_provenance_sha256=str(runtime_receipt["sha256"]),
            project_git_commit=str(runtime_receipt["git_commit"]),
            base_model_slug=config.model.slug,
            config_sha256=config_sha256,
            full_menu_sha256=menu_sha256,
            hyperparameters=hparams,
            world_size=distributed["world_size"],
        )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        do_train=True,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=distributed["gradient_accumulation_steps"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        seed=TRAINING_SEED,
        data_seed=TRAINING_SEED,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        disable_tqdm=distributed["rank"] != 0,
        log_on_each_node=False,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=(False if distributed["world_size"] > 1 else None),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    expected_update_steps = math.ceil(
        math.ceil(len(dataset) / (distributed["world_size"] * PER_DEVICE_BATCH_SIZE))
        / distributed["gradient_accumulation_steps"]
    )
    train_result = trainer.train()
    if (
        trainer.state.global_step != expected_update_steps
        or trainer.state.max_steps != expected_update_steps
    ):
        raise RuntimeError(
            "Trainer update-step count differs from the frozen global-batch protocol"
        )
    controller.assert_nonselected_rows_zero()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        current_hashes = {
            "sft_jsonl": sha256_file(data_path),
            "sft_manifest": sha256_file(manifest_path),
            "neuron_mask": sha256_file(mask_path),
            "runtime_provenance": sha256_file(runtime_provenance_path),
        }
        expected_hashes = {
            "sft_jsonl": sft_jsonl_sha256,
            "sft_manifest": sft_manifest_sha256,
            "neuron_mask": mask_sha256,
            "runtime_provenance": runtime_receipt["sha256"],
        }
        if current_hashes != expected_hashes:
            raise RuntimeError("A frozen Stage-07 input changed during training")
    trainer.accelerator.wait_for_everyone()
    normalize_peft_base_model_identity(model, config.model.slug)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        normalize_tokenizer_model_identity(tokenizer, config.model.slug)
        tokenizer.save_pretrained(str(output_dir))
        snapshot = selection.snapshot()
        snapshot.update(
            {
                "source_mask_file": mask_path.name,
                "source_mask_sha256": mask_sha256,
                "source_probe_union": probe_mask_receipt,
            }
        )
        snapshot_path = output_dir / "mask_snapshot.json"
        atomic_write_json(snapshot_path, snapshot)
        trainer_state = asdict(trainer.state)
        atomic_write_json(output_dir / "trainer_state.json", trainer_state)
        portable_metadata_audit = audit_portable_adapter_metadata(
            output_dir,
            forbidden_paths=(
                REPO_ROOT,
                config.paths.model,
                config.paths.dataset,
                config.paths.output_root,
                data_path,
                manifest_path,
                mask_path,
                runtime_provenance_path,
                output_dir,
            ),
        )
        train_manifest = {
            "schema_version": MASKED_LORA_SCHEMA_VERSION,
            "control_id": control_id,
            "condition_id": condition_id,
            "mode": args.mode,
            "random_mask_seed": args.random_seed,
            "training_seed": TRAINING_SEED,
            "base_model": config.model.slug,
            "config_sha256": config_sha256,
            "full_menu_sha256": menu_sha256,
            "inputs": {
                "sft_jsonl": {
                    "file": data_path.name,
                    "sha256": sft_jsonl_sha256,
                    "n_records": len(records),
                },
                "sft_manifest": {
                    "file": manifest_path.name,
                    "sha256": sft_manifest_sha256,
                    "input_bundle_sha256": sft_manifest["input_bundle_sha256"],
                },
                "neuron_mask": {
                    "file": mask_path.name,
                    "sha256": mask_sha256,
                    "schema_version": NEURON_MASK_SCHEMA_VERSION,
                },
            },
            "runtime_provenance": {
                "file": Path(runtime_receipt["path"]).name,
                "sha256": runtime_receipt["sha256"],
                "project_git_commit": runtime_receipt["git_commit"],
            },
            "hyperparameters": hparams,
            "dependency_versions": dependency_versions,
            "portable_model_identity": {
                "base_model_name_or_path": config.model.slug,
                "normalized_peft_adapters": list(normalized_adapters),
                "metadata_audit": portable_metadata_audit,
            },
            "distributed": distributed,
            "protocol_fingerprint": protocol_fingerprint,
            "parameter_counts": parameter_counts,
            "row_selection": {
                "indices_sha256": selection.indices_sha256,
                "target_union_sha256": selection.target_union_sha256,
                "probing_union_features_sha256": probe_mask_receipt[
                    "union_features_sha256"
                ],
                "selected_unique_rows": selection.selected_unique_rows,
                "snapshot_file": snapshot_path.name,
                "snapshot_sha256": sha256_file(snapshot_path),
            },
            "training_result": {
                "global_steps": trainer.state.global_step,
                "max_steps": trainer.state.max_steps,
                "expected_update_steps_per_epoch": expected_update_steps,
                "train_loss": train_result.training_loss,
                "metrics": train_result.metrics,
            },
            "post_training_assertion": "all non-selected LoRA-B rows are exactly zero",
        }
        atomic_write_json(output_dir / "train_manifest.json", train_manifest)
        print(
            f"Saved {condition_id} adapter: {output_dir}; "
            f"steps={trainer.state.global_step}, records={len(records)}, "
            f"raw/effective params={parameter_counts['raw_trainable_parameters']}/"
            f"{parameter_counts['effective_trainable_parameters']}"
        )
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
