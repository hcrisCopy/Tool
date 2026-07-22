"""Prepare Stage 7 SFT trajectories and train the single-GPU adapter panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from _stage_utils import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    StageRunner,
    audit_provenance,
    repo_path,
    require_files,
    require_unique,
)


MODEL_SLUG = "qwen3-4b-instruct-2507"
TRAIN_CONDITIONS = ("target", "dense", "random0", "random1", "random2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Stage 7 SFT data and/or train the fixed adapter "
            "panel. Paths are resolved relative to the repository root."
        ),
    )
    parser.add_argument(
        "--step",
        choices=("all", "prepare", "train"),
        default="all",
        help=(
            "Run both steps, prepare the SFT data only, or train adapters only "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"Input/output run root (default: {DEFAULT_RUN_ROOT})",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Primary mask JSON; defaults to Stage 5 rho0.003_signed",
    )
    parser.add_argument(
        "--causal-summary",
        type=Path,
        default=None,
        help="Stage 6 summary; defaults to stages/06_causal/conditions/summary.json",
    )
    parser.add_argument(
        "--causal-gate-passed",
        action="store_true",
        help="Required for training after manually checking the Stage 6 criterion",
    )
    parser.add_argument(
        "--train-conditions",
        nargs="+",
        choices=TRAIN_CONDITIONS,
        default=list(TRAIN_CONDITIONS),
        help="Adapters to train (default: target dense random0 random1 random2)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_unique(args.train_conditions, name="--train-conditions")

    config = repo_path(args.config)
    run_root = repo_path(args.run_root)
    data_dir = run_root / "data"
    labels_dir = run_root / "labels" / MODEL_SLUG
    primary_mask = (
        repo_path(args.mask)
        if args.mask is not None
        else run_root
        / "stages"
        / "05_probing"
        / "discovery"
        / "rho0.003_signed"
        / "tool_action_neurons.json"
    )
    causal_summary = (
        repo_path(args.causal_summary)
        if args.causal_summary is not None
        else run_root / "stages" / "06_causal" / "conditions" / "summary.json"
    )
    stage_root = run_root / "stages" / "07_training"
    sft_dir = stage_root / "sft"
    adapter_dir = stage_root / "adapters"
    manifest_dir = stage_root / "manifests"
    runtime_provenance = manifest_dir / "runtime_provenance.json"
    scoped_source = sft_dir / "train_scoped_current_seed0_fullrecord.json"
    sft_jsonl = sft_dir / "train_action_trajectories.jsonl"
    sft_manifest = sft_dir / "train_action_trajectories.manifest.json"

    require_files(
        [
            config,
            data_dir / "tasks_v1_train_category.json",
            data_dir / "tasks_v1_train_fulltools_category.json",
            labels_dir / "train_labels_no_reasoning_fulltools.json",
            labels_dir / "train_hard_no_tool_generations_fulltools.json",
            primary_mask,
        ],
        context="Stage 7 input",
    )

    sft_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    runner = StageRunner(stage_root / "logs" / "stage7_training.log")
    audit_provenance(runner, config=config, output=runtime_provenance)

    if args.step in {"all", "prepare"}:
        runner.run_module(
            "when2tool_action.scripts.run_eval",
            "--config",
            config,
            "--runtime-provenance",
            runtime_provenance,
            "--data",
            data_dir / "tasks_v1_train_category.json",
            "--labels",
            labels_dir / "train_labels_no_reasoning_fulltools.json",
            "--output",
            scoped_source,
            "--setting-name",
            "train_sft_current_no_reasoning_scoped",
            "--tool-scope",
            "scoped",
            "--prompt-mode",
            "current",
            "--reasoning-mode",
            "no_reasoning",
            "--record-mode",
            "full",
            "--seeds",
            0,
            "--resume",
        )
        runner.run_module(
            "when2tool_action.scripts.prepare_sft_trajectories",
            "--config",
            config,
            "--runtime-provenance",
            runtime_provenance,
            "--data",
            data_dir / "tasks_v1_train_fulltools_category.json",
            "--scoped-data",
            data_dir / "tasks_v1_train_category.json",
            "--labels",
            labels_dir / "train_labels_no_reasoning_fulltools.json",
            "--no-tool-generations",
            labels_dir / "train_hard_no_tool_generations_fulltools.json",
            "--scoped-trajectories",
            scoped_source,
            "--output",
            sft_jsonl,
            "--manifest",
            sft_manifest,
        )

    if args.step in {"all", "train"}:
        if not args.causal_gate_passed:
            raise RuntimeError(
                "Training is gated. Inspect "
                f"{causal_summary}, then rerun with --causal-gate-passed."
            )
        require_files(
            [causal_summary, sft_jsonl, sft_manifest],
            context="Stage 7 training prerequisite",
        )

        for condition in args.train_conditions:
            if condition == "target":
                mode = "target"
                output = adapter_dir / "target_neuron_lora"
                extra: list[object] = []
            elif condition == "dense":
                mode = "dense"
                output = adapter_dir / "dense_mlp_lora"
                extra = []
            else:
                random_seed = int(condition.removeprefix("random"))
                mode = "random"
                output = adapter_dir / f"random_neuron_lora_seed{random_seed}"
                extra = ["--random-seed", random_seed]

            runner.run_module(
                "when2tool_action.scripts.train_masked_lora",
                "--config",
                config,
                "--runtime-provenance",
                runtime_provenance,
                "--train-data",
                sft_jsonl,
                "--train-manifest",
                sft_manifest,
                "--mask",
                primary_mask,
                "--output-dir",
                output,
                "--mode",
                mode,
                *extra,
            )

    runner.message(f"Stage 7 completed for step={args.step}: {stage_root}")


if __name__ == "__main__":
    main()
