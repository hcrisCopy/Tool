"""Run and summarize the Stage 8 trained-adapter evaluation panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from _stage_utils import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    StageRunner,
    audit_provenance,
    repo_path,
    require_directories,
    require_files,
    require_unique,
)


MODEL_SLUG = "qwen3-4b-instruct-2507"
CONDITIONS = (
    "base_model",
    "target_neuron_lora",
    "dense_mlp_lora",
    "random_neuron_lora_seed0",
    "random_neuron_lora_seed1",
    "random_neuron_lora_seed2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and/or summarize the Stage 8 evaluation panel. "
            "Paths are resolved relative to the repository root."
        ),
    )
    parser.add_argument(
        "--step",
        choices=("all", "evaluation", "summary"),
        default="all",
        help=(
            "Run evaluation plus summary, evaluation only, or summary only "
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
        "--generation-seeds",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=[0, 1, 2],
        help="Strict seed panel to run/resume (default: 0 1 2)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_unique(args.generation_seeds, name="--generation-seeds")

    config = repo_path(args.config)
    run_root = repo_path(args.run_root)
    data_dir = run_root / "data"
    labels_dir = run_root / "labels" / MODEL_SLUG
    adapter_root = run_root / "stages" / "07_training" / "adapters"
    stage_root = run_root / "stages" / "08_evaluation"
    output_root = stage_root / "outputs"
    comparison_dir = stage_root / "comparison"
    manifest_dir = stage_root / "manifests"
    runtime_provenance = manifest_dir / "runtime_provenance.json"

    require_files(
        [
            config,
            data_dir / "tasks_v1_test_category.json",
            data_dir / "tasks_v1_test_fulltools_category.json",
            labels_dir / "test_labels_no_reasoning_fulltools.json",
        ],
        context="Stage 8 input",
    )
    if args.step in {"all", "evaluation"}:
        require_directories(
            [
                adapter_root / condition
                for condition in CONDITIONS
                if condition != "base_model"
            ],
            context="Stage 8 trained adapter",
        )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    runner = StageRunner(stage_root / "logs" / "stage8_evaluation.log")
    audit_provenance(runner, config=config, output=runtime_provenance)

    if args.step in {"all", "evaluation"}:
        for scope in ("scoped", "full"):
            data_path = (
                data_dir / "tasks_v1_test_category.json"
                if scope == "scoped"
                else data_dir / "tasks_v1_test_fulltools_category.json"
            )
            for condition in CONDITIONS:
                adapter_arguments: list[object] = []
                if condition != "base_model":
                    adapter_arguments = ["--adapter-dir", adapter_root / condition]
                runner.run_module(
                    "when2tool_action.scripts.run_trained_eval",
                    "--config",
                    config,
                    "--runtime-provenance",
                    runtime_provenance,
                    "--data",
                    data_path,
                    "--labels",
                    labels_dir / "test_labels_no_reasoning_fulltools.json",
                    "--output-root",
                    output_root,
                    "--tool-scope",
                    scope,
                    "--condition-id",
                    condition,
                    *adapter_arguments,
                    "--generation-seeds",
                    *args.generation_seeds,
                    "--resume",
                )

    if args.step in {"all", "summary"}:
        runner.run_module(
            "when2tool_action.scripts.summarize_trained_evals",
            "--output-root",
            output_root,
            "--output-dir",
            comparison_dir,
            "--require-complete",
        )

    runner.message(f"Stage 8 completed for step={args.step}: {stage_root}")


if __name__ == "__main__":
    main()
