"""Run the Stage 6 target-versus-random causal ablation panel."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the Stage 6 causal panel. Paths are "
            "resolved relative to the repository root."
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
        "--generation-seeds",
        nargs="+",
        type=int,
        choices=(0, 1, 2),
        default=[0],
        help="Generation seeds; use 0 for the primary panel",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_unique(args.generation_seeds, name="--generation-seeds")

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
    stage_root = run_root / "stages" / "06_causal"
    output_dir = stage_root / "conditions"
    manifest_dir = stage_root / "manifests"
    runtime_provenance = manifest_dir / "runtime_provenance.json"

    require_files(
        [
            config,
            data_dir / "tasks_v1_test_fulltools_category.json",
            labels_dir / "test_labels_no_reasoning_fulltools.json",
            primary_mask,
        ],
        context="Stage 6 input",
    )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    runner = StageRunner(stage_root / "logs" / "stage6_causal.log")
    audit_provenance(runner, config=config, output=runtime_provenance)
    runner.run_module(
        "when2tool_action.scripts.run_neuron_ablation",
        "--config",
        config,
        "--runtime-provenance",
        runtime_provenance,
        "--data",
        data_dir / "tasks_v1_test_fulltools_category.json",
        "--labels",
        labels_dir / "test_labels_no_reasoning_fulltools.json",
        "--mask",
        primary_mask,
        "--output-dir",
        output_dir,
        "--generation-seeds",
        *args.generation_seeds,
        "--resume",
    )
    runner.message(f"Stage 6 completed/resumed: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
