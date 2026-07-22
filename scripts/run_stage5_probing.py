"""Run Stage 5 activation extraction and FFN neuron probing."""

from __future__ import annotations

import argparse
from pathlib import Path

from _stage_utils import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    StageRunner,
    audit_provenance,
    positive_int,
    repo_path,
    require_files,
    require_unique,
)


MODEL_SLUG = "qwen3-4b-instruct-2507"
ALLOWED_RHOS = (0.001, 0.003, 0.005)
ALLOWED_VARIANTS = ("signed", "positive", "abs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5. Paths are resolved relative to the repository root. "
            "Use a new --run-root when reproducing completed formal outputs."
        ),
    )
    parser.add_argument(
        "--start",
        choices=("fresh", "activations", "discovery"),
        default="fresh",
        help=(
            "fresh runs extraction then discovery; activations runs extraction "
            "only; discovery reuses complete activation files"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--input-run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=(
            "Run root containing the frozen data/ and labels/ inputs "
            f"(default: {DEFAULT_RUN_ROOT})"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Output run root; defaults to --input-run-root",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Extraction batch size (default: 1)",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Torch device (default: cuda:0)"
    )
    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        choices=ALLOWED_RHOS,
        default=list(ALLOWED_RHOS),
        help="Frozen rho panel (default: 0.001 0.003 0.005)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ALLOWED_VARIANTS,
        default=list(ALLOWED_VARIANTS),
        help="Activation variants (default: signed positive abs)",
    )
    parser.add_argument(
        "--control-seed",
        type=int,
        default=42,
        help="Control sampling seed (default: 42)",
    )
    parser.add_argument(
        "--probe-c",
        type=float,
        default=0.0001,
        help="Logistic probe C (default: 0.0001)",
    )
    parser.add_argument(
        "--mean-chunk-size",
        type=positive_int,
        default=64,
        help="Activation mean chunk size (default: 64)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_unique(args.rhos, name="--rhos")
    require_unique(args.variants, name="--variants")

    config = repo_path(args.config)
    input_run_root = repo_path(args.input_run_root)
    run_root = repo_path(args.run_root or args.input_run_root)
    data_dir = input_run_root / "data"
    labels_dir = input_run_root / "labels" / MODEL_SLUG
    stage_root = run_root / "stages" / "05_probing"
    activations_dir = stage_root / "activations"
    discovery_dir = stage_root / "discovery"
    manifest_dir = stage_root / "manifests"
    runtime_provenance = manifest_dir / "runtime_provenance.json"

    required = [
        config,
        data_dir / "tasks_v1_train_fulltools_category.json",
        data_dir / "tasks_v1_test_fulltools_category.json",
        labels_dir / "train_labels_no_reasoning_fulltools.json",
        labels_dir / "test_labels_no_reasoning_fulltools.json",
    ]
    if args.start == "discovery":
        required.extend(
            [
                activations_dir / "down_proj_column_norms.pt",
                activations_dir / "train_mlp_lasttoken_fulltools.pt",
                activations_dir / "train_mlp_lasttoken_fulltools_manifest.json",
                activations_dir / "test_mlp_lasttoken_fulltools.pt",
                activations_dir / "test_mlp_lasttoken_fulltools_manifest.json",
            ]
        )
    require_files(required, context="Stage 5 input")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    runner = StageRunner(stage_root / "logs" / "stage5_probing.log")
    runner.message(f"Stage 5 input root: {input_run_root}")
    runner.message(f"Stage 5 output root: {run_root}")
    audit_provenance(runner, config=config, output=runtime_provenance)

    if args.start in {"fresh", "activations"}:
        runner.run_module(
            "when2tool_action.scripts.extract_mlp_activations",
            "--config",
            config,
            "--runtime-provenance",
            runtime_provenance,
            "--data-dir",
            data_dir,
            "--labels-dir",
            labels_dir,
            "--output-dir",
            activations_dir,
            "--batch-size",
            args.batch_size,
            "--device",
            args.device,
        )

    if args.start in {"fresh", "discovery"}:
        runner.run_module(
            "when2tool_action.scripts.probe_tool_action_neurons",
            "--config",
            config,
            "--runtime-provenance",
            runtime_provenance,
            "--activations-dir",
            activations_dir,
            "--train-labels",
            labels_dir / "train_labels_no_reasoning_fulltools.json",
            "--test-labels",
            labels_dir / "test_labels_no_reasoning_fulltools.json",
            "--output-dir",
            discovery_dir,
            "--rhos",
            *args.rhos,
            "--variants",
            *args.variants,
            "--control-seed",
            args.control_seed,
            "--probe-c",
            args.probe_c,
            "--mean-chunk-size",
            args.mean_chunk_size,
        )

    runner.message(
        "Stage 5 completed. Primary mask: "
        f"{discovery_dir / 'rho0.003_signed' / 'tool_action_neurons.json'}"
    )


if __name__ == "__main__":
    main()
