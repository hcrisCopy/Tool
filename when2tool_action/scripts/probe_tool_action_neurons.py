"""CLI for the frozen Precise-Shield-style tool-action neuron panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from when2tool_action.config import load_config
from when2tool_action.neuron_probing import (
    ACTIVATION_VARIANTS,
    ALLOWED_RHOS,
    DEFAULT_CONTROL_SEED,
    run_probing_suite,
)
from when2tool_action.provenance import validate_runtime_provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select masks from train MLP activations only, then evaluate fixed "
            "binary/four-action L2 logistic probes on test."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--activations-dir",
        type=Path,
        required=True,
        help="Directory produced by extract_mlp_activations.",
    )
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dedicated root; one new directory is created per rho/variant group.",
    )
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        required=True,
        help="Exact Stage-5 runtime_provenance.json used by activation extraction.",
    )
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=list(ALLOWED_RHOS),
        help="Unique subset of the frozen panel: 0.001 0.003 0.005.",
    )
    parser.add_argument(
        "--variants",
        choices=ACTIVATION_VARIANTS,
        nargs="+",
        default=list(ACTIVATION_VARIANTS),
    )
    parser.add_argument("--control-seed", type=int, default=DEFAULT_CONTROL_SEED)
    parser.add_argument(
        "--probe-c",
        type=float,
        default=None,
        help="L2 logistic C (default: config probe.c, registered as 0.0001).",
    )
    parser.add_argument("--mean-chunk-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    runtime_provenance = validate_runtime_provenance(
        config, args.runtime_provenance.resolve()
    )
    probe_c = config.probe.c if args.probe_c is None else args.probe_c
    written = run_probing_suite(
        config=config,
        activation_dir=args.activations_dir.resolve(),
        train_labels_path=args.train_labels.resolve(),
        test_labels_path=args.test_labels.resolve(),
        output_dir=args.output_dir.resolve(),
        rhos=tuple(args.rhos),
        variants=tuple(args.variants),
        control_seed=args.control_seed,
        probe_c=probe_c,
        mean_chunk_size=args.mean_chunk_size,
        runtime_provenance=runtime_provenance,
    )
    print(
        json.dumps(
            {
                "groups": [str(path) for path in written],
                "selection_split": "train",
                "test_used_for_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
