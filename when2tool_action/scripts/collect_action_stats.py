"""CLI for strict action-level behavior statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from when2tool_action.stats import collect_action_statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate When2Tool evaluation traces and create action statistics/figures."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more evaluation JSON files (top-level runs list or one-run object).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Strict hard-no-tool label artifacts, typically train then "
            "test, for split-preserving difficulty/category/necessity counts."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Exact behavior task JSON whose SHA256 is recorded by every artifact.",
    )
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        required=True,
        help="Registered runtime_provenance.json referenced by behavior artifacts.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Require every setting to contain exactly this formal seed panel."
        ),
    )
    parser.add_argument(
        "--expected-settings",
        nargs="+",
        required=True,
        help="Require the exact formal setting panel, with no missing or extra setting.",
    )
    parser.add_argument(
        "--analysis-protocol",
        choices=["fulltools", "scoped_adapted", "scoped_original_w2t"],
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacing this command's files in an existing output directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = collect_action_statistics(
        args.outputs,
        args.output_dir,
        labels_paths=args.labels,
        overwrite=args.overwrite,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_seeds=args.expected_seeds,
        expected_settings=args.expected_settings,
        analysis_protocol=args.analysis_protocol,
        data_path=args.data,
        runtime_provenance_path=args.runtime_provenance,
    )
    print(
        json.dumps(
            {
                "schema_version": summary["schema_version"],
                "n_rows": summary["n_rows"],
                "n_settings": summary["n_settings"],
                "n_runs": summary["n_runs"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
