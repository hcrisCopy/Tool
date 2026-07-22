"""Build the strict statistics-stage handoff manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from when2tool_action.stage_handoff import write_stage_handoff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash the completed formal statistics-stage artifacts. Logs and "
            "temporary work are never included."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--behavior-commit", required=True)
    parser.add_argument("--statistics-commit", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit replacing an existing handoff manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = write_stage_handoff(
        args.run_root,
        output=args.output,
        behavior_commit=args.behavior_commit,
        statistics_commit=args.statistics_commit,
        overwrite=args.overwrite,
    )
    print(
        "Published statistics-stage handoff: "
        f"{payload['totals']['file_count']} files, "
        f"{payload['totals']['bytes']} bytes"
    )


if __name__ == "__main__":
    main()
