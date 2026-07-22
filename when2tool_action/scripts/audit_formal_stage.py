"""CLI for the fail-closed final formal-stage semantic audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from when2tool_action.formal_stage_audit import write_formal_stage_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read and cross-validate the completed formal behavior/statistics "
            "stage, then atomically publish one semantic audit receipt."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--behavior-commit", required=True)
    parser.add_argument("--statistics-commit", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit replacing an existing audit receipt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = write_formal_stage_audit(
        args.run_root,
        config_path=args.config,
        output=args.output,
        behavior_commit=args.behavior_commit,
        statistics_commit=args.statistics_commit,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "audit_complete": payload["audit_complete"],
                "checked_file_count": payload["totals"]["checked_file_count"],
                "checked_bytes": payload["totals"]["checked_bytes"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
