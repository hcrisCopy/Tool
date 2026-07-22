"""Publish the scoped/full by six-condition trained HF comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from when2tool_action.adapter_evaluation import publish_trained_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <output-root>/comparison",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else output_root / "comparison"
    )
    summary = publish_trained_comparison(
        output_root,
        output_dir,
        require_complete=args.require_complete,
    )
    print(
        f"Published {summary['panel_status']} trained comparison: {output_dir} "
        f"({summary['completed_cells']}/{summary['expected_cells']} cells)",
        flush=True,
    )


if __name__ == "__main__":
    main()
