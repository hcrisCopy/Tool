"""Explicit CLI for importing an audited original scoped When2Tool baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from when2tool_action.config import REPO_ROOT, load_config
from when2tool_action.legacy_scoped import import_legacy_scoped_baseline


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and import the original pinned P_env When2Tool baseline. "
            "This command never runs implicitly from the statistics pipeline."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--legacy-root",
        required=True,
        help="Legacy model run root containing baseline/, labels/, and hidden/",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination model run root containing data/, labels/, and probes/",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hardlink avoids duplicating multi-GB hidden tensors; no fallback is used",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    receipt = import_legacy_scoped_baseline(
        load_config(args.config),
        _resolve(args.legacy_root),
        _resolve(args.output_root),
        transfer_mode=args.transfer_mode,
        overwrite=args.overwrite,
    )
    print(f"Imported original scoped baseline: {receipt}")


if __name__ == "__main__":
    main()
