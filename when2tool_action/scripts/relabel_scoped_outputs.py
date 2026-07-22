"""Relabel a complete scoped prompt matrix without regenerating behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.scoped_relabel import relabel_scoped_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--target-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    require_inputs(config)
    receipt = relabel_scoped_outputs(
        config,
        input_paths=[Path(path) for path in args.inputs],
        source_labels_path=Path(args.source_labels),
        target_labels_path=Path(args.target_labels),
        output_dir=Path(args.output_dir),
        protocol_id=args.protocol_id,
        overwrite=args.overwrite,
    )
    print(
        f"Published {receipt['n_source_files']} relabeled scoped artifacts "
        f"for protocol {receipt['protocol_id']}"
    )


if __name__ == "__main__":
    main()
