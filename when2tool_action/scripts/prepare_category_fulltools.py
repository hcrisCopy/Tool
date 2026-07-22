from __future__ import annotations

import argparse
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.data import write_augmented_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else config.run_root / "data"
    )
    manifest = write_augmented_data(
        config.paths.dataset, output_dir, overwrite=args.overwrite
    )
    print(f"Wrote strict category/fulltools data to {output_dir}")
    print(f"Full-menu SHA256: {manifest['full_menu_sha256']}")


if __name__ == "__main__":
    main()
