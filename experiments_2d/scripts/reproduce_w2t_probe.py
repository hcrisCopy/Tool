"""Materialize and run the pinned When2Tool all-layer binary probe."""

from __future__ import annotations

import argparse

from experiments_2d.baseline import materialize_w2t_inputs, run_pinned_w2t_all_probe
from experiments_2d.config import load_config, require_input_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments_2d/configs/qwen3_4b_instruct_2507.yaml",
    )
    parser.add_argument("--materialize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_input_paths(config)
    if args.materialize_only:
        output = materialize_w2t_inputs(config)
    else:
        output = run_pinned_w2t_all_probe(config)
    print(output)


if __name__ == "__main__":
    main()

