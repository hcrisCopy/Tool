from __future__ import annotations

import argparse
from pathlib import Path

from when2tool_action.config import load_config
from when2tool_action.probes import run_pinned_binary_probe, train_action_probes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--binary-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    probe_dir = (
        Path(args.probe_dir).resolve()
        if args.probe_dir
        else config.run_root / "probes" / "fulltools"
    )
    binary = run_pinned_binary_probe(probe_dir, overwrite=args.overwrite)
    print(f"Binary probe: {binary}")
    if not args.binary_only:
        action = train_action_probes(probe_dir, c=config.probe.c, overwrite=args.overwrite)
        print(f"Action probes: {action}")


if __name__ == "__main__":
    main()
