"""Write a strict model/data/runtime provenance manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.io_utils import atomic_write_json
from when2tool_action.provenance import build_provenance, validate_runtime_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    output = (
        Path(args.output).resolve()
        if args.output
        else config.run_root / "manifests" / "runtime_provenance.json"
    )
    if output.exists() and not args.overwrite:
        validated = validate_runtime_provenance(config, output)
        print(
            "Validated and reused existing provenance manifest: "
            f"{output} ({validated['sha256']})"
        )
        return
    if output.exists():
        print(
            "WARNING: --overwrite will create a new manifest hash; existing "
            "evaluation checkpoints will no longer be resumable against it."
        )
    artifact = build_provenance(config)
    atomic_write_json(output, artifact, overwrite=args.overwrite)
    print(f"Wrote provenance manifest: {output}")


if __name__ == "__main__":
    main()
