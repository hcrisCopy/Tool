"""CLI for full-tools Qwen3 SwiGLU activation extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from when2tool_action.config import load_config, require_inputs
from when2tool_action.data import load_task_json
from when2tool_action.mlp_activations import extract_all_activations
from when2tool_action.provenance import validate_runtime_provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture every Qwen3 mlp.down_proj input at the final non-padding "
            "full-tools prompt token. Outputs are never written outside --output-dir."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing tasks_v1_{train,test}_fulltools_category.json.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing formal full-tools hard-no-tool label artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dedicated data directory for float16 tensors and manifests.",
    )
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        required=True,
        help="Stage-5 runtime_provenance.json created for the current clean commit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Right-padded prompt batch size (default: config extraction_batch_size).",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    require_inputs(config)
    data_dir = args.data_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()
    runtime_provenance_path = args.runtime_provenance.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)
    task_paths = {
        split: data_dir / f"tasks_v1_{split}_fulltools_category.json"
        for split in ("train", "test")
    }
    label_paths = {
        split: labels_dir / f"{split}_labels_no_reasoning_fulltools.json"
        for split in ("train", "test")
    }
    tasks = {
        split: load_task_json(path, expected_scope="full")
        for split, path in task_paths.items()
    }
    runtime_provenance = validate_runtime_provenance(
        config, runtime_provenance_path
    )
    batch_size = (
        config.extraction_batch_size
        if args.batch_size is None
        else args.batch_size
    )
    extract_all_activations(
        tasks,
        task_paths=task_paths,
        label_paths=label_paths,
        config=config,
        output_dir=output_dir,
        batch_size=batch_size,
        device=args.device,
        runtime_provenance=runtime_provenance,
    )
    print(f"Saved MLP activation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
