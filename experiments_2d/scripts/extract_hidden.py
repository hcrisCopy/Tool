"""Extract h_0..h_36 for the PDF prompt variants."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from experiments_2d.config import load_config, require_input_paths
from experiments_2d.constants import HIDDEN_PROTOCOL_REVISION
from experiments_2d.data import load_single_hop_split, smoke_subset
from experiments_2d.hidden import extract_block_hidden
from experiments_2d.io_utils import atomic_torch_save, atomic_write_json
from experiments_2d.upstream import all_candidate_menu_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments_2d/configs/qwen3_4b_instruct_2507.yaml",
    )
    parser.add_argument(
        "--variant",
        choices=("P_env", "P_all", "P_no_schema", "all"),
        default="all",
    )
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_input_paths(config)
    if args.smoke and args.split != "both":
        raise ValueError("--smoke always uses the 45-cell train subset; omit --split")
    splits = ("train", "test") if args.split == "both" else (args.split,)
    if args.smoke:
        splits = ("train",)
    variants = (
        ("P_env", "P_all", "P_no_schema")
        if args.variant == "all"
        else (args.variant,)
    )
    mode = "smoke" if args.smoke else "full"

    for split in splits:
        tasks = load_single_hop_split(config.paths.dataset, split)
        if args.smoke:
            tasks = smoke_subset(tasks)
        for variant in variants:
            output_dir = config.run_root / "hidden" / mode / variant
            expected_outputs = (
                output_dir / f"{split}_hidden.pt",
                output_dir / f"{split}_metadata.json",
                output_dir / f"{split}_manifest.json",
            )
            existing = [path for path in expected_outputs if path.exists()]
            if existing:
                raise FileExistsError(
                    "Hidden outputs already exist; archive them before rerunning: "
                    + ", ".join(str(path) for path in existing)
                )
            hidden, metadata, validation = extract_block_hidden(tasks, config, variant)
            atomic_torch_save(output_dir / f"{split}_hidden.pt", hidden)
            atomic_write_json(output_dir / f"{split}_metadata.json", metadata)
            atomic_write_json(
                output_dir / f"{split}_manifest.json",
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "split": split,
                    "mode": mode,
                    "prompt_variant": variant,
                    "shape": list(hidden.shape),
                    "dtype": str(hidden.dtype),
                    "extraction_batch_size": config.extraction_batch_size,
                    "enable_thinking": False,
                    "protocol_revision": HIDDEN_PROTOCOL_REVISION,
                    "p_all_menu_sha256": all_candidate_menu_sha256(),
                    "definition": (
                        "h0=embedding/block-1 input; h_l=raw decoder block l output; "
                        "last pre-generation prompt token"
                    ),
                    "validation": validation,
                },
            )
            print(f"Saved {variant}/{split} hidden states under {output_dir}")


if __name__ == "__main__":
    main()
