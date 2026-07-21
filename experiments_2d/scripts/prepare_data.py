"""Audit and materialize the official When2Tool Parquet files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from experiments_2d.config import load_config, require_input_paths
from experiments_2d.data import (
    base_metadata,
    dataset_count_rows,
    load_single_hop_split,
    parquet_path,
)
from experiments_2d.io_utils import atomic_write_csv, atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments_2d/configs/qwen3_4b_instruct_2507.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_input_paths(config)

    tasks_by_split = {
        split: load_single_hop_split(config.paths.dataset, split)
        for split in ("train", "test")
    }
    train_ids = {task["id"] for task in tasks_by_split["train"]}
    test_ids = {task["id"] for task in tasks_by_split["test"]}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ValueError(f"Train/test task ids overlap: {overlap[:20]}")

    data_dir = config.run_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, tasks in tasks_by_split.items():
        atomic_write_json(data_dir / f"tasks_v1_{split}.json", tasks)
        atomic_write_json(data_dir / f"{split}_meta_base.json", base_metadata(tasks, split))

    count_rows = dataset_count_rows(tasks_by_split)
    atomic_write_csv(
        data_dir / "dataset_counts.csv",
        count_rows,
        fieldnames=["split", "category", "difficulty", "n"],
    )

    source_files = {}
    for config_name, multi_hop in (("single_hop", False), ("multi_hop", True)):
        for split in ("train", "test"):
            source = parquet_path(config.paths.dataset, split, multi_hop=multi_hop)
            source_files[f"{config_name}/{split}"] = {
                "name": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
    atomic_write_json(
        data_dir / "dataset_audit.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": source_files,
            "single_hop": {
                split: {
                    "n": len(tasks),
                    "min_id": min(task["id"] for task in tasks),
                    "max_id": max(task["id"] for task in tasks),
                }
                for split, tasks in tasks_by_split.items()
            },
            "train_test_id_overlap": 0,
        },
    )
    print(f"Materialized and audited When2Tool under {data_dir}")


if __name__ == "__main__":
    main()

