from __future__ import annotations

import argparse
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.data import load_task_json, smoke_subset
from when2tool_action.hidden import extract_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tool-scope", choices=["scoped", "full"], default="full")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.run_root / "data"
    labels_dir = (
        Path(args.labels_dir).resolve()
        if args.labels_dir
        else config.run_root / "labels" / config.model.slug
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else config.run_root
        / "probes"
        / (
            ("fulltools" if args.tool_scope == "full" else "scoped")
            + ("_smoke" if args.smoke else "")
        )
    )
    tasks = {
        split: load_task_json(
            data_dir
            / (
                f"tasks_v1_{split}_fulltools_category.json"
                if args.tool_scope == "full"
                else f"tasks_v1_{split}_category.json"
            ),
            expected_scope=args.tool_scope,
        )
        for split in ("train", "test")
    }
    if args.smoke:
        tasks = {split: smoke_subset(rows) for split, rows in tasks.items()}
    label_suffix = "_smoke" if args.smoke else ""
    labels = {
        split: labels_dir
        / f"{split}_labels_no_reasoning_{'fulltools' if args.tool_scope == 'full' else 'scoped'}{label_suffix}.json"
        for split in ("train", "test")
    }
    extract_all(
        tasks,
        labels,
        config,
        output_dir,
        tool_scope=args.tool_scope,
        overwrite=args.overwrite,
    )
    print(f"Saved {args.tool_scope} residual features to {output_dir}")


if __name__ == "__main__":
    main()
