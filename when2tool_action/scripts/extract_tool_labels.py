from __future__ import annotations

import argparse
import json
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.data import load_task_json, smoke_subset
from when2tool_action.io_utils import atomic_write_json
from when2tool_action.labels import build_label_rows, label_artifact
from when2tool_action.runtime import EvaluationSetting, evaluate
from when2tool_action.upstream import build_agent, set_generation_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tool-scope", choices=["scoped", "full"], default="full")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.run_root / "data"
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else config.run_root / "labels" / config.model.slug
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    scope_tag = "fulltools" if args.tool_scope == "full" else "scoped"
    targets = {
        split: output_dir / f"{split}_labels_no_reasoning_{scope_tag}{suffix}.json"
        for split in ("train", "test")
    }
    raw_targets = {
        split: output_dir / f"{split}_hard_no_tool_generations_{scope_tag}{suffix}.json"
        for split in ("train", "test")
    }
    for path in [*targets.values(), *raw_targets.values()]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")

    seed = config.generation.seeds[0]
    setting = EvaluationSetting(
        name=f"hard_no_tool_no_reasoning_{scope_tag}",
        tool_scope=args.tool_scope,
        prompt_mode="hard_no_tool",
        require_reasoning=False,
        record_mode="lite",
    )
    agent = build_agent(config)
    set_generation_seed(agent, seed, config.generation.repetition_penalty)
    for split in ("train", "test"):
        data_name = (
            f"tasks_v1_{split}_fulltools_category.json"
            if args.tool_scope == "full"
            else f"tasks_v1_{split}_category.json"
        )
        tasks = load_task_json(data_dir / data_name, expected_scope=args.tool_scope)
        if args.smoke:
            tasks = smoke_subset(tasks)
        rows = evaluate(
            tasks,
            agent,
            setting,
            seed=seed,
            run_id=f"labels_seed_{seed}",
            max_rounds=config.generation.label_hidden_extraction_max_rounds,
            max_model_len=config.generation.max_model_len,
        )
        labels = build_label_rows(
            rows, split=split, seed=seed, tool_scope=args.tool_scope
        )
        atomic_write_json(
            raw_targets[split],
            {
                "schema_version": "when2tool-action-v1",
                "split": split,
                "seed": seed,
                "setting": setting.name,
                "rows": rows,
            },
            overwrite=args.overwrite,
        )
        atomic_write_json(
            targets[split],
            label_artifact(
                labels,
                model_slug=config.model.slug,
                split=split,
                seed=seed,
                tool_scope=args.tool_scope,
            ),
            overwrite=args.overwrite,
        )
        print(f"Wrote {len(labels)} {split} labels to {targets[split]}")


if __name__ == "__main__":
    main()
