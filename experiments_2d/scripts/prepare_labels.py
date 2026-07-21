"""Generate model-specific no-tool labels with an explicit random seed."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone

from experiments_2d.config import load_config, require_input_paths
from experiments_2d.data import load_single_hop_split, smoke_subset
from experiments_2d.io_utils import atomic_write_csv, atomic_write_json
from experiments_2d.labels import generate_no_tool_labels
from experiments_2d.upstream import EXPECTED_COMMIT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments_2d/configs/qwen3_4b_instruct_2507.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    return parser.parse_args()


def _stats(results: list[dict]) -> list[dict]:
    counts: Counter[tuple[str, str, int]] = Counter()
    correct: Counter[tuple[str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for row in results:
        key = (row["category"], row["difficulty"])
        counts[(key[0], key[1], row["tool_necessary"])] += 1
        correct[key] += row["no_tool_correct"]
        totals[key] += 1
    rows = []
    for key in sorted(totals):
        n = totals[key]
        rows.append(
            {
                "category": key[0],
                "difficulty": key[1],
                "n": n,
                "no_tool_acc": correct[key] / n,
                "tool_necessary_ratio": counts[(key[0], key[1], 1)] / n,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    require_input_paths(config)
    if args.smoke and args.split != "both":
        raise ValueError("--smoke always uses the 45-cell train subset; omit --split")
    if args.all_seeds and args.seed is not None:
        raise ValueError("Use either --all-seeds or --seed, not both")
    if args.seed is not None and args.seed not in config.generation.seeds:
        raise ValueError(f"--seed must be one of {config.generation.seeds}")

    splits = ("train", "test") if args.split == "both" else (args.split,)
    if args.smoke:
        splits = ("train",)
    if args.all_seeds:
        seeds = config.generation.seeds
    elif args.seed is not None:
        seeds = (args.seed,)
    else:
        seeds = (config.generation.seeds[0],)

    for seed in seeds:
        for split in splits:
            tasks = load_single_hop_split(config.paths.dataset, split)
            if args.smoke:
                tasks = smoke_subset(tasks)
            mode = "smoke" if args.smoke else "full"
            output_dir = config.run_root / "labels" / mode / f"seed_{seed}"
            expected_outputs = (
                output_dir / f"{split}_no_tool_outputs.json",
                output_dir / f"{split}_label_stats.csv",
                output_dir / f"{split}_manifest.json",
            )
            existing = [path for path in expected_outputs if path.exists()]
            if existing:
                raise FileExistsError(
                    "Label outputs already exist; archive them before rerunning: "
                    + ", ".join(str(path) for path in existing)
                )
            results = generate_no_tool_labels(tasks, config, seed)
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output_dir / f"{split}_no_tool_outputs.json", results)
            atomic_write_csv(
                output_dir / f"{split}_label_stats.csv",
                _stats(results),
                fieldnames=[
                    "category",
                    "difficulty",
                    "n",
                    "no_tool_acc",
                    "tool_necessary_ratio",
                ],
            )
            atomic_write_json(
                output_dir / f"{split}_manifest.json",
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "split": split,
                    "mode": mode,
                    "seed": seed,
                    "n": len(results),
                    "task_ids": [row["id"] for row in results],
                    "completed": sum(row["completed"] for row in results),
                    "no_tool_correct": sum(row["no_tool_correct"] for row in results),
                    "upstream_commit": EXPECTED_COMMIT,
                    "upstream_default_seed_replication": seed == 0,
                    "enable_thinking": False,
                    "max_new_tokens": config.generation.max_new_tokens,
                    "max_rounds": config.generation.max_rounds,
                    "temperature": config.generation.temperature,
                    "top_p": config.generation.top_p,
                    "top_k": config.generation.top_k,
                    "repetition_penalty": config.generation.repetition_penalty,
                    "do_sample": config.generation.do_sample,
                },
            )
            print(f"Saved {len(results)} {split} labels under {output_dir}")


if __name__ == "__main__":
    main()
