"""Run the complete seeded scoped-tool Prompt-only/Reason-then-Act matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.data import load_task_json
from when2tool_action.io_utils import atomic_write_json, canonical_json_sha256
from when2tool_action.runtime import EvaluationSetting, attach_gold_actions, evaluate
from when2tool_action.upstream import build_agent, set_generation_seed


PROMPT_MODES = ("force_tool", "current", "necessary_tool", "sparse_tool", "no_tool")
REASONING_MODES = ("no_reasoning", "reasoning")


def _labels(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact.get("rows") if isinstance(artifact, dict) else None
    if not isinstance(rows, list):
        raise TypeError(f"Malformed label artifact {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-modes", nargs="*", default=None)
    parser.add_argument("--reasoning-modes", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    prompts = tuple(args.prompt_modes) if args.prompt_modes else PROMPT_MODES
    reasoning_modes = (
        tuple(args.reasoning_modes) if args.reasoning_modes else REASONING_MODES
    )
    if not set(prompts) <= set(PROMPT_MODES):
        raise ValueError("Unsupported prompt mode")
    if not set(reasoning_modes) <= set(REASONING_MODES):
        raise ValueError("Unsupported reasoning mode")
    seeds = tuple(args.seeds) if args.seeds else config.generation.seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique")
    tasks = load_task_json(Path(args.data).resolve(), expected_scope="scoped")
    tasks = attach_gold_actions(tasks, _labels(Path(args.labels).resolve()))
    output_dir = Path(args.output_dir).resolve()
    targets = {
        (prompt, reasoning): output_dir / f"{prompt}_{reasoning}_scoped.json"
        for reasoning in reasoning_modes
        for prompt in prompts
    }
    for path in targets.values():
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    agent = build_agent(config)
    task_hash = canonical_json_sha256([task["id"] for task in tasks])
    for reasoning in reasoning_modes:
        for prompt in prompts:
            name = f"{prompt}_{reasoning}_scoped"
            setting = EvaluationSetting(
                name=name,
                tool_scope="scoped",
                prompt_mode=prompt,
                require_reasoning=reasoning == "reasoning",
                record_mode="lite",
            )
            artifact = {
                "schema_version": SCHEMA_VERSION,
                "upstream_commit": UPSTREAM_COMMIT,
                "config": {
                    "model": config.model.slug,
                    "setting": name,
                    "tool_scope": "scoped",
                    "prompt_mode": prompt,
                    "reasoning_mode": reasoning,
                    "seeds": list(seeds),
                    "task_ids_sha256": task_hash,
                    "safety_policy": "trusted-code-subprocess-and-benchmark-range-guards",
                },
                "runs": [],
            }
            for run_index, seed in enumerate(seeds):
                set_generation_seed(agent, seed, config.generation.repetition_penalty)
                run_id = f"run_{run_index}_seed_{seed}"
                rows = evaluate(
                    tasks,
                    agent,
                    setting,
                    seed=seed,
                    run_id=run_id,
                    max_rounds=config.generation.max_rounds,
                    max_model_len=config.generation.max_model_len,
                )
                artifact["runs"].append(
                    {"run_id": run_id, "seed": seed, "setting": name, "rows": rows}
                )
                atomic_write_json(targets[(prompt, reasoning)], artifact, overwrite=True)
                print(f"Checkpointed {name} {run_id}", flush=True)


if __name__ == "__main__":
    main()
