from __future__ import annotations

import argparse
import json
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.data import load_task_json, smoke_subset
from when2tool_action.io_utils import atomic_write_json, canonical_json_sha256
from when2tool_action.runtime import (
    EvaluationSetting,
    attach_gold_actions,
    evaluate,
)
from when2tool_action.upstream import build_agent, full_menu_sha256, set_generation_seed


def _load_labels(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not isinstance(artifact.get("rows"), list):
        raise TypeError(f"{path} is not a label artifact")
    return artifact["rows"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--setting-name", required=True)
    parser.add_argument("--tool-scope", choices=["scoped", "full"], required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=["current", "necessary_tool", "sparse_tool", "force_tool", "no_tool"],
        required=True,
    )
    parser.add_argument(
        "--reasoning-mode", choices=["reasoning", "no_reasoning"], required=True
    )
    parser.add_argument("--record-mode", choices=["off", "lite", "full"], default="lite")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    require_inputs(config)
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    tasks = load_task_json(Path(args.data).resolve(), expected_scope=args.tool_scope)
    if args.smoke:
        tasks = smoke_subset(tasks)
    label_rows = _load_labels(Path(args.labels).resolve())
    selected_ids = {task["id"] for task in tasks}
    label_rows = [row for row in label_rows if row.get("id") in selected_ids]
    tasks = attach_gold_actions(tasks, label_rows)
    seeds = tuple(args.seeds) if args.seeds else config.generation.seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a non-empty unique list")
    setting = EvaluationSetting(
        name=args.setting_name,
        tool_scope=args.tool_scope,
        prompt_mode=args.prompt_mode,
        require_reasoning=args.reasoning_mode == "reasoning",
        record_mode=args.record_mode,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            "model": config.model.slug,
            "setting": setting.name,
            "tool_scope": setting.tool_scope,
            "prompt_mode": setting.prompt_mode,
            "reasoning_mode": args.reasoning_mode,
            "record_mode": setting.record_mode,
            "seeds": list(seeds),
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "repetition_penalty": config.generation.repetition_penalty,
            "max_new_tokens": config.generation.max_new_tokens,
            "max_rounds": config.generation.max_rounds,
            "max_model_len": config.generation.max_model_len,
            "full_menu_sha256": full_menu_sha256(),
            "task_ids_sha256": canonical_json_sha256([task["id"] for task in tasks]),
            "smoke": args.smoke,
        },
        "runs": [],
    }
    agent = build_agent(config)
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
            {
                "run_id": run_id,
                "seed": seed,
                "setting": setting.name,
                "rows": rows,
            }
        )
        atomic_write_json(output_path, artifact, overwrite=True)
        print(f"Checkpointed {run_id} to {output_path}", flush=True)


if __name__ == "__main__":
    main()
