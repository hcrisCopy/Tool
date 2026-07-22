from __future__ import annotations

import argparse
import json
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.data import load_task_json
from when2tool_action.io_utils import atomic_write_json, canonical_json_sha256
from when2tool_action.prefill import compute_prefills
from when2tool_action.runtime import EvaluationSetting, attach_gold_actions, evaluate
from when2tool_action.upstream import build_agent, full_menu_sha256, set_generation_seed


def _labels(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not isinstance(artifact.get("rows"), list):
        raise TypeError(f"Malformed labels: {path}")
    return artifact["rows"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tool-scope", choices=["scoped", "full"], default="full")
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    tasks = load_task_json(Path(args.data).resolve(), expected_scope=args.tool_scope)
    tasks = attach_gold_actions(tasks, _labels(Path(args.labels).resolve()))
    task_ids = [task["id"] for task in tasks]
    thresholds = tuple(args.thresholds) if args.thresholds else config.probe.thresholds
    if tuple(sorted(set(thresholds))) != tuple(sorted(thresholds)):
        raise ValueError("Thresholds must be unique")
    if any(value not in config.probe.thresholds for value in thresholds):
        raise ValueError("Thresholds must come from the registered sweep")
    seeds = tuple(args.seeds) if args.seeds else config.generation.seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique")
    output_dir = Path(args.output_dir).resolve()
    output_paths = {
        threshold: output_dir / f"probe_prefill_t{threshold:.1f}_{'fulltools' if args.tool_scope == 'full' else 'scoped'}.json"
        for threshold in thresholds
    }
    for path in output_paths.values():
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    agent = build_agent(config)
    for threshold in thresholds:
        prefills, decisions = compute_prefills(
            Path(args.probe_dir).resolve(),
            task_ids,
            threshold=threshold,
            temperature=config.probe.temperature,
        )
        setting = EvaluationSetting(
            name=f"probe_prefill_t{threshold:.1f}_{'fulltools' if args.tool_scope == 'full' else 'scoped'}",
            tool_scope=args.tool_scope,
            prompt_mode="current",
            require_reasoning=False,
            record_mode="lite",
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "config": {
                "model": config.model.slug,
                "setting": setting.name,
                "tool_scope": args.tool_scope,
                "prompt_mode": "current",
                "reasoning_mode": "no_reasoning",
                "probe_scope": f"{args.tool_scope}-adapted",
                "probe_temperature": config.probe.temperature,
                "probe_threshold": threshold,
                "prefill_mode": "soft",
                "seeds": list(seeds),
                "full_menu_sha256": full_menu_sha256(),
                "task_ids_sha256": canonical_json_sha256(task_ids),
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
                prefills=prefills,
            )
            for row in rows:
                row.update(decisions[row["id"]])
            artifact["runs"].append(
                {"run_id": run_id, "seed": seed, "setting": setting.name, "rows": rows}
            )
            atomic_write_json(output_paths[threshold], artifact, overwrite=True)
            print(f"Checkpointed {setting.name} {run_id}", flush=True)


if __name__ == "__main__":
    main()
