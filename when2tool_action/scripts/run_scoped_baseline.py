"""Run the complete seeded scoped-tool Prompt-only/Reason-then-Act matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from when2tool_action.config import load_config, require_inputs
from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.data import load_task_json
from when2tool_action.eval_resume import (
    PreparedEvaluationArtifact,
    initialize_evaluation_artifacts,
    prepare_evaluation_artifact,
    validate_evaluation_artifact_for_resume,
)
from when2tool_action.io_utils import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from when2tool_action.provenance import validate_runtime_provenance
from when2tool_action.runtime import EvaluationSetting, attach_gold_actions, evaluate
from when2tool_action.upstream import build_agent, full_menu_sha256, set_generation_seed


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
    output_policy = parser.add_mutually_exclusive_group()
    output_policy.add_argument("--overwrite", action="store_true")
    output_policy.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    runtime_provenance = validate_runtime_provenance(config)
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
    data_path = Path(args.data).resolve()
    labels_path = Path(args.labels).resolve()
    tasks = load_task_json(data_path, expected_scope="scoped")
    tasks = attach_gold_actions(tasks, _labels(labels_path))
    output_dir = Path(args.output_dir).resolve()
    targets = {
        (prompt, reasoning): output_dir / f"{prompt}_{reasoning}_scoped.json"
        for reasoning in reasoning_modes
        for prompt in prompts
    }
    task_ids = [task["id"] for task in tasks]
    expected_row_fields = {
        task["id"]: {"gold_action": task["gold_action"]} for task in tasks
    }
    task_hash = canonical_json_sha256(task_ids)
    config_sha256 = sha256_file(config.source)
    data_sha256 = sha256_file(data_path)
    labels_sha256 = sha256_file(labels_path)
    templates: dict[tuple[str, str], dict] = {}
    prepared_targets: dict[tuple[str, str], PreparedEvaluationArtifact] = {}
    for reasoning in reasoning_modes:
        for prompt in prompts:
            name = f"{prompt}_{reasoning}_scoped"
            artifact_template = {
                "schema_version": SCHEMA_VERSION,
                "upstream_commit": UPSTREAM_COMMIT,
                "config": {
                    "model": config.model.slug,
                    "config_sha256": config_sha256,
                    "data_sha256": data_sha256,
                    "labels_sha256": labels_sha256,
                    "runtime_provenance_sha256": runtime_provenance["sha256"],
                    "project_git_commit": runtime_provenance["git_commit"],
                    "setting": name,
                    "tool_scope": "scoped",
                    "prompt_mode": prompt,
                    "reasoning_mode": reasoning,
                    "record_mode": "lite",
                    "seeds": list(seeds),
                    "temperature": config.generation.temperature,
                    "top_p": config.generation.top_p,
                    "top_k": config.generation.top_k,
                    "repetition_penalty": config.generation.repetition_penalty,
                    "max_new_tokens": config.generation.max_new_tokens,
                    "max_rounds": config.generation.behavior_evaluation_max_rounds,
                    "max_model_len": config.generation.max_model_len,
                    "full_menu_sha256": full_menu_sha256(),
                    "task_ids_sha256": task_hash,
                    "smoke": False,
                    "safety_policy": "trusted-code-subprocess-and-benchmark-range-guards",
                },
                "runs": [],
            }
            key = (prompt, reasoning)
            templates[key] = artifact_template
            prepared_targets[key] = prepare_evaluation_artifact(
                targets[key],
                template=artifact_template,
                task_ids=task_ids,
                overwrite=args.overwrite,
                resume=args.resume,
                expected_row_fields=expected_row_fields,
            )

    initialize_evaluation_artifacts(list(prepared_targets.values()))

    if all(item.completed_runs == len(seeds) for item in prepared_targets.values()):
        print("All scoped baseline targets are already complete and validated.", flush=True)
        return

    agent = build_agent(config)
    for reasoning in reasoning_modes:
        for prompt in prompts:
            key = (prompt, reasoning)
            prepared = prepared_targets[key]
            if prepared.completed_runs == len(seeds):
                print(f"Validated complete target: {targets[key]}", flush=True)
                continue
            name = f"{prompt}_{reasoning}_scoped"
            setting = EvaluationSetting(
                name=name,
                tool_scope="scoped",
                prompt_mode=prompt,
                require_reasoning=reasoning == "reasoning",
                record_mode="lite",
            )
            artifact = prepared.artifact
            for run_index in range(prepared.completed_runs, len(seeds)):
                seed = seeds[run_index]
                set_generation_seed(agent, seed, config.generation.repetition_penalty)
                run_id = f"run_{run_index}_seed_{seed}"
                rows = evaluate(
                    tasks,
                    agent,
                    setting,
                    seed=seed,
                    run_id=run_id,
                    max_rounds=config.generation.behavior_evaluation_max_rounds,
                    max_model_len=config.generation.max_model_len,
                )
                artifact["runs"].append(
                    {"run_id": run_id, "seed": seed, "setting": name, "rows": rows}
                )
                validate_evaluation_artifact_for_resume(
                    artifact,
                    template=templates[key],
                    task_ids=task_ids,
                    expected_row_fields=expected_row_fields,
                )
                atomic_write_json(targets[key], artifact, overwrite=True)
                print(f"Checkpointed {name} {run_id}", flush=True)


if __name__ == "__main__":
    main()
