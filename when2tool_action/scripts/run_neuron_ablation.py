"""Run the full-tools HF neuron-ablation matrix with atomic checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from when2tool_action.config import load_config, require_inputs
from when2tool_action.constants import (
    EXPECTED_SPLIT_SIZES,
    SCHEMA_VERSION,
    UPSTREAM_COMMIT,
)
from when2tool_action.data import load_task_json, smoke_subset
from when2tool_action.hf_agent import (
    QWEN3_4B_INTERMEDIATE_SIZE,
    build_hf_causal_agent,
)
from when2tool_action.io_utils import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from when2tool_action.neuron_ablation import (
    ABLATION_SCHEMA_VERSION,
    AblationCondition,
    NeuronAblationHooks,
    build_ablation_conditions,
    compute_ablation_metrics,
    load_neuron_mask,
    write_ablation_reports,
)
from when2tool_action.provenance import validate_runtime_provenance
from when2tool_action.runtime import EvaluationSetting, attach_gold_actions, evaluate
from when2tool_action.upstream import full_menu_sha256


def _ensure_single_rank() -> None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE and RANK must be integers") from error
    if world_size != 1 or rank != 0:
        raise RuntimeError(
            "This handoff runner is intentionally single-process/single-GPU; "
            "only rank 0 may write checkpoints"
        )


def _load_stage_runtime_provenance(config: Any, path: str | Path) -> dict[str, Any]:
    return validate_runtime_provenance(config, Path(path).resolve())


def _load_labels(path: Path, *, model_slug: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid label JSON: {path}") from error
    rows = artifact.get("rows") if isinstance(artifact, dict) else None
    if not isinstance(rows, list) or not rows:
        raise TypeError(f"{path} is not a non-empty label artifact")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": model_slug,
        "split": "test",
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
        "n": len(rows),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(
                f"Label artifact {key!r} is {artifact.get(key)!r}; expected {value!r}"
            )
    return rows


def _checkpoint_path(
    output_dir: Path, condition: AblationCondition, generation_seed: int
) -> Path:
    return output_dir / "trajectories" / condition.condition_id / f"seed_{generation_seed}.json"


def _common_metadata(
    *,
    config: Any,
    data_path: Path,
    labels_path: Path,
    mask_path: Path,
    mask_source_sha256: str,
    task_ids: list[int],
    runtime_provenance: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    if sha256_file(mask_path) != mask_source_sha256:
        raise RuntimeError("Neuron mask changed after validation")
    return {
        "model": config.model.slug,
        "backend": "transformers-hf",
        "backend_comparison_note": (
            "All causal conditions, including no_mask, use this HF backend; "
            "do not numerically compare these rows directly with vLLM outputs."
        ),
        "config_sha256": sha256_file(config.source),
        "data_sha256": sha256_file(data_path),
        "labels_sha256": sha256_file(labels_path),
        "mask_sha256": mask_source_sha256,
        "runtime_provenance_sha256": runtime_provenance["sha256"],
        "runtime_provenance_filename": Path(runtime_provenance["path"]).name,
        "project_git_commit": runtime_provenance["git_commit"],
        "task_ids_sha256": canonical_json_sha256(task_ids),
        "full_menu_sha256": full_menu_sha256(),
        "tool_scope": "full",
        "prompt_mode": "current",
        "reasoning_mode": "no_reasoning",
        "record_mode": "full",
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "top_k": config.generation.top_k,
        "repetition_penalty": config.generation.repetition_penalty,
        "max_new_tokens": config.generation.max_new_tokens,
        "max_rounds": config.generation.behavior_evaluation_max_rounds,
        "max_model_len": config.generation.max_model_len,
        "num_hidden_layers": config.model.num_hidden_layers,
        "hidden_size": config.model.hidden_size,
        "intermediate_size": QWEN3_4B_INTERMEDIATE_SIZE,
        "smoke": smoke,
    }


def _expected_checkpoint(
    common: dict[str, Any], condition: AblationCondition, generation_seed: int
) -> dict[str, Any]:
    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "config": common,
        "condition": condition.to_dict(),
        "generation_seed": generation_seed,
        "run_id": f"hf_{condition.condition_id}_seed_{generation_seed}",
    }


def _validate_checkpoint(
    payload: Any,
    *,
    expected: dict[str, Any],
    task_ids: list[int],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Ablation checkpoint must be an object")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Ablation checkpoint field {key!r} differs from protocol")
    rows = payload.get("rows")
    metrics = payload.get("metrics")
    if not isinstance(rows, list) or not isinstance(metrics, dict):
        raise TypeError("Ablation checkpoint must contain rows and metrics")
    if [row.get("id") for row in rows if isinstance(row, dict)] != task_ids:
        raise ValueError("Ablation checkpoint task IDs/order differ from current panel")
    condition = expected["condition"]
    recomputed = compute_ablation_metrics(
        rows, masked_class=condition.get("masked_class")
    )
    if metrics != recomputed:
        raise ValueError("Ablation checkpoint metrics do not match trajectory rows")
    for row in rows:
        if row.get("run_id") != expected["run_id"]:
            raise ValueError("Ablation row run_id differs from checkpoint")
        if row.get("seed") != expected["generation_seed"]:
            raise ValueError("Ablation row generation seed differs from checkpoint")
        if row.get("tool_scope") != "full":
            raise ValueError("Ablation row is not full-tools")
        if row.get("reasoning_mode") != "no_reasoning":
            raise ValueError("Ablation row is not no_reasoning")
        if not isinstance(row.get("trace"), list) or not isinstance(row.get("output"), list):
            raise ValueError("Ablation row lacks full trajectory recording")
    return payload


def _read_checkpoint(
    path: Path,
    *,
    expected: dict[str, Any],
    task_ids: list[int],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid ablation checkpoint JSON: {path}") from error
    return _validate_checkpoint(payload, expected=expected, task_ids=task_ids)


def _select_conditions(
    conditions: list[AblationCondition], requested: list[str] | None
) -> list[AblationCondition]:
    by_id = {condition.condition_id: condition for condition in conditions}
    if requested is None or requested == ["all"]:
        return conditions
    if "all" in requested:
        raise ValueError("Use --conditions all alone")
    if len(requested) != len(set(requested)):
        raise ValueError("--conditions contains duplicates")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown ablation conditions: {unknown}")
    return [by_id[condition_id] for condition_id in requested]


def _preflight_selected(
    paths: list[Path], *, overwrite: bool, resume: bool
) -> None:
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    existing = [path for path in paths if path.exists()]
    non_files = [path for path in existing if not path.is_file()]
    if non_files:
        raise IsADirectoryError(non_files[0])
    if existing and not overwrite and not resume:
        formatted = "\n".join(f"  - {path}" for path in existing[:20])
        raise FileExistsError(
            "Ablation checkpoints already exist; pass --resume or --overwrite:\n"
            + formatted
        )


def _prepare_condition_manifest(
    output_dir: Path,
    *,
    common: dict[str, Any],
    conditions: list[AblationCondition],
    generation_seeds: tuple[int, ...],
    task_ids: list[int],
    overwrite: bool,
    resume: bool,
) -> None:
    path = output_dir / "condition_manifest.json"
    manifest = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "manifest_type": "hf-neuron-ablation-condition-matrix",
        "config": common,
        "generation_seeds": list(generation_seeds),
        "condition_count": len(conditions),
        "seed_condition_count": len(conditions) * len(generation_seeds),
        "conditions": [condition.to_dict() for condition in conditions],
    }
    if path.exists():
        if not path.is_file():
            raise IsADirectoryError(path)
        if not overwrite and not resume:
            raise FileExistsError(
                f"Condition manifest exists; pass --resume or --overwrite: {path}"
            )
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid condition manifest JSON: {path}") from error
        if resume and existing != manifest:
            old_seeds = existing.get("generation_seeds")
            new_seeds = manifest["generation_seeds"]
            if not isinstance(old_seeds, list) or not old_seeds:
                raise ValueError("Existing condition manifest has invalid seed panel")
            old_protocol = {
                key: value
                for key, value in existing.items()
                if key not in {"generation_seeds", "seed_condition_count"}
            }
            new_protocol = {
                key: value
                for key, value in manifest.items()
                if key not in {"generation_seeds", "seed_condition_count"}
            }
            if old_protocol != new_protocol:
                raise ValueError(
                    "Existing condition manifest differs outside generation seeds"
                )
            if (
                len(new_seeds) <= len(old_seeds)
                or new_seeds[: len(old_seeds)] != old_seeds
            ):
                raise ValueError(
                    "Resume seed change must be a strict append-only ordered superset"
                )
            # The old panel must be complete and valid before its declaration is
            # expanded.  This prevents a manifest from getting ahead of data.
            for condition in conditions:
                for seed in old_seeds:
                    checkpoint_path = _checkpoint_path(output_dir, condition, seed)
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(
                            f"Cannot extend seed panel; old checkpoint missing: {checkpoint_path}"
                        )
                    _read_checkpoint(
                        checkpoint_path,
                        expected=_expected_checkpoint(common, condition, seed),
                        task_ids=task_ids,
                    )
            atomic_write_json(path, manifest, overwrite=True)
    if overwrite or not path.exists():
        atomic_write_json(path, manifest, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 25-condition HF causal matrix: shared no_mask, four target "
            "masks, and five random controls per class."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-provenance", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Default/all runs all 25; otherwise pass exact condition IDs.",
    )
    parser.add_argument(
        "--generation-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Subset of the registered generation seeds; default is all [0,1,2].",
    )
    parser.add_argument("--smoke", action="store_true")
    output_policy = parser.add_mutually_exclusive_group()
    output_policy.add_argument("--overwrite", action="store_true")
    output_policy.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _ensure_single_rank()
    config = load_config(args.config)
    require_inputs(config)
    runtime_provenance = _load_stage_runtime_provenance(
        config, args.runtime_provenance
    )
    data_path = Path(args.data).resolve()
    labels_path = Path(args.labels).resolve()
    mask_path = Path(args.mask).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    tasks = load_task_json(data_path, expected_scope="full")
    if len(tasks) != EXPECTED_SPLIT_SIZES["test"]:
        raise ValueError(
            f"Causal evaluation requires the full test split of "
            f"{EXPECTED_SPLIT_SIZES['test']} tasks, got {len(tasks)}"
        )
    if args.smoke:
        tasks = smoke_subset(tasks)
    label_rows = _load_labels(labels_path, model_slug=config.model.slug)
    selected_task_ids = {task["id"] for task in tasks}
    label_rows = [row for row in label_rows if row.get("id") in selected_task_ids]
    tasks = attach_gold_actions(tasks, label_rows)
    task_ids = [task["id"] for task in tasks]
    mask = load_neuron_mask(
        mask_path,
        num_hidden_layers=config.model.num_hidden_layers,
        intermediate_size=QWEN3_4B_INTERMEDIATE_SIZE,
        expected_model={
            "slug": config.model.slug,
            "architecture": config.model.architecture,
            "num_hidden_layers": config.model.num_hidden_layers,
            "hidden_size": config.model.hidden_size,
            "intermediate_size": QWEN3_4B_INTERMEDIATE_SIZE,
        },
        expected_rho=0.003,
        expected_variant="signed",
        require_train_selection=True,
    )
    conditions = build_ablation_conditions(
        mask, intermediate_size=QWEN3_4B_INTERMEDIATE_SIZE
    )
    selected_conditions = _select_conditions(conditions, args.conditions)
    generation_seeds = (
        tuple(args.generation_seeds)
        if args.generation_seeds is not None
        else config.generation.seeds
    )
    if not generation_seeds or len(generation_seeds) != len(set(generation_seeds)):
        raise ValueError("Generation seeds must be non-empty and unique")
    if not set(generation_seeds) <= set(config.generation.seeds):
        raise ValueError(
            f"Generation seeds must be a subset of {config.generation.seeds}"
        )

    common = _common_metadata(
        config=config,
        data_path=data_path,
        labels_path=labels_path,
        mask_path=mask_path,
        mask_source_sha256=mask.source_sha256,
        task_ids=task_ids,
        runtime_provenance=runtime_provenance,
        smoke=args.smoke,
    )
    selected_paths = [
        _checkpoint_path(output_dir, condition, seed)
        for condition in selected_conditions
        for seed in generation_seeds
    ]
    _preflight_selected(
        selected_paths, overwrite=args.overwrite, resume=args.resume
    )
    _prepare_condition_manifest(
        output_dir,
        common=common,
        conditions=conditions,
        generation_seeds=tuple(generation_seeds),
        task_ids=task_ids,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    # Resume validation happens before allocating model weights.
    pending: list[tuple[AblationCondition, int, Path, dict[str, Any]]] = []
    for condition in selected_conditions:
        for seed in generation_seeds:
            path = _checkpoint_path(output_dir, condition, seed)
            expected = _expected_checkpoint(common, condition, seed)
            if path.exists() and args.resume:
                _read_checkpoint(path, expected=expected, task_ids=task_ids)
                print(f"Validated completed checkpoint: {path}", flush=True)
            else:
                pending.append((condition, seed, path, expected))

    setting = EvaluationSetting(
        name="current_no_reasoning_fulltools_hf_causal",
        tool_scope="full",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="full",
    )
    if pending:
        agent = build_hf_causal_agent(
            config, expected_intermediate_size=QWEN3_4B_INTERMEDIATE_SIZE
        )
        for condition, seed, path, expected in pending:
            agent.set_seed(seed)
            hook_context: Any
            hooks: NeuronAblationHooks | None = None
            if condition.kind == "no_mask":
                hook_context = nullcontext()
            else:
                hooks = NeuronAblationHooks(
                    agent.layers,
                    condition.layers,
                    intermediate_size=QWEN3_4B_INTERMEDIATE_SIZE,
                    require_exercised=True,
                )
                hook_context = hooks
            print(
                f"Running {condition.condition_id} generation_seed={seed} "
                f"selected_neurons={condition.selected_count}",
                flush=True,
            )
            with hook_context:
                rows = evaluate(
                    tasks,
                    agent,
                    setting,
                    seed=seed,
                    run_id=expected["run_id"],
                    max_rounds=config.generation.behavior_evaluation_max_rounds,
                    max_model_len=config.generation.max_model_len,
                )
            if sha256_file(mask_path) != mask.source_sha256:
                raise RuntimeError("Neuron mask changed while causal evaluation was running")
            metrics = compute_ablation_metrics(
                rows, masked_class=condition.masked_class
            )
            checkpoint = {
                **expected,
                "hook_call_counts": (
                    {str(layer): count for layer, count in sorted(hooks.call_counts.items())}
                    if hooks is not None
                    else {}
                ),
                "metrics": metrics,
                "rows": rows,
            }
            _validate_checkpoint(checkpoint, expected=expected, task_ids=task_ids)
            atomic_write_json(path, checkpoint, overwrite=args.overwrite)
            print(f"Atomically checkpointed: {path}", flush=True)

    # Load the entire compatible panel already present in this output directory.
    checkpoints: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in generation_seeds:
            path = _checkpoint_path(output_dir, condition, seed)
            if path.is_file():
                checkpoints.append(
                    _read_checkpoint(
                        path,
                        expected=_expected_checkpoint(common, condition, seed),
                        task_ids=task_ids,
                    )
                )
    summary = write_ablation_reports(
        checkpoints,
        output_dir,
        expected_condition_ids=[condition.condition_id for condition in conditions],
        expected_generation_seeds=generation_seeds,
    )
    print(
        f"Causal panel status: {summary['panel_status']} "
        f"({summary['completed_seed_condition_count']}/"
        f"{summary['expected_seed_condition_count']} seed-condition checkpoints)",
        flush=True,
    )


if __name__ == "__main__":
    main()
