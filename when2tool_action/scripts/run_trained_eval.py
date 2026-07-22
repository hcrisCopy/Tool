"""Evaluate one base/PEFT condition on scoped or full tools using HF only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from when2tool_action.adapter_evaluation import (
    ADAPTER_EVAL_SCHEMA_VERSION,
    BASE_CONDITION_ID,
    AdapterIdentity,
    compute_adapter_metrics,
    inspect_adapter,
    load_adapter_into_agent,
    load_frozen_labels,
    validate_condition_id,
    write_adapter_summary,
)
from when2tool_action.config import load_config, require_inputs
from when2tool_action.constants import EXPECTED_SPLIT_SIZES
from when2tool_action.data import load_task_json
from when2tool_action.hf_agent import build_hf_causal_agent
from when2tool_action.io_utils import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
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
            "Trained evaluation is a single-process HF runner; only rank 0 writes"
        )


def _load_stage_runtime_provenance(config: Any, path: str | Path) -> dict[str, Any]:
    return validate_runtime_provenance(config, Path(path).resolve())


def _input_hashes(
    data_path: Path,
    labels_path: Path,
    condition_id: str,
    adapter_dir: Path | None,
    *,
    model_slug: str,
) -> tuple[dict[str, str], AdapterIdentity]:
    adapter = inspect_adapter(
        condition_id, adapter_dir, base_model_slug=model_slug
    )
    return {
        "data_sha256": sha256_file(data_path),
        "labels_sha256": sha256_file(labels_path),
    }, adapter


def _assert_inputs_unchanged(
    expected_hashes: dict[str, str],
    expected_adapter: AdapterIdentity,
    *,
    data_path: Path,
    labels_path: Path,
    adapter_dir: Path | None,
    model_slug: str,
) -> None:
    actual_hashes, actual_adapter = _input_hashes(
        data_path,
        labels_path,
        expected_adapter.condition_id,
        adapter_dir,
        model_slug=model_slug,
    )
    if actual_hashes != expected_hashes:
        raise RuntimeError("Evaluation data or frozen labels changed during the run")
    if actual_adapter != expected_adapter:
        raise RuntimeError("PEFT adapter files changed during the run")


def _expected_checkpoint(
    *,
    common: dict[str, Any],
    generation_seed: int,
) -> dict[str, Any]:
    condition_id = common["adapter"]["condition_id"]
    scope = common["tool_scope"]
    return {
        "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
        "config": common,
        "condition_id": condition_id,
        "tool_scope": scope,
        "generation_seed": generation_seed,
        "run_id": f"hf_{condition_id}_{scope}_seed_{generation_seed}",
    }


def _validate_checkpoint(
    payload: Any,
    *,
    expected: dict[str, Any],
    task_ids: list[int],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Trained-evaluation checkpoint must be an object")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint field {key!r} differs from current protocol")
    rows = payload.get("rows")
    metrics = payload.get("metrics")
    if not isinstance(rows, list) or not isinstance(metrics, dict):
        raise TypeError("Checkpoint must contain rows and metrics")
    if [row.get("id") for row in rows if isinstance(row, dict)] != task_ids:
        raise ValueError("Checkpoint task IDs/order differ from current data")
    if compute_adapter_metrics(rows) != metrics:
        raise ValueError("Checkpoint metrics differ from trajectory rows")
    for row in rows:
        if row.get("run_id") != expected["run_id"]:
            raise ValueError("Trajectory run_id differs from checkpoint")
        if row.get("seed") != expected["generation_seed"]:
            raise ValueError("Trajectory generation seed differs from checkpoint")
        if row.get("tool_scope") != expected["tool_scope"]:
            raise ValueError("Trajectory tool scope differs from checkpoint")
        if row.get("reasoning_mode") != "no_reasoning":
            raise ValueError("Trajectory is not no_reasoning")
        if not isinstance(row.get("trace"), list) or not isinstance(row.get("output"), list):
            raise ValueError("Trajectory lacks full record-mode fields")
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
        raise ValueError(f"Invalid checkpoint JSON: {path}") from error
    return _validate_checkpoint(payload, expected=expected, task_ids=task_ids)


def _prepare_evaluation_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    common: dict[str, Any],
    task_ids: list[int],
    overwrite: bool,
    resume: bool,
) -> None:
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise IsADirectoryError(manifest_path)
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid evaluation manifest: {manifest_path}") from error
        if not resume and not overwrite:
            raise FileExistsError(
                f"Evaluation manifest exists; pass --resume or --overwrite: {manifest_path}"
            )
        if resume and existing != manifest:
            old_seeds = existing.get("generation_seeds")
            new_seeds = manifest.get("generation_seeds")
            if not isinstance(old_seeds, list) or not old_seeds:
                raise ValueError("Existing evaluation manifest has invalid seed panel")
            old_protocol = {
                key: value for key, value in existing.items() if key != "generation_seeds"
            }
            new_protocol = {
                key: value for key, value in manifest.items() if key != "generation_seeds"
            }
            if old_protocol != new_protocol:
                raise ValueError(
                    "Existing evaluation manifest differs outside generation seeds"
                )
            if (
                not isinstance(new_seeds, list)
                or len(new_seeds) <= len(old_seeds)
                or new_seeds[: len(old_seeds)] != old_seeds
            ):
                raise ValueError(
                    "Resume seed change must be a strict append-only ordered superset"
                )
            for seed in old_seeds:
                checkpoint_path = output_dir / "trajectories" / f"seed_{seed}.json"
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Cannot extend seed panel; old checkpoint missing: {checkpoint_path}"
                    )
                _read_checkpoint(
                    checkpoint_path,
                    expected=_expected_checkpoint(
                        common=common, generation_seed=seed
                    ),
                    task_ids=task_ids,
                )
            atomic_write_json(manifest_path, manifest, overwrite=True)
    if overwrite or not manifest_path.exists():
        atomic_write_json(manifest_path, manifest, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one base/target/random/dense condition using the same strict "
            "HF backend. Run scoped and full as separate invocations."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-provenance", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tool-scope", choices=("scoped", "full"), required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Required for adapter conditions and forbidden for base_model.",
    )
    parser.add_argument("--generation-seeds", nargs="+", type=int, default=None)
    output_policy = parser.add_mutually_exclusive_group()
    output_policy.add_argument("--overwrite", action="store_true")
    output_policy.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _ensure_single_rank()
    condition_id = validate_condition_id(args.condition_id)
    adapter_dir = Path(args.adapter_dir).resolve() if args.adapter_dir else None
    config = load_config(args.config)
    require_inputs(config)
    runtime = _load_stage_runtime_provenance(config, args.runtime_provenance)
    data_path = Path(args.data).resolve()
    labels_path = Path(args.labels).resolve()
    output_root = Path(args.output_root).resolve()
    output_dir = output_root / args.tool_scope / condition_id
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)

    tasks = load_task_json(data_path, expected_scope=args.tool_scope)
    if len(tasks) != EXPECTED_SPLIT_SIZES["test"]:
        raise ValueError(
            f"Trained evaluation requires {EXPECTED_SPLIT_SIZES['test']} test tasks, "
            f"got {len(tasks)}"
        )
    label_rows, labels_sha256 = load_frozen_labels(
        labels_path, model_slug=config.model.slug
    )
    tasks = attach_gold_actions(tasks, label_rows)
    task_ids = [task["id"] for task in tasks]
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

    input_hashes, adapter = _input_hashes(
        data_path,
        labels_path,
        condition_id,
        adapter_dir,
        model_slug=config.model.slug,
    )
    if input_hashes["labels_sha256"] != labels_sha256:
        raise AssertionError("Frozen label hash changed during initial loading")
    common = {
        "model": config.model.slug,
        "backend": "transformers-hf",
        "backend_contract": {
            "all_base_dense_random_target_conditions": "transformers-hf",
            "vllm_probe_prefill": (
                "context-only comparison; backend differs and cannot be used for "
                "direct adapter causal attribution"
            ),
        },
        "adapter": adapter.snapshot(),
        "config_sha256": sha256_file(config.source),
        **input_hashes,
        "runtime_provenance_sha256": runtime["sha256"],
        "runtime_provenance_filename": Path(runtime["path"]).name,
        "project_git_commit": runtime["git_commit"],
        "task_ids_sha256": canonical_json_sha256(task_ids),
        "tool_scope": args.tool_scope,
        "full_menu_sha256": full_menu_sha256(),
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
    }
    manifest = {
        "schema_version": ADAPTER_EVAL_SCHEMA_VERSION,
        "manifest_type": "trained-hf-evaluation",
        "condition_id": condition_id,
        "tool_scope": args.tool_scope,
        "generation_seeds": list(generation_seeds),
        "config": common,
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    seed_paths = {
        seed: output_dir / "trajectories" / f"seed_{seed}.json"
        for seed in generation_seeds
    }
    existing = [path for path in seed_paths.values() if path.exists()]
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(
            "Seed checkpoints exist; pass --resume or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    _prepare_evaluation_manifest(
        manifest_path,
        manifest,
        output_dir=output_dir,
        common=common,
        task_ids=task_ids,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    pending: list[tuple[int, Path, dict[str, Any]]] = []
    for seed, path in seed_paths.items():
        expected = _expected_checkpoint(common=common, generation_seed=seed)
        if path.exists() and args.resume:
            _read_checkpoint(path, expected=expected, task_ids=task_ids)
            print(f"Validated completed checkpoint: {path}", flush=True)
        else:
            pending.append((seed, path, expected))

    if pending:
        agent = build_hf_causal_agent(config)
        if condition_id != BASE_CONDITION_ID:
            assert adapter_dir is not None
            load_adapter_into_agent(
                agent,
                condition_id=condition_id,
                adapter_dir=adapter_dir,
            )
        setting = EvaluationSetting(
            name=f"{condition_id}_{args.tool_scope}_current_no_reasoning_hf",
            tool_scope=args.tool_scope,
            prompt_mode="current",
            require_reasoning=False,
            record_mode="full",
        )
        for seed, path, expected in pending:
            _assert_inputs_unchanged(
                input_hashes,
                adapter,
                data_path=data_path,
                labels_path=labels_path,
                adapter_dir=adapter_dir,
                model_slug=config.model.slug,
            )
            agent.set_seed(seed)
            rows = evaluate(
                tasks,
                agent,
                setting,
                seed=seed,
                run_id=expected["run_id"],
                max_rounds=config.generation.behavior_evaluation_max_rounds,
                max_model_len=config.generation.max_model_len,
            )
            _assert_inputs_unchanged(
                input_hashes,
                adapter,
                data_path=data_path,
                labels_path=labels_path,
                adapter_dir=adapter_dir,
                model_slug=config.model.slug,
            )
            checkpoint = {
                **expected,
                "metrics": compute_adapter_metrics(rows),
                "rows": rows,
            }
            _validate_checkpoint(checkpoint, expected=expected, task_ids=task_ids)
            atomic_write_json(path, checkpoint, overwrite=args.overwrite)
            print(f"Atomically checkpointed: {path}", flush=True)

    checkpoints = [
        _read_checkpoint(
            path,
            expected=_expected_checkpoint(common=common, generation_seed=seed),
            task_ids=task_ids,
        )
        for seed, path in seed_paths.items()
    ]
    _assert_inputs_unchanged(
        input_hashes,
        adapter,
        data_path=data_path,
        labels_path=labels_path,
        adapter_dir=adapter_dir,
        model_slug=config.model.slug,
    )
    per_seed = [
        {
            "condition_id": condition_id,
            "tool_scope": args.tool_scope,
            "generation_seed": checkpoint["generation_seed"],
            **checkpoint["metrics"],
        }
        for checkpoint in checkpoints
    ]
    summary = write_adapter_summary(
        output_dir,
        metadata={
            "condition_id": condition_id,
            "tool_scope": args.tool_scope,
            "backend": "transformers-hf",
            "adapter_bundle_sha256": adapter.bundle_sha256,
            "data_sha256": input_hashes["data_sha256"],
            "labels_sha256": input_hashes["labels_sha256"],
            "generation_seeds": list(generation_seeds),
            "vllm_probe_prefill_is_context_only": True,
        },
        per_seed_rows=per_seed,
    )
    print(
        f"Completed {condition_id}/{args.tool_scope}: "
        f"{summary['aggregate']['n_generation_seeds']} generation seeds",
        flush=True,
    )


if __name__ == "__main__":
    main()
