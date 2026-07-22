from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from when2tool_action.config import load_config, require_inputs
from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.data import load_task_json, smoke_subset
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
from when2tool_action.prefill import compute_prefills
from when2tool_action.provenance import validate_runtime_provenance
from when2tool_action.runtime import EvaluationSetting, attach_gold_actions, evaluate
from when2tool_action.upstream import build_agent, full_menu_sha256, set_generation_seed


ADAPTED_PROTOCOL = "adapted"
ORIGINAL_W2T_PROTOCOL = "scoped_original_w2t"
ORIGINAL_PROBE_SCOPE = "scoped-original-pinned"
ORIGINAL_ADAPTATION_STATUS = "original-pinned-not-current-scoped-adapted"
ORIGINAL_ALLOWED_CLAIM = "original When2Tool scoped binary baseline reproduction"


@dataclass(frozen=True)
class ProbeProtocolSpec:
    output_scope: str
    probe_scope: str


def resolve_probe_protocol(protocol: str, tool_scope: str) -> ProbeProtocolSpec:
    """Resolve an explicit protocol without inspecting the probe directory name."""

    if protocol == ADAPTED_PROTOCOL:
        if tool_scope not in {"scoped", "full"}:
            raise ValueError(f"Unsupported tool scope: {tool_scope}")
        return ProbeProtocolSpec(
            output_scope="fulltools" if tool_scope == "full" else "scoped",
            probe_scope=f"{tool_scope}-adapted",
        )
    if protocol == ORIGINAL_W2T_PROTOCOL:
        if tool_scope != "scoped":
            raise ValueError(
                "scoped_original_w2t is only compatible with --tool-scope scoped"
            )
        return ProbeProtocolSpec(
            output_scope=ORIGINAL_W2T_PROTOCOL,
            probe_scope=ORIGINAL_PROBE_SCOPE,
        )
    raise ValueError(f"Unsupported probe protocol: {protocol}")


def _json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise TypeError(f"Malformed {context}: {path}")
    return artifact


def _labels(path: Path) -> dict[str, Any]:
    artifact = _json_object(path, "labels")
    if not isinstance(artifact.get("rows"), list):
        raise TypeError(f"Malformed labels: {path}")
    return artifact


def _expect(mapping: dict[str, Any], key: str, expected: Any, context: str) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise ValueError(f"{context}.{key}: {actual!r} != {expected!r}")


def validate_original_protocol_metadata(
    receipt: dict[str, Any],
    labels: dict[str, Any],
    *,
    model_slug: str,
    config_sha256: str,
    label_seed: int,
    task_ids: list[int],
    n_layers: int,
    hidden_dim: int,
    probe_c: float,
) -> None:
    """Pure validation of the imported original-W2T receipt and labels."""

    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": ORIGINAL_W2T_PROTOCOL,
        "model": model_slug,
        "upstream_commit": UPSTREAM_COMMIT,
        "config_sha256": config_sha256,
    }.items():
        _expect(receipt, key, expected, "migration_receipt")
    compatibility = receipt.get("compatibility")
    if not isinstance(compatibility, dict):
        raise TypeError("migration_receipt.compatibility must be an object")
    for key, expected in {
        "probe_scope": ORIGINAL_PROBE_SCOPE,
        "current_scoped_adapted": False,
        "allowed_claim": ORIGINAL_ALLOWED_CLAIM,
    }.items():
        _expect(compatibility, key, expected, "migration_receipt.compatibility")
    probe = receipt.get("probe_validation")
    if not isinstance(probe, dict):
        raise TypeError("migration_receipt.probe_validation must be an object")
    for key, expected in {
        "C": probe_c,
        "layer": "all",
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "official_double_standard_scaler_reconstructed": True,
        "all_saved_metrics_recomputed": True,
    }.items():
        _expect(probe, key, expected, "migration_receipt.probe_validation")
    splits = receipt.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "test"}:
        raise ValueError("migration_receipt.splits must contain exactly train and test")
    test_split = splits["test"]
    if not isinstance(test_split, dict):
        raise TypeError("migration_receipt.splits.test must be an object")
    for key, expected in {
        "n": len(task_ids),
        "task_ids_sha256": canonical_json_sha256(task_ids),
        "hidden_shape": [len(task_ids), n_layers, hidden_dim],
        "hidden_dtype": "torch.float32",
    }.items():
        _expect(test_split, key, expected, "migration_receipt.splits.test")

    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": model_slug,
        "split": "test",
        "seed": label_seed,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "scoped",
        "protocol_id": ORIGINAL_W2T_PROTOCOL,
        "adaptation_status": ORIGINAL_ADAPTATION_STATUS,
        "n": len(task_ids),
    }.items():
        _expect(labels, key, expected, "labels")
    rows = labels.get("rows")
    if not isinstance(rows, list) or len(rows) != len(task_ids):
        raise ValueError("labels.rows has the wrong length")
    row_ids: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"labels.rows[{index}] must be an object")
        for key, expected in {
            "seed": label_seed,
            "tool_scope": "scoped",
            "label_protocol": ORIGINAL_W2T_PROTOCOL,
        }.items():
            _expect(row, key, expected, f"labels.rows[{index}]")
        task_id = row.get("id")
        if not isinstance(task_id, int):
            raise TypeError(f"labels.rows[{index}].id must be an integer")
        row_ids.append(task_id)
    if row_ids != task_ids:
        raise ValueError("Original-W2T labels and evaluation task IDs/order differ")


def validate_original_protocol_files(
    *,
    receipt: dict[str, Any],
    probe_dir: Path,
    labels_path: Path,
    run_root: Path,
) -> None:
    """Verify that receipt hashes bind the exact files selected for evaluation."""

    hashes = receipt.get("destination_artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise TypeError(
            "migration_receipt.destination_artifact_sha256 must be a non-empty object"
        )
    resolved_root = run_root.resolve()
    resolved_targets: dict[Path, str] = {}
    for relative, expected_hash in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise TypeError("Migration receipt artifact hashes must map strings to strings")
        target = (resolved_root / relative).resolve()
        if target != resolved_root and resolved_root not in target.parents:
            raise ValueError(f"Migration receipt path escapes run root: {relative}")
        if not target.is_file():
            raise FileNotFoundError(target)
        if sha256_file(target) != expected_hash:
            raise ValueError(f"Migration receipt SHA256 mismatch: {target}")
        if target in resolved_targets:
            raise ValueError(f"Migration receipt aliases one destination twice: {target}")
        resolved_targets[target] = expected_hash
    required = [
        *(probe_dir.resolve() / name for name in (
            "probe_no_reasoning.pt",
            "test_hidden_no_reasoning.pt",
            "test_labels_no_reasoning.json",
        )),
        labels_path.resolve(),
    ]
    missing = [str(path) for path in required if path not in resolved_targets]
    if missing:
        raise ValueError(
            "Migration receipt does not bind selected evaluation artifacts: "
            + ", ".join(missing)
        )
    data_manifest = resolved_root / "data" / "data_manifest.json"
    if not data_manifest.is_file():
        raise FileNotFoundError(data_manifest)
    _expect(
        receipt,
        "destination_data_manifest_sha256",
        sha256_file(data_manifest),
        "migration_receipt",
    )
    data_hashes = receipt.get("destination_data_files_sha256")
    expected_data_paths = {
        f"data/tasks_v1_{split}_category.json" for split in ("train", "test")
    }
    if not isinstance(data_hashes, dict) or set(data_hashes) != expected_data_paths:
        raise ValueError(
            "migration_receipt.destination_data_files_sha256 must bind exactly "
            "the scoped train/test task files"
        )
    for relative, expected_hash in data_hashes.items():
        if not isinstance(expected_hash, str):
            raise TypeError("Migration receipt data hashes must be strings")
        target = (resolved_root / relative).resolve()
        if resolved_root not in target.parents:
            raise ValueError(f"Migration receipt data path escapes run root: {relative}")
        if not target.is_file():
            raise FileNotFoundError(target)
        if sha256_file(target) != expected_hash:
            raise ValueError(f"Migration receipt data SHA256 mismatch: {target}")


def build_probe_prefill_artifact_template(
    *,
    config: Any,
    setting: EvaluationSetting,
    probe_protocol: str,
    probe_scope: str,
    threshold: float,
    seeds: tuple[int, ...],
    task_ids: list[int],
    smoke: bool,
    config_sha256: str,
    data_sha256: str,
    labels_sha256: str,
    probe_inputs_sha256: dict[str, str],
    probe_decisions_sha256: str,
    runtime_provenance_sha256: str,
    project_git_commit: str,
) -> dict[str, Any]:
    """Build the complete immutable protocol used for strict resume checks."""

    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            "model": config.model.slug,
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "labels_sha256": labels_sha256,
            "probe_inputs_sha256": dict(probe_inputs_sha256),
            "probe_decisions_sha256": probe_decisions_sha256,
            "runtime_provenance_sha256": runtime_provenance_sha256,
            "project_git_commit": project_git_commit,
            "setting": setting.name,
            "tool_scope": setting.tool_scope,
            "prompt_mode": setting.prompt_mode,
            "reasoning_mode": "no_reasoning",
            "record_mode": setting.record_mode,
            "probe_protocol": probe_protocol,
            "probe_scope": probe_scope,
            "probe_training_label_seed": config.generation.seeds[0],
            "probe_temperature": config.probe.temperature,
            "probe_threshold": threshold,
            "prefill_mode": "soft",
            "seeds": list(seeds),
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "repetition_penalty": config.generation.repetition_penalty,
            "max_new_tokens": config.generation.max_new_tokens,
            "max_rounds": config.generation.max_rounds,
            "max_model_len": config.generation.max_model_len,
            "full_menu_sha256": full_menu_sha256(),
            "task_ids_sha256": canonical_json_sha256(task_ids),
            "smoke": smoke,
        },
        "runs": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tool-scope", choices=["scoped", "full"], default="full")
    parser.add_argument(
        "--probe-protocol",
        choices=[ADAPTED_PROTOCOL, ORIGINAL_W2T_PROTOCOL],
        default=ADAPTED_PROTOCOL,
        help=(
            "Explicit probe provenance. The default is the current adapted "
            "pipeline; scoped_original_w2t requires an audited migration receipt."
        ),
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    output_policy = parser.add_mutually_exclusive_group()
    output_policy.add_argument("--overwrite", action="store_true")
    output_policy.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_inputs(config)
    runtime_provenance = validate_runtime_provenance(config)
    protocol = resolve_probe_protocol(args.probe_protocol, args.tool_scope)
    if args.probe_protocol == ORIGINAL_W2T_PROTOCOL and args.smoke:
        raise ValueError(
            "scoped_original_w2t is a full imported test-split artifact and does "
            "not support --smoke"
        )
    data_path = Path(args.data).resolve()
    tasks = load_task_json(data_path, expected_scope=args.tool_scope)
    if args.smoke:
        tasks = smoke_subset(tasks)
    labels_path = Path(args.labels).resolve()
    labels = _labels(labels_path)
    probe_dir = Path(args.probe_dir).resolve()
    if args.probe_protocol == ORIGINAL_W2T_PROTOCOL:
        receipt = _json_object(
            probe_dir / "migration_receipt.json", "migration receipt"
        )
        validate_original_protocol_metadata(
            receipt,
            labels,
            model_slug=config.model.slug,
            config_sha256=sha256_file(config.source),
            label_seed=config.generation.seeds[0],
            task_ids=[task["id"] for task in tasks],
            n_layers=config.model.num_hidden_layers + 1,
            hidden_dim=config.model.hidden_size,
            probe_c=config.probe.c,
        )
        validate_original_protocol_files(
            receipt=receipt,
            probe_dir=probe_dir,
            labels_path=labels_path,
            run_root=config.run_root,
        )
    tasks = attach_gold_actions(tasks, labels["rows"])
    task_ids = [task["id"] for task in tasks]
    thresholds = tuple(args.thresholds) if args.thresholds else config.probe.thresholds
    if tuple(sorted(set(thresholds))) != tuple(sorted(thresholds)):
        raise ValueError("Thresholds must be unique")
    if any(value not in config.probe.thresholds for value in thresholds):
        raise ValueError("Thresholds must come from the registered sweep")
    seeds = tuple(args.seeds) if args.seeds else config.generation.seeds
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and unique")
    if args.probe_protocol == ORIGINAL_W2T_PROTOCOL and any(
        seed not in config.generation.seeds for seed in seeds
    ):
        raise ValueError(
            "scoped_original_w2t evaluation seeds must come from the registered sweep"
        )
    output_dir = Path(args.output_dir).resolve()
    output_paths = {
        threshold: output_dir
        / f"probe_prefill_t{threshold:.1f}_{protocol.output_scope}{'_smoke' if args.smoke else ''}.json"
        for threshold in thresholds
    }
    settings: dict[float, EvaluationSetting] = {}
    templates: dict[float, dict[str, Any]] = {}
    prepared_targets: dict[float, PreparedEvaluationArtifact] = {}
    prefills_by_threshold: dict[float, dict[int, str]] = {}
    decisions_by_threshold: dict[float, dict[int, dict[str, Any]]] = {}
    expected_fields_by_threshold: dict[float, dict[int, dict[str, Any]]] = {}
    probe_input_names = [
        "probe_no_reasoning.pt",
        "test_hidden_no_reasoning.pt",
        "test_labels_no_reasoning.json",
    ]
    if args.probe_protocol == ORIGINAL_W2T_PROTOCOL:
        probe_input_names.append("migration_receipt.json")
    probe_inputs_sha256 = {
        name: sha256_file(probe_dir / name) for name in probe_input_names
    }
    config_sha256 = sha256_file(config.source)
    data_sha256 = sha256_file(data_path)
    labels_sha256 = sha256_file(labels_path)
    for threshold in thresholds:
        prefills, decisions = compute_prefills(
            probe_dir,
            task_ids,
            threshold=threshold,
            temperature=config.probe.temperature,
        )
        decision_rows = [{"id": task_id, **decisions[task_id]} for task_id in task_ids]
        setting = EvaluationSetting(
            name=f"probe_prefill_t{threshold:.1f}_{protocol.output_scope}",
            tool_scope=args.tool_scope,
            prompt_mode="current",
            require_reasoning=False,
            record_mode="lite",
        )
        template = build_probe_prefill_artifact_template(
            config=config,
            setting=setting,
            probe_protocol=args.probe_protocol,
            probe_scope=protocol.probe_scope,
            threshold=threshold,
            seeds=seeds,
            task_ids=task_ids,
            smoke=args.smoke,
            config_sha256=config_sha256,
            data_sha256=data_sha256,
            labels_sha256=labels_sha256,
            probe_inputs_sha256=probe_inputs_sha256,
            probe_decisions_sha256=canonical_json_sha256(decision_rows),
            runtime_provenance_sha256=runtime_provenance["sha256"],
            project_git_commit=runtime_provenance["git_commit"],
        )
        settings[threshold] = setting
        templates[threshold] = template
        prefills_by_threshold[threshold] = prefills
        expected_fields = {
            task["id"]: {
                "gold_action": task["gold_action"],
                **decisions[task["id"]],
            }
            for task in tasks
        }
        decisions_by_threshold[threshold] = decisions
        expected_fields_by_threshold[threshold] = expected_fields
        prepared_targets[threshold] = prepare_evaluation_artifact(
            output_paths[threshold],
            template=template,
            task_ids=task_ids,
            overwrite=args.overwrite,
            resume=args.resume,
            expected_row_fields=expected_fields,
        )

    initialize_evaluation_artifacts(list(prepared_targets.values()))

    if all(
        prepared.completed_runs == len(seeds)
        for prepared in prepared_targets.values()
    ):
        print("All Probe&Prefill targets are already complete and validated.", flush=True)
        return

    agent = build_agent(config)
    for threshold in thresholds:
        prepared = prepared_targets[threshold]
        if prepared.completed_runs == len(seeds):
            print(f"Validated complete target: {output_paths[threshold]}", flush=True)
            continue
        prefills = prefills_by_threshold[threshold]
        decisions = decisions_by_threshold[threshold]
        expected_fields = expected_fields_by_threshold[threshold]
        setting = settings[threshold]
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
                max_rounds=config.generation.max_rounds,
                max_model_len=config.generation.max_model_len,
                prefills=prefills,
            )
            for row in rows:
                row.update(decisions[row["id"]])
            artifact["runs"].append(
                {"run_id": run_id, "seed": seed, "setting": setting.name, "rows": rows}
            )
            validate_evaluation_artifact_for_resume(
                artifact,
                template=templates[threshold],
                task_ids=task_ids,
                expected_row_fields=expected_fields,
            )
            atomic_write_json(output_paths[threshold], artifact, overwrite=True)
            print(f"Checkpointed {setting.name} {run_id}", flush=True)


if __name__ == "__main__":
    main()
