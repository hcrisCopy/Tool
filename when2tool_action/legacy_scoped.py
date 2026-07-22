"""Strict import of the already-computed original scoped When2Tool baseline.

The legacy probe used the original per-environment ``P_env`` prompt.  It is a
useful reproduction baseline, but it is not the scoped-adapted probe produced
by the current renderer.  This module deliberately keeps those namespaces
separate and performs every validation before writing an imported artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .constants import (
    CATEGORY_NAMES,
    DIFFICULTIES,
    ENV_TO_CATEGORY,
    EXPECTED_PER_ENV_DIFFICULTY,
    EXPECTED_SPLIT_SIZES,
    SCHEMA_VERSION,
    UPSTREAM_COMMIT,
)
from .data import load_task_json
from .io_utils import atomic_write_json, canonical_json_sha256, sha256_file
from .upstream import load_runtime, verify_upstream_checkout


PROTOCOL_ID = "scoped_original_w2t"
LEGACY_HIDDEN_PROTOCOL_REVISION = (
    "raw-block-v2__w2t-current-no-reasoning__thinking-false__"
    "pall-fixed-namespaced-unlabeled-menu"
)
LEGACY_LABEL_POLICY = "1) Do not use any tools in this task."
LEGACY_PROBE_NOTE = (
    "Inputs use the public hidden-state convention: h0, raw blocks 1..35, "
    "and final-RMSNorm(block 36). The upstream all-layer script is executed "
    "unchanged, including its two StandardScaler passes."
)
TRANSFER_FILES = (
    "train_hidden_no_reasoning.pt",
    "test_hidden_no_reasoning.pt",
    "train_labels_no_reasoning.json",
    "test_labels_no_reasoning.json",
    "baseline_input_manifest.json",
    "probe_no_reasoning.pt",
    "probe_results_no_reasoning.json",
)
EXPECTED_SOURCE_ARTIFACTS = 17
EXPECTED_DESTINATION_ARTIFACTS = 19
EXPECTED_AUDIT_SOURCES = 10


def legacy_audit_source_paths(legacy_root: Path, seed: int) -> tuple[Path, ...]:
    """Exact legacy evidence retained so the import remains independently auditable."""

    paths: list[Path] = []
    for split in ("train", "test"):
        label_dir = legacy_root / "labels" / "full" / f"seed_{seed}"
        paths.extend(
            (
                label_dir / f"{split}_no_tool_outputs.json",
                label_dir / f"{split}_label_stats.csv",
                label_dir / f"{split}_manifest.json",
            )
        )
        hidden_dir = legacy_root / "hidden" / "full" / "P_env"
        paths.extend(
            (
                hidden_dir / f"{split}_manifest.json",
                hidden_dir / f"{split}_metadata.json",
            )
        )
    return tuple(paths)


def _audit_source_relatives(seed: int) -> tuple[str, ...]:
    sentinel = Path("__legacy_root__")
    return tuple(
        path.relative_to(sentinel).as_posix()
        for path in legacy_audit_source_paths(sentinel, seed)
    )


def _expected_receipt_inventory(
    model_slug: str, seed: int
) -> tuple[
    set[str],
    set[str],
    dict[str, str],
]:
    """Return the exact source, destination, and audit mappings for an import."""

    probe_relative = PurePosixPath("probes") / PROTOCOL_ID
    source_paths = {
        (PurePosixPath("baseline") / "w2t_all" / name).as_posix()
        for name in TRANSFER_FILES
    }
    destination_paths = {
        (probe_relative / name).as_posix() for name in TRANSFER_FILES
    }
    audit_mapping: dict[str, str] = {}
    for source_relative in _audit_source_relatives(seed):
        destination_relative = (
            probe_relative / "audit_source" / PurePosixPath(source_relative)
        ).as_posix()
        source_paths.add(source_relative)
        destination_paths.add(destination_relative)
        audit_mapping[source_relative] = destination_relative
    destination_paths.update(
        {
            (
                PurePosixPath("labels")
                / model_slug
                / f"{split}_labels_no_reasoning_{PROTOCOL_ID}.json"
            ).as_posix()
            for split in ("train", "test")
        }
    )
    if len(source_paths) != EXPECTED_SOURCE_ARTIFACTS:
        raise AssertionError(
            f"Importer source inventory changed: {len(source_paths)} "
            f"!= {EXPECTED_SOURCE_ARTIFACTS}"
        )
    if len(destination_paths) != EXPECTED_DESTINATION_ARTIFACTS:
        raise AssertionError(
            f"Importer destination inventory changed: {len(destination_paths)} "
            f"!= {EXPECTED_DESTINATION_ARTIFACTS}"
        )
    if len(audit_mapping) != EXPECTED_AUDIT_SOURCES:
        raise AssertionError(
            f"Importer audit inventory changed: {len(audit_mapping)} "
            f"!= {EXPECTED_AUDIT_SOURCES}"
        )
    return source_paths, destination_paths, audit_mapping


def _resolved_receipt_path(root: Path, relative: str, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TypeError(f"{context} must be a non-empty POSIX relative path")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != relative
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{context} is not a canonical relative path: {relative!r}")
    resolved_root = root.resolve()
    target = resolved_root.joinpath(*parsed.parts).resolve()
    if target == resolved_root or resolved_root not in target.parents:
        raise ValueError(f"{context} escapes its registered root: {relative!r}")
    return target


def validate_imported_scoped_destination(
    receipt: Mapping[str, Any],
    *,
    output_root: Path,
    model_slug: str,
    label_seed: int,
    artifact_root: Path | None = None,
) -> None:
    """Validate a complete import without reading the legacy source directory.

    ``artifact_root`` may be a same-filesystem staging mirror.  Receipt paths
    always remain relative to the final ``output_root`` layout.
    """

    output_root = output_root.resolve()
    artifact_root = output_root if artifact_root is None else artifact_root.resolve()
    expected_sources, expected_destinations, expected_audit = (
        _expected_receipt_inventory(model_slug, label_seed)
    )

    source_hashes = receipt.get("source_artifact_sha256")
    destination_hashes = receipt.get("destination_artifact_sha256")
    if not isinstance(source_hashes, dict):
        raise TypeError("receipt.source_artifact_sha256 must be an object")
    if not isinstance(destination_hashes, dict):
        raise TypeError("receipt.destination_artifact_sha256 must be an object")
    if set(source_hashes) != expected_sources:
        raise ValueError(
            "Receipt source inventory differs from the fixed 17-file protocol: "
            f"missing={sorted(expected_sources-set(source_hashes))}, "
            f"extra={sorted(set(source_hashes)-expected_sources)}"
        )
    if set(destination_hashes) != expected_destinations:
        raise ValueError(
            "Receipt destination inventory differs from the fixed 19-file protocol: "
            f"missing={sorted(expected_destinations-set(destination_hashes))}, "
            f"extra={sorted(set(destination_hashes)-expected_destinations)}"
        )
    for relative, digest in source_hashes.items():
        _require_sha256(digest, f"receipt source hash {relative}")
    for relative, digest in destination_hashes.items():
        _require_sha256(digest, f"receipt destination hash {relative}")
        target = _resolved_receipt_path(
            artifact_root, relative, f"receipt destination path {relative}"
        )
        if not target.is_file():
            raise FileNotFoundError(target)
        _equal(
            sha256_file(target),
            digest,
            f"receipt destination file {relative} SHA256",
        )

    probe_relative = PurePosixPath("probes") / PROTOCOL_ID
    for name in TRANSFER_FILES:
        source_relative = (
            PurePosixPath("baseline") / "w2t_all" / name
        ).as_posix()
        destination_relative = (probe_relative / name).as_posix()
        _equal(
            destination_hashes[destination_relative],
            source_hashes[source_relative],
            f"receipt transferred mapping {name}",
        )

    audit = receipt.get("audit_source")
    if not isinstance(audit, dict):
        raise TypeError("receipt.audit_source must be an object")
    expected_audit_root = (probe_relative / "audit_source").as_posix()
    _equal(audit.get("root"), expected_audit_root, "receipt audit root")
    if audit.get("self_contained_after_legacy_source_removal") is not True:
        raise ValueError(
            "receipt audit source must explicitly be self-contained after source removal"
        )
    files = audit.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_AUDIT_SOURCES:
        raise ValueError(
            f"receipt.audit_source.files must contain exactly {EXPECTED_AUDIT_SOURCES} items"
        )
    seen_sources: set[str] = set()
    seen_destinations: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict) or set(entry) != {
            "source_relative_path",
            "destination_relative_path",
            "sha256",
        }:
            raise ValueError(f"receipt.audit_source.files[{index}] has invalid keys")
        source_relative = entry["source_relative_path"]
        destination_relative = entry["destination_relative_path"]
        digest = entry["sha256"]
        if not isinstance(source_relative, str) or not isinstance(
            destination_relative, str
        ):
            raise TypeError(f"receipt.audit_source.files[{index}] paths must be strings")
        if source_relative in seen_sources or destination_relative in seen_destinations:
            raise ValueError("receipt.audit_source.files contains duplicate paths")
        seen_sources.add(source_relative)
        seen_destinations.add(destination_relative)
        _equal(
            expected_audit.get(source_relative),
            destination_relative,
            f"receipt audit mapping {source_relative}",
        )
        _equal(
            digest,
            source_hashes.get(source_relative),
            f"receipt audit source hash {source_relative}",
        )
        _equal(
            digest,
            destination_hashes.get(destination_relative),
            f"receipt audit destination hash {destination_relative}",
        )
    if seen_sources != set(expected_audit) or seen_destinations != set(
        expected_audit.values()
    ):
        raise ValueError("receipt.audit_source.files differs from the fixed whitelist")

    data_manifest = output_root / "data" / "data_manifest.json"
    if not data_manifest.is_file():
        raise FileNotFoundError(data_manifest)
    _equal(
        receipt.get("destination_data_manifest_sha256"),
        sha256_file(data_manifest),
        "receipt destination data manifest SHA256",
    )
    expected_data_files = {
        f"data/tasks_v1_{split}_category.json" for split in ("train", "test")
    }
    data_file_hashes = receipt.get("destination_data_files_sha256")
    if not isinstance(data_file_hashes, dict) or set(data_file_hashes) != (
        expected_data_files
    ):
        raise ValueError(
            "receipt.destination_data_files_sha256 must bind exactly the scoped "
            "train/test task files"
        )
    for relative, digest in data_file_hashes.items():
        _require_sha256(digest, f"receipt data file hash {relative}")
        data_file = _resolved_receipt_path(
            output_root, relative, f"receipt data file path {relative}"
        )
        if not data_file.is_file():
            raise FileNotFoundError(data_file)
        _equal(
            sha256_file(data_file),
            digest,
            f"receipt data file {relative} SHA256",
        )


@dataclass(frozen=True)
class ValidatedSplit:
    hidden: torch.Tensor
    baseline_meta: list[dict[str, Any]]
    converted_rows: list[dict[str, Any]]
    source_hashes: dict[str, str]
    task_ids_sha256: str
    necessary: int


def _object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _array(path: Path) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"{path} must contain a JSON array")
    return value


def _equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def _binary(value: Any, context: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise TypeError(f"{context} must be the integer 0 or 1")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} is not a lowercase SHA256 digest")
    return value


def _relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _render_with_tokenizer(
    tokenizer: Any, messages: list[dict[str, str]], tools: list[dict[str, Any]]
) -> tuple[str, list[int]]:
    kwargs: dict[str, Any] = {
        "tools": tools,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if not isinstance(text, str) or not text:
        raise TypeError("Tokenizer returned an empty prompt")
    if not isinstance(token_ids, list) or not token_ids or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError("Tokenizer returned malformed prompt token IDs")
    roundtrip = tokenizer(text, add_special_tokens=False)["input_ids"]
    if token_ids != roundtrip:
        raise ValueError("Prompt text and token rendering differ")
    return text, token_ids


def render_legacy_prompt_contracts(
    tasks_by_split: Mapping[str, list[dict[str, Any]]], config: ExperimentConfig
) -> tuple[
    dict[str, dict[int, tuple[str, list[int]]]],
    dict[str, dict[int, tuple[str, list[int]]]],
]:
    """Render the exact legacy label and hidden prompt contracts.

    Label generation used pinned ``init_state`` and therefore includes the
    pinned ListManipulation contract.  The old P_env hidden extractor used the
    same pinned system/user text and tool schemas, but omitted that extra
    ListManipulation system message.  Keeping both renderers explicit prevents
    the imported probe from being mislabeled as current scoped-adapted.
    """

    if not config.paths.model.is_dir():
        raise FileNotFoundError(config.paths.model)
    verify_upstream_checkout()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.paths.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    utils, _, _ = load_runtime()
    tool_format = utils.detect_tool_format(str(config.paths.model))
    if tool_format != "xml":
        raise ValueError("The registered Qwen legacy baseline requires XML tool calls")
    system_prompt = utils.get_system_prompt(tool_format)
    label_prompts: dict[str, dict[int, tuple[str, list[int]]]] = {}
    hidden_prompts: dict[str, dict[int, tuple[str, list[int]]]] = {}
    for split, tasks in tasks_by_split.items():
        label_prompts[split] = {}
        hidden_prompts[split] = {}
        for task in tasks:
            state = utils.init_state(
                task,
                system_prompt,
                record_mode="lite",
                prompt_mode="hard_no_tool",
                require_reasoning=False,
                tool_format="xml",
                tokenizer=tokenizer,
            )
            label_prompts[split][task["id"]] = _render_with_tokenizer(
                tokenizer, state["messages"], state["tools"]
            )
            _, scoped_tools = utils.build_envs(task)
            hidden_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": utils.build_user_message(
                        task["instruction"], "current", require_reasoning=False
                    ),
                },
            ]
            hidden_prompts[split][task["id"]] = _render_with_tokenizer(
                tokenizer, hidden_messages, scoped_tools
            )
    return label_prompts, hidden_prompts


def _validate_label_manifest(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    split: str,
    config: ExperimentConfig,
) -> None:
    expected = {
        "split": split,
        "mode": "full",
        "seed": config.generation.seeds[0],
        "n": len(rows),
        "task_ids": [row["id"] for row in rows],
        "completed": sum(bool(row.get("completed")) for row in rows),
        "no_tool_correct": sum(_binary(row.get("no_tool_correct"), "label") for row in rows),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_seed_extension": True,
        "published_rng_replay": False,
        "split_independent_seed_reset": True,
        "vllm_enable_v1_multiprocessing": False,
        "single_gpu_adaptation": True,
        "enable_thinking": False,
        "max_new_tokens": config.generation.max_new_tokens,
        "max_rounds": config.generation.max_rounds,
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "top_k": config.generation.top_k,
        "repetition_penalty": config.generation.repetition_penalty,
        "do_sample": True,
    }
    for key, value in expected.items():
        _equal(manifest.get(key), value, f"{split} label manifest {key}")


def _validate_baseline_labels(
    artifact: dict[str, Any], *, split: str, expected_n: int
) -> list[dict[str, Any]]:
    _equal(artifact.get("reasoning_mode"), "no_reasoning", f"{split} baseline mode")
    _equal(artifact.get("split"), split, f"{split} baseline split")
    metadata = artifact.get("task_meta")
    labels = artifact.get("no_tool_correct")
    if not isinstance(metadata, list) or not isinstance(labels, list):
        raise TypeError(f"{split} baseline labels must contain two lists")
    if len(metadata) != expected_n or len(labels) != expected_n:
        raise ValueError(f"{split} baseline label count is not {expected_n}")
    ids: set[int] = set()
    for index, row in enumerate(metadata):
        if not isinstance(row, dict):
            raise TypeError(f"{split} baseline metadata row {index} is not an object")
        required = {
            "id", "difficulty", "env", "category", "no_tool_correct",
            "tool_necessary", "first_sentence",
        }
        if set(row) != required:
            raise ValueError(
                f"{split} baseline metadata row {index} keys differ: "
                f"missing={sorted(required-set(row))}, extra={sorted(set(row)-required)}"
            )
        task_id = row["id"]
        if not isinstance(task_id, int) or task_id in ids:
            raise ValueError(f"{split} invalid or duplicate baseline task ID {task_id!r}")
        ids.add(task_id)
        if row["difficulty"] not in DIFFICULTIES:
            raise ValueError(f"{split} task {task_id} has invalid difficulty")
        env = row["env"]
        if env not in ENV_TO_CATEGORY or row["category"] != ENV_TO_CATEGORY[env]:
            raise ValueError(f"{split} task {task_id} has invalid env/category")
        no_tool = _binary(row["no_tool_correct"], f"{split} task {task_id}")
        necessary = _binary(row["tool_necessary"], f"{split} task {task_id}")
        if necessary != 1 - no_tool or labels[index] != no_tool:
            raise ValueError(f"{split} task {task_id} has inconsistent binary labels")
        _equal(row["first_sentence"], "", f"{split} task {task_id} first_sentence")
    return metadata


def _validate_legacy_label_rows(
    rows: list[Any],
    baseline_meta: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    rendered_prompts: Mapping[int, tuple[str, list[int]]],
    *,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    if len(rows) != len(tasks) or len(baseline_meta) != len(tasks):
        raise ValueError(f"{split} task/legacy/baseline counts differ")
    converted: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, (untyped, base, task) in enumerate(
        zip(rows, baseline_meta, tasks, strict=True)
    ):
        if not isinstance(untyped, dict):
            raise TypeError(f"{split} legacy label row {index} is not an object")
        row = untyped
        required = {
            "id", "difficulty", "env", "category", "tool_type", "seed",
            "prompt_variant", "prompt_hash", "rounds", "completed",
            "final_response", "boxed_answer", "cleaned_answer", "gold_answer",
            "no_tool_correct", "tool_necessary", "tool_calls",
            "generation_tokens", "prefill_tokens", "reasoning_mode",
            "upstream_commit", "enable_thinking", "trace",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"{split} legacy label row {index} misses {sorted(missing)}")
        task_id = row["id"]
        if not isinstance(task_id, int) or task_id in seen:
            raise ValueError(f"{split} invalid or duplicate legacy ID {task_id!r}")
        seen.add(task_id)
        _equal(task_id, task["id"], f"{split} task order")
        _equal(base["id"], task_id, f"{split} baseline ID order")
        env = task["gold_env_name"]
        category = ENV_TO_CATEGORY[env]
        for key, expected in {
            "difficulty": task["difficulty"],
            "env": env,
            "category": category,
            "tool_type": category,
            "seed": seed,
            "prompt_variant": "P_env",
            "reasoning_mode": "no_reasoning",
            "upstream_commit": UPSTREAM_COMMIT,
            "enable_thinking": False,
            "tool_calls": 0,
        }.items():
            _equal(row.get(key), expected, f"{split} task {task_id} {key}")
        if type(row.get("completed")) is not bool:
            raise TypeError(f"{split} task {task_id} completed must be bool")
        for key in ("rounds", "generation_tokens", "prefill_tokens"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise TypeError(f"{split} task {task_id} {key} must be nonnegative int")
        no_tool = _binary(row["no_tool_correct"], f"{split} task {task_id}")
        necessary = _binary(row["tool_necessary"], f"{split} task {task_id}")
        if necessary != 1 - no_tool:
            raise ValueError(f"{split} task {task_id} binary labels are inconsistent")
        expected_base = {
            "id": task_id,
            "difficulty": task["difficulty"],
            "env": env,
            "category": category,
            "no_tool_correct": no_tool,
            "tool_necessary": necessary,
            "first_sentence": "",
        }
        _equal(base, expected_base, f"{split} task {task_id} baseline metadata")
        trace = row["trace"]
        if not isinstance(trace, list) or not trace or not isinstance(trace[0], dict):
            raise ValueError(f"{split} task {task_id} lacks a first-round trace")
        prompt_text = trace[0].get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text:
            raise ValueError(f"{split} task {task_id} first prompt is empty")
        _equal(trace[0].get("round"), 1, f"{split} task {task_id} first trace round")
        prompt_hash = _require_sha256(
            row["prompt_hash"], f"{split} task {task_id} prompt_hash"
        )
        _equal(prompt_hash, _sha256_text(prompt_text), f"{split} task {task_id} trace hash")
        rendered_text, _ = rendered_prompts[task_id]
        _equal(prompt_text, rendered_text, f"{split} task {task_id} pinned prompt")
        if LEGACY_LABEL_POLICY not in prompt_text:
            raise ValueError(f"{split} task {task_id} is not a hard-no-tool prompt")
        if task["instruction"] not in prompt_text:
            raise ValueError(f"{split} task {task_id} prompt omits its instruction")
        for tool_name in task["gold_tools"]:
            if tool_name not in prompt_text:
                raise ValueError(
                    f"{split} task {task_id} prompt omits gold tool {tool_name}"
                )
        if row["gold_answer"] != task["expected"]["answer"]:
            raise ValueError(f"{split} task {task_id} gold answer differs from data")
        converted.append(
            {
                "id": task_id,
                "split": split,
                "difficulty": task["difficulty"],
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "gold_env_name": env,
                "gold_tools": list(task["gold_tools"]),
                "no_tool_correct": no_tool,
                "tool_necessary": necessary,
                "gold_action": category if necessary else "NONE",
                "seed": seed,
                "prompt_mode": "hard_no_tool",
                "reasoning_mode": "no_reasoning",
                "tool_scope": "scoped",
                "label_protocol": PROTOCOL_ID,
                "source_prompt_variant": "P_env",
                "source_prompt_sha256": prompt_hash,
                "final_response": row["final_response"],
                "episode_done": row["completed"],
                "rounds": row["rounds"],
            }
        )
    if set(rendered_prompts) != seen:
        raise ValueError(f"{split} rendered-prompt ID set differs from labels")
    return converted


def _validate_hidden_prompt_manifest(
    legacy_root: Path,
    tasks: list[dict[str, Any]],
    rendered_prompts: Mapping[int, tuple[str, list[int]]],
    *,
    split: str,
    expected_shape: tuple[int, int, int],
) -> dict[str, str]:
    source_dir = legacy_root / "hidden" / "full" / "P_env"
    manifest_path = source_dir / f"{split}_manifest.json"
    metadata_path = source_dir / f"{split}_metadata.json"
    manifest = _object(manifest_path)
    expected = {
        "split": split,
        "mode": "full",
        "prompt_variant": "P_env",
        "shape": list(expected_shape),
        "dtype": "torch.float32",
        "extraction_batch_size": 1,
        "enable_thinking": False,
        "protocol_revision": LEGACY_HIDDEN_PROTOCOL_REVISION,
    }
    for key, value in expected.items():
        _equal(manifest.get(key), value, f"{split} hidden manifest {key}")
    _require_sha256(manifest.get("p_all_menu_sha256"), f"{split} P_all menu hash")
    metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_value, list) or len(metadata_value) != len(tasks):
        raise ValueError(f"{split} hidden metadata has wrong row count")
    for row, task in zip(metadata_value, tasks, strict=True):
        if not isinstance(row, dict):
            raise TypeError(f"{split} hidden metadata contains a non-object")
        task_id = task["id"]
        text, token_ids = rendered_prompts[task_id]
        for key, value in {
            "id": task_id,
            "prompt_variant": "P_env",
            "prompt_hash": _sha256_text(text),
            "input_tokens": len(token_ids),
            "decision_index": len(token_ids) - 1,
            "decision_token_id": token_ids[-1],
        }.items():
            _equal(row.get(key), value, f"{split} task {task_id} hidden metadata {key}")
    return {
        _relative_key(manifest_path, legacy_root): sha256_file(manifest_path),
        _relative_key(metadata_path, legacy_root): sha256_file(metadata_path),
    }


def _load_and_validate_hidden(
    path: Path, *, expected_shape: tuple[int, int, int]
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    hidden = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    if tuple(hidden.shape) != expected_shape:
        raise ValueError(f"{path} shape {tuple(hidden.shape)} != {expected_shape}")
    if hidden.dtype != torch.float32:
        raise TypeError(f"{path} dtype {hidden.dtype} != torch.float32")
    for start in range(0, len(hidden), 64):
        if not bool(torch.isfinite(hidden[start : start + 64]).all()):
            raise ValueError(f"{path} contains NaN or infinity")
    return hidden


def _validate_breakdown_row(
    value: Any, *, context: str, expected_n: int | None = None
) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    n = value.get("n")
    necessary = value.get("n_necessary")
    if type(n) is not int or n <= 0 or (expected_n is not None and n != expected_n):
        raise ValueError(f"{context} has invalid n={n!r}")
    if type(necessary) is not int or not 0 <= necessary <= n:
        raise ValueError(f"{context} has invalid n_necessary={necessary!r}")
    matrix = value.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        or any(type(cell) is not int or cell < 0 for row in matrix for cell in row)
    ):
        raise ValueError(f"{context} has malformed confusion matrix")
    if sum(sum(row) for row in matrix) != n or sum(matrix[1]) != necessary:
        raise ValueError(f"{context} confusion counts are inconsistent")
    expected_accuracy = round((matrix[0][0] + matrix[1][1]) / n, 4)
    _equal(value.get("accuracy"), expected_accuracy, f"{context} accuracy")
    auroc = value.get("auroc")
    one_class = necessary in (0, n)
    if one_class and auroc is not None:
        raise ValueError(f"{context} AUROC must be null for a one-class group")
    if not one_class and (
        not isinstance(auroc, (int, float)) or not 0.0 <= float(auroc) <= 1.0
    ):
        raise ValueError(f"{context} has invalid AUROC")


def _breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metadata: list[dict[str, Any]],
    key: str,
    order: list[str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    keys = np.asarray([row[key] for row in metadata], dtype=object)
    for value in order:
        mask = keys == value
        if not bool(mask.any()):
            continue
        truth = y_true[mask]
        prediction = y_pred[mask]
        probability = y_prob[mask]
        unique = np.unique(truth)
        output[value] = {
            "n": int(mask.sum()),
            "n_necessary": int(truth.sum()),
            "accuracy": round(float(accuracy_score(truth, prediction)), 4),
            "auroc": (
                round(float(roc_auc_score(truth, probability)), 4)
                if len(unique) > 1
                else None
            ),
            "confusion_matrix": confusion_matrix(
                truth, prediction, labels=[0, 1]
            ).tolist(),
        }
    return output


def _validate_probe_and_recompute(
    probe_path: Path,
    result_path: Path,
    hidden_by_split: Mapping[str, torch.Tensor],
    metadata_by_split: Mapping[str, list[dict[str, Any]]],
    *,
    n_layers: int,
    hidden_dim: int,
) -> dict[str, Any]:
    result = _object(result_path)
    expected_result = {
        "mode": "no_reasoning",
        "C": 0.0001,
        "all_layers": True,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "best_layer": "all",
        "train_envs": None,
    }
    for key, value in expected_result.items():
        _equal(result.get(key), value, f"probe result {key}")
    for key in ("best_train_acc", "best_test_acc", "best_test_auroc"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"probe result {key} is invalid")
    per_difficulty = result.get("per_difficulty")
    per_env = result.get("per_env")
    if not isinstance(per_difficulty, dict) or set(per_difficulty) != set(DIFFICULTIES):
        raise ValueError("probe per_difficulty keys differ from the benchmark")
    if not isinstance(per_env, dict) or set(per_env) != set(ENV_TO_CATEGORY):
        raise ValueError("probe per_env keys differ from the benchmark")
    test_n = len(metadata_by_split["test"])
    for difficulty, value in per_difficulty.items():
        _validate_breakdown_row(
            value,
            context=f"probe per_difficulty.{difficulty}",
            expected_n=test_n // len(DIFFICULTIES),
        )
    expected_env_n = EXPECTED_PER_ENV_DIFFICULTY["test"] * len(DIFFICULTIES)
    for env, value in per_env.items():
        _validate_breakdown_row(
            value, context=f"probe per_env.{env}", expected_n=expected_env_n
        )

    if not probe_path.is_file():
        raise FileNotFoundError(probe_path)
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    if not isinstance(probe, dict):
        raise TypeError("Legacy probe artifact must be a dictionary")
    for key, value in {
        "C": 0.0001,
        "mode": "no_reasoning",
        "layer": "all",
        "n_layers": n_layers,
    }.items():
        _equal(probe.get(key), value, f"probe artifact {key}")
    features = n_layers * hidden_dim
    tensors: dict[str, torch.Tensor] = {}
    for key in ("coef", "scaler_mean", "scaler_scale"):
        value = probe.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (features,):
            raise ValueError(f"probe {key} must have shape {(features,)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"probe {key} contains NaN or infinity")
        tensors[key] = value
    if not bool((tensors["scaler_scale"] > 0).all()):
        raise ValueError("probe scaler_scale must be positive")
    intercept = probe.get("intercept")
    if not isinstance(intercept, (int, float)) or not math.isfinite(float(intercept)):
        raise ValueError("probe intercept is not finite")

    train_raw = hidden_by_split["train"].reshape(len(hidden_by_split["train"]), -1).numpy()
    test_raw = hidden_by_split["test"].reshape(len(hidden_by_split["test"]), -1).numpy()
    first_scaler = StandardScaler().fit(train_raw)
    np.testing.assert_allclose(
        first_scaler.mean_, tensors["scaler_mean"].numpy(), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        first_scaler.scale_, tensors["scaler_scale"].numpy(), rtol=0, atol=1e-12
    )
    train_first = first_scaler.transform(train_raw)
    test_first = first_scaler.transform(test_raw)
    # The pinned upstream all-layer branch applies StandardScaler twice.  The
    # second scaler was not serialized, so reconstruct it exactly from the
    # validated training hidden states before recomputing every saved metric.
    second_scaler = StandardScaler().fit(train_first)
    train_second = second_scaler.transform(train_first)
    test_second = second_scaler.transform(test_first)
    classifier = LogisticRegression(C=0.0001, solver="lbfgs")
    classifier.classes_ = np.asarray([0, 1], dtype=np.int64)
    classifier.coef_ = tensors["coef"].numpy().reshape(1, -1)
    classifier.intercept_ = np.asarray([float(intercept)], dtype=np.float64)
    classifier.n_features_in_ = features
    train_prediction = classifier.predict(train_second)
    test_prediction = classifier.predict(test_second)
    test_probability = classifier.predict_proba(test_second)[:, 1]
    train_truth = np.asarray(
        [row["tool_necessary"] for row in metadata_by_split["train"]], dtype=np.int64
    )
    test_truth = np.asarray(
        [row["tool_necessary"] for row in metadata_by_split["test"]], dtype=np.int64
    )
    if len(np.unique(train_truth)) != 2 or len(np.unique(test_truth)) != 2:
        raise ValueError("Full legacy probe validation requires both binary classes")
    recomputed = {
        "best_train_acc": round(float(accuracy_score(train_truth, train_prediction)), 4),
        "best_test_acc": round(float(accuracy_score(test_truth, test_prediction)), 4),
        "best_test_auroc": round(float(roc_auc_score(test_truth, test_probability)), 4),
        "per_difficulty": _breakdown(
            test_truth,
            test_prediction,
            test_probability,
            metadata_by_split["test"],
            "difficulty",
            list(DIFFICULTIES),
        ),
        "per_env": _breakdown(
            test_truth,
            test_prediction,
            test_probability,
            metadata_by_split["test"],
            "env",
            sorted(ENV_TO_CATEGORY),
        ),
    }
    for key, value in recomputed.items():
        _equal(result.get(key), value, f"recomputed probe {key}")
    return recomputed


def _atomic_transfer(
    source: Path, destination: Path, *, mode: str, overwrite: bool
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, suffix=".import.tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        if mode == "hardlink":
            os.link(source, temporary)
        elif mode == "copy":
            shutil.copy2(source, temporary)
        else:
            raise ValueError(f"Unsupported transfer mode {mode}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _label_artifact(
    rows: list[dict[str, Any]], *, config: ExperimentConfig, split: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": config.model.slug,
        "split": split,
        "seed": config.generation.seeds[0],
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "scoped",
        "protocol_id": PROTOCOL_ID,
        "adaptation_status": "original-pinned-not-current-scoped-adapted",
        "n": len(rows),
        "rows": rows,
    }


def import_legacy_scoped_baseline(
    config: ExperimentConfig,
    legacy_root: Path,
    output_root: Path,
    *,
    transfer_mode: str = "hardlink",
    overwrite: bool = False,
) -> Path:
    """Validate and import an original scoped baseline only when called."""

    legacy_root = legacy_root.resolve()
    output_root = output_root.resolve()
    if not legacy_root.is_dir():
        raise FileNotFoundError(legacy_root)
    if (
        legacy_root == output_root
        or legacy_root in output_root.parents
        or output_root in legacy_root.parents
    ):
        raise ValueError("Legacy and destination roots must be separate directories")
    if legacy_root.name != config.model.slug or output_root.name != config.model.slug:
        raise ValueError(
            "Legacy and destination model-run directories must match config.model.slug"
        )
    if transfer_mode not in {"hardlink", "copy"}:
        raise ValueError("transfer_mode must be hardlink or copy")
    data_dir = output_root / "data"
    tasks_by_split = {
        split: load_task_json(
            data_dir / f"tasks_v1_{split}_category.json", expected_scope="scoped"
        )
        for split in ("train", "test")
    }
    data_file_hashes = {
        f"data/tasks_v1_{split}_category.json": sha256_file(
            data_dir / f"tasks_v1_{split}_category.json"
        )
        for split in ("train", "test")
    }
    data_manifest_path = data_dir / "data_manifest.json"
    data_manifest = _object(data_manifest_path)
    data_manifest_sha256 = sha256_file(data_manifest_path)
    _equal(data_manifest.get("schema_version"), SCHEMA_VERSION, "data manifest schema")
    data_splits = data_manifest.get("splits")
    if not isinstance(data_splits, dict):
        raise TypeError("data manifest splits must be an object")
    for split, tasks in tasks_by_split.items():
        split_item = data_splits.get(split)
        if not isinstance(split_item, dict):
            raise TypeError(f"data manifest splits.{split} must be an object")
        _equal(split_item.get("n"), len(tasks), f"data manifest {split} count")
        _equal(
            split_item.get("ids_sha256"),
            canonical_json_sha256([task["id"] for task in tasks]),
            f"data manifest {split} ID hash",
        )
        _equal(
            split_item.get("category_file"),
            f"tasks_v1_{split}_category.json",
            f"data manifest {split} scoped filename",
        )
    label_prompts, hidden_prompts = render_legacy_prompt_contracts(
        tasks_by_split, config
    )
    baseline_dir = legacy_root / "baseline" / "w2t_all"
    baseline_manifest_path = baseline_dir / "baseline_input_manifest.json"
    baseline_manifest = _object(baseline_manifest_path)
    for key, value in {
        "upstream_commit": UPSTREAM_COMMIT,
        "regularization_argument": 10000,
        "sklearn_C": 0.0001,
        "note": LEGACY_PROBE_NOTE,
    }.items():
        _equal(baseline_manifest.get(key), value, f"baseline manifest {key}")
    split_manifest = baseline_manifest.get("splits")
    if not isinstance(split_manifest, dict) or set(split_manifest) != {"train", "test"}:
        raise ValueError("baseline manifest must contain exactly train/test splits")

    validated: dict[str, ValidatedSplit] = {}
    source_hashes: dict[str, str] = {
        _relative_key(baseline_manifest_path, legacy_root): sha256_file(
            baseline_manifest_path
        )
    }
    for split in ("train", "test"):
        expected_shape = (
            EXPECTED_SPLIT_SIZES[split],
            config.model.num_hidden_layers + 1,
            config.model.hidden_size,
        )
        item = split_manifest[split]
        if not isinstance(item, dict):
            raise TypeError(f"baseline manifest splits.{split} must be an object")
        _equal(item.get("shape"), list(expected_shape), f"baseline {split} shape")
        hidden_path = baseline_dir / f"{split}_hidden_no_reasoning.pt"
        baseline_labels_path = baseline_dir / f"{split}_labels_no_reasoning.json"
        hidden_hash = sha256_file(hidden_path)
        baseline_labels_hash = sha256_file(baseline_labels_path)
        _equal(
            hidden_hash,
            _require_sha256(item.get("hidden_sha256"), f"baseline {split} hidden hash"),
            f"baseline {split} hidden SHA256",
        )
        _equal(
            baseline_labels_hash,
            _require_sha256(item.get("labels_sha256"), f"baseline {split} labels hash"),
            f"baseline {split} labels SHA256",
        )
        hidden = _load_and_validate_hidden(hidden_path, expected_shape=expected_shape)
        baseline_meta = _validate_baseline_labels(
            _object(baseline_labels_path),
            split=split,
            expected_n=EXPECTED_SPLIT_SIZES[split],
        )
        _equal(item.get("first_id"), baseline_meta[0]["id"], f"baseline {split} first ID")
        _equal(item.get("last_id"), baseline_meta[-1]["id"], f"baseline {split} last ID")
        raw_path = (
            legacy_root
            / "labels"
            / "full"
            / f"seed_{config.generation.seeds[0]}"
            / f"{split}_no_tool_outputs.json"
        )
        raw_rows = _array(raw_path)
        label_manifest_path = raw_path.with_name(f"{split}_manifest.json")
        _validate_label_manifest(
            _object(label_manifest_path), raw_rows, split=split, config=config
        )
        converted = _validate_legacy_label_rows(
            raw_rows,
            baseline_meta,
            tasks_by_split[split],
            label_prompts[split],
            split=split,
            seed=config.generation.seeds[0],
        )
        prompt_hashes = _validate_hidden_prompt_manifest(
            legacy_root,
            tasks_by_split[split],
            hidden_prompts[split],
            split=split,
            expected_shape=expected_shape,
        )
        split_hashes = {
            _relative_key(hidden_path, legacy_root): hidden_hash,
            _relative_key(baseline_labels_path, legacy_root): baseline_labels_hash,
            _relative_key(raw_path, legacy_root): sha256_file(raw_path),
            _relative_key(label_manifest_path, legacy_root): sha256_file(
                label_manifest_path
            ),
            **prompt_hashes,
        }
        source_hashes.update(split_hashes)
        validated[split] = ValidatedSplit(
            hidden=hidden,
            baseline_meta=baseline_meta,
            converted_rows=converted,
            source_hashes=split_hashes,
            task_ids_sha256=canonical_json_sha256(
                [task["id"] for task in tasks_by_split[split]]
            ),
            necessary=sum(row["tool_necessary"] for row in baseline_meta),
        )

    probe_path = baseline_dir / "probe_no_reasoning.pt"
    result_path = baseline_dir / "probe_results_no_reasoning.json"
    recomputed = _validate_probe_and_recompute(
        probe_path,
        result_path,
        {split: item.hidden for split, item in validated.items()},
        {split: item.baseline_meta for split, item in validated.items()},
        n_layers=config.model.num_hidden_layers + 1,
        hidden_dim=config.model.hidden_size,
    )
    source_hashes[_relative_key(probe_path, legacy_root)] = sha256_file(probe_path)
    source_hashes[_relative_key(result_path, legacy_root)] = sha256_file(result_path)
    audit_sources = legacy_audit_source_paths(
        legacy_root, config.generation.seeds[0]
    )
    for source in audit_sources:
        source_hashes[_relative_key(source, legacy_root)] = sha256_file(source)

    probe_output = output_root / "probes" / PROTOCOL_ID
    audit_destinations = {
        source: probe_output / "audit_source" / source.relative_to(legacy_root)
        for source in audit_sources
    }
    labels_output = output_root / "labels" / config.model.slug
    label_paths = {
        split: labels_output
        / f"{split}_labels_no_reasoning_{PROTOCOL_ID}.json"
        for split in ("train", "test")
    }
    receipt_path = probe_output / "migration_receipt.json"
    artifact_targets = [
        *(probe_output / name for name in TRANSFER_FILES),
        *audit_destinations.values(),
        *label_paths.values(),
    ]
    targets = [*artifact_targets, receipt_path]
    expected_sources, expected_destinations, _ = _expected_receipt_inventory(
        config.model.slug, config.generation.seeds[0]
    )
    if set(source_hashes) != expected_sources:
        raise ValueError(
            "Validated legacy source inventory differs from the fixed protocol: "
            f"missing={sorted(expected_sources-set(source_hashes))}, "
            f"extra={sorted(set(source_hashes)-expected_sources)}"
        )
    target_relatives = {
        _relative_key(path, output_root) for path in artifact_targets
    }
    if target_relatives != expected_destinations:
        raise AssertionError("Importer target construction differs from receipt protocol")
    if len(targets) != len(set(targets)):
        raise AssertionError("Importer target paths are not unique")
    for relative in sorted(target_relatives | {_relative_key(receipt_path, output_root)}):
        _resolved_receipt_path(output_root, relative, f"import target {relative}")
    existing = [path for path in targets if os.path.lexists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite imported artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    invalid_targets = [
        path
        for path in existing
        if path.is_dir() and not path.is_symlink()
    ]
    if invalid_targets:
        raise ValueError(
            "Importer targets must be files, not directories: "
            + ", ".join(str(path) for path in invalid_targets)
        )

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{PROTOCOL_ID}.staging-", dir=output_root)
    ).resolve()
    publish_started = False
    try:
        def staged(final_path: Path) -> Path:
            return staging_root / final_path.relative_to(output_root)

        destination_hashes: dict[str, str] = {}
        for name in TRANSFER_FILES:
            source = baseline_dir / name
            source_relative = _relative_key(source, legacy_root)
            expected_source_hash = source_hashes[source_relative]
            _equal(
                sha256_file(source),
                expected_source_hash,
                f"source changed after validation: {source_relative}",
            )
            destination = probe_output / name
            staged_destination = staged(destination)
            _atomic_transfer(
                source,
                staged_destination,
                mode=transfer_mode,
                overwrite=False,
            )
            _equal(
                sha256_file(source),
                expected_source_hash,
                f"source changed during transfer: {source_relative}",
            )
            destination_hash = sha256_file(staged_destination)
            _equal(
                destination_hash,
                expected_source_hash,
                f"staged transferred artifact {name} SHA256",
            )
            destination_hashes[_relative_key(destination, output_root)] = (
                destination_hash
            )

        audit_receipt: list[dict[str, str]] = []
        for source, destination in audit_destinations.items():
            source_relative = _relative_key(source, legacy_root)
            expected_source_hash = source_hashes[source_relative]
            _equal(
                sha256_file(source),
                expected_source_hash,
                f"audit source changed after validation: {source_relative}",
            )
            staged_destination = staged(destination)
            _atomic_transfer(
                source,
                staged_destination,
                mode=transfer_mode,
                overwrite=False,
            )
            _equal(
                sha256_file(source),
                expected_source_hash,
                f"audit source changed during transfer: {source_relative}",
            )
            destination_hash = sha256_file(staged_destination)
            _equal(
                destination_hash,
                expected_source_hash,
                f"staged audit source {source_relative} SHA256",
            )
            destination_relative = _relative_key(destination, output_root)
            destination_hashes[destination_relative] = destination_hash
            audit_receipt.append(
                {
                    "source_relative_path": source_relative,
                    "destination_relative_path": destination_relative,
                    "sha256": destination_hash,
                }
            )

        for split, path in label_paths.items():
            staged_path = staged(path)
            atomic_write_json(
                staged_path,
                _label_artifact(
                    validated[split].converted_rows, config=config, split=split
                ),
                overwrite=False,
            )
            destination_hashes[_relative_key(path, output_root)] = sha256_file(
                staged_path
            )

        receipt = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": config.model.slug,
            "upstream_commit": UPSTREAM_COMMIT,
            "config_sha256": sha256_file(config.source),
            "transfer_mode": transfer_mode,
            "source_artifact_sha256": dict(sorted(source_hashes.items())),
            "destination_data_manifest_sha256": data_manifest_sha256,
            "destination_data_files_sha256": dict(sorted(data_file_hashes.items())),
            "destination_artifact_sha256": dict(sorted(destination_hashes.items())),
            "audit_source": {
                "root": _relative_key(
                    probe_output / "audit_source", output_root
                ),
                "files": audit_receipt,
                "self_contained_after_legacy_source_removal": True,
            },
            "splits": {
                split: {
                    "n": len(validated[split].baseline_meta),
                    "tool_necessary": validated[split].necessary,
                    "task_ids_sha256": validated[split].task_ids_sha256,
                    "hidden_shape": list(validated[split].hidden.shape),
                    "hidden_dtype": str(validated[split].hidden.dtype),
                }
                for split in ("train", "test")
            },
            "probe_validation": {
                "C": 0.0001,
                "layer": "all",
                "n_layers": config.model.num_hidden_layers + 1,
                "hidden_dim": config.model.hidden_size,
                "official_double_standard_scaler_reconstructed": True,
                "all_saved_metrics_recomputed": True,
                "recomputed": recomputed,
            },
            "prompt_validation": {
                "label_prompt": "exact pinned scoped hard-no-tool/no-reasoning render",
                "hidden_prompt": "exact legacy original P_env/current/no-reasoning render",
                "all_prompt_hashes_recomputed": True,
            },
            "compatibility": {
                "probe_scope": "scoped-original-pinned",
                "current_scoped_adapted": False,
                "reason": (
                    "The legacy P_env hidden renderer omitted the additional "
                    "ListManipulation system contract used by the current renderer."
                ),
                "model_weight_hash_available_in_legacy_run": False,
                "allowed_claim": (
                    "original When2Tool scoped binary baseline reproduction"
                ),
            },
        }
        staged_receipt = staged(receipt_path)
        atomic_write_json(staged_receipt, receipt, overwrite=False)
        validate_imported_scoped_destination(
            receipt,
            output_root=output_root,
            artifact_root=staging_root,
            model_slug=config.model.slug,
            label_seed=config.generation.seeds[0],
        )

        if os.path.lexists(receipt_path):
            invalidated_receipt = staging_root / ".previous_receipt.invalidated"
            os.replace(receipt_path, invalidated_receipt)
        publish_started = True
        for destination in sorted(
            artifact_targets, key=lambda path: _relative_key(path, output_root)
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged(destination), destination)
        validate_imported_scoped_destination(
            receipt,
            output_root=output_root,
            model_slug=config.model.slug,
            label_seed=config.generation.seeds[0],
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_receipt, receipt_path)
        validate_imported_scoped_destination(
            _object(receipt_path),
            output_root=output_root,
            model_slug=config.model.slug,
            label_seed=config.generation.seeds[0],
        )
        return receipt_path
    except Exception as error:
        if publish_started:
            if os.path.lexists(receipt_path):
                if receipt_path.is_dir() and not receipt_path.is_symlink():
                    raise RuntimeError(
                        "Scoped import failed after publication began, and the "
                        f"receipt path became a directory: {receipt_path}"
                    ) from error
                receipt_path.unlink()
            raise RuntimeError(
                "Scoped import failed after publication began. A valid migration "
                "receipt was deliberately withheld; partial destination files may "
                "remain. Fix the reported cause and rerun with --overwrite."
            ) from error
        raise
    finally:
        shutil.rmtree(staging_root)
