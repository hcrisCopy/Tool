"""Strict, deterministic inventory for the completed statistics stage.

The handoff manifest is intentionally content-agnostic: earlier pipeline
steps validate the semantics of labels, evaluations, probes, and statistics.
This module freezes the exact files handed to the next stage without copying
paths outside the run root or embedding file contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .io_utils import atomic_write_json, canonical_json_sha256


SCHEMA_VERSION = "when2tool-stage-handoff.v1"
MANIFEST_TYPE = "statistics-stage-handoff"

MANAGED_CATEGORIES = (
    "data",
    "labels",
    "manifests",
    "probes",
    "outputs",
    "analysis",
    "reports",
)
EXCLUDED_TOP_LEVEL_DIRECTORIES = (
    "logs",
    "smoke",
    "tmp",
    "staging",
    "cache",
)

ANALYSIS_PROTOCOLS = (
    "fulltools",
    "scoped_adapted",
    "scoped_original_w2t",
)
THRESHOLDS = ("0.1", "0.3", "0.5", "0.7", "0.9")
SCOPED_PROMPT_STEMS = tuple(
    f"{prompt}_{reasoning}_scoped"
    for reasoning in ("no_reasoning", "reasoning")
    for prompt in (
        "force_tool",
        "current",
        "necessary_tool",
        "sparse_tool",
        "no_tool",
    )
)

FORMAL_BEHAVIOR_OUTPUTS = tuple(
    sorted(
        (
            "outputs/fulltools/current_no_reasoning_fulltools.json",
            "outputs/fulltools/necessary_tool_no_reasoning_fulltools.json",
            "outputs/fulltools/sparse_tool_no_reasoning_fulltools.json",
            *(
                f"outputs/fulltools/probe_prefill_t{threshold}_fulltools.json"
                for threshold in THRESHOLDS
            ),
            *(
                f"outputs/scoped_adapted/{stem}.json"
                for stem in SCOPED_PROMPT_STEMS
            ),
            *(
                f"outputs/scoped_adapted/probe_prefill_t{threshold}_scoped.json"
                for threshold in THRESHOLDS
            ),
            *(
                f"outputs/scoped_original_w2t/{stem}.json"
                for stem in SCOPED_PROMPT_STEMS
            ),
            *(
                "outputs/scoped_original_w2t/"
                f"probe_prefill_t{threshold}_scoped_original_w2t.json"
                for threshold in THRESHOLDS
            ),
        )
    )
)

REQUIRED_ARTIFACTS = tuple(
    sorted(
        (
            "data/data_manifest.json",
            "manifests/runtime_provenance.json",
            "probes/scoped_original_w2t/migration_receipt.json",
            "outputs/scoped_original_w2t/relabel_receipt.json",
            *(
                f"analysis/{protocol}/summary.json"
                for protocol in ANALYSIS_PROTOCOLS
            ),
        )
    )
)

_EXACT_OUTPUT_FILES = frozenset(
    (*FORMAL_BEHAVIOR_OUTPUTS, "outputs/scoped_original_w2t/relabel_receipt.json")
)
_EXACT_OUTPUT_DIRECTORIES = frozenset(
    {
        "outputs/fulltools",
        "outputs/scoped_adapted",
        "outputs/scoped_original_w2t",
    }
)
_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_TEMP_TOKEN_PATTERN = re.compile(r"(?:^|[._-])(?:tmp|temp)(?:[._-]|$)")
_LOG_TOKEN_PATTERN = re.compile(r"(?:^|[._-])logs?(?:[._-]|$)")
_HIDDEN_STAGE_PATTERN = re.compile(r"(?:^|[._-])stage(?:[._-]|$)")


def _validate_commit(value: str, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be a canonical lowercase 40- or 64-hex commit ID"
        )
    return value


def _resolve_output(path: Path) -> Path:
    raw = Path(path)
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    if absolute.is_symlink():
        raise ValueError("Stage handoff output must not be a symlink")
    return absolute.resolve()


def _forbidden_component(component: str) -> str | None:
    """Return the contamination class for a managed path component."""

    lowered = component.casefold()
    if "smoke" in lowered:
        return "smoke"
    if "staging" in lowered:
        return "staging"
    if "cache" in lowered:
        return "cache"
    if _TEMP_TOKEN_PATTERN.search(lowered):
        return "temporary"
    if _LOG_TOKEN_PATTERN.search(lowered):
        return "logs"
    # Transactional publishers use hidden names such as
    # .action-stats-stage-*.  Do not reject legitimate reports/stages or a
    # report named STAGE_STATISTICS_*.md.
    if lowered.startswith(".") and _HIDDEN_STAGE_PATTERN.search(lowered):
        return "staging"
    return None


def _relative_posix(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Artifact escapes run root: {path}") from error
    key = relative.as_posix()
    parsed = PurePosixPath(key)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"Non-canonical artifact path: {key!r}")
    return key


def _validate_output_target(run_root: Path, output: Path) -> None:
    if output.exists() and (output.is_symlink() or not output.is_file()):
        raise ValueError("Stage handoff output must be a regular file when it exists")
    try:
        relative = output.relative_to(run_root).as_posix()
    except ValueError:
        return
    if relative in REQUIRED_ARTIFACTS or relative in FORMAL_BEHAVIOR_OUTPUTS:
        raise ValueError(f"Stage handoff output aliases a required artifact: {relative}")
    parts = PurePosixPath(relative).parts
    if parts[0] != "manifests":
        raise ValueError(
            "A stage handoff output inside run root must be placed under manifests/"
        )
    for component in parts[1:]:
        contamination = _forbidden_component(component)
        if contamination is not None:
            raise ValueError(
                f"Stage handoff output uses forbidden {contamination} path: {relative}"
            )


def _directory_entries(
    category_root: Path,
    *,
    run_root: Path,
    excluded_output: Path,
) -> tuple[list[Path], list[Path]]:
    """Discover regular files and directories without following symlinks."""

    files: list[Path] = []
    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(
        category_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current_path / name
            relative = _relative_posix(path, run_root)
            contamination = _forbidden_component(name)
            if contamination is not None:
                raise ValueError(
                    f"Forbidden {contamination} directory in managed artifacts: {relative}"
                )
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink directory is not allowed: {relative}")
            if not stat.S_ISDIR(mode):
                raise ValueError(f"Non-directory entry discovered as directory: {relative}")
            directories.append(path)
        for name in file_names:
            path = current_path / name
            if path == excluded_output:
                continue
            relative = _relative_posix(path, run_root)
            contamination = _forbidden_component(name)
            if contamination is not None:
                raise ValueError(
                    f"Forbidden {contamination} file in managed artifacts: {relative}"
                )
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlink file is not allowed: {relative}")
            if not stat.S_ISREG(mode):
                raise ValueError(f"Non-regular artifact is not allowed: {relative}")
            files.append(path)
    return files, directories


def _discover_managed_files(run_root: Path, excluded_output: Path) -> list[Path]:
    files: list[Path] = []
    for category in MANAGED_CATEGORIES:
        category_root = run_root / category
        if not category_root.exists():
            raise FileNotFoundError(f"Missing managed directory: {category}")
        mode = category_root.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError(f"Managed category must be a real directory: {category}")
        category_files, category_directories = _directory_entries(
            category_root,
            run_root=run_root,
            excluded_output=excluded_output,
        )
        if category == "outputs":
            actual_directories = {
                _relative_posix(path, run_root) for path in category_directories
            }
            if actual_directories != _EXACT_OUTPUT_DIRECTORIES:
                raise ValueError(
                    "Formal output directory inventory mismatch: "
                    f"missing={sorted(_EXACT_OUTPUT_DIRECTORIES-actual_directories)}, "
                    f"extra={sorted(actual_directories-_EXACT_OUTPUT_DIRECTORIES)}"
                )
        if not category_files:
            raise ValueError(f"Managed category is empty: {category}")
        files.extend(category_files)
    return sorted(files, key=lambda path: _relative_posix(path, run_root))


def _file_signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)


def _hash_stable_file(path: Path) -> tuple[int, str, tuple[int, int, int, int]]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Artifact is no longer a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        if _file_signature(opened_before) != _file_signature(before):
            raise RuntimeError(f"Artifact changed while opening it: {path}")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    after = path.lstat()
    signature = _file_signature(after)
    if (
        _file_signature(before) != _file_signature(opened_after)
        or _file_signature(before) != signature
    ):
        raise RuntimeError(f"Artifact changed while hashing it: {path}")
    return after.st_size, digest.hexdigest(), signature


def _validate_required_json_objects(run_root: Path) -> None:
    for relative in REQUIRED_ARTIFACTS:
        path = run_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing required regular artifact: {relative}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Required artifact is not valid UTF-8 JSON: {relative}") from error
        if not isinstance(value, dict):
            raise TypeError(f"Required artifact must be a JSON object: {relative}")


def _validate_exact_output_inventory(paths: Iterable[Path], run_root: Path) -> None:
    actual = {
        _relative_posix(path, run_root)
        for path in paths
        if _relative_posix(path, run_root).startswith("outputs/")
    }
    if actual != _EXACT_OUTPUT_FILES:
        raise ValueError(
            "Formal output inventory mismatch: "
            f"missing={sorted(_EXACT_OUTPUT_FILES - actual)}, "
            f"extra={sorted(actual - _EXACT_OUTPUT_FILES)}"
        )


def build_stage_handoff(
    run_root: Path,
    *,
    output: Path,
    behavior_commit: str,
    statistics_commit: str,
) -> dict[str, Any]:
    """Build a deterministic handoff payload without writing any file."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output_path = _resolve_output(output)
    _validate_output_target(root, output_path)
    behavior_commit = _validate_commit(behavior_commit, "behavior_commit")
    statistics_commit = _validate_commit(statistics_commit, "statistics_commit")

    paths = _discover_managed_files(root, output_path)
    _validate_exact_output_inventory(paths, root)
    _validate_required_json_objects(root)

    artifacts: list[dict[str, Any]] = []
    signatures: dict[str, tuple[int, int, int, int]] = {}
    for path in paths:
        relative = _relative_posix(path, root)
        size, digest, signature = _hash_stable_file(path)
        category = PurePosixPath(relative).parts[0]
        artifacts.append(
            {
                "category": category,
                "path": relative,
                "bytes": size,
                "sha256": digest,
            }
        )
        signatures[relative] = signature

    # Detect files added, removed, replaced, or modified during inventorying.
    final_paths = _discover_managed_files(root, output_path)
    final_relatives = [_relative_posix(path, root) for path in final_paths]
    if final_relatives != [artifact["path"] for artifact in artifacts]:
        raise RuntimeError("Managed artifact inventory changed while building handoff")
    for path in final_paths:
        relative = _relative_posix(path, root)
        if _file_signature(path.lstat()) != signatures[relative]:
            raise RuntimeError(f"Managed artifact changed after hashing: {relative}")

    category_summary: dict[str, dict[str, int]] = {}
    for category in MANAGED_CATEGORIES:
        members = [item for item in artifacts if item["category"] == category]
        category_summary[category] = {
            "file_count": len(members),
            "bytes": sum(item["bytes"] for item in members),
        }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "code_commits": {
            "behavior": behavior_commit,
            "statistics": statistics_commit,
        },
        "inventory_policy": {
            "path_base": "run_root",
            "managed_categories": list(MANAGED_CATEGORIES),
            "excluded_top_level_directories": list(EXCLUDED_TOP_LEVEL_DIRECTORIES),
            "logs_hashed": False,
            "manifest_self_hashed": False,
            "symlinks_allowed": False,
        },
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "formal_behavior_outputs": list(FORMAL_BEHAVIOR_OUTPUTS),
        "artifacts": artifacts,
        "category_summary": category_summary,
        "totals": {
            "file_count": len(artifacts),
            "bytes": sum(item["bytes"] for item in artifacts),
        },
    }
    payload["inventory_sha256"] = canonical_json_sha256(artifacts)
    return payload


def write_stage_handoff(
    run_root: Path,
    *,
    output: Path,
    behavior_commit: str,
    statistics_commit: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate, build, and atomically publish one stage handoff manifest."""

    output_path = _resolve_output(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite explicitly"
        )
    payload = build_stage_handoff(
        run_root,
        output=output_path,
        behavior_commit=behavior_commit,
        statistics_commit=statistics_commit,
    )
    atomic_write_json(output_path, payload, overwrite=overwrite)
    return payload
