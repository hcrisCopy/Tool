"""Reproducibility manifest for a completed experiment workspace."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, ExperimentConfig
from .constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from .io_utils import sha256_bytes, sha256_file
from .upstream import full_menu_sha256, verify_upstream_checkout


MODEL_IDENTITY_SUFFIXES = {
    ".json",
    ".model",
    ".safetensors",
    ".tiktoken",
    ".txt",
}
RUNTIME_PACKAGES = (
    "accelerate",
    "numpy",
    "pandas",
    "peft",
    "pyyaml",
    "scikit-learn",
    "torch",
    "transformers",
    "vllm",
)


def identity_files(root: Path) -> list[Path]:
    """Return model files that determine weights, tokenizer, and configuration."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in MODEL_IDENTITY_SUFFIXES
            and ".git" not in path.relative_to(root).parts
            and ".cache" not in path.relative_to(root).parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"No safetensors model identity found under {root}")
    return files


def _hashed_files(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(
            "Refusing to register provenance from a dirty code checkout:\n" + status
        )
    return {"commit": commit, "worktree_clean": True}


def _runtime() -> dict[str, Any]:
    versions = {
        package: importlib.metadata.version(package) for package in RUNTIME_PACKAGES
    }
    import torch

    cuda: dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_0_name": torch.cuda.get_device_name(0),
                "device_0_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": cuda,
    }


def build_provenance(config: ExperimentConfig) -> dict[str, Any]:
    """Hash code inputs, generated task files, and every model identity file."""

    verify_upstream_checkout()
    data_dir = config.run_root / "data"
    data_manifest = data_dir / "data_manifest.json"
    if not data_manifest.is_file():
        raise FileNotFoundError(data_manifest)
    task_files = sorted(data_dir.glob("tasks_v1_*_category.json"))
    expected_names = {
        "tasks_v1_train_category.json",
        "tasks_v1_test_category.json",
        "tasks_v1_train_fulltools_category.json",
        "tasks_v1_test_fulltools_category.json",
    }
    if {path.name for path in task_files} != expected_names:
        raise ValueError("Generated category/full-tools task file set is incomplete")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "runtime-and-input-provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            "path": config.source.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(config.source),
            "model_slug": config.model.slug,
        },
        "full_menu_sha256": full_menu_sha256(),
        "model_identity": {
            "files": _hashed_files(
                config.paths.model, identity_files(config.paths.model)
            )
        },
        "generated_data": {
            "manifest_sha256": sha256_file(data_manifest),
            "files": _hashed_files(data_dir, task_files),
        },
        "runtime": _runtime(),
    }


def validate_provenance_snapshot(
    registered: dict[str, Any], current: dict[str, Any]
) -> None:
    """Pure exact comparison, excluding only the registration timestamp."""

    if not isinstance(registered, dict) or not isinstance(current, dict):
        raise TypeError("Provenance snapshots must be JSON objects")
    if set(registered) != set(current):
        missing = sorted(set(current) - set(registered))
        extra = sorted(set(registered) - set(current))
        raise ValueError(
            f"Runtime provenance top-level keys differ: missing={missing}, extra={extra}"
        )
    created = registered.get("created_at_utc")
    if not isinstance(created, str) or not created.strip():
        raise TypeError("Runtime provenance created_at_utc must be a non-empty string")
    for section, expected in current.items():
        if section == "created_at_utc":
            continue
        actual = registered.get(section)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Runtime provenance section {section!r} does not match current state"
            )


def validate_runtime_provenance(
    config: ExperimentConfig, path: Path | None = None
) -> dict[str, Any]:
    """Validate and bind the registered code/model/data/runtime snapshot.

    Model files are re-hashed deliberately.  A matching model slug or file size
    is not sufficient evidence that behavior checkpoints used the same weights.
    """

    manifest_path = (
        path.resolve()
        if path is not None
        else config.run_root / "manifests" / "runtime_provenance.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing runtime provenance manifest: {manifest_path}; "
            "run when2tool_action.scripts.audit_provenance first"
        )
    payload = manifest_path.read_bytes()
    try:
        registered = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid runtime provenance JSON: {manifest_path}") from error
    current = build_provenance(config)
    validate_provenance_snapshot(registered, current)
    git = registered.get("git")
    if not isinstance(git, dict) or not isinstance(git.get("commit"), str):
        raise TypeError("Runtime provenance git.commit must be a string")
    return {
        "path": manifest_path,
        "sha256": sha256_bytes(payload),
        "git_commit": git["commit"],
        "model_identity_file_count": len(registered["model_identity"]["files"]),
    }
