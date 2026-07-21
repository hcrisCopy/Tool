"""Strict adapter around the pinned When2Tool submodule.

The upstream checkout supplies environment schemas for fair reproduction.  It is
never imported from an absolute machine-specific path, and its code-executor
implementation is not exposed through this adapter.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import REPO_ROOT


UPSTREAM_ROOT = REPO_ROOT / "third_party" / "when2tool"
EXPECTED_COMMIT = "66f100089d1f3f7e7f2acee279c4dbf6e7ae5e2c"


def verify_upstream_checkout() -> Path:
    """Require the exact audited upstream commit and return its root."""

    if not (UPSTREAM_ROOT / "envs" / "__init__.py").is_file():
        raise FileNotFoundError(
            "Pinned When2Tool submodule is missing. Run: "
            "git submodule update --init --recursive"
        )
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != EXPECTED_COMMIT:
        raise RuntimeError(
            f"When2Tool commit mismatch: expected {EXPECTED_COMMIT}, got {actual}"
        )
    return UPSTREAM_ROOT


@lru_cache(maxsize=1)
def _environment_registry() -> dict[str, type]:
    root = verify_upstream_checkout()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("envs")
    registry = getattr(module, "ENV_REGISTRY", None)
    if not isinstance(registry, dict):
        raise TypeError("Pinned When2Tool envs.ENV_REGISTRY is not a dictionary")
    return registry


def build_environment_tools(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the original P_env schemas without permitting tool execution."""

    environments = task.get("environments")
    if not isinstance(environments, list) or len(environments) != 1:
        raise ValueError(f"Task {task.get('id')} must contain exactly one environment")

    registry = _environment_registry()
    env_config = environments[0]
    env_name = env_config.get("name")
    if env_name == "CodeExecutorEnv":
        # Construction only exposes schemas.  Execution is deliberately absent here.
        pass
    if env_name not in registry:
        raise KeyError(f"Unsupported environment in task {task.get('id')}: {env_name}")

    parameters = deepcopy(env_config.get("parameters", {}))
    environment = registry[env_name](parameters=parameters)
    allowed = env_config.get("tools")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError(f"Task {task.get('id')} has no allowed tools")
    descriptions = environment.get_tool_descs(sorted(set(allowed)))
    if len(descriptions) != len(set(allowed)):
        raise ValueError(
            f"Task {task.get('id')} schema count does not match allowed tool count"
        )
    return [
        {"type": "function", "function": deepcopy(description)}
        for description in descriptions
    ]

