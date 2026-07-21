"""Strict adapter around the pinned When2Tool submodule.

The upstream checkout supplies environment schemas for fair reproduction.  It is
never imported from an absolute machine-specific path, and its code-executor
implementation is not exposed through this adapter.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .constants import ENV_TO_CATEGORY


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


@dataclass(frozen=True)
class ToolRoute:
    exposed_name: str
    category: str
    environment: str
    original_name: str


@lru_cache(maxsize=1)
def _all_candidate_menu() -> tuple[
    tuple[dict[str, Any], ...], tuple[ToolRoute, ...], str
]:
    """Build the fixed, category-unlabeled P_all menu and reverse routes."""

    root = verify_upstream_checkout()
    configured_environments = set(ENV_TO_CATEGORY)
    registry_environments = set(_environment_registry())
    if configured_environments != registry_environments:
        raise ValueError(
            "P_all environments differ from pinned registry: "
            f"configured_only={sorted(configured_environments - registry_environments)}, "
            f"registry_only={sorted(registry_environments - configured_environments)}"
        )
    output: list[dict[str, Any]] = []
    routes: list[ToolRoute] = []
    seen_names: set[str] = set()
    # Global environment/name order is independent of the A/B/C target.
    for environment in sorted(configured_environments):
        schema_path = root / "envs" / f"{environment}.json"
        if not schema_path.is_file():
            raise FileNotFoundError(schema_path)
        schemas = json.loads(schema_path.read_text(encoding="utf-8"))
        if not isinstance(schemas, list) or not schemas:
            raise TypeError(f"Invalid tool schema file: {schema_path}")
        for original in sorted(schemas, key=lambda item: str(item.get("name", ""))):
            original_name = original.get("name")
            if not isinstance(original_name, str) or not original_name:
                raise ValueError(f"Unnamed tool in {schema_path}")
            namespace = environment.removesuffix("Env").lower()
            fixed_name = f"{namespace}__{original_name}"
            if fixed_name in seen_names:
                raise ValueError(f"Duplicate P_all tool name: {fixed_name}")
            seen_names.add(fixed_name)
            function = deepcopy(original)
            # Preserve official description/parameters verbatim.  Adding A/B/C
            # text would leak the target that type onset is meant to discover.
            function["name"] = fixed_name
            output.append({"type": "function", "function": function})
            routes.append(
                ToolRoute(
                    exposed_name=fixed_name,
                    category=ENV_TO_CATEGORY[environment],
                    environment=environment,
                    original_name=original_name,
                )
            )
    if not output:
        raise AssertionError("Pinned P_all menu is empty")
    canonical = json.dumps(
        output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return tuple(output), tuple(routes), hashlib.sha256(canonical).hexdigest()


def build_all_candidate_tools() -> tuple[dict[str, Any], ...]:
    """Return the fixed-order menu shared by every P_all sample."""

    return _all_candidate_menu()[0]


def all_candidate_tool_routes() -> tuple[ToolRoute, ...]:
    """Return the frozen reverse-routing table for P_all calls."""

    return _all_candidate_menu()[1]


@lru_cache(maxsize=1)
def _route_by_name() -> dict[str, ToolRoute]:
    return {route.exposed_name: route for route in all_candidate_tool_routes()}


def resolve_all_candidate_tool(exposed_name: str) -> ToolRoute:
    """Resolve a known P_all name; unknown names fail explicitly."""

    try:
        return _route_by_name()[exposed_name]
    except KeyError as error:
        raise KeyError(f"Unknown P_all tool name: {exposed_name}") from error


def all_candidate_menu_sha256() -> str:
    return _all_candidate_menu()[2]


@lru_cache(maxsize=1)
def load_upstream_runtime() -> tuple[Any, Any]:
    """Import the exact pinned ``utils`` and ``model`` modules.

    The upstream source uses top-level imports, so its ``src`` and repository
    roots must be placed on ``sys.path``.  Existing modules with the same names
    are rejected unless they resolve to the pinned checkout.
    """

    root = verify_upstream_checkout()
    src = root / "src"
    for path in (src, root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    modules = []
    for name in ("utils", "model"):
        module = importlib.import_module(name)
        module_path = Path(module.__file__).resolve()
        if root.resolve() not in module_path.parents:
            raise ImportError(
                f"Imported {name} from {module_path}, outside pinned checkout {root}"
            )
        modules.append(module)
    return modules[0], modules[1]
