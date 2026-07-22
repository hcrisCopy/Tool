"""Audited adapter for the pinned official When2Tool checkout."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .constants import (
    ENV_TO_CATEGORY,
    EXPECTED_ENV_COUNT,
    EXPECTED_TOOL_COUNT,
    UPSTREAM_COMMIT,
)
from .io_utils import canonical_json_sha256


UPSTREAM_ROOT = REPO_ROOT / "third_party" / "when2tool"


def verify_upstream_checkout() -> Path:
    if not (UPSTREAM_ROOT / "envs" / "__init__.py").is_file():
        raise FileNotFoundError(
            "Pinned submodule missing; run: git submodule update --init --recursive"
        )
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"When2Tool commit mismatch: {actual}; expected {UPSTREAM_COMMIT}"
        )
    return UPSTREAM_ROOT


@lru_cache(maxsize=1)
def load_runtime() -> tuple[Any, Any, dict[str, type]]:
    root = verify_upstream_checkout()
    for path in (root / "src", root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    utils = importlib.import_module("utils")
    model = importlib.import_module("model")
    envs_module = importlib.import_module("envs")
    for module in (utils, model, envs_module):
        module_path = Path(module.__file__).resolve()
        if root.resolve() not in module_path.parents:
            raise ImportError(f"Imported {module.__name__} from outside pinned checkout")
    registry = getattr(envs_module, "ENV_REGISTRY", None)
    if not isinstance(registry, dict):
        raise TypeError("envs.ENV_REGISTRY is not a dictionary")
    if set(registry) != set(ENV_TO_CATEGORY):
        raise ValueError(
            "Environment registry differs from frozen category map: "
            f"registry_only={sorted(set(registry) - set(ENV_TO_CATEGORY))}, "
            f"map_only={sorted(set(ENV_TO_CATEGORY) - set(registry))}"
        )
    return utils, model, registry


@dataclass(frozen=True)
class ToolInfo:
    name: str
    environment: str
    category: str


@dataclass
class BuiltEnvironments:
    envs: list[tuple[Any, set[str]]]
    schemas: list[dict[str, Any]]
    route_map: dict[str, ToolInfo]
    menu_sha256: str


def _validate_task_environment(task: dict[str, Any]) -> dict[str, Any]:
    environments = task.get("environments")
    if not isinstance(environments, list) or len(environments) != 1:
        raise ValueError(f"Task {task.get('id')} must contain exactly one gold env")
    gold = environments[0]
    if not isinstance(gold, dict) or gold.get("name") not in ENV_TO_CATEGORY:
        raise ValueError(f"Task {task.get('id')} has invalid gold environment")
    if not isinstance(gold.get("parameters"), dict):
        raise TypeError(f"Task {task.get('id')} parameters must be a dictionary")
    tools = gold.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"Task {task.get('id')} has no gold tools")
    return gold


@lru_cache(maxsize=1)
def frozen_full_menu() -> tuple[tuple[dict[str, Any], ...], tuple[ToolInfo, ...], str]:
    """Build the task-independent 15-environment/33-tool schema menu."""

    _, _, registry = load_runtime()
    schemas: list[dict[str, Any]] = []
    infos: list[ToolInfo] = []
    names: set[str] = set()
    for env_name in sorted(registry):
        env = registry[env_name](parameters={})
        allowed = sorted(set(env.tool_list))
        descriptions = env.get_tool_descs(allowed)
        descriptions = sorted(descriptions, key=lambda row: str(row.get("name", "")))
        if len(descriptions) != len(allowed):
            raise ValueError(f"Schema/tool count mismatch for {env_name}")
        for description in descriptions:
            name = description.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Unnamed schema in {env_name}")
            if name in names:
                raise ValueError(f"Tool names are not globally unique: {name}")
            names.add(name)
            schemas.append({"type": "function", "function": deepcopy(description)})
            infos.append(ToolInfo(name, env_name, ENV_TO_CATEGORY[env_name]))
    if len(registry) != EXPECTED_ENV_COUNT or len(schemas) != EXPECTED_TOOL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ENV_COUNT} envs/{EXPECTED_TOOL_COUNT} tools; "
            f"got {len(registry)}/{len(schemas)}"
        )
    digest = canonical_json_sha256(schemas)
    return tuple(schemas), tuple(infos), digest


def full_menu_sha256() -> str:
    return frozen_full_menu()[2]


def build_environments(task: dict[str, Any], tool_scope: str) -> BuiltEnvironments:
    """Create fresh per-task envs while keeping the full schema menu fixed."""

    gold = _validate_task_environment(task)
    _, _, registry = load_runtime()
    if tool_scope not in {"scoped", "full"}:
        raise ValueError(f"Unknown tool scope: {tool_scope}")
    env_names = [gold["name"]] if tool_scope == "scoped" else sorted(registry)
    envs: list[tuple[Any, set[str]]] = []
    schemas: list[dict[str, Any]] = []
    route_map: dict[str, ToolInfo] = {}
    for env_name in env_names:
        parameters = deepcopy(gold["parameters"]) if env_name == gold["name"] else {}
        env = registry[env_name](parameters=parameters)
        env.include_metadata = False
        if tool_scope == "scoped":
            allowed = set(gold["tools"])
        else:
            allowed = set(env.tool_list)
        if env_name == "ListManipulationEnv":
            allowed &= {"append", "remove", "insert", "sort", "reverse"}
        if not allowed:
            raise ValueError(f"No allowed tools for {env_name}")
        envs.append((env, allowed))
        descriptions = sorted(
            env.get_tool_descs(sorted(allowed)), key=lambda row: str(row.get("name", ""))
        )
        if len(descriptions) != len(allowed):
            raise ValueError(f"Schema/tool count mismatch for {env_name}")
        for description in descriptions:
            name = description.get("name")
            if name in route_map:
                raise ValueError(f"Duplicate exposed tool name: {name}")
            schemas.append({"type": "function", "function": deepcopy(description)})
            route_map[name] = ToolInfo(name, env_name, ENV_TO_CATEGORY[env_name])
    if tool_scope == "full":
        frozen_schemas, frozen_infos, frozen_hash = frozen_full_menu()
        if schemas != list(frozen_schemas):
            raise ValueError(
                f"Task {task.get('id')} produced a non-canonical full-tool menu"
            )
        if tuple(route_map.values()) != frozen_infos:
            raise ValueError("Full-tool route map differs from frozen menu")
        digest = frozen_hash
    else:
        digest = canonical_json_sha256(schemas)
    return BuiltEnvironments(envs, schemas, route_map, digest)


def build_agent(config: Any) -> Any:
    import importlib.metadata

    expected_versions = {"transformers": "4.55.2", "vllm": "0.8.5"}
    actual_versions = {
        package: importlib.metadata.version(package) for package in expected_versions
    }
    if actual_versions != expected_versions:
        raise RuntimeError(
            f"Runtime version mismatch: {actual_versions}; expected {expected_versions}"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    existing_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if existing_mp not in {None, "0"}:
        raise RuntimeError("VLLM_ENABLE_V1_MULTIPROCESSING must be 0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    utils, model_module, _ = load_runtime()
    tool_format = utils.detect_tool_format(str(config.paths.model))
    system_prompt = utils.get_system_prompt(tool_format)
    agent = model_module.AgentModel(
        model_path=str(config.paths.model),
        backend="vllm",
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        max_new_tokens=config.generation.max_new_tokens,
        tensor_parallel_size=config.generation.tensor_parallel_size,
        max_model_len=config.generation.max_model_len,
        vllm_dtype="bfloat16",
        enable_thinking=False,
        system_prompt_override=system_prompt,
    )
    # Upstream context pre-check otherwise falls back silently to 32768.
    agent.max_model_len = config.generation.max_model_len
    return agent


def set_generation_seed(agent: Any, seed: int, repetition_penalty: float) -> None:
    """Inject a replayable vLLM seed without mutating the pinned source."""

    backend = getattr(agent, "engine", None)
    if backend is None or not hasattr(backend, "_sampling_params"):
        raise TypeError("Expected the pinned VLLM backend")
    original = getattr(backend, "_calltool_original_sampling_params", None)
    if original is None:
        original = backend._sampling_params
        backend._calltool_original_sampling_params = original

    def seeded_sampling_params() -> Any:
        generation = backend.generation_config
        values = {"n": 1, "max_tokens": backend.max_new_tokens, "seed": int(seed)}
        for name in ("temperature", "top_p", "top_k"):
            value = getattr(generation, name, None)
            if value is not None:
                values[name] = float(value) if name != "top_k" else int(value)
        values["repetition_penalty"] = float(repetition_penalty)
        return backend.SamplingParams(**values)

    backend._sampling_params = seeded_sampling_params
