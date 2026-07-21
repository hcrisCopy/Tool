"""Exact-format adapter for the pinned When2Tool all-layer probe baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .config import ExperimentConfig
from .constants import EXPECTED_SPLIT_SIZES, HIDDEN_PROTOCOL_REVISION
from .io_utils import atomic_torch_save, atomic_write_json, sha256_file
from .upstream import (
    EXPECTED_COMMIT,
    all_candidate_menu_sha256,
    verify_upstream_checkout,
)


def _load_final_norm(model_root: Path) -> tuple[torch.Tensor, float]:
    index_path = model_root / "model.safetensors.index.json"
    config_path = model_root / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Model safetensors index or config.json is missing")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or "model.norm.weight" not in weight_map:
        raise KeyError("model.norm.weight is absent from the safetensors index")
    shard_path = model_root / weight_map["model.norm.weight"]
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("model.norm.weight").clone()
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    epsilon = float(model_config["rms_norm_eps"])
    return weight, epsilon


def apply_public_final_norm(
    raw_hidden: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Replace raw h_L with the public API's final-RMSNorm representation."""

    if raw_hidden.ndim != 3 or raw_hidden.shape[-1] != weight.numel():
        raise ValueError("Raw hidden tensor and final norm weight have incompatible shapes")
    public = raw_hidden.clone()
    weight_bf16 = weight.to(torch.bfloat16)
    for start in range(0, len(public), chunk_size):
        raw_last = public[start : start + chunk_size, -1, :].to(torch.bfloat16)
        normalized_float = raw_last.float()
        variance = normalized_float.square().mean(dim=-1, keepdim=True)
        normalized_float = normalized_float * torch.rsqrt(variance + epsilon)
        normalized_bf16 = weight_bf16 * normalized_float.to(torch.bfloat16)
        public[start : start + chunk_size, -1, :] = normalized_bf16.float()
    return public


def _baseline_labels(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    metadata = [
        {
            "id": row["id"],
            "difficulty": row["difficulty"],
            "env": row["env"],
            "category": row["category"],
            "no_tool_correct": row["no_tool_correct"],
            "tool_necessary": row["tool_necessary"],
            "first_sentence": "",
        }
        for row in rows
    ]
    return {
        "reasoning_mode": "no_reasoning",
        "split": split,
        "no_tool_correct": [row["no_tool_correct"] for row in rows],
        "task_meta": metadata,
    }


def _require_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"{path}: {key}={manifest.get(key)!r} != {expected_value!r}"
            )
    return manifest


def materialize_w2t_inputs(config: ExperimentConfig) -> Path:
    """Create the exact files consumed by upstream train_probe.py."""

    output_dir = config.run_root / "baseline" / "w2t_all"
    output_dir.mkdir(parents=True, exist_ok=True)
    weight, epsilon = _load_final_norm(config.paths.model)
    input_manifest: dict[str, Any] = {
        "upstream_commit": EXPECTED_COMMIT,
        "regularization_argument": 10000,
        "sklearn_C": 0.0001,
        "note": (
            "Inputs use the public hidden-state convention: h0, raw blocks 1..35, "
            "and final-RMSNorm(block 36). The upstream all-layer script is executed "
            "unchanged, including its two StandardScaler passes."
        ),
        "splits": {},
    }
    for split in ("train", "test"):
        hidden_path = (
            config.run_root / "hidden" / "full" / "P_env" / f"{split}_hidden.pt"
        )
        labels_path = (
            config.run_root
            / "labels"
            / "full"
            / f"seed_{config.generation.seeds[0]}"
            / f"{split}_no_tool_outputs.json"
        )
        if not hidden_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Missing full P_env hidden states or labels for split={split}"
            )
        raw_hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
        rows = json.loads(labels_path.read_text(encoding="utf-8"))
        metadata_path = (
            config.run_root / "hidden" / "full" / "P_env" / f"{split}_metadata.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        hidden_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _require_manifest(
            metadata_path.with_name(f"{split}_manifest.json"),
            {
                "split": split,
                "mode": "full",
                "prompt_variant": "P_env",
                "dtype": "torch.float32",
                "extraction_batch_size": config.extraction_batch_size,
                "enable_thinking": False,
                "protocol_revision": HIDDEN_PROTOCOL_REVISION,
                "p_all_menu_sha256": all_candidate_menu_sha256(),
            },
        )
        label_manifest = _require_manifest(
            labels_path.with_name(f"{split}_manifest.json"),
            {
                "split": split,
                "mode": "full",
                "seed": config.generation.seeds[0],
                "upstream_commit": EXPECTED_COMMIT,
                "enable_thinking": False,
                "max_new_tokens": config.generation.max_new_tokens,
                "max_rounds": config.generation.max_rounds,
                "repetition_penalty": config.generation.repetition_penalty,
                "vllm_enable_v1_multiprocessing": False,
                "single_gpu_adaptation": True,
            },
        )
        expected_shape = (
            EXPECTED_SPLIT_SIZES[split],
            config.model.num_hidden_layers + 1,
            config.model.hidden_size,
        )
        if tuple(raw_hidden.shape) != expected_shape:
            raise ValueError(
                f"{split}: hidden shape {tuple(raw_hidden.shape)} != {expected_shape}"
            )
        if len(rows) != EXPECTED_SPLIT_SIZES[split]:
            raise ValueError(
                f"{split}: label count {len(rows)} != {EXPECTED_SPLIT_SIZES[split]}"
            )
        if label_manifest.get("n") != len(rows):
            raise ValueError(f"{split}: label manifest count differs from rows")
        hidden_ids = [row["id"] for row in hidden_metadata]
        label_ids = [row["id"] for row in rows]
        if hidden_ids != label_ids:
            raise ValueError(f"{split}: hidden metadata and label ID order differ")
        if any(row.get("upstream_commit") != EXPECTED_COMMIT for row in rows):
            raise ValueError(f"{split}: labels are not from the pinned upstream evaluator")
        if any(row.get("seed") != config.generation.seeds[0] for row in rows):
            raise ValueError(f"{split}: labels do not use the primary generation seed")
        public_hidden = apply_public_final_norm(raw_hidden, weight, epsilon)
        baseline_hidden_path = output_dir / f"{split}_hidden_no_reasoning.pt"
        baseline_label_path = output_dir / f"{split}_labels_no_reasoning.json"
        atomic_torch_save(baseline_hidden_path, public_hidden)
        atomic_write_json(baseline_label_path, _baseline_labels(rows, split))
        input_manifest["splits"][split] = {
            "shape": list(public_hidden.shape),
            "first_id": label_ids[0],
            "last_id": label_ids[-1],
            "hidden_sha256": sha256_file(baseline_hidden_path),
            "labels_sha256": sha256_file(baseline_label_path),
        }
    atomic_write_json(output_dir / "baseline_input_manifest.json", input_manifest)
    return output_dir


def run_pinned_w2t_all_probe(config: ExperimentConfig) -> Path:
    output_dir = materialize_w2t_inputs(config)
    upstream_root = verify_upstream_checkout()
    script = upstream_root / "src" / "train_probe.py"
    command = [
        sys.executable,
        str(script),
        "--data_dir",
        str(output_dir),
        "--output_dir",
        str(output_dir),
        "--mode",
        "no_reasoning",
        "--reg",
        "10000",
        "--all_layers",
    ]
    result_path = output_dir / "probe_results_no_reasoning.json"
    probe_path = output_dir / "probe_no_reasoning.pt"
    if result_path.exists() or probe_path.exists():
        raise FileExistsError(
            "Pinned baseline outputs already exist; archive the old run before rerunning"
        )
    subprocess.run(command, check=True, cwd=upstream_root)
    if not result_path.is_file() or not probe_path.is_file():
        raise FileNotFoundError("Pinned baseline script did not create its result JSON")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_result = {
        "mode": "no_reasoning",
        "C": 0.0001,
        "all_layers": True,
        "n_layers": config.model.num_hidden_layers + 1,
        "hidden_dim": config.model.hidden_size,
        "best_layer": "all",
    }
    for key, expected_value in expected_result.items():
        if result.get(key) != expected_value:
            raise ValueError(
                f"Pinned baseline result {key}={result.get(key)!r} != {expected_value!r}"
            )
    for metric in ("best_test_auroc", "best_test_acc"):
        value = result.get(metric)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Pinned baseline returned invalid {metric}={value!r}")
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    if probe.get("layer") != "all" or probe.get("n_layers") != expected_result["n_layers"]:
        raise ValueError("Pinned baseline probe metadata is inconsistent")
    expected_features = expected_result["n_layers"] * expected_result["hidden_dim"]
    if tuple(probe["coef"].shape) != (expected_features,):
        raise ValueError(
            f"Pinned baseline coefficient shape {tuple(probe['coef'].shape)} "
            f"!= {(expected_features,)}"
        )
    return result_path
