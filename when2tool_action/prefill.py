"""Exact-ID binary Probe&Prefill decisions for the full-menu prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


PREFILLS = {
    False: "I can solve this directly without using a tool.\n",
    True: "I need to use a tool for this question.\n",
}


def compute_prefills(
    probe_dir: Path,
    task_ids: list[int],
    *,
    threshold: float,
    temperature: float,
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be in (0,1)")
    if temperature <= 0:
        raise ValueError("Probe temperature must be positive")
    probe = torch.load(
        probe_dir / "probe_no_reasoning.pt", map_location="cpu", weights_only=True
    )
    if probe.get("layer") != "all":
        raise ValueError("Probe&Prefill requires the all-layer binary probe")
    hidden = torch.load(
        probe_dir / "test_hidden_no_reasoning.pt",
        map_location="cpu",
        weights_only=True,
    )
    metadata = json.loads(
        (probe_dir / "test_labels_no_reasoning.json").read_text(encoding="utf-8")
    )["task_meta"]
    hidden_ids = [row["id"] for row in metadata]
    if len(hidden_ids) != len(hidden) or len(hidden_ids) != len(set(hidden_ids)):
        raise ValueError("Hidden-state ID metadata is malformed")
    if hidden_ids != task_ids:
        raise ValueError("Evaluation tasks and hidden-state IDs/order differ")
    features = hidden.reshape(len(hidden), -1).numpy()
    scaler = StandardScaler()
    scaler.mean_ = probe["scaler_mean"].numpy()
    scaler.scale_ = probe["scaler_scale"].numpy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    scaled = scaler.transform(features)
    coef = probe["coef"].numpy().reshape(-1)
    intercept = float(probe["intercept"])
    if scaled.shape[1] != len(coef):
        raise ValueError("Probe coefficient dimension differs from hidden states")
    logits = scaled @ coef + intercept
    z = np.clip(logits / temperature, -80.0, 80.0)
    probabilities = 1.0 / (1.0 + np.exp(-z))
    prefills: dict[int, str] = {}
    decisions: dict[int, dict[str, Any]] = {}
    for task_id, logit, probability in zip(task_ids, logits, probabilities):
        use_tool = bool(probability >= threshold)
        prefills[task_id] = PREFILLS[use_tool]
        decisions[task_id] = {
            "probe_logit": float(logit),
            "probe_probability": float(probability),
            "probe_temperature": temperature,
            "probe_threshold": threshold,
            "probe_decision": "use_tool" if use_tool else "no_tool",
            "probe_prefill": PREFILLS[use_tool],
        }
    if set(prefills) != set(task_ids) or set(decisions) != set(task_ids):
        raise AssertionError("Probe&Prefill omitted a task")
    return prefills, decisions
