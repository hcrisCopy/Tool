"""Pinned binary baseline plus preregistered four-action residual probes."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .constants import ACTIONS, CATEGORY_TO_ENVS, UPSTREAM_COMMIT
from .io_utils import atomic_torch_save, atomic_write_json
from .upstream import verify_upstream_checkout


def run_pinned_binary_probe(probe_dir: Path, *, overwrite: bool) -> Path:
    output = probe_dir / "probe_results_no_reasoning.json"
    model_file = probe_dir / "probe_no_reasoning.pt"
    if (output.exists() or model_file.exists()) and not overwrite:
        raise FileExistsError("Binary probe output exists")
    if overwrite:
        for path in (output, model_file):
            if path.exists():
                path.unlink()
    script = verify_upstream_checkout() / "src" / "train_probe.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--data_dir",
            str(probe_dir),
            "--output_dir",
            str(probe_dir),
            "--mode",
            "no_reasoning",
            "--reg",
            "10000",
            "--all_layers",
        ],
        cwd=verify_upstream_checkout(),
        check=True,
    )
    if not output.is_file() or not model_file.is_file():
        raise FileNotFoundError("Pinned binary probe did not create both artifacts")
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("C") != 0.0001 or result.get("best_layer") != "all":
        raise ValueError("Pinned binary probe result violates registered protocol")
    return output


def _load(probe_dir: Path) -> tuple[torch.Tensor, torch.Tensor, list[dict], list[dict]]:
    hidden: dict[str, torch.Tensor] = {}
    meta: dict[str, list[dict]] = {}
    for split in ("train", "test"):
        hidden[split] = torch.load(
            probe_dir / f"{split}_hidden_no_reasoning.pt",
            map_location="cpu",
            weights_only=True,
        )
        artifact = json.loads(
            (probe_dir / f"{split}_labels_no_reasoning.json").read_text(encoding="utf-8")
        )
        meta[split] = artifact["task_meta"]
        if len(meta[split]) != len(hidden[split]):
            raise ValueError(f"{split} hidden/metadata length mismatch")
        ids = [row["id"] for row in meta[split]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split} duplicate metadata IDs")
    if hidden["train"].shape[1:] != hidden["test"].shape[1:]:
        raise ValueError("Train/test hidden shapes differ")
    return hidden["train"], hidden["test"], meta["train"], meta["test"]


def _metrics(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    labels = np.arange(prob.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_ovr_auroc": float(roc_auc_score(y_true, prob, labels=labels, multi_class="ovr", average="macro")),
        "per_class_recall": recall_score(
            y_true, pred, labels=labels, average=None, zero_division=0
        ).tolist(),
        "confusion": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }


def _fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    c: float,
) -> tuple[dict[str, Any], StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    clf = LogisticRegression(
        C=c, solver="lbfgs", max_iter=3000, random_state=42
    )
    clf.fit(train_scaled, train_y)
    pred = clf.predict(test_scaled)
    prob = clf.predict_proba(test_scaled)
    return _metrics(test_y, pred, prob), scaler, clf


def train_action_probes(
    probe_dir: Path, *, c: float, overwrite: bool
) -> Path:
    result_path = probe_dir / "action_probe_results.json"
    model_path = probe_dir / "action_probe_no_reasoning.pt"
    csv_path = probe_dir / "layerwise_action_probe.csv"
    plot_path = probe_dir / "layerwise_action_probe.png"
    for path in (result_path, model_path, csv_path, plot_path):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    train_h, test_h, train_meta, test_meta = _load(probe_dir)
    encoder = LabelEncoder().fit(list(ACTIONS))
    if tuple(encoder.classes_) != tuple(sorted(ACTIONS)):
        raise AssertionError("Unexpected LabelEncoder order")
    train_y = encoder.transform([row["gold_action"] for row in train_meta])
    test_y = encoder.transform([row["gold_action"] for row in test_meta])
    train_flat = train_h.reshape(len(train_h), -1).numpy()
    test_flat = test_h.reshape(len(test_h), -1).numpy()
    all_metrics, scaler, clf = _fit(train_flat, train_y, test_flat, test_y, c=c)
    all_metrics["classes"] = encoder.classes_.tolist()
    all_metrics["majority_baseline"] = float(
        max(np.bincount(test_y)) / len(test_y)
    )
    priors = np.bincount(train_y, minlength=len(ACTIONS)) / len(train_y)
    test_priors = np.bincount(test_y, minlength=len(ACTIONS)) / len(test_y)
    all_metrics["prior_matched_expected_accuracy"] = float(np.dot(priors, test_priors))

    layer_rows: list[dict[str, Any]] = []
    for layer in range(train_h.shape[1]):
        metrics, _, _ = _fit(
            train_h[:, layer, :].numpy(),
            train_y,
            test_h[:, layer, :].numpy(),
            test_y,
            c=c,
        )
        layer_rows.append(
            {
                "layer": layer,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_ovr_auroc": metrics["macro_ovr_auroc"],
            }
        )

    needed_train = np.array([row["tool_necessary"] == 1 for row in train_meta])
    needed_test = np.array([row["tool_necessary"] == 1 for row in test_meta])
    type_classes = ("A", "B", "C")
    type_encoder = LabelEncoder().fit(type_classes)
    type_train_y = type_encoder.transform(
        [row["category"] for row, keep in zip(train_meta, needed_train) if keep]
    )
    type_test_y = type_encoder.transform(
        [row["category"] for row, keep in zip(test_meta, needed_test) if keep]
    )
    type_metrics, _, _ = _fit(
        train_flat[needed_train], type_train_y,
        test_flat[needed_test], type_test_y,
        c=c,
    )
    type_metrics["classes"] = list(type_encoder.classes_)

    heldout: list[dict[str, Any]] = []
    for fold in range(5):
        held_envs = {CATEGORY_TO_ENVS[category][fold] for category in ("A", "B", "C")}
        train_mask = needed_train & np.array([row["env"] not in held_envs for row in train_meta])
        test_mask = needed_test & np.array([row["env"] in held_envs for row in test_meta])
        held_train_y = type_encoder.transform(
            [row["category"] for row, keep in zip(train_meta, train_mask) if keep]
        )
        held_test_y = type_encoder.transform(
            [row["category"] for row, keep in zip(test_meta, test_mask) if keep]
        )
        fold_metrics, _, _ = _fit(
            train_flat[train_mask], held_train_y,
            test_flat[test_mask], held_test_y,
            c=c,
        )
        heldout.append(
            {
                "fold": fold,
                "heldout_envs": sorted(held_envs),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **fold_metrics,
            }
        )

    result = {
        "upstream_commit": UPSTREAM_COMMIT,
        "C": c,
        "feature": "all-layer public residual hidden at final full-menu prompt token",
        "action_four_class": all_metrics,
        "type_only_needed_three_class": type_metrics,
        "environment_heldout_type": {
            "folds": heldout,
            "mean_accuracy": float(np.mean([row["accuracy"] for row in heldout])),
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in heldout])),
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in heldout])),
        },
        "layerwise": layer_rows,
    }
    atomic_torch_save(
        model_path,
        {
            "classes": encoder.classes_.tolist(),
            "coef": torch.from_numpy(clf.coef_),
            "intercept": torch.from_numpy(clf.intercept_),
            "scaler_mean": torch.from_numpy(scaler.mean_),
            "scaler_scale": torch.from_numpy(scaler.scale_),
            "C": c,
            "layer": "all",
            "n_layers": train_h.shape[1],
        },
        overwrite=overwrite,
    )
    atomic_write_json(result_path, result, overwrite=overwrite)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([row["layer"] for row in layer_rows], [row["balanced_accuracy"] for row in layer_rows], label="Balanced accuracy")
    ax.plot([row["layer"] for row in layer_rows], [row["macro_f1"] for row in layer_rows], label="Macro-F1")
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=1, label="Uniform reference")
    ax.set(xlabel="Residual hidden layer (0 = embedding)", ylabel="Score", title="Four-action linear decodability across layers")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return result_path
