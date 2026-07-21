"""Publication-oriented diagnostic plots for the non-probe onset stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _base_axes(title: str) -> tuple[Any, Any]:
    figure, axes = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axes.set_title(title)
    axes.set_xlabel("Decoder block (1-based)")
    axes.set_ylabel("Label-shuffle Z (3-layer smoothed)")
    axes.grid(alpha=0.25, linewidth=0.7)
    return figure, axes


def plot_necessity_onset(result: dict[str, Any], path: Path) -> None:
    layers = np.arange(1, len(result["common_smoothed_z"]) + 1)
    figure, axes = _base_axes("Tool-necessity residual-write onset")
    for category, color in zip(("A", "B", "C"), ("#4C78A8", "#F58518", "#54A24B")):
        axes.plot(
            layers,
            result["by_category"][category]["smoothed_z"],
            color=color,
            alpha=0.68,
            linewidth=1.5,
            label=f"Type {category}",
        )
    axes.plot(
        layers,
        result["common_smoothed_z"],
        color="#B22222",
        linewidth=2.5,
        label="Common min(A,B,C)",
    )
    onset = result["onset"]["onset_layer"]
    if onset is not None:
        axes.axvline(onset, color="#222222", linestyle="--", linewidth=1.2, label=f"onset={onset}")
    axes.legend(frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_type_onset(result: dict[str, Any], path: Path) -> None:
    layers = np.arange(1, len(result["overall_smoothed_z"]) + 1)
    figure, axes = _base_axes("Tool-type residual-write onset")
    for category, color in zip(("A", "B", "C"), ("#4C78A8", "#F58518", "#54A24B")):
        axes.plot(
            layers,
            result["category_smoothed_z"][category],
            color=color,
            alpha=0.8,
            linewidth=1.7,
            label=f"Type {category}",
        )
    axes.plot(
        layers,
        result["overall_smoothed_z"],
        color="#7A3E9D",
        linewidth=2.5,
        label="Mean type score",
    )
    onset = result["onset"]["onset_layer"]
    if onset is not None:
        axes.axvline(onset, color="#222222", linestyle="--", linewidth=1.2, label=f"onset={onset}")
    axes.legend(frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)

