"""Non-probe residual-write onset analysis from the experiment protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


EPSILON = 1e-6


@dataclass(frozen=True)
class ContrastCurve:
    observed_score: np.ndarray
    null_mean: np.ndarray
    null_std: np.ndarray
    z_score: np.ndarray
    smoothed_z: np.ndarray
    max_score_fwer_p: float
    n_positive: int
    n_negative: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_score": self.observed_score.tolist(),
            "null_mean": self.null_mean.tolist(),
            "null_std": self.null_std.tolist(),
            "z_score": self.z_score.tolist(),
            "smoothed_z": self.smoothed_z.tolist(),
            "max_score_fwer_p": self.max_score_fwer_p,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


def standardize_residual_writes(
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert h_0..h_L into FP32 standardized r_1..r_L."""

    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError(f"Expected [N,L+1,D] hidden tensor, got {tuple(hidden.shape)}")
    values = hidden.float()
    residual = values[:, 1:, :] - values[:, :-1, :]
    mean = residual.mean(dim=0)
    std = residual.std(dim=0, unbiased=False)
    standardized = (residual - mean) / (std + EPSILON)
    if not torch.isfinite(standardized).all():
        raise FloatingPointError("Non-finite standardized residual write values")
    return standardized, mean, std


def _smooth_three(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Smoothing expects a one-dimensional layer curve")
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    # PDF §5.3: centered 3-point mean internally; at layer 1/L average only
    # the adjacent available layers (a 2-point boundary mean).
    smoothed = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        left = max(0, index - 1)
        right = min(len(values), index + 2)
        smoothed[index] = float(np.mean(values[left:right]))
    return smoothed


def _layer_scores_from_masks(
    values: torch.Tensor,
    masks: np.ndarray,
    *,
    device: str,
    permutation_batch_size: int = 25,
) -> np.ndarray:
    """Vectorized Welch T^2 layer scores for observed and permuted masks."""

    if values.ndim != 3:
        raise ValueError("values must have shape [N,L,D]")
    n_samples, n_layers, hidden_size = values.shape
    if masks.ndim != 2 or masks.shape[1] != n_samples:
        raise ValueError("Permutation masks do not match contrast sample count")
    flattened = values.reshape(n_samples, n_layers * hidden_size).to(
        device=device, dtype=torch.float32
    )
    squared = flattened.square()
    total_sum = flattened.sum(dim=0, keepdim=True)
    total_square = squared.sum(dim=0, keepdim=True)
    output: list[torch.Tensor] = []

    for start in range(0, masks.shape[0], permutation_batch_size):
        mask = torch.from_numpy(
            masks[start : start + permutation_batch_size].astype(np.float32)
        ).to(device)
        n_positive = mask.sum(dim=1, keepdim=True)
        n_negative = n_samples - n_positive
        if (n_positive < 2).any() or (n_negative < 2).any():
            raise ValueError("Every contrast permutation requires at least two samples per side")
        positive_sum = mask @ flattened
        positive_square = mask @ squared
        negative_sum = total_sum - positive_sum
        negative_square = total_square - positive_square

        positive_mean = positive_sum / n_positive
        negative_mean = negative_sum / n_negative
        positive_var = (
            positive_square - positive_sum.square() / n_positive
        ) / (n_positive - 1)
        negative_var = (
            negative_square - negative_sum.square() / n_negative
        ) / (n_negative - 1)
        positive_var.clamp_min_(0)
        negative_var.clamp_min_(0)
        denominator = torch.sqrt(
            positive_var / n_positive + negative_var / n_negative + EPSILON
        )
        welch = (positive_mean - negative_mean) / denominator
        layer_scores = welch.square().reshape(-1, n_layers, hidden_size).mean(dim=2)
        output.append(layer_scores.cpu())

    return torch.cat(output, dim=0).numpy().astype(np.float64, copy=False)


def _stratified_masks(
    labels: np.ndarray,
    strata: np.ndarray,
    n_shuffles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    masks = np.empty((n_shuffles + 1, len(labels)), dtype=bool)
    masks[0] = labels
    unique_strata = sorted(set(str(value) for value in strata))
    stratum_indices = [
        np.flatnonzero(strata.astype(str) == stratum) for stratum in unique_strata
    ]
    for row in range(1, n_shuffles + 1):
        permuted = labels.copy()
        for indices in stratum_indices:
            permuted[indices] = rng.permutation(labels[indices])
        masks[row] = permuted
    return masks


def _environment_category_masks(
    categories: np.ndarray,
    environments: np.ndarray,
    target: str,
    n_shuffles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    masks = np.empty((n_shuffles + 1, len(categories)), dtype=bool)
    masks[0] = categories == target
    unique_environments = sorted(set(str(value) for value in environments))
    category_by_environment: dict[str, str] = {}
    for environment in unique_environments:
        observed = set(categories[environments.astype(str) == environment].tolist())
        if len(observed) != 1:
            raise ValueError(f"Environment {environment} has categories {observed}")
        category_by_environment[environment] = observed.pop()
    assignments = np.array(
        [category_by_environment[environment] for environment in unique_environments]
    )
    unique_categories, category_counts = np.unique(assignments, return_counts=True)
    if set(unique_categories.tolist()) != {"A", "B", "C"} or int(category_counts.min()) < 3:
        raise ValueError(
            "Type permutation requires all categories and at least three represented "
            "environments per category"
        )
    environment_array = environments.astype(str)
    for row in range(1, n_shuffles + 1):
        shuffled = rng.permutation(assignments)
        mapped = dict(zip(unique_environments, shuffled, strict=True))
        masks[row] = np.array(
            [mapped[environment] == target for environment in environment_array]
        )
    return masks


def contrast_curve(
    values: torch.Tensor,
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    n_shuffles: int,
    rng: np.random.Generator,
    device: str,
    strata: np.ndarray | None = None,
    type_categories: np.ndarray | None = None,
    type_environments: np.ndarray | None = None,
    type_target: str | None = None,
) -> ContrastCurve:
    positive = np.asarray(positive, dtype=bool)
    negative = np.asarray(negative, dtype=bool)
    if positive.shape != negative.shape or positive.shape != (values.shape[0],):
        raise ValueError("Contrast masks must match the full value tensor")
    if np.any(positive & negative):
        raise ValueError("Positive and negative contrast sets overlap")
    pool = positive | negative
    pooled_values = values[torch.from_numpy(pool)]
    observed = positive[pool]
    if int(observed.sum()) < 2 or int((~observed).sum()) < 2:
        raise ValueError(
            f"Contrast has n_positive={int(observed.sum())}, "
            f"n_negative={int((~observed).sum())}"
        )

    if type_target is None:
        if strata is None:
            raise ValueError("Necessity contrast requires permutation strata")
        masks = _stratified_masks(observed, strata[pool], n_shuffles, rng)
    else:
        if type_categories is None or type_environments is None:
            raise ValueError("Type contrast requires category and environment arrays")
        masks = _environment_category_masks(
            type_categories[pool],
            type_environments[pool],
            type_target,
            n_shuffles,
            rng,
        )
        if not np.array_equal(masks[0], observed):
            raise AssertionError("Observed type mask differs from target category mask")

    scores = _layer_scores_from_masks(pooled_values, masks, device=device)
    observed_score = scores[0]
    null = scores[1:]
    null_mean = null.mean(axis=0)
    null_std = null.std(axis=0, ddof=1)
    z_score = (observed_score - null_mean) / (null_std + EPSILON)
    smoothed = _smooth_three(z_score)
    p_fwer = float(
        (1 + np.sum(null.max(axis=1) >= observed_score.max())) / (len(null) + 1)
    )
    return ContrastCurve(
        observed_score=observed_score,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        smoothed_z=smoothed,
        max_score_fwer_p=p_fwer,
        n_positive=int(observed.sum()),
        n_negative=int((~observed).sum()),
    )


def select_onset(
    curve: np.ndarray,
    *,
    peak_fraction: float,
    max_window: int,
) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    peak_value = float(values.max())
    peak_index = int(values.argmax())
    threshold = peak_fraction * peak_value
    candidates = np.flatnonzero(values >= threshold) if peak_value > 0 else np.array([])
    if len(candidates) == 0:
        return {
            "onset_layer": None,
            "peak_layer": peak_index + 1,
            "peak_value": peak_value,
            "peak_ratio": None,
            "background_median": None,
            "peak_prominence": None,
            "window": [],
        }
    onset_index = int(candidates[0])
    background_mask = np.ones(len(values), dtype=bool)
    background_mask[max(0, onset_index - 2) : min(len(values), onset_index + 3)] = False
    background = values[background_mask]
    background_median = float(np.median(background))
    peak_ratio = float(values[onset_index] / max(background_median, EPSILON))

    default_left = max(0, onset_index - 1)
    default_right = min(len(values) - 1, onset_index + 1)
    window_indices = list(range(default_left, default_right + 1))
    half_height = 0.5 * float(values[onset_index])
    left = onset_index
    right = onset_index
    while left > 0 and values[left - 1] >= half_height:
        left -= 1
    while right + 1 < len(values) and values[right + 1] >= half_height:
        right += 1
    if right - left + 1 > len(window_indices):
        width = min(max_window, right - left + 1)
        start = min(max(left, onset_index - width // 2), right - width + 1)
        window_indices = list(range(start, start + width))

    return {
        "onset_layer": onset_index + 1,
        "peak_layer": peak_index + 1,
        "peak_value": peak_value,
        "peak_ratio": peak_ratio,
        "background_median": background_median,
        "peak_prominence": float(values[onset_index] - background_median),
        "window": [index + 1 for index in window_indices],
    }


def masks_from_metadata(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "category": np.array([row["category"] for row in rows]),
        "difficulty": np.array([row["difficulty"] for row in rows]),
        "environment": np.array([row["env"] for row in rows]),
        "tool_necessary": np.array(
            [int(row["tool_necessary"]) for row in rows], dtype=np.int8
        ),
    }


def clean_set_masks(metadata: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for category in ("A", "B", "C"):
        output[f"E_{category}"] = (
            (metadata["category"] == category)
            & (metadata["difficulty"] == "easy")
            & (metadata["tool_necessary"] == 0)
        )
        output[f"H_{category}"] = (
            (metadata["category"] == category)
            & (metadata["difficulty"] == "hard")
            & (metadata["tool_necessary"] == 1)
        )
    return output


def run_onset_analysis(
    hidden_necessity: torch.Tensor,
    hidden_type: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    n_shuffles: int,
    seed: int,
    peak_fraction: float,
    max_window: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if hidden_necessity.shape != hidden_type.shape:
        raise ValueError("Necessity and type hidden tensors have different shapes")
    if hidden_necessity.shape[0] != len(rows):
        raise ValueError("Hidden rows and label metadata have different lengths")
    necessity_residual, necessity_mean, necessity_std = standardize_residual_writes(
        hidden_necessity
    )
    type_residual, type_mean, type_std = standardize_residual_writes(hidden_type)
    metadata = masks_from_metadata(rows)
    clean = clean_set_masks(metadata)
    rng = np.random.default_rng(seed)

    necessity_curves: dict[str, ContrastCurve] = {}
    for category in ("A", "B", "C"):
        necessity_curves[category] = contrast_curve(
            necessity_residual,
            clean[f"H_{category}"],
            clean[f"E_{category}"],
            n_shuffles=n_shuffles,
            rng=rng,
            device=device,
            strata=metadata["environment"],
        )
    necessity_common = np.min(
        np.stack(
            [necessity_curves[category].smoothed_z for category in ("A", "B", "C")]
        ),
        axis=0,
    )

    type_curves: dict[str, dict[str, ContrastCurve]] = {}
    type_category_curves: dict[str, np.ndarray] = {}
    for category in ("A", "B", "C"):
        by_state: dict[str, ContrastCurve] = {}
        for state in ("E", "H"):
            positive = clean[f"{state}_{category}"]
            negative = np.zeros(len(rows), dtype=bool)
            for other in ("A", "B", "C"):
                if other != category:
                    negative |= clean[f"{state}_{other}"]
            by_state[state] = contrast_curve(
                type_residual,
                positive,
                negative,
                n_shuffles=n_shuffles,
                rng=rng,
                device=device,
                type_categories=metadata["category"],
                type_environments=metadata["environment"],
                type_target=category,
            )
        type_curves[category] = by_state
        type_category_curves[category] = np.minimum(
            by_state["E"].smoothed_z, by_state["H"].smoothed_z
        )
    type_overall = np.mean(
        np.stack([type_category_curves[c] for c in ("A", "B", "C")]), axis=0
    )

    necessity_onset = select_onset(
        necessity_common, peak_fraction=peak_fraction, max_window=max_window
    )
    type_onset = select_onset(
        type_overall, peak_fraction=peak_fraction, max_window=max_window
    )
    n_layers = hidden_necessity.shape[1] - 1
    necessity_onset["early_layer_gate"] = bool(
        necessity_onset["onset_layer"] is not None
        and necessity_onset["onset_layer"] / n_layers <= 0.45
    )
    necessity_onset["peak_ratio_gate"] = bool(
        necessity_onset["peak_ratio"] is not None
        and necessity_onset["peak_ratio"] >= 1.5
    )

    result = {
        "n_shuffles": n_shuffles,
        "seed": seed,
        "protocol_conventions": {
            "standardization": "(r - train_mean) / (train_std + 1e-6)",
            "permutation_z": "(score - null_mean) / (null_std + 1e-6)",
            "smoothing": "centered three-point mean; two available layers at boundaries",
            "fwhm_reference": "0.5 * smoothed_z_at_selected_onset",
            "necessity_null": "label permutation within environment strata",
            "type_null": "environment-to-category assignment permutation in blocks",
        },
        "clean_set_counts": {
            key: int(mask.sum()) for key, mask in sorted(clean.items())
        },
        "necessity": {
            "by_category": {
                category: curve.as_dict()
                for category, curve in necessity_curves.items()
            },
            "common_smoothed_z": necessity_common.tolist(),
            "onset": necessity_onset,
        },
        "type": {
            "by_category_state": {
                category: {
                    state: curve.as_dict() for state, curve in states.items()
                }
                for category, states in type_curves.items()
            },
            "category_smoothed_z": {
                category: values.tolist()
                for category, values in type_category_curves.items()
            },
            "overall_smoothed_z": type_overall.tolist(),
            "onset": type_onset,
        },
    }
    scalers = {
        "necessity_residual_mean": necessity_mean,
        "necessity_residual_std": necessity_std,
        "type_residual_mean": type_mean,
        "type_residual_std": type_std,
    }
    return result, scalers


def curve_rows(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    necessity = result["necessity"]
    type_result = result["type"]
    for index in range(len(necessity["common_smoothed_z"])):
        yield {
            "layer": index + 1,
            "necessity_A": necessity["by_category"]["A"]["smoothed_z"][index],
            "necessity_B": necessity["by_category"]["B"]["smoothed_z"][index],
            "necessity_C": necessity["by_category"]["C"]["smoothed_z"][index],
            "necessity_common": necessity["common_smoothed_z"][index],
            "type_A": type_result["category_smoothed_z"]["A"][index],
            "type_B": type_result["category_smoothed_z"]["B"][index],
            "type_C": type_result["category_smoothed_z"]["C"][index],
            "type_overall": type_result["overall_smoothed_z"][index],
        }
