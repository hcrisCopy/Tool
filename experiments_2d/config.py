"""Strict configuration loading with paths anchored at the repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required configuration key: {context}.{key}")
    return mapping[key]


def _resolve_repo_path(value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(
            f"{key} must be repository-relative, got absolute path: {value}"
        )
    return (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class ExperimentPaths:
    model: Path
    dataset: Path
    output_root: Path


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    head_dim: int
    torch_dtype: str
    transformers_version: str


@dataclass(frozen=True)
class GenerationSpec:
    seeds: tuple[int, ...]
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    max_rounds: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float


@dataclass(frozen=True)
class AnalysisSpec:
    seed: int
    n_label_shuffles_smoke: int
    n_label_shuffles_final: int
    n_onset_bootstraps: int
    n_selection_bootstraps: int
    n_metric_bootstraps: int
    onset_peak_fraction: float
    onset_smoothing_width: int
    max_onset_window: int
    stability_frequency: float
    sign_agreement: float
    necessity_top_fraction: float
    necessity_top_cap: int
    category_top_k: int
    category_margin: float
    necessity_penalty: float
    n_random_controls: int
    causal_permutations: int
    patch_alphas: tuple[float, ...]
    mask_fractions: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    paths: ExperimentPaths
    model: ModelSpec
    generation: GenerationSpec
    analysis: AnalysisSpec
    extraction_batch_size: int

    @property
    def run_root(self) -> Path:
        return self.paths.output_root / self.model.slug


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Top-level configuration must be a mapping")

    raw_paths = _require(raw, "paths", "root")
    raw_model = _require(raw, "model", "root")
    raw_generation = _require(raw, "generation", "root")
    raw_analysis = _require(raw, "analysis", "root")
    for name, value in (
        ("paths", raw_paths),
        ("model", raw_model),
        ("generation", raw_generation),
        ("analysis", raw_analysis),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a mapping")

    paths = ExperimentPaths(
        model=_resolve_repo_path(_require(raw_paths, "model", "paths"), "paths.model"),
        dataset=_resolve_repo_path(
            _require(raw_paths, "dataset", "paths"), "paths.dataset"
        ),
        output_root=_resolve_repo_path(
            _require(raw_paths, "output_root", "paths"), "paths.output_root"
        ),
    )
    model = ModelSpec(
        slug=str(_require(raw_model, "slug", "model")),
        architecture=str(_require(raw_model, "architecture", "model")),
        num_hidden_layers=int(_require(raw_model, "num_hidden_layers", "model")),
        hidden_size=int(_require(raw_model, "hidden_size", "model")),
        head_dim=int(_require(raw_model, "head_dim", "model")),
        torch_dtype=str(_require(raw_model, "torch_dtype", "model")),
        transformers_version=str(
            _require(raw_model, "transformers_version", "model")
        ),
    )
    seeds = tuple(int(seed) for seed in _require(raw_generation, "seeds", "generation"))
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("generation.seeds must be a non-empty list of unique integers")
    generation = GenerationSpec(
        seeds=seeds,
        temperature=float(_require(raw_generation, "temperature", "generation")),
        top_p=float(_require(raw_generation, "top_p", "generation")),
        top_k=int(_require(raw_generation, "top_k", "generation")),
        max_new_tokens=int(
            _require(raw_generation, "max_new_tokens", "generation")
        ),
        max_rounds=int(_require(raw_generation, "max_rounds", "generation")),
        max_model_len=int(
            _require(raw_generation, "max_model_len", "generation")
        ),
        tensor_parallel_size=int(
            _require(raw_generation, "tensor_parallel_size", "generation")
        ),
        gpu_memory_utilization=float(
            _require(raw_generation, "gpu_memory_utilization", "generation")
        ),
    )
    analysis = AnalysisSpec(
        seed=int(_require(raw_analysis, "seed", "analysis")),
        n_label_shuffles_smoke=int(
            _require(raw_analysis, "n_label_shuffles_smoke", "analysis")
        ),
        n_label_shuffles_final=int(
            _require(raw_analysis, "n_label_shuffles_final", "analysis")
        ),
        n_onset_bootstraps=int(
            _require(raw_analysis, "n_onset_bootstraps", "analysis")
        ),
        n_selection_bootstraps=int(
            _require(raw_analysis, "n_selection_bootstraps", "analysis")
        ),
        n_metric_bootstraps=int(
            _require(raw_analysis, "n_metric_bootstraps", "analysis")
        ),
        onset_peak_fraction=float(
            _require(raw_analysis, "onset_peak_fraction", "analysis")
        ),
        onset_smoothing_width=int(
            _require(raw_analysis, "onset_smoothing_width", "analysis")
        ),
        max_onset_window=int(
            _require(raw_analysis, "max_onset_window", "analysis")
        ),
        stability_frequency=float(
            _require(raw_analysis, "stability_frequency", "analysis")
        ),
        sign_agreement=float(_require(raw_analysis, "sign_agreement", "analysis")),
        necessity_top_fraction=float(
            _require(raw_analysis, "necessity_top_fraction", "analysis")
        ),
        necessity_top_cap=int(
            _require(raw_analysis, "necessity_top_cap", "analysis")
        ),
        category_top_k=int(_require(raw_analysis, "category_top_k", "analysis")),
        category_margin=float(
            _require(raw_analysis, "category_margin", "analysis")
        ),
        necessity_penalty=float(
            _require(raw_analysis, "necessity_penalty", "analysis")
        ),
        n_random_controls=int(
            _require(raw_analysis, "n_random_controls", "analysis")
        ),
        causal_permutations=int(
            _require(raw_analysis, "causal_permutations", "analysis")
        ),
        patch_alphas=tuple(
            float(value) for value in _require(raw_analysis, "patch_alphas", "analysis")
        ),
        mask_fractions=tuple(
            float(value) for value in _require(raw_analysis, "mask_fractions", "analysis")
        ),
    )

    batch_size = int(_require(raw, "extraction_batch_size", "root"))
    if batch_size < 1:
        raise ValueError("extraction_batch_size must be positive")
    if not 0.0 < analysis.onset_peak_fraction <= 1.0:
        raise ValueError("analysis.onset_peak_fraction must be in (0, 1]")
    if analysis.max_onset_window not in (1, 3, 5):
        raise ValueError("analysis.max_onset_window must be one of 1, 3, 5")
    if analysis.onset_smoothing_width != 3:
        raise ValueError("analysis.onset_smoothing_width must remain fixed at 3")
    if generation.tensor_parallel_size != 1:
        raise ValueError("The Qwen3-4B single-GPU protocol requires tensor_parallel_size=1")
    if not 0.0 < generation.gpu_memory_utilization < 1.0:
        raise ValueError("generation.gpu_memory_utilization must be in (0, 1)")

    return ExperimentConfig(
        config_path=config_path,
        paths=paths,
        model=model,
        generation=generation,
        analysis=analysis,
        extraction_batch_size=batch_size,
    )


def require_input_paths(config: ExperimentConfig) -> None:
    """Fail loudly if model or dataset inputs are absent."""

    if not config.paths.model.is_dir():
        raise FileNotFoundError(f"Model directory not found: {config.paths.model}")
    if not config.paths.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {config.paths.dataset}")

