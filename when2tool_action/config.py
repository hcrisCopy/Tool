"""Strict project-relative experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing configuration key {context}.{key}")
    return mapping[key]


def _resolve_relative(value: str, key: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{key} must be repository-relative, got {value}")
    return (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class Paths:
    model: Path
    dataset: Path
    output_root: Path


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    architecture: str
    num_hidden_layers: int
    hidden_size: int


@dataclass(frozen=True)
class GenerationSpec:
    seeds: tuple[int, ...]
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int
    max_rounds: int
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float


@dataclass(frozen=True)
class ProbeSpec:
    c: float
    temperature: float
    thresholds: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    source: Path
    paths: Paths
    model: ModelSpec
    generation: GenerationSpec
    probe: ProbeSpec
    extraction_batch_size: int
    statistics_seed: int
    bootstrap_replicates: int

    @property
    def run_root(self) -> Path:
        return self.paths.output_root / self.model.slug


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.is_absolute():
        source = (REPO_ROOT / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping")
    p = _require(raw, "paths", "root")
    m = _require(raw, "model", "root")
    g = _require(raw, "generation", "root")
    q = _require(raw, "probe", "root")
    s = _require(raw, "statistics", "root")
    for name, value in (("paths", p), ("model", m), ("generation", g), ("probe", q), ("statistics", s)):
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a mapping")
    seeds = tuple(int(value) for value in _require(g, "seeds", "generation"))
    if seeds != (0, 1, 2):
        raise ValueError("The registered statistics protocol requires seeds [0, 1, 2]")
    thresholds = tuple(float(value) for value in _require(q, "thresholds", "probe"))
    if thresholds != (0.1, 0.3, 0.5, 0.7, 0.9):
        raise ValueError("Probe thresholds must be exactly [0.1,0.3,0.5,0.7,0.9]")
    config = ExperimentConfig(
        source=source,
        paths=Paths(
            model=_resolve_relative(str(_require(p, "model", "paths")), "paths.model"),
            dataset=_resolve_relative(str(_require(p, "dataset", "paths")), "paths.dataset"),
            output_root=_resolve_relative(str(_require(p, "output_root", "paths")), "paths.output_root"),
        ),
        model=ModelSpec(
            slug=str(_require(m, "slug", "model")),
            architecture=str(_require(m, "architecture", "model")),
            num_hidden_layers=int(_require(m, "num_hidden_layers", "model")),
            hidden_size=int(_require(m, "hidden_size", "model")),
        ),
        generation=GenerationSpec(
            seeds=seeds,
            temperature=float(_require(g, "temperature", "generation")),
            top_p=float(_require(g, "top_p", "generation")),
            top_k=int(_require(g, "top_k", "generation")),
            repetition_penalty=float(_require(g, "repetition_penalty", "generation")),
            max_new_tokens=int(_require(g, "max_new_tokens", "generation")),
            max_rounds=int(_require(g, "max_rounds", "generation")),
            max_model_len=int(_require(g, "max_model_len", "generation")),
            tensor_parallel_size=int(_require(g, "tensor_parallel_size", "generation")),
            gpu_memory_utilization=float(_require(g, "gpu_memory_utilization", "generation")),
        ),
        probe=ProbeSpec(
            c=float(_require(q, "c", "probe")),
            temperature=float(_require(q, "temperature", "probe")),
            thresholds=thresholds,
        ),
        extraction_batch_size=int(_require(raw, "extraction_batch_size", "root")),
        statistics_seed=int(_require(s, "seed", "statistics")),
        bootstrap_replicates=int(_require(s, "bootstrap_replicates", "statistics")),
    )
    if config.generation.tensor_parallel_size != 1:
        raise ValueError("This single-GPU protocol requires tensor_parallel_size=1")
    if not 0 < config.generation.gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be in (0,1)")
    if config.extraction_batch_size < 1:
        raise ValueError("extraction_batch_size must be positive")
    return config


def require_inputs(config: ExperimentConfig) -> None:
    if not config.paths.model.is_dir():
        raise FileNotFoundError(config.paths.model)
    if not config.paths.dataset.is_dir():
        raise FileNotFoundError(config.paths.dataset)
