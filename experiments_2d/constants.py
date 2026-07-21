"""Experiment constants shared by data, probing, and evaluation code."""

from __future__ import annotations

from typing import Final


CATEGORY_TO_ENVS: Final[dict[str, tuple[str, ...]]] = {
    "A": (
        "CalculatorEnv",
        "StatisticsEnv",
        "CountingEnv",
        "MatrixEnv",
        "PrimeEnv",
    ),
    "B": (
        "RetrieverEnv",
        "HistoricalYearEnv",
        "GameRuleEnv",
        "HashEnv",
        "DecodingEnv",
    ),
    "C": (
        "ListManipulationEnv",
        "DateTimeEnv",
        "CodeExecutorEnv",
        "ScheduleEnv",
        "RegexMatchEnv",
    ),
}

ENV_TO_CATEGORY: Final[dict[str, str]] = {
    env: category
    for category, environments in CATEGORY_TO_ENVS.items()
    for env in environments
}

CATEGORY_NAMES: Final[dict[str, str]] = {
    "A": "computational-scale",
    "B": "knowledge-boundary",
    "C": "reliable-execution",
}

DIFFICULTIES: Final[tuple[str, ...]] = ("easy", "medium", "hard")
PROMPT_VARIANTS: Final[tuple[str, ...]] = ("P_env", "P_all", "P_no_schema")

# Expected cardinalities from the official single-hop benchmark release.
EXPECTED_SPLIT_SIZES: Final[dict[str, int]] = {"train": 900, "test": 2250}
EXPECTED_PER_ENV_DIFFICULTY: Final[dict[str, int]] = {"train": 20, "test": 50}

