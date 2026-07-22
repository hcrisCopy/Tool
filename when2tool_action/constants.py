"""Frozen benchmark and action-taxonomy constants."""

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
ACTIONS: Final[tuple[str, ...]] = ("NONE", "A", "B", "C")
DIFFICULTIES: Final[tuple[str, ...]] = ("easy", "medium", "hard")
EXPECTED_SPLIT_SIZES: Final[dict[str, int]] = {"train": 900, "test": 2250}
EXPECTED_PER_ENV_DIFFICULTY: Final[dict[str, int]] = {"train": 20, "test": 50}
EXPECTED_ENV_COUNT: Final[int] = 15
EXPECTED_TOOL_COUNT: Final[int] = 33
UPSTREAM_COMMIT: Final[str] = "66f100089d1f3f7e7f2acee279c4dbf6e7ae5e2c"
SCHEMA_VERSION: Final[str] = "when2tool-action-v1"

# This is a derived functional action taxonomy.  The When2Tool paper used these
# three groups for necessity analysis; it did not evaluate full-menu routing.
TAXONOMY_NOTE: Final[str] = (
    "Derived functional action categories adapted from When2Tool's three "
    "tool-necessity groups; not an upstream routing label."
)
