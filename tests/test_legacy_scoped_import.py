from __future__ import annotations

import hashlib

import pytest
import torch

from when2tool_action.constants import UPSTREAM_COMMIT
from when2tool_action.legacy_scoped import (
    LEGACY_LABEL_POLICY,
    PROTOCOL_ID,
    _load_and_validate_hidden,
    _validate_legacy_label_rows,
)


def _fixture() -> tuple[list[dict], list[dict], list[dict], dict]:
    task = {
        "id": 101,
        "difficulty": "easy",
        "instruction": "What is 2 + 2?",
        "expected": {"answer": "4"},
        "gold_env_name": "CalculatorEnv",
        "gold_tools": ["evaluate_expression"],
    }
    prompt = (
        "SYSTEM evaluate_expression\nWhat is 2 + 2?\n"
        f"{LEGACY_LABEL_POLICY}\n"
    )
    row = {
        "id": 101,
        "difficulty": "easy",
        "env": "CalculatorEnv",
        "category": "A",
        "tool_type": "A",
        "seed": 0,
        "prompt_variant": "P_env",
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rounds": 1,
        "completed": True,
        "final_response": "\\boxed{4}",
        "boxed_answer": "4",
        "cleaned_answer": "4",
        "gold_answer": "4",
        "no_tool_correct": 1,
        "tool_necessary": 0,
        "tool_calls": 0,
        "generation_tokens": 3,
        "prefill_tokens": 10,
        "reasoning_mode": "no_reasoning",
        "upstream_commit": UPSTREAM_COMMIT,
        "enable_thinking": False,
        "trace": [{"round": 1, "prompt_text": prompt}],
    }
    baseline = {
        "id": 101,
        "difficulty": "easy",
        "env": "CalculatorEnv",
        "category": "A",
        "no_tool_correct": 1,
        "tool_necessary": 0,
        "first_sentence": "",
    }
    rendered = {101: (prompt, [1, 2, 3])}
    return [row], [baseline], [task], rendered


def test_legacy_label_fixture_is_converted_without_claiming_adaptation() -> None:
    rows, baseline, tasks, rendered = _fixture()
    converted = _validate_legacy_label_rows(
        rows,
        baseline,
        tasks,
        rendered,
        split="test",
        seed=0,
    )
    assert converted[0]["gold_action"] == "NONE"
    assert converted[0]["label_protocol"] == PROTOCOL_ID
    assert converted[0]["tool_scope"] == "scoped"


def test_legacy_label_fixture_rejects_prompt_hash_mismatch() -> None:
    rows, baseline, tasks, rendered = _fixture()
    rows[0]["prompt_hash"] = "0" * 64
    with pytest.raises(ValueError, match="trace hash"):
        _validate_legacy_label_rows(
            rows,
            baseline,
            tasks,
            rendered,
            split="test",
            seed=0,
        )


def test_hidden_fixture_requires_float32_shape_and_finite_values(tmp_path) -> None:
    path = tmp_path / "hidden.pt"
    torch.save(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), path)
    hidden = _load_and_validate_hidden(path, expected_shape=(2, 3, 4))
    assert tuple(hidden.shape) == (2, 3, 4)

    torch.save(torch.full((2, 3, 4), float("nan"), dtype=torch.float32), path)
    with pytest.raises(ValueError, match="NaN or infinity"):
        _load_and_validate_hidden(path, expected_shape=(2, 3, 4))
