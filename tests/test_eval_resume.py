from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.eval_resume import (
    initialize_evaluation_artifacts,
    prepare_evaluation_artifact,
    validate_evaluation_artifact_for_resume,
)
from when2tool_action.io_utils import atomic_write_json, canonical_json_sha256


TASK_IDS = [101, 102]


def _template() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            "model": "qwen3-4b-instruct-2507",
            "config_sha256": "c" * 64,
            "data_sha256": "d" * 64,
            "labels_sha256": "e" * 64,
            "runtime_provenance_sha256": "r" * 64,
            "project_git_commit": "commit",
            "setting": "current_no_reasoning_fulltools",
            "tool_scope": "full",
            "prompt_mode": "current",
            "reasoning_mode": "no_reasoning",
            "record_mode": "lite",
            "seeds": [0, 1, 2],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_new_tokens": 2048,
            "max_rounds": 12,
            "max_model_len": 32768,
            "full_menu_sha256": "a" * 64,
            "task_ids_sha256": canonical_json_sha256(TASK_IDS),
            "smoke": False,
        },
        "runs": [],
    }


def _run(index: int, seed: int) -> dict:
    run_id = f"run_{index}_seed_{seed}"
    setting = _template()["config"]["setting"]
    return {
        "run_id": run_id,
        "seed": seed,
        "setting": setting,
        "rows": [
            {
                "id": task_id,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "seed": seed,
                "setting": setting,
                "tool_scope": "full",
                "gold_action": "NONE",
                "routed_tool_events": [],
                "tool_calls": 0,
                "final_correct": True,
                "pred_action": "NONE",
                "termination_reason": "boxed_answer",
                "tool_parse_failures": 0,
            }
            for task_id in TASK_IDS
        ],
    }


def _artifact(run_count: int = 1) -> dict:
    artifact = _template()
    artifact["runs"] = [_run(index, index) for index in range(run_count)]
    return artifact


def test_valid_partial_and_complete_prefixes() -> None:
    assert (
        validate_evaluation_artifact_for_resume(
            _artifact(1), template=_template(), task_ids=TASK_IDS
        )
        == 1
    )
    assert (
        validate_evaluation_artifact_for_resume(
            _artifact(3), template=_template(), task_ids=TASK_IDS
        )
        == 3
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model", "other-model"),
        ("config_sha256", "0" * 64),
        ("data_sha256", "1" * 64),
        ("labels_sha256", "2" * 64),
        ("runtime_provenance_sha256", "3" * 64),
        ("project_git_commit", "other-commit"),
        ("setting", "other-setting"),
        ("tool_scope", "scoped"),
        ("prompt_mode", "force_tool"),
        ("reasoning_mode", "reasoning"),
        ("record_mode", "full"),
        ("seeds", [0, 2, 1]),
        ("temperature", 0.8),
        ("top_p", 0.9),
        ("max_rounds", 11),
        ("task_ids_sha256", "b" * 64),
    ],
)
def test_rejects_any_config_mismatch(field: str, bad_value: object) -> None:
    artifact = _artifact()
    artifact["config"][field] = bad_value
    with pytest.raises(ValueError, match=f"config\\.{field}"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )


def test_rejects_schema_upstream_and_extra_keys() -> None:
    artifact = _artifact()
    artifact["schema_version"] = "old"
    with pytest.raises(ValueError, match="schema_version"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )

    artifact = _artifact()
    artifact["upstream_commit"] = "wrong"
    with pytest.raises(ValueError, match="upstream_commit"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )

    artifact = _artifact()
    artifact["config"]["unregistered"] = True
    with pytest.raises(ValueError, match="config keys"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )


def test_rejects_nonprefix_runs_and_duplicate_runs() -> None:
    artifact = _template()
    artifact["runs"] = [_run(1, 1)]
    with pytest.raises(ValueError, match=r"runs\[0\]\.run_id"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )

    artifact = _artifact(2)
    artifact["runs"][1] = deepcopy(artifact["runs"][0])
    with pytest.raises(ValueError, match=r"runs\[1\]\.run_id"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )


def test_rejects_missing_reordered_or_duplicate_task_ids() -> None:
    artifact = _artifact()
    artifact["runs"][0]["rows"].reverse()
    with pytest.raises(ValueError, match="task ID order"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )

    artifact = _artifact()
    artifact["runs"][0]["rows"][1]["id"] = TASK_IDS[0]
    with pytest.raises(ValueError, match="duplicate task IDs"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )

    artifact = _artifact()
    artifact["runs"][0]["rows"].pop()
    with pytest.raises(ValueError, match="task ID order"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )


@pytest.mark.parametrize(
    ("row_field", "bad_value"),
    [
        ("schema_version", "old"),
        ("run_id", "wrong"),
        ("seed", 2),
        ("setting", "wrong"),
        ("tool_scope", "scoped"),
    ],
)
def test_rejects_row_identity_mismatch(row_field: str, bad_value: object) -> None:
    artifact = _artifact()
    artifact["runs"][0]["rows"][0][row_field] = bad_value
    with pytest.raises(ValueError, match=row_field):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(gold_action="INVALID"), "gold_action"),
        (lambda row: row.update(tool_calls=1), "len\\(routed_tool_events\\)"),
        (lambda row: row.update(final_correct=1), "final_correct"),
        (lambda row: row.update(pred_action="A"), "pred_action"),
        (lambda row: row.update(termination_reason="unknown"), "termination_reason"),
        (lambda row: row.update(tool_parse_failures=-1), "tool_parse_failures"),
    ],
)
def test_rejects_invalid_statistics_row_contract(mutation, message: str) -> None:
    artifact = _artifact()
    mutation(artifact["runs"][0]["rows"][0])
    with pytest.raises((TypeError, ValueError), match=message):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )


def test_rejects_unknown_routed_category_and_missing_result() -> None:
    artifact = _artifact()
    row = artifact["runs"][0]["rows"][0]
    row.update(
        routed_tool_events=[{"category": "unknown", "result": {}}],
        tool_calls=1,
        pred_action="INVALID",
    )
    with pytest.raises(ValueError, match="category must be one of"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )

    row["routed_tool_events"] = [{"category": "A"}]
    row["pred_action"] = "A"
    with pytest.raises(ValueError, match="missing 'result'"):
        validate_evaluation_artifact_for_resume(
            artifact, template=_template(), task_ids=TASK_IDS
        )


def test_expected_probe_fields_are_exactly_validated() -> None:
    artifact = _artifact()
    expected = {
        task_id: {
            "gold_action": "NONE",
            "probe_probability": 0.2,
            "probe_decision": "no_tool",
        }
        for task_id in TASK_IDS
    }
    for row in artifact["runs"][0]["rows"]:
        row.update(expected[row["id"]])
    assert (
        validate_evaluation_artifact_for_resume(
            artifact,
            template=_template(),
            task_ids=TASK_IDS,
            expected_row_fields=expected,
        )
        == 1
    )
    artifact["runs"][0]["rows"][0]["probe_probability"] = 0.3
    with pytest.raises(ValueError, match="probe_probability"):
        validate_evaluation_artifact_for_resume(
            artifact,
            template=_template(),
            task_ids=TASK_IDS,
            expected_row_fields=expected,
        )


def test_prepare_requires_explicit_policy_for_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "eval.json"
    atomic_write_json(target, _artifact(), overwrite=False)

    with pytest.raises(FileExistsError, match="--resume"):
        prepare_evaluation_artifact(
            target,
            template=_template(),
            task_ids=TASK_IDS,
            overwrite=False,
            resume=False,
        )

    prepared = prepare_evaluation_artifact(
        target,
        template=_template(),
        task_ids=TASK_IDS,
        overwrite=False,
        resume=True,
    )
    assert prepared.completed_runs == 1
    assert prepared.artifact["runs"][0]["seed"] == 0

    restarted = prepare_evaluation_artifact(
        target,
        template=_template(),
        task_ids=TASK_IDS,
        overwrite=True,
        resume=False,
    )
    assert restarted.completed_runs == 0
    assert restarted.artifact == _template()
    assert json.loads(target.read_text(encoding="utf-8"))["runs"]
    initialize_evaluation_artifacts([restarted])
    assert json.loads(target.read_text(encoding="utf-8"))["runs"] == []


def test_prepare_rejects_resume_and_overwrite_together(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_evaluation_artifact(
            tmp_path / "eval.json",
            template=_template(),
            task_ids=TASK_IDS,
            overwrite=True,
            resume=True,
        )


def test_prepare_new_target_starts_from_seed_zero(tmp_path: Path) -> None:
    prepared = prepare_evaluation_artifact(
        tmp_path / "eval.json",
        template=_template(),
        task_ids=TASK_IDS,
        overwrite=False,
        resume=True,
    )
    assert prepared.completed_runs == 0
    assert prepared.artifact == _template()
    assert not (tmp_path / "eval.json").exists()
    initialize_evaluation_artifacts([prepared])
    assert json.loads((tmp_path / "eval.json").read_text(encoding="utf-8")) == _template()


def test_later_preflight_failure_leaves_earlier_overwrite_target_untouched(
    tmp_path: Path,
) -> None:
    first_target = tmp_path / "first.json"
    second_target = tmp_path / "second.json"
    atomic_write_json(first_target, _artifact(3), overwrite=False)
    malformed = _artifact(1)
    malformed["config"]["model"] = "wrong"
    atomic_write_json(second_target, malformed, overwrite=False)

    first = prepare_evaluation_artifact(
        first_target,
        template=_template(),
        task_ids=TASK_IDS,
        overwrite=True,
        resume=False,
    )
    with pytest.raises(ValueError, match="config.model"):
        prepare_evaluation_artifact(
            second_target,
            template=_template(),
            task_ids=TASK_IDS,
            overwrite=False,
            resume=True,
        )
    assert json.loads(first_target.read_text(encoding="utf-8"))["runs"] == _artifact(3)[
        "runs"
    ]
    assert first.needs_initialization is True


def test_each_target_keeps_its_own_resume_position(tmp_path: Path) -> None:
    complete_target = tmp_path / "complete.json"
    partial_target = tmp_path / "partial.json"
    atomic_write_json(complete_target, _artifact(3), overwrite=False)
    atomic_write_json(partial_target, _artifact(1), overwrite=False)

    complete = prepare_evaluation_artifact(
        complete_target,
        template=_template(),
        task_ids=TASK_IDS,
        overwrite=False,
        resume=True,
    )
    partial = prepare_evaluation_artifact(
        partial_target,
        template=_template(),
        task_ids=TASK_IDS,
        overwrite=False,
        resume=True,
    )

    assert complete.completed_runs == 3
    assert partial.completed_runs == 1
