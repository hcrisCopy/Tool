from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.evaluation_contract import classify_action_outcome
from when2tool_action.io_utils import canonical_json_sha256, sha256_file
from when2tool_action import scoped_relabel
from when2tool_action.scoped_relabel import (
    ALLOWED_ROW_MUTATIONS,
    RECEIPT_NAME,
    relabel_scoped_outputs,
)


PROTOCOL = "scoped_original_w2t"
SETTING = "current_no_reasoning_scoped"


def _label_artifact(
    task_ids: list[int], necessities: list[int], *, target: bool
) -> dict:
    rows = []
    for task_id, necessary in zip(task_ids, necessities):
        rows.append(
            {
                "id": task_id,
                "split": "test",
                "difficulty": "easy",
                "category": "A",
                "tool_necessary": necessary,
                "no_tool_correct": 1 - necessary,
                "gold_action": "A" if necessary else "NONE",
                "seed": 0,
                "prompt_mode": "hard_no_tool",
                "reasoning_mode": "no_reasoning",
                "tool_scope": "scoped",
                **({"label_protocol": PROTOCOL} if target else {}),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": "model-slug",
        "split": "test",
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "scoped",
        "n": len(rows),
        "rows": rows,
        **({"protocol_id": PROTOCOL} if target else {}),
    }


def _event() -> dict:
    return {
        "round": 1,
        "tool_name": "evaluate_expression",
        "arguments": {"expression": "2+2"},
        "recognized": True,
        "environment": "CalculatorEnv",
        "category": "A",
        "arguments_valid": True,
        "result_success": True,
        "result": {"success": True, "value": 4},
        "routed": True,
    }


def _row(
    task_id: int,
    *,
    run_id: str,
    seed: int,
    gold_action: str,
    necessary: int,
) -> dict:
    events = [_event()] if task_id == 0 else []
    categories = [event["category"] for event in events]
    pred_action = categories[0] if categories else "NONE"
    final_correct = task_id % 2 == 0
    error_type = classify_action_outcome(
        gold_action,
        pred_action,
        final_correct,
        False,
    )
    first = events[0] if events else None
    return {
        "id": task_id,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "seed": seed,
        "setting": SETTING,
        "tool_scope": "scoped",
        "category": "A",
        "difficulty": "easy",
        "gold_action": gold_action,
        "tool_necessary": necessary,
        "no_tool_correct": 1 - necessary,
        "routed_tool_events": events,
        "tool_calls": len(events),
        "total_tool_calls": len(events),
        "pred_action": pred_action,
        "tool_call_categories": categories,
        "unique_tool_call_categories": list(dict.fromkeys(categories)),
        "n_tool_call_categories": len(set(categories)),
        "mixed_category_calls": False,
        "invalid_tool_calls": 0,
        "first_tool_name": first["tool_name"] if first else None,
        "first_tool_category": first["category"] if first else None,
        "first_tool_environment": first["environment"] if first else None,
        "first_arguments_valid": bool(first and first["arguments_valid"]),
        "final_correct": final_correct,
        "error_type": error_type,
        "episode_done": True,
        "termination_reason": "boxed_answer",
        "tool_parse_failures": 0,
        "final_response": f"generated answer for task {task_id} seed {seed}",
        "generation_tokens": 7 + task_id,
        "prefill_tokens": 0,
    }


def _evaluation_artifact(
    task_ids: list[int],
    necessities: list[int],
    *,
    setting: str,
    labels_sha256: str,
    provenance_sha256: str,
) -> dict:
    runs = []
    for run_index, seed in enumerate((0, 1, 2)):
        run_id = f"run_{run_index}_seed_{seed}"
        rows = []
        for task_id, necessary in zip(task_ids, necessities):
            row = _row(
                task_id,
                run_id=run_id,
                seed=seed,
                gold_action="A" if necessary else "NONE",
                necessary=necessary,
            )
            row["setting"] = setting
            rows.append(row)
        runs.append({"run_id": run_id, "seed": seed, "setting": setting, "rows": rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            "model": "model-slug",
            "setting": setting,
            "tool_scope": "scoped",
            "seeds": [0, 1, 2],
            "task_ids_sha256": canonical_json_sha256(task_ids),
            "labels_sha256": labels_sha256,
            "runtime_provenance_sha256": provenance_sha256,
            "project_git_commit": "commit",
            "smoke": False,
        },
        "runs": runs,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: tuple[str, ...] = (SETTING,),
) -> tuple[SimpleNamespace, list[Path], Path, Path, Path, list[int], list[int]]:
    task_ids = [0, 1, 2, 3]
    source_necessities = [1, 0, 1, 0]
    target_necessities = [0, 1, 0, 1]
    source_labels = tmp_path / "source_labels.json"
    target_labels = tmp_path / "target_labels.json"
    provenance = tmp_path / "runtime_provenance.json"
    _write_json(
        source_labels,
        _label_artifact(task_ids, source_necessities, target=False),
    )
    _write_json(
        target_labels,
        _label_artifact(task_ids, target_necessities, target=True),
    )
    provenance.write_text("registered provenance\n", encoding="utf-8")
    provenance_sha = sha256_file(provenance)
    monkeypatch.setattr(
        scoped_relabel,
        "validate_runtime_provenance",
        lambda config: {
            "path": provenance,
            "sha256": provenance_sha,
            "git_commit": "commit",
        },
    )
    input_paths = []
    for setting in settings:
        source = tmp_path / "sources" / f"{setting}.json"
        _write_json(
            source,
            _evaluation_artifact(
                task_ids,
                source_necessities,
                setting=setting,
                labels_sha256=sha256_file(source_labels),
                provenance_sha256=provenance_sha,
            ),
        )
        input_paths.append(source)
    config = SimpleNamespace(model=SimpleNamespace(slug="model-slug"))
    output_dir = tmp_path / "derived"
    return (
        config,
        input_paths,
        source_labels,
        target_labels,
        output_dir,
        task_ids,
        source_necessities,
    )


def _derive(fixture, *, overwrite: bool = False, expected_settings=None):
    config, inputs, source_labels, target_labels, output_dir, task_ids, _ = fixture
    return relabel_scoped_outputs(
        config,
        input_paths=inputs,
        source_labels_path=source_labels,
        target_labels_path=target_labels,
        output_dir=output_dir,
        protocol_id=PROTOCOL,
        overwrite=overwrite,
        expected_settings=expected_settings or [SETTING],
        expected_task_count=len(task_ids),
    )


def test_relabel_changes_only_allowed_fields_and_recomputes_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _derive(fixture)
    _, inputs, source_labels, target_labels, output_dir, task_ids, _ = fixture
    source = json.loads(inputs[0].read_text(encoding="utf-8"))
    derived_path = output_dir / inputs[0].name
    derived = json.loads(derived_path.read_text(encoding="utf-8"))

    allowed = set(ALLOWED_ROW_MUTATIONS)
    for source_run, derived_run in zip(source["runs"], derived["runs"]):
        for source_row, derived_row in zip(source_run["rows"], derived_run["rows"]):
            assert set(source_row) == set(derived_row)
            assert {
                key: value for key, value in source_row.items() if key not in allowed
            } == {
                key: value for key, value in derived_row.items() if key not in allowed
            }
    first = derived["runs"][0]["rows"][0]
    second = derived["runs"][0]["rows"][1]
    assert first["gold_action"] == "NONE"
    assert first["error_type"] == "over_call"
    assert second["gold_action"] == "A"
    assert second["error_type"] == "under_call"
    assert first["tool_necessary"] == 0
    assert first["no_tool_correct"] == 1
    assert derived["config"]["labels_sha256"] == sha256_file(target_labels)
    assert derived["config"]["source_labels_sha256"] == sha256_file(source_labels)
    assert derived["derivation"]["protocol_id"] == PROTOCOL
    assert derived["derivation"]["n_task_ids"] == len(task_ids)

    stored_receipt = json.loads((output_dir / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert stored_receipt == receipt
    assert receipt["changed_task_count"] == len(task_ids)
    assert receipt["artifacts"][0]["output_sha256"] == sha256_file(derived_path)
    assert receipt["artifacts"][0]["immutable_rows_sha256"] == derived["derivation"][
        "immutable_rows_sha256"
    ]


def test_incomplete_seed_panel_and_bad_task_ids_fail_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    source = fixture[1][0]
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["runs"].pop()
    _write_json(source, payload)
    with pytest.raises(ValueError, match="expected 3 complete runs"):
        _derive(fixture)
    assert not fixture[4].exists()

    fixture = _fixture(tmp_path / "bad-id", monkeypatch)
    source = fixture[1][0]
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["runs"][1]["rows"][0]["id"] = 999
    _write_json(source, payload)
    with pytest.raises(ValueError, match="task ID order"):
        _derive(fixture)
    assert not fixture[4].exists()


def test_incomplete_labels_and_formal_default_count_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    target_labels = fixture[3]
    payload = json.loads(target_labels.read_text(encoding="utf-8"))
    payload["rows"].pop()
    payload["n"] -= 1
    _write_json(target_labels, payload)
    with pytest.raises(ValueError, match="expected 4 label rows"):
        _derive(fixture)
    assert not fixture[4].exists()

    fixture = _fixture(tmp_path / "formal-count", monkeypatch)
    config, inputs, source_labels, target_labels, output_dir, _, _ = fixture
    with pytest.raises(ValueError, match="expected 2250 label rows"):
        relabel_scoped_outputs(
            config,
            input_paths=inputs,
            source_labels_path=source_labels,
            target_labels_path=target_labels,
            output_dir=output_dir,
            protocol_id=PROTOCOL,
            expected_settings=[SETTING],
        )
    assert not output_dir.exists()


def test_later_bad_input_is_transactional_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = (SETTING, "necessary_tool_no_reasoning_scoped")
    fixture = _fixture(tmp_path, monkeypatch, settings=settings)
    bad = fixture[1][1]
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["runs"][2]["rows"].pop()
    _write_json(bad, payload)
    with pytest.raises(ValueError, match="task ID order/count"):
        _derive(fixture, expected_settings=settings)
    assert not fixture[4].exists()


def test_default_overwrite_refusal_preserves_published_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _derive(fixture)
    output_path = fixture[4] / fixture[1][0].name
    receipt_path = fixture[4] / RECEIPT_NAME
    output_before = output_path.read_bytes()
    receipt_before = receipt_path.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _derive(fixture)
    assert output_path.read_bytes() == output_before
    assert receipt_path.read_bytes() == receipt_before
