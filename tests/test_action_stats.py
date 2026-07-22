import json
import math
from pathlib import Path

import pandas as pd
import pytest

from when2tool_action.constants import SCHEMA_VERSION as ACTION_SCHEMA_VERSION
from when2tool_action.constants import UPSTREAM_COMMIT
from when2tool_action.io_utils import canonical_json_sha256, sha256_file
from when2tool_action import stats as stats_module
from when2tool_action.stats import (
    BOOTSTRAP_METRICS,
    build_current_relative_tradeoff,
    build_gold_action_final_accuracy,
    build_multicall_tables,
    build_needed_category_analysis,
    build_none_analysis,
    build_run_diagnostics,
    collect_action_statistics,
    compute_per_run_metrics,
    load_evaluation_outputs,
    paired_bootstrap_comparisons,
    validate_expected_seed_panel,
)


def _row(
    task_id: int,
    run_id: str,
    seed: int,
    setting: str,
    gold: str,
    categories: list[str],
    correct: bool,
) -> dict:
    return {
        "id": task_id,
        "run_id": run_id,
        "seed": seed,
        "setting": setting,
        "difficulty": "easy",
        "category": gold if gold in {"A", "B", "C"} else "A",
        "gold_action": gold,
        "routed_tool_events": [
            {
                "category": category,
                "result": {"success": category != "unknown"},
            }
            for category in categories
        ],
        "tool_calls": len(categories),
        "final_correct": correct,
        "termination_reason": "boxed_answer",
        "tool_parse_failures": 0,
    }


def _payload(setting: str, outcomes: list[tuple[str, list[str], bool]]) -> dict:
    runs = []
    for seed in (0, 1):
        run_id = f"run-{seed}"
        rows = [
            _row(index, run_id, seed, setting, gold, categories, correct)
            for index, (gold, categories, correct) in enumerate(outcomes)
        ]
        runs.append({"run_id": run_id, "seed": seed, "setting": setting, "rows": rows})
    return {"schema_version": "test", "runs": runs}


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _label_artifact(
    split: str, entries: list[tuple[int, str, int]]
) -> dict:
    rows = [
        {
            "id": task_id,
            "split": split,
            "difficulty": "easy",
            "category": category,
            "tool_necessary": necessary,
            "no_tool_correct": 1 - necessary,
            "gold_action": category if necessary else "NONE",
            "seed": 0,
            "prompt_mode": "hard_no_tool",
            "reasoning_mode": "no_reasoning",
            "tool_scope": "full",
        }
        for task_id, category, necessary in entries
    ]
    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": "model-slug",
        "split": split,
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "full",
        "n": len(rows),
        "rows": rows,
    }


def _formal_payload(
    setting: str,
    outcomes: list[tuple[str, list[str], bool]],
    *,
    labels_sha256: str,
    seeds: tuple[int, ...] = (0, 1),
    tool_scope: str = "full",
    data_sha256: str = "b" * 64,
    runtime_provenance_sha256: str = "c" * 64,
) -> dict:
    task_ids = list(range(len(outcomes)))
    runs = []
    for run_index, seed in enumerate(seeds):
        run_id = f"run_{run_index}_seed_{seed}"
        rows = []
        for task_id, (gold, categories, correct) in enumerate(outcomes):
            row = _row(task_id, run_id, seed, setting, gold, categories, correct)
            row.update(
                {
                    "schema_version": ACTION_SCHEMA_VERSION,
                    "tool_scope": tool_scope,
                }
            )
            rows.append(row)
        runs.append(
            {"run_id": run_id, "seed": seed, "setting": setting, "rows": rows}
        )
    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            "model": "model-slug",
            "config_sha256": "a" * 64,
            "data_sha256": data_sha256,
            "labels_sha256": labels_sha256,
            "runtime_provenance_sha256": runtime_provenance_sha256,
            "project_git_commit": "project-commit",
            "setting": setting,
            "tool_scope": tool_scope,
            "seeds": list(seeds),
            "full_menu_sha256": "d" * 64,
            "task_ids_sha256": canonical_json_sha256(task_ids),
            "smoke": False,
        },
        "runs": runs,
    }


def _write_reference_inputs(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "tasks.json"
    provenance = tmp_path / "runtime_provenance.json"
    data.write_text('{"tasks": "test"}\n', encoding="utf-8")
    provenance.write_text(
        json.dumps(
            {
                "manifest_type": "runtime-and-input-provenance",
                "git": {"commit": "project-commit"},
            }
        ),
        encoding="utf-8",
    )
    return data, provenance


def _write_formal_inputs(
    tmp_path: Path,
    *,
    settings: tuple[str, ...] = ("current",),
    seeds: tuple[int, ...] = (0, 1),
) -> tuple[list[Path], Path, Path, Path, Path]:
    data, provenance = _write_reference_inputs(tmp_path)
    train_labels = tmp_path / "train_labels.json"
    test_labels = tmp_path / "test_labels.json"
    entries = [(0, "A", 0), (1, "A", 1), (2, "B", 1), (3, "C", 1)]
    train_entries = [(100 + task_id, category, necessary) for task_id, category, necessary in entries]
    train_labels.write_text(
        json.dumps(_label_artifact("train", train_entries)), encoding="utf-8"
    )
    test_labels.write_text(
        json.dumps(_label_artifact("test", entries)), encoding="utf-8"
    )
    outcomes = [
        ("NONE", [], True),
        ("A", ["A"], True),
        ("B", ["B"], True),
        ("C", ["C"], True),
    ]
    outputs: list[Path] = []
    for setting in settings:
        path = tmp_path / f"{setting}.json"
        _write(
            path,
            _formal_payload(
                setting,
                outcomes,
                labels_sha256=sha256_file(test_labels),
                seeds=seeds,
                data_sha256=sha256_file(data),
                runtime_provenance_sha256=sha256_file(provenance),
            ),
        )
        outputs.append(path)
    return outputs, train_labels, test_labels, data, provenance


def test_derives_actions_sequences_hierarchy_and_metrics(tmp_path: Path) -> None:
    path = tmp_path / "outputs.json"
    _write(
        path,
        _payload(
            "current",
            [
                ("NONE", [], True),
                ("A", ["A"], True),
                ("B", [], False),
                ("C", ["A", "C"], False),
                ("NONE", ["unknown"], False),
            ],
        ),
    )
    frame = load_evaluation_outputs([path])

    first_run = frame[frame["run_id"] == "run-0"].sort_values("id")
    assert first_run["pred_action"].tolist() == ["NONE", "A", "NONE", "A", "INVALID"]
    assert first_run["category_sequence_text"].tolist() == [
        "NONE",
        "A",
        "NONE",
        "A>C",
        "INVALID",
    ]
    assert first_run["mixed_calls"].tolist() == [False, False, False, True, False]
    assert first_run["mixed_category_calls"].tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert first_run["n_tool_call_categories"].tolist() == [0, 1, 0, 2, 1]
    assert first_run["outcome"].tolist() == [
        "success",
        "success",
        "under_call",
        "wrong_category",
        "invalid_tool",
    ]

    metrics = compute_per_run_metrics(frame)
    row = metrics.iloc[0]
    assert row["final_accuracy"] == pytest.approx(0.4)
    assert row["total_tool_calls"] == 4
    assert row["avg_tool_calls"] == pytest.approx(0.8)
    assert row["tool_call_rate"] == pytest.approx(0.8)
    assert row["action_accuracy"] == pytest.approx(0.4)
    assert row["no_call_precision"] == pytest.approx(0.5)
    assert row["no_call_recall"] == pytest.approx(0.5)
    assert row["category_accuracy_needed_given_call"] == pytest.approx(0.5)
    assert row["multi_call_rate"] == pytest.approx(0.2)
    assert row["mixed_over_multicall_rate"] == pytest.approx(1.0)
    assert row["recall_NONE"] == pytest.approx(0.5)
    assert row["recall_A"] == pytest.approx(1.0)
    assert row["recall_B"] == pytest.approx(0.0)
    assert row["recall_C"] == pytest.approx(0.0)
    assert row["majority_accuracy_baseline"] == pytest.approx(0.4)
    assert row["prior_matched_expected_accuracy"] == pytest.approx(0.28)


@pytest.mark.parametrize(
    ("gold", "categories", "correct", "expected"),
    [
        ("NONE", [], False, "direct_answer_wrong"),
        ("NONE", ["A"], True, "over_call"),
        ("A", [], True, "under_call"),
        ("A", ["B"], True, "wrong_category"),
        ("A", ["A"], False, "correct_category_wrong_answer"),
        ("A", ["A", "not-a-category"], True, "invalid_tool"),
    ],
)
def test_outcome_hierarchy(
    tmp_path: Path,
    gold: str,
    categories: list[str],
    correct: bool,
    expected: str,
) -> None:
    path = tmp_path / f"{expected}.json"
    _write(path, _payload("current", [(gold, categories, correct)]))
    frame = load_evaluation_outputs([path])
    assert set(frame["outcome"]) == {expected}


def test_plan_required_category_none_and_multicall_tables(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.json"
    _write(
        path,
        _payload(
            "current",
            [
                ("NONE", [], True),
                ("NONE", [], False),
                ("NONE", ["A"], False),
                ("NONE", ["B"], False),
                ("NONE", ["C"], False),
                ("NONE", ["unknown"], False),
                ("A", ["A"], True),
                ("A", ["A"], False),
                ("A", [], False),
                ("A", ["B"], False),
                ("B", ["B", "C"], True),
                ("C", ["C", "A"], False),
                ("C", ["C", "C"], True),
                ("C", ["C", "C"], True),
            ],
        ),
    )
    frame = load_evaluation_outputs([path])
    run = frame[frame["run_id"] == "run-0"]

    metrics = compute_per_run_metrics(frame)
    metric = metrics[metrics["run_id"] == "run-0"].iloc[0]
    assert metric["no_call_precision"] == pytest.approx(2 / 3)
    assert metric["no_call_recall"] == pytest.approx(2 / 6)
    assert metric["category_accuracy_needed_given_call"] == pytest.approx(6 / 7)
    assert metric["tool_call_rate"] == pytest.approx(15 / 14)
    assert metric["multi_call_rate"] == pytest.approx(4 / 14)
    assert metric["mixed_over_multicall_rate"] == pytest.approx(0.5)

    needed = build_needed_category_analysis(run)
    action_a = needed[needed["gold_action"] == "A"].iloc[0]
    assert action_a["n_tasks"] == 4
    assert action_a["call_correct_category_count"] == 2
    assert action_a["under_call_count"] == 1
    assert action_a["wrong_category_count"] == 1
    assert action_a["correct_category_wrong_answer_count"] == 1
    assert action_a["needed_final_success_count"] == 1
    assert action_a["call_correct_category_rate"] == pytest.approx(0.5)

    none = build_none_analysis(run).iloc[0]
    assert none["n_tasks"] == 6
    assert none["no_call_correct_count"] == 1
    assert none["direct_answer_wrong_count"] == 1
    for category in ("A", "B", "C", "INVALID"):
        assert none[f"overcall_{category}_count"] == 1
    assert none["overcall_total_count"] == 4
    assert none["overcall_total_rate"] == pytest.approx(4 / 6)
    assert none["any_invalid_tool_count"] == 1

    multicall, multicall_by_gold = build_multicall_tables(run)
    summary = multicall.iloc[0]
    assert summary["multi_call_count"] == 4
    assert summary["mixed_multicall_count"] == 2
    assert summary["any_tool_call_count"] == 11
    assert summary["mixed_over_all_rate"] == pytest.approx(2 / 14)
    assert summary["mixed_over_multicall_rate"] == pytest.approx(0.5)
    assert summary["mixed_category_task_count"] == 2
    assert summary["top_1_category_sequence"] == "B->C"
    assert summary["top_1_sequence_count"] == 1
    top_sequences = json.loads(summary["top_category_sequences_json"])
    assert top_sequences[0] == {
        "rank": 1,
        "category_sequence": "B->C",
        "count": 1,
        "rate_within_multicall": 0.25,
        "rate_within_mixed": 0.5,
    }
    gold_b = multicall_by_gold[multicall_by_gold["gold_action"] == "B"].iloc[0]
    assert gold_b["multi_call_count"] == 1
    assert gold_b["top_1_category_sequence"] == "B->C"


def test_invalid_call_counts_in_category_accuracy_and_mixed_is_valid_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid_and_mixed.json"
    _write(
        path,
        _payload(
            "current",
            [
                ("A", ["A"], True),
                ("A", ["unknown"], False),
                ("A", ["A", "unknown"], False),
                ("B", ["A", "B"], True),
                ("C", ["unknown", "unknown"], False),
                ("A", ["A", "A"], True),
            ],
        ),
    )
    frame = load_evaluation_outputs([path])
    metric = compute_per_run_metrics(frame).iloc[0]
    assert metric["category_accuracy_needed_given_call_n"] == 6
    assert metric["category_accuracy_needed_given_call"] == pytest.approx(3 / 6)
    assert metric["category_accuracy_needed_given_valid_call_n"] == 4
    assert metric["category_accuracy_needed_given_valid_call"] == pytest.approx(3 / 4)

    run = frame[frame["run_id"] == "run-0"].sort_values("id")
    assert run["mixed_calls"].tolist() == [False, False, False, True, False, False]
    assert run["mixed_calls_including_invalid"].tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    multicall, _ = build_multicall_tables(run)
    summary = multicall.iloc[0]
    assert summary["mixed_multicall_count"] == 1
    assert summary["mixed_multicall_including_invalid_count"] == 2
    assert summary["top_1_category_sequence"] == "A->B"


def test_core_summary_uses_three_seed_population_sd(tmp_path: Path) -> None:
    runs = []
    calls_by_seed = (([], []), (["A"], []), (["A"], ["B"]))
    for seed, calls in enumerate(calls_by_seed):
        run_id = f"run-{seed}"
        rows = [
            _row(task_id, run_id, seed, "current", "NONE", categories, True)
            for task_id, categories in enumerate(calls)
        ]
        runs.append(
            {"run_id": run_id, "seed": seed, "setting": "current", "rows": rows}
        )
    path = tmp_path / "three_seed.json"
    _write(path, {"schema_version": "test", "runs": runs})
    frame = load_evaluation_outputs([path])
    none = build_none_analysis(frame)
    summary = stats_module.summarize_core_run_table(
        none, group_columns=["setting"]
    )
    row = summary[summary["metric"] == "overcall_total_rate"].iloc[0]
    assert row["n_runs"] == 3
    assert row["mean"] == pytest.approx(0.5)
    assert row["population_sd"] == pytest.approx(math.sqrt(1 / 6))


def test_rejects_call_count_mismatch_and_duplicate_keys(tmp_path: Path) -> None:
    mismatch = tmp_path / "mismatch.json"
    payload = _payload("current", [("A", ["A"], True)])
    payload["runs"][0]["rows"][0]["tool_calls"] = 0
    _write(mismatch, payload)
    with pytest.raises(ValueError, match=r"len\(routed_tool_events\)"):
        load_evaluation_outputs([mismatch])

    duplicate = tmp_path / "duplicate.json"
    payload = _payload("current", [("A", ["A"], True)])
    payload["runs"][0]["rows"].append(dict(payload["runs"][0]["rows"][0]))
    _write(duplicate, payload)
    with pytest.raises(ValueError, match=r"Duplicate \(setting, run_id, id\)"):
        load_evaluation_outputs([duplicate])


def test_formal_diagnostics_and_gold_action_accuracy(tmp_path: Path) -> None:
    path = tmp_path / "formal.json"
    payload = _payload(
        "current",
        [
            ("NONE", [], True),
            ("A", ["A"], False),
            ("B", ["B"], True),
            ("C", ["C"], True),
        ],
    )
    for run in payload["runs"]:
        run["rows"][1]["termination_reason"] = "max_rounds"
        run["rows"][1]["tool_parse_failures"] = 2
        run["rows"][1]["routed_tool_events"][0]["result"] = {
            "success": False,
            "message": "[SAFETY_REJECTED] test guard",
        }
    _write(path, payload)
    frame = load_evaluation_outputs([path])
    diagnostics, diagnostic_summary = build_run_diagnostics(frame)
    row = diagnostics.iloc[0]
    assert row["max_rounds_count"] == 1
    assert row["total_tool_parse_failures"] == 2
    assert row["rows_with_safety_rejection_count"] == 1
    assert row["total_safety_rejections"] == 1
    assert set(diagnostic_summary["n_runs"]) == {2}

    per_run, summary = build_gold_action_final_accuracy(frame)
    action_a = per_run[per_run["gold_action"] == "A"]
    assert set(action_a["final_accuracy"]) == {0.0}
    action_a_summary = summary[
        (summary["gold_action"] == "A") & (summary["metric"] == "final_accuracy")
    ].iloc[0]
    assert action_a_summary["mean"] == pytest.approx(0.0)
    assert action_a_summary["population_sd"] == pytest.approx(0.0)


def test_formal_schema_rejects_missing_diagnostic_evidence(tmp_path: Path) -> None:
    missing_termination = tmp_path / "missing_termination.json"
    payload = _payload("current", [("A", ["A"], True)])
    del payload["runs"][0]["rows"][0]["termination_reason"]
    _write(missing_termination, payload)
    with pytest.raises(ValueError, match="termination_reason"):
        load_evaluation_outputs([missing_termination])

    missing_result = tmp_path / "missing_result.json"
    payload = _payload("current", [("A", ["A"], True)])
    del payload["runs"][0]["rows"][0]["routed_tool_events"][0]["result"]
    _write(missing_result, payload)
    with pytest.raises(ValueError, match="formal safety diagnostics"):
        load_evaluation_outputs([missing_result])


def test_requires_identical_id_sets_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    payload = _payload("current", [("NONE", [], True), ("A", ["A"], True)])
    payload["runs"][1]["rows"].pop()
    _write(path, payload)
    with pytest.raises(ValueError, match="same ID set"):
        load_evaluation_outputs([path])


def test_formal_seed_panel_is_exact_and_checked_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "two_seeds.json"
    labels = tmp_path / "test_labels.json"
    data, provenance = _write_reference_inputs(tmp_path)
    labels.write_text(
        json.dumps(_label_artifact("test", [(0, "A", 0), (1, "A", 1)])),
        encoding="utf-8",
    )
    _write(
        path,
        _formal_payload(
            "current",
            [("NONE", [], True), ("A", ["A"], True)],
            labels_sha256=sha256_file(labels),
            data_sha256=sha256_file(data),
            runtime_provenance_sha256=sha256_file(provenance),
        ),
    )
    output_dir = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match=r"expected=\[0, 1, 2\].*actual=\[0, 1\]"):
        collect_action_statistics(
            [path],
            output_dir,
            labels_paths=[labels],
            expected_seeds=(0, 1, 2),
            expected_settings=("current",),
            analysis_protocol="fulltools",
            data_path=data,
            runtime_provenance_path=provenance,
            n_bootstrap=20,
        )

    assert not output_dir.exists()
    frame = load_evaluation_outputs([path])
    assert validate_expected_seed_panel(frame, (0, 1)) == (0, 1)
    with pytest.raises(ValueError, match="must be unique"):
        validate_expected_seed_panel(frame, (0, 0))


def test_collect_requires_explicit_formal_panels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit expected seed panel"):
        collect_action_statistics(
            [tmp_path / "unused.json"],
            tmp_path / "stats",
            labels_paths=[tmp_path / "unused-labels.json"],
            expected_settings=("current",),
            analysis_protocol="fulltools",
            n_bootstrap=20,
        )
    assert not (tmp_path / "stats").exists()


def test_paired_bootstrap_is_deterministic_and_signed(tmp_path: Path) -> None:
    left_path = tmp_path / "a.json"
    right_path = tmp_path / "b.json"
    _write(
        left_path,
        _payload(
            "a",
            [
                ("NONE", ["A"], False),
                ("A", [], False),
                ("B", ["A"], False),
                ("C", [], False),
            ],
        ),
    )
    _write(
        right_path,
        _payload(
            "b",
            [
                ("NONE", [], True),
                ("A", ["A"], True),
                ("B", ["B"], True),
                ("C", ["C"], True),
            ],
        ),
    )
    frame = load_evaluation_outputs([left_path, right_path])
    first = paired_bootstrap_comparisons(frame, n_bootstrap=100, bootstrap_seed=7)
    second = paired_bootstrap_comparisons(frame, n_bootstrap=100, bootstrap_seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == set(BOOTSTRAP_METRICS)
    expected_deltas = {
        "final_accuracy": 1.0,
        "action_accuracy": 1.0,
        "avg_tool_calls": 0.25,
        "total_tool_calls_per_run": 1.0,
        "balanced_accuracy": 1.0,
        "macro_f1": 1.0,
        "toolneed_f1": 0.6,
        "recall_NONE": 1.0,
        "recall_A": 1.0,
        "recall_B": 1.0,
        "recall_C": 1.0,
        "overcall_rate": -1.0,
    }
    for metric, expected in expected_deltas.items():
        row = first[first["metric"] == metric].iloc[0]
        assert row["setting_a"] == "a"
        assert row["setting_b"] == "b"
        assert row["delta_b_minus_a"] == pytest.approx(expected)
        assert 0 < row["n_bootstrap_valid"] <= 100

    clone = frame[frame["setting"] == "a"].copy()
    clone["setting"] = "clone_of_a"
    paired = paired_bootstrap_comparisons(
        pd.concat([frame, clone], ignore_index=True),
        n_bootstrap=100,
        bootstrap_seed=7,
    )
    identical = paired[
        (paired["setting_a"] == "a") & (paired["setting_b"] == "clone_of_a")
    ]
    assert len(identical) == len(BOOTSTRAP_METRICS)
    assert (identical[["delta_b_minus_a", "ci95_low", "ci95_high"]] == 0.0).all().all()


def test_current_relative_tradeoff_is_seed_paired(tmp_path: Path) -> None:
    current_path = tmp_path / "current.json"
    sparse_path = tmp_path / "sparse.json"
    _write(
        current_path,
        _payload(
            "current_no_reasoning_fulltools",
            [
                ("NONE", ["A"], True),
                ("A", ["A", "A"], True),
                ("B", ["B"], True),
                ("C", ["C"], True),
            ],
        ),
    )
    _write(
        sparse_path,
        _payload(
            "sparse_tool_no_reasoning_fulltools",
            [
                ("NONE", [], True),
                ("A", ["A"], False),
                ("B", ["B"], True),
                ("C", [], False),
            ],
        ),
    )
    frame = load_evaluation_outputs([current_path, sparse_path])
    per_run = compute_per_run_metrics(frame)
    tradeoff, summary = build_current_relative_tradeoff(per_run)
    sparse = tradeoff[
        tradeoff["setting"] == "sparse_tool_no_reasoning_fulltools"
    ]
    assert len(sparse) == 2
    assert set(sparse["reference_setting"]) == {"current_no_reasoning_fulltools"}
    assert set(sparse["tool_calls_saved"]) == {3.0}
    assert set(sparse["tc_reduction"]) == {0.6}
    assert set(sparse["accuracy_loss"]) == {0.5}
    assert sparse["cost_per_saved_call"].tolist() == pytest.approx([1 / 6, 1 / 6])
    reduction = summary[
        (summary["setting"] == "sparse_tool_no_reasoning_fulltools")
        & (summary["metric"] == "tc_reduction")
    ].iloc[0]
    assert reduction["mean"] == pytest.approx(0.6)
    assert reduction["population_sd"] == pytest.approx(0.0)


def test_per_difficulty_behavior_and_tradeoff_are_seed_paired(tmp_path: Path) -> None:
    current_path = tmp_path / "current_difficulty.json"
    sparse_path = tmp_path / "sparse_difficulty.json"
    current = _payload(
        "current_no_reasoning_fulltools",
        [("A", ["A"], True), ("B", ["B"], True), ("C", ["C"], True)],
    )
    sparse = _payload(
        "sparse_tool_no_reasoning_fulltools",
        [("A", [], False), ("B", [], False), ("C", [], False)],
    )
    for payload in (current, sparse):
        for run in payload["runs"]:
            for row, difficulty in zip(run["rows"], ("easy", "medium", "hard")):
                row["difficulty"] = difficulty
    _write(current_path, current)
    _write(sparse_path, sparse)
    frame = load_evaluation_outputs([current_path, sparse_path])
    per_difficulty = stats_module.compute_per_difficulty_metrics(frame)
    assert set(per_difficulty["difficulty"]) == {"easy", "medium", "hard"}
    tradeoff, summary = stats_module.build_difficulty_current_relative_tradeoff(
        per_difficulty
    )
    sparse_rows = tradeoff[
        tradeoff["setting"] == "sparse_tool_no_reasoning_fulltools"
    ]
    assert len(sparse_rows) == 6
    assert set(sparse_rows["tool_calls_saved"]) == {1.0}
    assert set(sparse_rows["accuracy_loss"]) == {1.0}
    reduction = summary[
        (summary["setting"] == "sparse_tool_no_reasoning_fulltools")
        & (summary["metric"] == "tc_reduction")
    ]
    assert len(reduction) == 3
    assert set(reduction["n_runs"]) == {2}
    assert set(reduction["mean"]) == {1.0}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("no_tool_correct"), "no_tool_correct"),
        (lambda row: row.update(no_tool_correct=True), "integer 0 or 1"),
        (lambda row: row.update(no_tool_correct=0), "disagrees"),
    ],
)
def test_label_distribution_requires_explicit_consistent_no_tool_correct(
    tmp_path: Path, mutation, message: str
) -> None:
    payload = _label_artifact("test", [(0, "A", 0)])
    mutation(payload["rows"][0])
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=message):
        stats_module.load_label_distributions([path])


def test_full_collection_writes_tables_plots_and_label_distribution(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs.json"
    data, provenance = _write_reference_inputs(tmp_path)
    train_labels = tmp_path / "train_labels.json"
    test_labels = tmp_path / "test_labels.json"
    output_dir = tmp_path / "stats"
    train_labels.write_text(
        json.dumps(
            _label_artifact(
                "train",
                [(100, "A", 0), (101, "A", 1), (102, "B", 1), (103, "C", 0)],
            )
        ),
        encoding="utf-8",
    )
    test_labels.write_text(
        json.dumps(
            _label_artifact(
                "test",
                [(0, "A", 0), (1, "A", 1), (2, "B", 1), (3, "C", 1)],
            )
        ),
        encoding="utf-8",
    )
    _write(
        outputs,
        _formal_payload(
            "current",
            [
                ("NONE", [], True),
                ("A", ["A"], True),
                ("B", ["B"], True),
                ("C", ["C"], True),
            ],
            labels_sha256=sha256_file(test_labels),
            data_sha256=sha256_file(data),
            runtime_provenance_sha256=sha256_file(provenance),
        ),
    )

    summary = collect_action_statistics(
        [outputs],
        output_dir,
        labels_paths=[train_labels, test_labels],
        n_bootstrap=20,
        bootstrap_seed=3,
        expected_seeds=(0, 1),
        expected_settings=("current",),
        analysis_protocol="fulltools",
        data_path=data,
        runtime_provenance_path=provenance,
    )
    assert summary["n_rows"] == 8
    assert summary["expected_seeds"] == [0, 1]
    expected = {
        "summary.json",
        "derived_action_rows.csv",
        "per_run_metrics.csv",
        "setting_metric_summary.csv",
        "difficulty_per_run_metrics.csv",
        "difficulty_metric_summary.csv",
        "confusion_counts.csv",
        "confusion_row_normalized.csv",
        "action_recall_summary.csv",
        "outcome_counts.csv",
        "outcome_rates.csv",
        "class_outcome_rates.csv",
        "needed_category_analysis.csv",
        "needed_category_analysis_summary.csv",
        "none_analysis.csv",
        "none_analysis_summary.csv",
        "multicall_summary.csv",
        "multicall_metric_summary.csv",
        "multicall_by_gold.csv",
        "multicall_by_gold_summary.csv",
        "paired_bootstrap_comparisons.csv",
        "label_distribution.csv",
        "run_diagnostics.csv",
        "run_diagnostic_summary.csv",
        "gold_action_final_accuracy_per_run.csv",
        "gold_action_final_accuracy_summary.csv",
        "current_relative_tradeoff_per_run.csv",
        "current_relative_tradeoff_summary.csv",
        "difficulty_current_relative_tradeoff_per_run.csv",
        "difficulty_current_relative_tradeoff_summary.csv",
        "confusion_heatmap.png",
        "recall_bars.png",
        "error_stacked.png",
        "accuracy_vs_total_tc.png",
        "accuracy_vs_total_tc_by_difficulty.png",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    assert [item["path"] for item in summary["labels_files"]] == [
        str(train_labels),
        str(test_labels),
    ]
    assert all(len(item["sha256"]) == 64 for item in summary["labels_files"])
    distribution = pd.read_csv(output_dir / "label_distribution.csv")
    assert set(distribution["split"]) == {"train", "test"}
    assert "no_tool_correct" in distribution
    assert summary["analysis_protocol"] == "fulltools"
    assert summary["label_protocol"] == "adapted"
    assert summary["expected_settings"] == ["current"]
    assert summary["input_artifacts"] == [
        {
            "path": str(outputs.resolve()),
            "sha256": sha256_file(outputs),
            "setting": "current",
            "model": "model-slug",
            "tool_scope": "full",
            "labels_sha256": sha256_file(test_labels),
            "runtime_provenance_sha256": sha256_file(provenance),
            "project_git_commit": "project-commit",
        }
    ]
    assert summary["behavior_generation_git_commit"] == "project-commit"
    assert summary["statistics_code_git_commit"] != "project-commit"
    assert summary["referenced_inputs"] == {
        "data": {"path": str(data.resolve()), "sha256": sha256_file(data)},
        "runtime_provenance": {
            "path": str(provenance.resolve()),
            "sha256": sha256_file(provenance),
        },
    }
    assert summary["metric_definitions"]["no_call_precision"] == (
        "P(gold NONE | predicted NONE)."
    )
    assert "single-hop benchmark" in summary["metric_definitions"]["tool_call_rate"]
    assert "more than one" in summary["metric_definitions"]["multi_call_rate"]
    assert (
        "mixed-category paths only"
        in summary["table_definitions"]["multicall_summary.csv"]
    )
    confusion = pd.read_csv(output_dir / "confusion_counts.csv")
    assert list(confusion.columns[-5:]) == ["NONE", "A", "B", "C", "INVALID"]
    for filename in (
        "confusion_heatmap.png",
        "recall_bars.png",
        "error_stacked.png",
        "accuracy_vs_total_tc.png",
        "accuracy_vs_total_tc_by_difficulty.png",
    ):
        assert (output_dir / filename).stat().st_size > 0
    for csv_path in output_dir.glob("*.csv"):
        table = pd.read_csv(csv_path)
        assert table.columns[:2].tolist() == ["analysis_protocol", "label_protocol"]
        if len(table):
            assert set(table["analysis_protocol"]) == {"fulltools"}
            assert set(table["label_protocol"]) == {"adapted"}
    with pytest.raises(FileExistsError):
        collect_action_statistics(
            [outputs],
            output_dir,
            labels_paths=[train_labels, test_labels],
            n_bootstrap=20,
            expected_seeds=(0, 1),
            expected_settings=("current",),
            analysis_protocol="fulltools",
            data_path=data,
            runtime_provenance_path=provenance,
        )


def test_formal_collection_rejects_wrong_setting_and_cross_artifact_provenance(
    tmp_path: Path,
) -> None:
    outputs, train_labels, test_labels, data, provenance = _write_formal_inputs(
        tmp_path, settings=("current", "necessary_tool_no_reasoning_fulltools")
    )
    common = {
        "labels_paths": [train_labels, test_labels],
        "expected_seeds": (0, 1),
        "analysis_protocol": "fulltools",
        "data_path": data,
        "runtime_provenance_path": provenance,
        "n_bootstrap": 10,
    }
    with pytest.raises(ValueError, match="Formal setting panel mismatch"):
        collect_action_statistics(
            outputs,
            tmp_path / "wrong-settings",
            expected_settings=("current", "sparse_tool_no_reasoning_fulltools"),
            **common,
        )
    assert not (tmp_path / "wrong-settings").exists()

    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    payload["config"]["runtime_provenance_sha256"] = "e" * 64
    _write(outputs[1], payload)
    with pytest.raises(ValueError, match="runtime_provenance_sha256"):
        collect_action_statistics(
            outputs,
            tmp_path / "wrong-provenance",
            expected_settings=(
                "current",
                "necessary_tool_no_reasoning_fulltools",
            ),
            **common,
        )
    assert not (tmp_path / "wrong-provenance").exists()


def test_formal_collection_binds_test_label_sha_gold_and_difficulty(
    tmp_path: Path,
) -> None:
    outputs, train_labels, test_labels, data, provenance = _write_formal_inputs(
        tmp_path
    )
    common = {
        "labels_paths": [train_labels, test_labels],
        "expected_seeds": (0, 1),
        "expected_settings": ("current",),
        "analysis_protocol": "fulltools",
        "data_path": data,
        "runtime_provenance_path": provenance,
        "n_bootstrap": 10,
    }
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    payload["config"]["labels_sha256"] = "e" * 64
    _write(outputs[0], payload)
    with pytest.raises(ValueError, match="Test labels sha256"):
        collect_action_statistics(outputs, tmp_path / "wrong-sha", **common)
    assert not (tmp_path / "wrong-sha").exists()

    payload["config"]["labels_sha256"] = sha256_file(test_labels)
    for run in payload["runs"]:
        run["rows"][1]["gold_action"] = "B"
    _write(outputs[0], payload)
    with pytest.raises(ValueError, match="gold_action values"):
        collect_action_statistics(outputs, tmp_path / "wrong-gold", **common)
    assert not (tmp_path / "wrong-gold").exists()

    for run in payload["runs"]:
        run["rows"][1]["gold_action"] = "A"
        run["rows"][1]["difficulty"] = "medium"
    _write(outputs[0], payload)
    with pytest.raises(ValueError, match="difficulty values"):
        collect_action_statistics(outputs, tmp_path / "wrong-difficulty", **common)
    assert not (tmp_path / "wrong-difficulty").exists()


def test_formal_collection_verifies_exact_data_and_provenance_files(
    tmp_path: Path,
) -> None:
    outputs, train_labels, test_labels, data, provenance = _write_formal_inputs(
        tmp_path
    )
    common = {
        "labels_paths": [train_labels, test_labels],
        "expected_seeds": (0, 1),
        "expected_settings": ("current",),
        "analysis_protocol": "fulltools",
        "n_bootstrap": 10,
    }
    wrong_data = tmp_path / "wrong_tasks.json"
    wrong_data.write_text('{"tasks": "different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Behavior data SHA256"):
        collect_action_statistics(
            outputs,
            tmp_path / "wrong-data",
            data_path=wrong_data,
            runtime_provenance_path=provenance,
            **common,
        )
    assert not (tmp_path / "wrong-data").exists()

    wrong_provenance = tmp_path / "wrong_runtime_provenance.json"
    wrong_provenance.write_text(
        json.dumps(
            {
                "manifest_type": "runtime-and-input-provenance",
                "git": {"commit": "project-commit"},
                "unexpected": "different hash",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Runtime provenance SHA256"):
        collect_action_statistics(
            outputs,
            tmp_path / "wrong-provenance-file",
            data_path=data,
            runtime_provenance_path=wrong_provenance,
            **common,
        )
    assert not (tmp_path / "wrong-provenance-file").exists()


def test_statistics_staging_failure_preserves_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, train_labels, test_labels, data, provenance = _write_formal_inputs(
        tmp_path
    )
    destination = tmp_path / "stats"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("old", encoding="utf-8")

    def fail_plot(*args, **kwargs):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(stats_module, "_plot_recalls", fail_plot)
    with pytest.raises(RuntimeError, match="plot failed"):
        collect_action_statistics(
            outputs,
            destination,
            labels_paths=[train_labels, test_labels],
            expected_seeds=(0, 1),
            expected_settings=("current",),
            analysis_protocol="fulltools",
            data_path=data,
            runtime_provenance_path=provenance,
            overwrite=True,
            n_bootstrap=10,
        )
    assert sentinel.read_text(encoding="utf-8") == "old"
    assert {path.name for path in destination.iterdir()} == {"sentinel.txt"}
    assert not list(tmp_path.glob(".stats.stage-*"))


def test_statistics_publish_failure_rolls_back_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, train_labels, test_labels, data, provenance = _write_formal_inputs(
        tmp_path
    )
    destination = tmp_path / "stats"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("old", encoding="utf-8")

    def tiny_plot(*args, **kwargs):
        path = Path(args[-1])
        path.write_bytes(b"plot")

    for name in (
        "_plot_confusions",
        "_plot_recalls",
        "_plot_error_stack",
        "_plot_accuracy_vs_calls",
        "_plot_accuracy_vs_calls_by_difficulty",
    ):
        monkeypatch.setattr(stats_module, name, tiny_plot)

    real_replace = stats_module.os.replace

    def fail_staged_publish(source, target):
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name.startswith(".stats.stage-") and target_path == destination:
            raise OSError("publish failed")
        return real_replace(source, target)

    monkeypatch.setattr(stats_module.os, "replace", fail_staged_publish)
    with pytest.raises(OSError, match="publish failed"):
        collect_action_statistics(
            outputs,
            destination,
            labels_paths=[train_labels, test_labels],
            expected_seeds=(0, 1),
            expected_settings=("current",),
            analysis_protocol="fulltools",
            data_path=data,
            runtime_provenance_path=provenance,
            overwrite=True,
            n_bootstrap=10,
        )
    assert sentinel.read_text(encoding="utf-8") == "old"
    assert {path.name for path in destination.iterdir()} == {"sentinel.txt"}
    assert not list(tmp_path.glob(".stats.stage-*"))
    assert not list(tmp_path.glob(".stats.backup-*"))


def test_statistics_successful_overwrite_replaces_directory_without_stale_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, train_labels, test_labels, data, provenance = _write_formal_inputs(
        tmp_path
    )
    destination = tmp_path / "stats"
    destination.mkdir()
    (destination / "stale.csv").write_text("stale", encoding="utf-8")

    def tiny_plot(*args, **kwargs):
        path = Path(args[-1])
        path.write_bytes(b"plot")

    for name in (
        "_plot_confusions",
        "_plot_recalls",
        "_plot_error_stack",
        "_plot_accuracy_vs_calls",
        "_plot_accuracy_vs_calls_by_difficulty",
    ):
        monkeypatch.setattr(stats_module, name, tiny_plot)
    summary = collect_action_statistics(
        outputs,
        destination,
        labels_paths=[train_labels, test_labels],
        expected_seeds=(0, 1),
        expected_settings=("current",),
        analysis_protocol="fulltools",
        data_path=data,
        runtime_provenance_path=provenance,
        overwrite=True,
        n_bootstrap=10,
    )
    assert not (destination / "stale.csv").exists()
    assert (destination / "summary.json").is_file()
    assert all((destination / item["path"]).is_file() for item in summary["published_files"])
