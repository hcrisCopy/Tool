import json
from pathlib import Path

import pandas as pd
import pytest

from when2tool_action.constants import SCHEMA_VERSION as ACTION_SCHEMA_VERSION
from when2tool_action.constants import UPSTREAM_COMMIT
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
    _write(path, _payload("current", [("NONE", [], True), ("A", ["A"], True)]))
    output_dir = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match=r"expected=\[0, 1, 2\].*actual=\[0, 1\]"):
        collect_action_statistics(
            [path],
            output_dir,
            expected_seeds=(0, 1, 2),
            n_bootstrap=20,
        )

    assert not output_dir.exists()
    frame = load_evaluation_outputs([path])
    assert validate_expected_seed_panel(frame, (0, 1)) == (0, 1)
    with pytest.raises(ValueError, match="must be unique"):
        validate_expected_seed_panel(frame, (0, 0))


def test_single_seed_smoke_requires_omitting_expected_panel(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    payload = _payload("current_smoke", [("NONE", [], True), ("A", ["A"], True)])
    payload["runs"] = payload["runs"][:1]
    _write(path, payload)

    frame = load_evaluation_outputs([path])
    with pytest.raises(ValueError, match=r"actual=\[0\]"):
        validate_expected_seed_panel(frame, (0, 1, 2))
    summary = collect_action_statistics(
        [path], tmp_path / "smoke_stats", n_bootstrap=20
    )
    assert summary["expected_seeds"] is None
    assert summary["n_runs"] == 1


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


def test_full_collection_writes_tables_plots_and_label_distribution(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs.json"
    train_labels = tmp_path / "train_labels.json"
    test_labels = tmp_path / "test_labels.json"
    output_dir = tmp_path / "stats"
    _write(
        outputs,
        _payload(
            "current",
            [
                ("NONE", [], True),
                ("A", ["A"], True),
                ("B", ["B"], True),
                ("C", ["C"], True),
            ],
        ),
    )
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
                [(0, "A", 0), (1, "A", 1), (2, "B", 1), (3, "C", 0)],
            )
        ),
        encoding="utf-8",
    )

    summary = collect_action_statistics(
        [outputs],
        output_dir,
        labels_paths=[train_labels, test_labels],
        n_bootstrap=20,
        bootstrap_seed=3,
        expected_seeds=(0, 1),
    )
    assert summary["n_rows"] == 8
    assert summary["expected_seeds"] == [0, 1]
    expected = {
        "summary.json",
        "derived_action_rows.csv",
        "per_run_metrics.csv",
        "setting_metric_summary.csv",
        "confusion_counts.csv",
        "confusion_row_normalized.csv",
        "action_recall_summary.csv",
        "outcome_counts.csv",
        "outcome_rates.csv",
        "class_outcome_rates.csv",
        "needed_category_analysis.csv",
        "none_analysis.csv",
        "multicall_summary.csv",
        "multicall_by_gold.csv",
        "paired_bootstrap_comparisons.csv",
        "label_distribution.csv",
        "run_diagnostics.csv",
        "run_diagnostic_summary.csv",
        "gold_action_final_accuracy_per_run.csv",
        "gold_action_final_accuracy_summary.csv",
        "current_relative_tradeoff_per_run.csv",
        "current_relative_tradeoff_summary.csv",
        "confusion_heatmap.png",
        "recall_bars.png",
        "error_stacked.png",
        "accuracy_vs_total_tc.png",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    assert [item["path"] for item in summary["labels_files"]] == [
        str(train_labels),
        str(test_labels),
    ]
    assert all(len(item["sha256"]) == 64 for item in summary["labels_files"])
    distribution = pd.read_csv(output_dir / "label_distribution.csv")
    assert set(distribution["split"]) == {"train", "test"}
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
    ):
        assert (output_dir / filename).stat().st_size > 0
    with pytest.raises(FileExistsError):
        collect_action_statistics([outputs], output_dir, n_bootstrap=20)
