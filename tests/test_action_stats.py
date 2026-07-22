import json
from pathlib import Path

import pandas as pd
import pytest

from when2tool_action.stats import (
    build_multicall_tables,
    build_needed_category_analysis,
    build_none_analysis,
    collect_action_statistics,
    compute_per_run_metrics,
    load_evaluation_outputs,
    paired_bootstrap_comparisons,
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
        "routed_tool_events": [{"category": category} for category in categories],
        "tool_calls": len(categories),
        "final_correct": correct,
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


def test_requires_identical_id_sets_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    payload = _payload("current", [("NONE", [], True), ("A", ["A"], True)])
    payload["runs"][1]["rows"].pop()
    _write(path, payload)
    with pytest.raises(ValueError, match="same ID set"):
        load_evaluation_outputs([path])


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
    final = first[first["metric"] == "final_accuracy"].iloc[0]
    action = first[first["metric"] == "action_accuracy"].iloc[0]
    assert final["setting_a"] == "a"
    assert final["setting_b"] == "b"
    assert final["delta_b_minus_a"] == pytest.approx(1.0)
    assert action["delta_b_minus_a"] == pytest.approx(1.0)


def test_full_collection_writes_tables_plots_and_label_distribution(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs.json"
    labels = tmp_path / "labels.json"
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
    labels.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": index,
                        "split": "test",
                        "difficulty": "easy",
                        "category": category,
                        "tool_necessary": necessary,
                        "gold_action": category if necessary else "NONE",
                    }
                    for index, (category, necessary) in enumerate(
                        [("A", 0), ("A", 1), ("B", 1), ("C", 0)]
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = collect_action_statistics(
        [outputs],
        output_dir,
        labels_path=labels,
        n_bootstrap=20,
        bootstrap_seed=3,
    )
    assert summary["n_rows"] == 8
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
        "confusion_heatmap.png",
        "recall_bars.png",
        "error_stacked.png",
        "accuracy_vs_total_tc.png",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
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
