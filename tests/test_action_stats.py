import json
from pathlib import Path

import pandas as pd
import pytest

from when2tool_action.stats import (
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
    assert row["action_accuracy"] == pytest.approx(0.4)
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
        ("A", ["A"], False, "answer_wrong"),
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
        "paired_bootstrap_comparisons.csv",
        "label_distribution.csv",
        "confusion_heatmap.png",
        "recall_bars.png",
        "error_stacked.png",
        "accuracy_vs_total_tc.png",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
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
