from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from when2tool_action.eval_resume import PreparedEvaluationArtifact
from when2tool_action.scripts import run_eval, run_scoped_baseline


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source=tmp_path / "config.yaml",
        model=SimpleNamespace(slug="model-slug"),
        generation=SimpleNamespace(
            seeds=(0,),
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            max_new_tokens=64,
            label_hidden_extraction_max_rounds=12,
            behavior_evaluation_max_rounds=2,
            max_model_len=1024,
        ),
    )


def _common_mocks(module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "require_inputs", lambda value: None)
    monkeypatch.setattr(
        module,
        "validate_runtime_provenance",
        lambda value: {"sha256": "r" * 64, "git_commit": "commit"},
    )
    monkeypatch.setattr(module, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(module, "full_menu_sha256", lambda: "f" * 64)
    monkeypatch.setattr(
        module,
        "load_task_json",
        lambda path, expected_scope: [{"id": 101}],
    )
    monkeypatch.setattr(
        module,
        "attach_gold_actions",
        lambda tasks, rows: [dict(task, gold_action="NONE") for task in tasks],
    )


def test_overwrite_initializes_empty_checkpoint_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _common_mocks(run_eval, tmp_path, monkeypatch)
    target = tmp_path / "evaluation.json"
    target.write_text('{"stale": true}\n', encoding="utf-8")
    monkeypatch.setattr(run_eval, "_load_labels", lambda path: [{"id": 101}])

    def fail_model_load(config):
        checkpoint = json.loads(target.read_text(encoding="utf-8"))
        assert checkpoint["runs"] == []
        assert checkpoint["config"]["max_rounds"] == 2
        raise RuntimeError("model load failed")

    monkeypatch.setattr(run_eval, "build_agent", fail_model_load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_eval",
            "--config",
            "unused.yaml",
            "--data",
            "unused-data.json",
            "--labels",
            "unused-labels.json",
            "--output",
            str(target),
            "--setting-name",
            "current_no_reasoning_fulltools",
            "--tool-scope",
            "full",
            "--prompt-mode",
            "current",
            "--reasoning-mode",
            "no_reasoning",
            "--seeds",
            "0",
            "--overwrite",
        ],
    )

    with pytest.raises(RuntimeError, match="model load failed"):
        run_eval.main()
    assert json.loads(target.read_text(encoding="utf-8"))["runs"] == []


def test_scoped_matrix_finishes_all_preflights_before_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _common_mocks(run_scoped_baseline, tmp_path, monkeypatch)
    monkeypatch.setattr(run_scoped_baseline, "_labels", lambda path: [{"id": 101}])
    calls = 0

    def preflight(
        path, *, template, task_ids, overwrite, resume, expected_row_fields
    ):
        nonlocal calls
        calls += 1
        assert template["config"]["max_rounds"] == 2
        if calls == 2:
            raise ValueError("later target preflight failed")
        return PreparedEvaluationArtifact(path.resolve(), template, 0, True)

    monkeypatch.setattr(run_scoped_baseline, "prepare_evaluation_artifact", preflight)
    monkeypatch.setattr(
        run_scoped_baseline,
        "initialize_evaluation_artifacts",
        lambda targets: pytest.fail("initialization ran before every preflight passed"),
    )
    monkeypatch.setattr(
        run_scoped_baseline,
        "build_agent",
        lambda config: pytest.fail("model loaded before every preflight passed"),
    )
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_scoped_baseline",
            "--config",
            "unused.yaml",
            "--data",
            "unused-data.json",
            "--labels",
            "unused-labels.json",
            "--output-dir",
            str(output_dir),
            "--prompt-modes",
            "current",
            "necessary_tool",
            "--reasoning-modes",
            "no_reasoning",
            "--seeds",
            "0",
            "--resume",
        ],
    )

    with pytest.raises(ValueError, match="later target preflight failed"):
        run_scoped_baseline.main()
    assert not output_dir.exists()
