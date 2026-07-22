from pathlib import Path

import pytest

from when2tool_action import hidden


def test_hidden_output_preflight_covers_both_splits_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "probes"
    output_dir.mkdir()
    existing_test_artifact = output_dir / "test_hidden_manifest.json"
    existing_test_artifact.write_text("already here", encoding="utf-8")

    model_load_attempted = False

    def fail_if_model_is_loaded(_config: object) -> tuple[object, object]:
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("model loading must happen after the global preflight")

    monkeypatch.setattr(hidden, "_load_model", fail_if_model_is_loaded)

    with pytest.raises(FileExistsError, match="test_hidden_manifest.json"):
        hidden.extract_all(
            {"train": [], "test": []},
            {"train": tmp_path / "train.json", "test": tmp_path / "test.json"},
            object(),
            output_dir,
            tool_scope="full",
            overwrite=False,
        )

    assert not model_load_attempted
    assert hidden.hidden_output_paths(output_dir) == (
        output_dir / "train_hidden_no_reasoning.pt",
        output_dir / "train_labels_no_reasoning.json",
        output_dir / "train_hidden_manifest.json",
        output_dir / "test_hidden_no_reasoning.pt",
        output_dir / "test_labels_no_reasoning.json",
        output_dir / "test_hidden_manifest.json",
    )


def test_hidden_output_preflight_rejects_a_file_output_root(tmp_path: Path) -> None:
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("collision", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        hidden.preflight_hidden_outputs(output_path, overwrite=True)
