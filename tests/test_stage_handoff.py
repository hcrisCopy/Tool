from __future__ import annotations

import json
from pathlib import Path

import pytest

from when2tool_action.stage_handoff import (
    FORMAL_BEHAVIOR_OUTPUTS,
    MANAGED_CATEGORIES,
    REQUIRED_ARTIFACTS,
    build_stage_handoff,
    write_stage_handoff,
)


BEHAVIOR_COMMIT = "a" * 40
STATISTICS_COMMIT = "b" * 40


def _write_json(path: Path, value: object | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ok": True} if value is None else value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _formal_fixture(root: Path) -> None:
    for category in MANAGED_CATEGORIES:
        (root / category).mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_ARTIFACTS:
        _write_json(root / relative)
    for relative in FORMAL_BEHAVIOR_OUTPUTS:
        _write_json(root / relative, {"setting": Path(relative).stem})
    _write_json(root / "labels/qwen/test_labels.json")
    _write_json(root / "probes/fulltools/probe_results.json")
    report = root / "reports/stages/STAGE_STATISTICS_QWEN3_4B.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Stage report\n", encoding="utf-8")


def test_build_is_relative_deterministic_and_excludes_logs(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    log = root / "logs/formal.log"
    log.parent.mkdir(parents=True)
    log.write_text("first\n", encoding="utf-8")
    output = root / "manifests/stage_handoff.json"

    first = write_stage_handoff(
        root,
        output=output,
        behavior_commit=BEHAVIOR_COMMIT,
        statistics_commit=STATISTICS_COMMIT,
    )
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == first
    assert first["code_commits"] == {
        "behavior": BEHAVIOR_COMMIT,
        "statistics": STATISTICS_COMMIT,
    }
    assert first["inventory_policy"]["logs_hashed"] is False
    assert first["category_summary"]["outputs"]["file_count"] == 39
    paths = [entry["path"] for entry in first["artifacts"]]
    assert paths == sorted(paths)
    assert "manifests/stage_handoff.json" not in paths
    assert not any(path.startswith("logs/") for path in paths)
    assert all(not Path(path).is_absolute() and "\\" not in path for path in paths)
    assert str(root) not in json.dumps(first)

    log.write_text("changed but still excluded\n", encoding="utf-8")
    second = write_stage_handoff(
        root,
        output=output,
        behavior_commit=BEHAVIOR_COMMIT,
        statistics_commit=STATISTICS_COMMIT,
        overwrite=True,
    )
    assert second == first


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    output = tmp_path / "handoff.json"
    write_stage_handoff(
        root,
        output=output,
        behavior_commit=BEHAVIOR_COMMIT,
        statistics_commit=STATISTICS_COMMIT,
    )
    before = output.read_bytes()

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_stage_handoff(
            root,
            output=output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "relative",
    (
        "probes/smoke/result.json",
        "analysis/.action-stats-stage-dead/result.csv",
        "data/cache/item.bin",
        "labels/shard.tmp",
        "reports/logs/console.txt",
    ),
)
def test_rejects_nonformal_work_inside_managed_directories(
    tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    contaminated = root / relative
    contaminated.parent.mkdir(parents=True, exist_ok=True)
    contaminated.write_bytes(b"not formal")

    with pytest.raises(ValueError, match="Forbidden"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )


def test_requires_exact_formal_output_panel(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    missing = root / FORMAL_BEHAVIOR_OUTPUTS[0]
    missing.unlink()

    with pytest.raises(ValueError, match="Formal output inventory mismatch"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )

    _write_json(missing)
    _write_json(root / "outputs/fulltools/unexpected.json")
    with pytest.raises(ValueError, match="unexpected.json"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )

    (root / "outputs/fulltools/unexpected.json").unlink()
    (root / "outputs/unused").mkdir()
    with pytest.raises(ValueError, match="directory inventory mismatch"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )


def test_requires_core_json_objects_and_canonical_commits(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    runtime = root / "manifests/runtime_provenance.json"
    runtime.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="JSON object"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )

    _write_json(runtime)
    with pytest.raises(ValueError, match="behavior_commit"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit="not-a-commit",
            statistics_commit=STATISTICS_COMMIT,
        )


def test_output_inside_run_root_must_be_under_manifests(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    with pytest.raises(ValueError, match="under manifests"):
        build_stage_handoff(
            root,
            output=root / "reports/handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )
