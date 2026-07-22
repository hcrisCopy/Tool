from __future__ import annotations

import json
from pathlib import Path

import pytest

from when2tool_action.io_utils import canonical_json_sha256, sha256_file
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
        if relative == "manifests/formal_stage_audit.json":
            continue
        _write_json(root / relative)
    for relative in FORMAL_BEHAVIOR_OUTPUTS:
        _write_json(root / relative, {"setting": Path(relative).stem})
    _write_json(root / "labels/qwen/test_labels.json")
    _write_json(root / "probes/fulltools/probe_results.json")
    report = root / "reports/stages/STAGE_STATISTICS_QWEN3_4B.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Stage report\n", encoding="utf-8")
    critical = sorted(
        {
            *FORMAL_BEHAVIOR_OUTPUTS,
            *(
                relative
                for relative in REQUIRED_ARTIFACTS
                if "formal_stage_audit" not in relative
            ),
        }
    )
    checked = [
        {
            "path_base": "run_root",
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in critical
    ]
    _write_json(
        root / "manifests/formal_stage_audit.json",
        {
            "schema_version": "when2tool-formal-stage-audit.v1",
            "manifest_type": "formal-statistics-stage-semantic-audit",
            "audit_complete": True,
            "model": "qwen3-4b-instruct-2507",
            "code_commits": {
                "behavior": BEHAVIOR_COMMIT,
                "statistics": STATISTICS_COMMIT,
            },
            "registered_protocol": {
                "seeds": [0, 1, 2],
                "train_task_count": 900,
                "task_count": 2250,
                "behavior_max_rounds": 10,
                "probe_temperature": 2.0,
                "probe_thresholds": [0.1, 0.3, 0.5, 0.7, 0.9],
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20260722,
            },
            "statistics": {
                "protocols": {
                    protocol: {
                        **counts,
                        "summary_sha256": sha256_file(
                            root / "analysis" / protocol / "summary.json"
                        ),
                    }
                    for protocol, counts in {
                        "fulltools": {
                            "n_settings": 8,
                            "n_runs": 24,
                            "n_rows": 54000,
                        },
                        "scoped_adapted": {
                            "n_settings": 15,
                            "n_runs": 45,
                            "n_rows": 101250,
                        },
                        "scoped_original_w2t": {
                            "n_settings": 15,
                            "n_runs": 45,
                            "n_rows": 101250,
                        },
                    }.items()
                },
                "summary_count": 3,
                "publication_hashes_verified": True,
            },
            "behavior": {
                "n_artifacts": len(FORMAL_BEHAVIOR_OUTPUTS),
                "exact_inventory_verified": True,
                "config_and_provenance_verified": True,
                "setting_modes_action_rows_and_menus_verified": True,
            },
            "probe_prefill": [
                {
                    "directory": directory,
                    "probe_protocol": probe_protocol,
                    "probe_directory": probe_directory,
                    "n_task_ids": 2250,
                    "probe_input_hashes_verified": True,
                    "probability_sigmoid_verified": True,
                    "probability_logit_temperature_invariant": True,
                    "decision_threshold_consistent": True,
                    "use_tool_sets_nested": True,
                }
                for directory, probe_protocol, probe_directory in (
                    ("fulltools", "adapted", "probes/fulltools"),
                    ("scoped_adapted", "adapted", "probes/scoped"),
                    (
                        "scoped_original_w2t",
                        "scoped_original_w2t",
                        "probes/scoped_original_w2t",
                    ),
                )
            ],
            "scoped_relabel": {
                "source_and_target_labels_bound": True,
                "immutable_behavior_fields_preserved": True,
                "receipt_hashes_verified": True,
            },
            "migration": {
                "original_protocol_metadata_validation_passed": True,
                "destination_only_validation_passed": True,
            },
            "checked_files": checked,
            "checked_files_sha256": canonical_json_sha256(checked),
        },
    )


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
        "manifests/.ipynb_checkpoints/runtime_provenance-checkpoint.json",
        "analysis/fulltools/previous-backup/result.csv",
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


def test_requires_completed_semantic_audit_receipt(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    (root / "manifests/formal_stage_audit.json").unlink()

    with pytest.raises(FileNotFoundError, match="formal_stage_audit"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )


def test_semantic_audit_commit_and_file_bindings_must_match_handoff(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    receipt_path = root / "manifests/formal_stage_audit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["code_commits"]["statistics"] = "c" * 40
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="code_commits"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )

    _formal_fixture(root)
    changed = root / FORMAL_BEHAVIOR_OUTPUTS[0]
    _write_json(changed, {"changed": True})
    with pytest.raises(ValueError, match="no longer matches"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("behavior", "setting_modes_action_rows_and_menus_verified"),
        ("scoped_relabel", "source_and_target_labels_bound"),
        ("migration", "original_protocol_metadata_validation_passed"),
    ),
)
def test_handoff_requires_new_semantic_audit_contract_flags(
    tmp_path: Path, section: str, field: str
) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    receipt_path = root / "manifests/formal_stage_audit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[section][field] = False
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match=field):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )


def test_handoff_requires_probe_probability_and_input_audit_flags(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    receipt_path = root / "manifests/formal_stage_audit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["probe_prefill"][0]["probability_sigmoid_verified"] = False
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="probability_sigmoid_verified"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda receipt: receipt.pop("statistics"), "statistics must be an object"),
        (
            lambda receipt: receipt["statistics"].__setitem__(
                "publication_hashes_verified", False
            ),
            "publication_hashes_verified",
        ),
        (
            lambda receipt: receipt["statistics"]["protocols"]["fulltools"].__setitem__(
                "n_rows", 53999
            ),
            "fulltools n_rows",
        ),
    ),
)
def test_handoff_requires_exact_statistics_contract(
    tmp_path: Path, mutate, match: str
) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    receipt_path = root / "manifests/formal_stage_audit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(receipt)
    _write_json(receipt_path, receipt)

    with pytest.raises((TypeError, ValueError), match=match):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
        )


def test_statistics_summary_sha_is_cross_bound_to_checked_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _formal_fixture(root)
    receipt_path = root / "manifests/formal_stage_audit.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["statistics"]["protocols"]["scoped_adapted"]["summary_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="cross-bound to checked_files"):
        build_stage_handoff(
            root,
            output=tmp_path / "handoff.json",
            behavior_commit=BEHAVIOR_COMMIT,
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
