from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch

from when2tool_action.constants import UPSTREAM_COMMIT
from when2tool_action.legacy_scoped import (
    EXPECTED_AUDIT_SOURCES,
    EXPECTED_DESTINATION_ARTIFACTS,
    EXPECTED_SOURCE_ARTIFACTS,
    LEGACY_LABEL_POLICY,
    PROTOCOL_ID,
    TRANSFER_FILES,
    _atomic_transfer,
    _expected_receipt_inventory,
    _load_and_validate_hidden,
    _validate_legacy_label_rows,
    legacy_audit_source_paths,
    validate_imported_scoped_destination,
)
from when2tool_action.io_utils import sha256_file
from when2tool_action.scripts.run_probe_prefill import validate_original_protocol_files


def _fixture() -> tuple[list[dict], list[dict], list[dict], dict]:
    task = {
        "id": 101,
        "difficulty": "easy",
        "instruction": "What is 2 + 2?",
        "expected": {"answer": "4"},
        "gold_env_name": "CalculatorEnv",
        "gold_tools": ["evaluate_expression"],
    }
    prompt = (
        "SYSTEM evaluate_expression\nWhat is 2 + 2?\n"
        f"{LEGACY_LABEL_POLICY}\n"
    )
    row = {
        "id": 101,
        "difficulty": "easy",
        "env": "CalculatorEnv",
        "category": "A",
        "tool_type": "A",
        "seed": 0,
        "prompt_variant": "P_env",
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rounds": 1,
        "completed": True,
        "final_response": "\\boxed{4}",
        "boxed_answer": "4",
        "cleaned_answer": "4",
        "gold_answer": "4",
        "no_tool_correct": 1,
        "tool_necessary": 0,
        "tool_calls": 0,
        "generation_tokens": 3,
        "prefill_tokens": 10,
        "reasoning_mode": "no_reasoning",
        "upstream_commit": UPSTREAM_COMMIT,
        "enable_thinking": False,
        "trace": [{"round": 1, "prompt_text": prompt}],
    }
    baseline = {
        "id": 101,
        "difficulty": "easy",
        "env": "CalculatorEnv",
        "category": "A",
        "no_tool_correct": 1,
        "tool_necessary": 0,
        "first_sentence": "",
    }
    rendered = {101: (prompt, [1, 2, 3])}
    return [row], [baseline], [task], rendered


def test_legacy_label_fixture_is_converted_without_claiming_adaptation() -> None:
    rows, baseline, tasks, rendered = _fixture()
    converted = _validate_legacy_label_rows(
        rows,
        baseline,
        tasks,
        rendered,
        split="test",
        seed=0,
    )
    assert converted[0]["gold_action"] == "NONE"
    assert converted[0]["label_protocol"] == PROTOCOL_ID
    assert converted[0]["tool_scope"] == "scoped"


def test_legacy_label_fixture_rejects_prompt_hash_mismatch() -> None:
    rows, baseline, tasks, rendered = _fixture()
    rows[0]["prompt_hash"] = "0" * 64
    with pytest.raises(ValueError, match="trace hash"):
        _validate_legacy_label_rows(
            rows,
            baseline,
            tasks,
            rendered,
            split="test",
            seed=0,
        )


def test_hidden_fixture_requires_float32_shape_and_finite_values(tmp_path) -> None:
    path = tmp_path / "hidden.pt"
    torch.save(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), path)
    hidden = _load_and_validate_hidden(path, expected_shape=(2, 3, 4))
    assert tuple(hidden.shape) == (2, 3, 4)

    torch.save(torch.full((2, 3, 4), float("nan"), dtype=torch.float32), path)
    with pytest.raises(ValueError, match="NaN or infinity"):
        _load_and_validate_hidden(path, expected_shape=(2, 3, 4))


def test_audit_source_inventory_is_exact_and_overwrite_is_explicit(tmp_path) -> None:
    paths = legacy_audit_source_paths(tmp_path / "legacy", 0)
    relative = [path.relative_to(tmp_path / "legacy").as_posix() for path in paths]
    assert relative == [
        "labels/full/seed_0/train_no_tool_outputs.json",
        "labels/full/seed_0/train_label_stats.csv",
        "labels/full/seed_0/train_manifest.json",
        "hidden/full/P_env/train_manifest.json",
        "hidden/full/P_env/train_metadata.json",
        "labels/full/seed_0/test_no_tool_outputs.json",
        "labels/full/seed_0/test_label_stats.csv",
        "labels/full/seed_0/test_manifest.json",
        "hidden/full/P_env/test_manifest.json",
        "hidden/full/P_env/test_metadata.json",
    ]
    assert not any(
        forbidden in path
        for path in relative
        for forbidden in ("P_all", "P_no_schema", "diagnostic", "onset")
    )

    source = tmp_path / "source.json"
    destination = tmp_path / "audit_source" / "source.json"
    source.write_text("first", encoding="utf-8")
    _atomic_transfer(source, destination, mode="copy", overwrite=False)
    source.write_text("second", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _atomic_transfer(source, destination, mode="copy", overwrite=False)
    _atomic_transfer(source, destination, mode="copy", overwrite=True)
    assert destination.read_text(encoding="utf-8") == "second"


def _receipt_fixture_without_legacy_source(tmp_path):
    model_slug = "qwen3-4b-instruct-2507"
    run_root = tmp_path / model_slug
    legacy_root = tmp_path / "legacy" / model_slug
    source_paths, destination_paths, audit_mapping = _expected_receipt_inventory(
        model_slug, 0
    )
    assert len(source_paths) == EXPECTED_SOURCE_ARTIFACTS
    assert len(destination_paths) == EXPECTED_DESTINATION_ARTIFACTS
    assert len(audit_mapping) == EXPECTED_AUDIT_SOURCES

    transfer_mapping = {
        f"baseline/w2t_all/{name}": f"probes/{PROTOCOL_ID}/{name}"
        for name in TRANSFER_FILES
    }
    source_to_destination = {**transfer_mapping, **audit_mapping}
    source_hashes = {}
    destination_hashes = {}
    for source_relative, destination_relative in source_to_destination.items():
        payload = f"fixture::{source_relative}\n"
        source = legacy_root / source_relative
        destination = run_root / destination_relative
        source.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(payload, encoding="utf-8")
        destination.write_text(payload, encoding="utf-8")
        source_hashes[source_relative] = sha256_file(source)
        destination_hashes[destination_relative] = sha256_file(destination)

    for split in ("train", "test"):
        relative = (
            f"labels/{model_slug}/{split}_labels_no_reasoning_{PROTOCOL_ID}.json"
        )
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"split": split}), encoding="utf-8")
        destination_hashes[relative] = sha256_file(path)

    assert set(source_hashes) == source_paths
    assert set(destination_hashes) == destination_paths
    data_file_hashes = {}
    for split in ("train", "test"):
        relative = f"data/tasks_v1_{split}_category.json"
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"split": split}]), encoding="utf-8")
        data_file_hashes[relative] = sha256_file(path)
    data_manifest = run_root / "data" / "data_manifest.json"
    data_manifest.parent.mkdir(parents=True, exist_ok=True)
    data_manifest.write_text("{}\n", encoding="utf-8")
    receipt = {
        "source_artifact_sha256": dict(sorted(source_hashes.items())),
        "destination_artifact_sha256": dict(sorted(destination_hashes.items())),
        "destination_data_manifest_sha256": sha256_file(data_manifest),
        "destination_data_files_sha256": dict(sorted(data_file_hashes.items())),
        "audit_source": {
            "root": f"probes/{PROTOCOL_ID}/audit_source",
            "files": [
                {
                    "source_relative_path": source_relative,
                    "destination_relative_path": destination_relative,
                    "sha256": source_hashes[source_relative],
                }
                for source_relative, destination_relative in audit_mapping.items()
            ],
            "self_contained_after_legacy_source_removal": True,
        },
    }
    shutil.rmtree(legacy_root.parent)
    assert not legacy_root.exists()
    return receipt, run_root, model_slug


def test_receipt_inventory_and_original_file_validation_survive_source_deletion(
    tmp_path,
) -> None:
    receipt, run_root, model_slug = _receipt_fixture_without_legacy_source(tmp_path)
    validate_imported_scoped_destination(
        receipt,
        output_root=run_root,
        model_slug=model_slug,
        label_seed=0,
    )
    probe_dir = run_root / "probes" / PROTOCOL_ID
    validate_original_protocol_files(
        receipt=receipt,
        probe_dir=probe_dir,
        labels_path=(
            run_root
            / "labels"
            / model_slug
            / f"test_labels_no_reasoning_{PROTOCOL_ID}.json"
        ),
        run_root=run_root,
    )


def test_receipt_inventory_rejects_missing_source_and_bad_audit_mapping(tmp_path) -> None:
    receipt, run_root, model_slug = _receipt_fixture_without_legacy_source(tmp_path)
    missing_source = json.loads(json.dumps(receipt))
    missing_source["source_artifact_sha256"].pop(
        next(iter(missing_source["source_artifact_sha256"]))
    )
    with pytest.raises(ValueError, match="fixed 17-file protocol"):
        validate_imported_scoped_destination(
            missing_source,
            output_root=run_root,
            model_slug=model_slug,
            label_seed=0,
        )

    bad_audit = json.loads(json.dumps(receipt))
    bad_audit["audit_source"]["files"][0]["destination_relative_path"] = (
        bad_audit["audit_source"]["files"][1]["destination_relative_path"]
    )
    with pytest.raises(ValueError, match="duplicate paths|audit mapping"):
        validate_imported_scoped_destination(
            bad_audit,
            output_root=run_root,
            model_slug=model_slug,
            label_seed=0,
        )

    (run_root / "data" / "tasks_v1_test_category.json").write_text(
        '[{"split":"test","drifted":true}]', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="data file .* SHA256"):
        validate_imported_scoped_destination(
            receipt,
            output_root=run_root,
            model_slug=model_slug,
            label_seed=0,
        )
