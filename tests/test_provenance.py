from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from when2tool_action import provenance
from when2tool_action.provenance import (
    identity_files,
    validate_provenance_snapshot,
    validate_runtime_provenance,
)
from when2tool_action.scripts import audit_provenance


def test_model_identity_inventory_is_sorted_and_excludes_cache(tmp_path: Path) -> None:
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text("not identity", encoding="utf-8")
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "ignored.json").write_text("{}", encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in identity_files(tmp_path)] == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_model_identity_requires_weights(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No safetensors"):
        identity_files(tmp_path)


def _snapshot(created: str = "2026-01-01T00:00:00+00:00") -> dict:
    return {
        "schema_version": "v1",
        "manifest_type": "runtime-and-input-provenance",
        "created_at_utc": created,
        "git": {"commit": "abc", "worktree_clean": True},
        "upstream_commit": "upstream",
        "config": {"sha256": "c" * 64},
        "full_menu_sha256": "m" * 64,
        "model_identity": {
            "files": [{"path": "model.safetensors", "bytes": 3, "sha256": "w" * 64}]
        },
        "generated_data": {"manifest_sha256": "d" * 64, "files": []},
        "runtime": {"python": "3.11"},
    }


def test_snapshot_comparison_ignores_only_registration_timestamp() -> None:
    registered = _snapshot("old")
    current = _snapshot("new")
    validate_provenance_snapshot(registered, current)

    for section in ("git", "model_identity", "generated_data", "runtime"):
        tampered = deepcopy(registered)
        tampered[section] = {"tampered": True}
        with pytest.raises(ValueError, match=section):
            validate_provenance_snapshot(tampered, current)


def test_runtime_validation_binds_the_exact_validated_manifest_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = _snapshot("registered")
    current = _snapshot("current-check-time")
    path = tmp_path / "runtime_provenance.json"
    payload = json.dumps(registered, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    monkeypatch.setattr(provenance, "build_provenance", lambda config: current)

    validated = validate_runtime_provenance(SimpleNamespace(), path)
    assert validated["git_commit"] == "abc"
    assert validated["model_identity_file_count"] == 1
    assert validated["sha256"] == provenance.sha256_bytes(payload)


def test_audit_reuses_valid_existing_manifest_without_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "runtime_provenance.json"
    target.write_text("registered\n", encoding="utf-8")
    config = SimpleNamespace(run_root=tmp_path)
    monkeypatch.setattr(audit_provenance, "load_config", lambda path: config)
    monkeypatch.setattr(audit_provenance, "require_inputs", lambda value: None)
    monkeypatch.setattr(
        audit_provenance,
        "validate_runtime_provenance",
        lambda value, path: {"sha256": "a" * 64},
    )
    monkeypatch.setattr(
        audit_provenance,
        "build_provenance",
        lambda value: pytest.fail("valid existing manifest was rebuilt"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["audit_provenance", "--config", "unused.yaml", "--output", str(target)],
    )

    audit_provenance.main()
    assert target.read_text(encoding="utf-8") == "registered\n"
