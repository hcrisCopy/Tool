from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import when2tool_action.formal_stage_audit as formal_stage_audit_module
from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.evaluation_contract import classify_action_outcome
from when2tool_action.formal_stage_audit import (
    ANALYSIS_SPECS,
    AUDIT_RECEIPT_RELATIVE,
    BEHAVIOR_MODES,
    FULL_SETTINGS,
    PUBLISHED_ANALYSIS_FILES,
    RELABEL_ALLOWED_ROW_MUTATIONS,
    RELABEL_PROTOCOL,
    SCOPED_PROMPT_SETTINGS,
    FormalProtocol,
    _default_scoped_menu_builder,
    _immutable_runs_sha,
    write_formal_stage_audit,
)
from when2tool_action.io_utils import canonical_json_sha256


BEHAVIOR_COMMIT = "a" * 40
STATISTICS_COMMIT = "b" * 40
FULL_MENU_SHA = "f" * 64
SCOPED_MENU_SHA_BY_ID = {0: "0" * 64, 1: "1" * 64}


def _test_scoped_menu_builder(task) -> str:
    return SCOPED_MENU_SHA_BY_ID[task["id"]]


@pytest.fixture(autouse=True)
def _stub_local_scoped_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # The desktop Anaconda NumPy binary crashes while importing the full pinned
    # env registry. Production/CLI keeps the real lazy builder; focused tests
    # replace only that dependency and separately verify delegation below.
    monkeypatch.setattr(
        formal_stage_audit_module,
        "_default_scoped_menu_builder",
        _test_scoped_menu_builder,
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FormalFixture:
    root: Path
    repository: Path
    config: Path
    formal: FormalProtocol

    @property
    def audit_output(self) -> Path:
        return self.root.joinpath(*AUDIT_RECEIPT_RELATIVE.parts)


def _label_payload(split: str, *, original: bool, model: str, tool_scope: str) -> dict:
    if split == "train":
        rows = [
            {
                "id": 100,
                "category": "A",
                "difficulty": "easy",
                "gold_action": "A",
                "tool_necessary": 1,
                "no_tool_correct": 0,
            }
        ]
    elif original:
        rows = [
            {
                "id": 0,
                "category": "A",
                "difficulty": "easy",
                "gold_action": "NONE",
                "tool_necessary": 0,
                "no_tool_correct": 1,
            },
            {
                "id": 1,
                "category": "B",
                "difficulty": "hard",
                "gold_action": "B",
                "tool_necessary": 1,
                "no_tool_correct": 0,
            },
        ]
    else:
        rows = [
            {
                "id": 0,
                "category": "A",
                "difficulty": "easy",
                "gold_action": "A",
                "tool_necessary": 1,
                "no_tool_correct": 0,
            },
            {
                "id": 1,
                "category": "B",
                "difficulty": "hard",
                "gold_action": "NONE",
                "tool_necessary": 0,
                "no_tool_correct": 1,
            },
        ]
    for row in rows:
        row.update(
            {
                "split": split,
                "seed": 0,
                "prompt_mode": "hard_no_tool",
                "reasoning_mode": "no_reasoning",
                "tool_scope": tool_scope,
            }
        )
        if original:
            row["label_protocol"] = RELABEL_PROTOCOL
    result = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": model,
        "split": split,
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": tool_scope,
        "rows": rows,
        "n": len(rows),
    }
    if original:
        result["protocol_id"] = RELABEL_PROTOCOL
        result["adaptation_status"] = "original-pinned-not-current-scoped-adapted"
    return result


def _probe_decision(task_id: int, threshold: float) -> dict:
    logit = -2.0 if task_id == 0 else 2.0
    probability = 1.0 / (1.0 + math.exp(-(logit / 2.0)))
    use_tool = probability >= threshold
    return {
        "probe_logit": logit,
        "probe_probability": probability,
        "probe_temperature": 2.0,
        "probe_threshold": threshold,
        "probe_decision": "use_tool" if use_tool else "no_tool",
        "probe_prefill": (
            "I need to use a tool for this question.\n"
            if use_tool
            else "I can solve this directly without using a tool.\n"
        ),
    }


def _task_payload(task_id: int, *, tool_scope: str) -> dict:
    if task_id == 1:
        category = "B"
        environment = "RetrieverEnv"
        tools = ["search_corpus", "read_doc"]
        parameters = {"corpus": []}
        difficulty = "hard"
    else:
        category = "A"
        environment = "CalculatorEnv"
        tools = ["evaluate_expression"]
        parameters = {}
        difficulty = "easy"
    return {
        "id": task_id,
        "difficulty": difficulty,
        "multi_step": False,
        "instruction": f"fixture task {task_id}",
        "environments": [
            {
                "name": environment,
                "tools": tools,
                "parameters": parameters,
            }
        ],
        "expected": {"answer": "fixture"},
        "tags": [],
        "category": category,
        "category_name": "fixture-category",
        "gold_env_name": environment,
        "gold_tools": tools,
        "tool_scope": tool_scope,
    }


def _behavior_payload(
    *,
    setting: str,
    tool_scope: str,
    labels: dict,
    labels_sha: str,
    data_sha: str,
    runtime_sha: str,
    config_sha: str,
    formal: FormalProtocol,
    scoped_menu_by_id: dict[int, str],
    probe_protocol: str | None = None,
    probe_scope: str | None = None,
    threshold: float | None = None,
    probe_inputs_sha256: dict[str, str] | None = None,
) -> dict:
    rows_by_id = {row["id"]: row for row in labels["rows"]}
    task_ids = [0, 1]
    config = {
        "model": formal.model_slug,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
        "labels_sha256": labels_sha,
        "runtime_provenance_sha256": runtime_sha,
        "project_git_commit": BEHAVIOR_COMMIT,
        "setting": setting,
        "tool_scope": tool_scope,
        "prompt_mode": BEHAVIOR_MODES[setting].prompt_mode,
        "reasoning_mode": BEHAVIOR_MODES[setting].reasoning_mode,
        "record_mode": BEHAVIOR_MODES[setting].record_mode,
        "seeds": list(formal.seeds),
        "full_menu_sha256": FULL_MENU_SHA,
        "task_ids_sha256": canonical_json_sha256(task_ids),
        "smoke": False,
        "max_rounds": formal.max_rounds,
    }
    if threshold is not None:
        if probe_inputs_sha256 is None:
            raise AssertionError("Probe inputs are required for Probe&Prefill")
        decisions = [
            {"id": task_id, **_probe_decision(task_id, threshold)}
            for task_id in task_ids
        ]
        config.update(
            {
                "probe_protocol": probe_protocol,
                "probe_scope": probe_scope,
                "probe_training_label_seed": 0,
                "probe_inputs_sha256": probe_inputs_sha256,
                "probe_threshold": threshold,
                "probe_temperature": formal.probe_temperature,
                "probe_decisions_sha256": canonical_json_sha256(decisions),
                "prefill_mode": "soft",
            }
        )
    runs = []
    for run_index, seed in enumerate(formal.seeds):
        run_id = f"run_{run_index}_seed_{seed}"
        rows = []
        for task_id in task_ids:
            label = rows_by_id[task_id]
            final_correct = task_id == 0
            pred_action = "NONE"
            row = {
                "schema_version": SCHEMA_VERSION,
                "id": task_id,
                "run_id": run_id,
                "seed": seed,
                "setting": setting,
                "tool_scope": tool_scope,
                "category": label["category"],
                "difficulty": label["difficulty"],
                "gold_action": label["gold_action"],
                "tool_necessary": label["tool_necessary"],
                "no_tool_correct": label["no_tool_correct"],
                "menu_sha256": (
                    FULL_MENU_SHA
                    if tool_scope == "full"
                    else scoped_menu_by_id[task_id]
                ),
                "routed_tool_events": [],
                "tool_calls": 0,
                "total_tool_calls": 0,
                "final_correct": final_correct,
                "pred_action": pred_action,
                "tool_call_categories": [],
                "unique_tool_call_categories": [],
                "n_tool_call_categories": 0,
                "mixed_category_calls": False,
                "invalid_tool_calls": 0,
                "first_tool_name": None,
                "first_tool_category": None,
                "first_tool_environment": None,
                "first_env_correct": False,
                "exact_tool_allowed": False,
                "first_arguments_valid": False,
                "gold_env_name": f"env-{label['category']}",
                "gold_tools": [f"tool-{label['category']}"],
                "episode_done": True,
                "termination_reason": "boxed_answer",
                "tool_parse_failures": 0,
                "error_type": classify_action_outcome(
                    label["gold_action"], pred_action, final_correct, False
                ),
            }
            if threshold is not None:
                row.update(_probe_decision(task_id, threshold))
            rows.append(row)
        runs.append({"run_id": run_id, "seed": seed, "setting": setting, "rows": rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "config": config,
        "runs": runs,
    }


def _make_fixture(tmp_path: Path) -> FormalFixture:
    formal = FormalProtocol(train_task_count=1, task_count=2)
    repository = tmp_path / "code"
    config = repository / "when2tool_action/configs/qwen.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("formal: true\n", encoding="utf-8")
    config_sha = _sha(config)
    root = tmp_path / "run"
    root.mkdir()

    data_dir = root / "data"
    data_dir.mkdir()
    scoped_test_tasks = [
        _task_payload(task_id, tool_scope="scoped") for task_id in (0, 1)
    ]
    _write_json(
        data_dir / "tasks_v1_train_category.json",
        [_task_payload(100, tool_scope="scoped")],
    )
    _write_json(data_dir / "tasks_v1_test_category.json", scoped_test_tasks)
    _write_json(
        data_dir / "tasks_v1_test_fulltools_category.json",
        [_task_payload(task_id, tool_scope="full") for task_id in (0, 1)],
    )
    _write_json(data_dir / "data_manifest.json", {"schema_version": SCHEMA_VERSION})
    scoped_menu_by_id = dict(SCOPED_MENU_SHA_BY_ID)

    labels_dir = root / "labels" / formal.model_slug
    label_payloads: dict[str, dict] = {}
    for stem, original in (
        ("fulltools", False),
        ("scoped", False),
        (RELABEL_PROTOCOL, True),
    ):
        for split in ("train", "test"):
            payload = _label_payload(
                split,
                original=original,
                model=formal.model_slug,
                tool_scope="full" if stem == "fulltools" else "scoped",
            )
            path = labels_dir / f"{split}_labels_no_reasoning_{stem}.json"
            _write_json(path, payload)
            label_payloads[f"{stem}:{split}"] = payload

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "runtime-and-input-provenance",
        "upstream_commit": UPSTREAM_COMMIT,
        "git": {"commit": BEHAVIOR_COMMIT, "worktree_clean": True},
        "config": {
            "path": config.relative_to(repository).as_posix(),
            "sha256": config_sha,
            "model_slug": formal.model_slug,
        },
        "full_menu_sha256": FULL_MENU_SHA,
    }
    provenance_path = root / "manifests/runtime_provenance.json"
    _write_json(provenance_path, provenance)
    runtime_sha = _sha(provenance_path)

    probe_inputs_by_directory: dict[str, dict[str, str]] = {}
    for probe_directory in ("fulltools", "scoped", "scoped_original_w2t"):
        directory = root / "probes" / probe_directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "probe_no_reasoning.pt").write_bytes(
            f"{probe_directory}:probe\n".encode()
        )
        (directory / "test_hidden_no_reasoning.pt").write_bytes(
            f"{probe_directory}:hidden\n".encode()
        )
        _write_json(
            directory / "test_labels_no_reasoning.json",
            {"task_meta": [{"id": 0}, {"id": 1}]},
        )

    original_probe_dir = root / "probes" / RELABEL_PROTOCOL
    original_test_labels = (
        labels_dir / f"test_labels_no_reasoning_{RELABEL_PROTOCOL}.json"
    )
    migration_receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": RELABEL_PROTOCOL,
        "model": formal.model_slug,
        "upstream_commit": UPSTREAM_COMMIT,
        "config_sha256": config_sha,
        "compatibility": {
            "probe_scope": "scoped-original-pinned",
            "current_scoped_adapted": False,
            "allowed_claim": "original When2Tool scoped binary baseline reproduction",
        },
        "probe_validation": {
            "C": 0.0001,
            "layer": "all",
            "n_layers": 37,
            "hidden_dim": 2560,
            "official_double_standard_scaler_reconstructed": True,
            "all_saved_metrics_recomputed": True,
        },
        "splits": {
            "train": {},
            "test": {
                "n": formal.task_count,
                "task_ids_sha256": canonical_json_sha256([0, 1]),
                "hidden_shape": [formal.task_count, 37, 2560],
                "hidden_dtype": "torch.float32",
            },
        },
        "destination_artifact_sha256": {
            path.relative_to(root).as_posix(): _sha(path)
            for path in (
                original_probe_dir / "probe_no_reasoning.pt",
                original_probe_dir / "test_hidden_no_reasoning.pt",
                original_probe_dir / "test_labels_no_reasoning.json",
                original_test_labels,
            )
        },
        "destination_data_files_sha256": {
            (data_dir / "tasks_v1_train_category.json")
            .relative_to(root)
            .as_posix(): _sha(data_dir / "tasks_v1_train_category.json"),
            (data_dir / "tasks_v1_test_category.json")
            .relative_to(root)
            .as_posix(): _sha(data_dir / "tasks_v1_test_category.json"),
        },
        "destination_data_manifest_sha256": _sha(data_dir / "data_manifest.json"),
    }
    _write_json(original_probe_dir / "migration_receipt.json", migration_receipt)
    for probe_directory in ("fulltools", "scoped", RELABEL_PROTOCOL):
        names = [
            "probe_no_reasoning.pt",
            "test_hidden_no_reasoning.pt",
            "test_labels_no_reasoning.json",
        ]
        if probe_directory == RELABEL_PROTOCOL:
            names.append("migration_receipt.json")
        probe_inputs_by_directory[probe_directory] = {
            name: _sha(root / "probes" / probe_directory / name) for name in names
        }

    # Direct fulltools/scoped-adapted behavior and original P&P behavior.
    for spec in ANALYSIS_SPECS:
        labels_path = labels_dir / f"test_labels_no_reasoning_{spec.label_stem}.json"
        labels = _read(labels_path)
        data_path = data_dir / spec.data_filename
        for setting in spec.settings:
            if spec.protocol == RELABEL_PROTOCOL and setting in SCOPED_PROMPT_SETTINGS:
                continue
            threshold = None
            if setting.startswith("probe_prefill_t"):
                threshold = float(setting.split("_t", 1)[1].split("_", 1)[0])
            if spec.protocol == "fulltools":
                probe_protocol, probe_scope = "adapted", "full-adapted"
                probe_directory = "fulltools"
            elif spec.protocol == "scoped_adapted":
                probe_protocol, probe_scope = "adapted", "scoped-adapted"
                probe_directory = "scoped"
            else:
                probe_protocol, probe_scope = RELABEL_PROTOCOL, "scoped-original-pinned"
                probe_directory = RELABEL_PROTOCOL
            payload = _behavior_payload(
                setting=setting,
                tool_scope=spec.tool_scope,
                labels=labels,
                labels_sha=_sha(labels_path),
                data_sha=_sha(data_path),
                runtime_sha=runtime_sha,
                config_sha=config_sha,
                formal=formal,
                scoped_menu_by_id=scoped_menu_by_id,
                probe_protocol=probe_protocol,
                probe_scope=probe_scope,
                threshold=threshold,
                probe_inputs_sha256=(
                    probe_inputs_by_directory[probe_directory]
                    if threshold is not None
                    else None
                ),
            )
            _write_json(
                root / "outputs" / spec.output_directory / f"{setting}.json", payload
            )

    # Strict derived copies for the ten original-W2T scoped prompt settings.
    source_labels_path = labels_dir / "test_labels_no_reasoning_scoped.json"
    target_labels_path = (
        labels_dir / f"test_labels_no_reasoning_{RELABEL_PROTOCOL}.json"
    )
    target_by_id = {row["id"]: row for row in _read(target_labels_path)["rows"]}
    relabel_artifacts = []
    for setting in SCOPED_PROMPT_SETTINGS:
        source_path = root / "outputs/scoped_adapted" / f"{setting}.json"
        target_path = root / "outputs/scoped_original_w2t" / f"{setting}.json"
        source = _read(source_path)
        source_sha = _sha(source_path)
        target = copy.deepcopy(source)
        source_immutable = _immutable_runs_sha(source["runs"])
        changes = {key: 0 for key in RELABEL_ALLOWED_ROW_MUTATIONS}
        for run in target["runs"]:
            for row in run["rows"]:
                label = target_by_id[row["id"]]
                replacements = {
                    "gold_action": label["gold_action"],
                    "error_type": classify_action_outcome(
                        label["gold_action"],
                        row["pred_action"],
                        row["final_correct"],
                        row["invalid_tool_calls"] > 0,
                    ),
                    "tool_necessary": label["tool_necessary"],
                    "no_tool_correct": label["no_tool_correct"],
                }
                for key, value in replacements.items():
                    changes[key] += int(row[key] != value)
                    row[key] = value
        target["config"].update(
            {
                "labels_sha256": _sha(target_labels_path),
                "derivation_protocol_id": RELABEL_PROTOCOL,
                "source_evaluation_sha256": source_sha,
                "source_labels_sha256": _sha(source_labels_path),
                "target_labels_sha256": _sha(target_labels_path),
            }
        )
        target["derivation"] = {
            "derivation_type": "scoped-behavior-gold-action-relabel",
            "protocol_id": RELABEL_PROTOCOL,
            "source_filename": source_path.name,
            "source_sha256": source_sha,
            "source_labels_filename": source_labels_path.name,
            "source_labels_sha256": _sha(source_labels_path),
            "target_labels_filename": target_labels_path.name,
            "target_labels_sha256": _sha(target_labels_path),
            "runtime_provenance_sha256": runtime_sha,
            "project_git_commit": BEHAVIOR_COMMIT,
            "allowed_row_mutations": list(RELABEL_ALLOWED_ROW_MUTATIONS),
            "immutable_rows_sha256": source_immutable,
            "n_runs": len(formal.seeds),
            "n_task_ids": formal.task_count,
            "task_ids_sha256": canonical_json_sha256([0, 1]),
        }
        _write_json(target_path, target)
        relabel_artifacts.append(
            {
                "setting": setting,
                "source_filename": source_path.name,
                "source_sha256": source_sha,
                "output_filename": target_path.name,
                "output_sha256": _sha(target_path),
                "immutable_rows_sha256": source_immutable,
                "error_type_changed_rows": changes["error_type"],
                "tool_necessary_changed_rows": changes["tool_necessary"],
                "no_tool_correct_changed_rows": changes["no_tool_correct"],
            }
        )
    relabel_receipt = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "scoped-behavior-relabel-receipt",
        "protocol_id": RELABEL_PROTOCOL,
        "model": formal.model_slug,
        "tool_scope": "scoped",
        "expected_seeds": list(formal.seeds),
        "n_task_ids": formal.task_count,
        "task_ids_sha256": canonical_json_sha256([0, 1]),
        "n_source_files": len(SCOPED_PROMPT_SETTINGS),
        "changed_task_count": 2,
        "allowed_row_mutations": list(RELABEL_ALLOWED_ROW_MUTATIONS),
        "source_labels": {
            "filename": source_labels_path.name,
            "sha256": _sha(source_labels_path),
        },
        "target_labels": {
            "filename": target_labels_path.name,
            "sha256": _sha(target_labels_path),
        },
        "runtime_provenance": {
            "filename": provenance_path.name,
            "sha256": runtime_sha,
            "project_git_commit": BEHAVIOR_COMMIT,
        },
        "artifacts": sorted(relabel_artifacts, key=lambda item: item["setting"]),
    }
    _write_json(
        root / "outputs/scoped_original_w2t/relabel_receipt.json", relabel_receipt
    )

    # Statistics publications are exact directories with 34 hash-bound files.
    for spec in ANALYSIS_SPECS:
        directory = root / "analysis" / spec.protocol
        published = []
        for name in PUBLISHED_ANALYSIS_FILES:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{spec.protocol}:{name}\n".encode())
            published.append({"path": name, "sha256": _sha(path)})
        behavior_paths = [
            root / "outputs" / spec.output_directory / f"{setting}.json"
            for setting in spec.settings
        ]
        train_label = labels_dir / f"train_labels_no_reasoning_{spec.label_stem}.json"
        test_label = labels_dir / f"test_labels_no_reasoning_{spec.label_stem}.json"
        summary = {
            "schema_version": "when2tool_action_stats.v3",
            "manifest_type": "action-statistics-publication-receipt",
            "publication_complete": True,
            "analysis_protocol": spec.protocol,
            "label_protocol": spec.label_protocol,
            "behavior_generation_git_commit": BEHAVIOR_COMMIT,
            "statistics_code_git_commit": STATISTICS_COMMIT,
            "expected_seeds": list(formal.seeds),
            "expected_settings": list(spec.settings),
            "n_rows": spec.expected_rows(formal),
            "n_task_ids": formal.task_count,
            "n_settings": len(spec.settings),
            "n_runs": spec.expected_runs(formal),
            "paired_bootstrap": {
                "n_bootstrap": formal.bootstrap_samples,
                "seed": formal.bootstrap_seed,
            },
            "settings": {
                setting: {"n_runs": len(formal.seeds), "metrics": {}}
                for setting in spec.settings
            },
            "input_files": [str(path.resolve()) for path in behavior_paths],
            "input_artifacts": [
                {
                    "path": str(
                        (
                            root / "outputs" / spec.output_directory / f"{setting}.json"
                        ).resolve()
                    ),
                    "sha256": _sha(
                        root / "outputs" / spec.output_directory / f"{setting}.json"
                    ),
                    "setting": setting,
                    "model": formal.model_slug,
                    "tool_scope": spec.tool_scope,
                    "labels_sha256": _sha(test_label),
                    "runtime_provenance_sha256": runtime_sha,
                    "project_git_commit": BEHAVIOR_COMMIT,
                }
                for setting in spec.settings
            ],
            "labels_file": None,
            "labels_files": [
                {"path": str(train_label.resolve()), "sha256": _sha(train_label)},
                {"path": str(test_label.resolve()), "sha256": _sha(test_label)},
            ],
            "referenced_inputs": {
                "data": {
                    "path": str((data_dir / spec.data_filename).resolve()),
                    "sha256": _sha(data_dir / spec.data_filename),
                },
                "runtime_provenance": {
                    "path": str(provenance_path.resolve()),
                    "sha256": runtime_sha,
                },
            },
            "published_files": published,
        }
        _write_json(directory / "summary.json", summary)

    return FormalFixture(root=root, repository=repository, config=config, formal=formal)


def _no_op_migration(receipt, context):
    assert receipt["destination_artifact_sha256"]
    assert context.root.is_dir()
    assert context.model_slug == "qwen3-4b-instruct-2507"
    assert context.config_sha256 == _sha(
        context.root.parent / "code/when2tool_action/configs/qwen.yaml"
    )
    assert context.task_ids == (0, 1)


def _rewrite_json(path: Path, mutate) -> None:
    payload = _read(path)
    mutate(payload)
    _write_json(path, payload)


def _rebind_behavior_summary(fixture: FormalFixture, behavior_path: Path) -> None:
    relative_parts = behavior_path.relative_to(fixture.root).parts
    output_directory = relative_parts[1]
    spec = next(
        spec for spec in ANALYSIS_SPECS if spec.output_directory == output_directory
    )
    summary_path = fixture.root / "analysis" / spec.protocol / "summary.json"
    summary = _read(summary_path)
    for item in summary["input_artifacts"]:
        if Path(item["path"]).resolve() == behavior_path.resolve():
            item["sha256"] = _sha(behavior_path)
            break
    else:
        raise AssertionError(behavior_path)
    _write_json(summary_path, summary)


def _audit_fixture(
    fixture: FormalFixture, *, migration_validator=_no_op_migration
) -> dict:
    return write_formal_stage_audit(
        fixture.root,
        config_path=fixture.config,
        output=fixture.audit_output,
        behavior_commit=BEHAVIOR_COMMIT,
        statistics_commit=STATISTICS_COMMIT,
        repository_root=fixture.repository,
        formal=fixture.formal,
        migration_validator=migration_validator,
        scoped_menu_builder=_test_scoped_menu_builder,
    )


def test_complete_formal_stage_is_audited_and_overwrite_is_explicit(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = write_formal_stage_audit(
        fixture.root,
        config_path=fixture.config,
        output=fixture.audit_output,
        behavior_commit=BEHAVIOR_COMMIT,
        statistics_commit=STATISTICS_COMMIT,
        repository_root=fixture.repository,
        formal=fixture.formal,
        migration_validator=_no_op_migration,
    )
    assert payload["audit_complete"] is True
    assert payload["behavior"]["n_artifacts"] == 38
    assert payload["statistics"]["summary_count"] == 3
    assert len(payload["probe_prefill"]) == 3
    assert payload["migration"]["destination_only_validation_passed"] is True
    assert _read(fixture.audit_output) == payload

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )
    second = write_formal_stage_audit(
        fixture.root,
        config_path=fixture.config,
        output=fixture.audit_output,
        behavior_commit=BEHAVIOR_COMMIT,
        statistics_commit=STATISTICS_COMMIT,
        repository_root=fixture.repository,
        formal=fixture.formal,
        migration_validator=_no_op_migration,
        overwrite=True,
    )
    assert second == payload


def test_failed_overwrite_preserves_previous_audit_receipt(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    write_formal_stage_audit(
        fixture.root,
        config_path=fixture.config,
        output=fixture.audit_output,
        behavior_commit=BEHAVIOR_COMMIT,
        statistics_commit=STATISTICS_COMMIT,
        repository_root=fixture.repository,
        formal=fixture.formal,
        migration_validator=_no_op_migration,
    )
    before = fixture.audit_output.read_bytes()
    behavior = fixture.root / "outputs/fulltools" / f"{FULL_SETTINGS[0]}.json"
    _rewrite_json(
        behavior,
        lambda payload: payload["config"].__setitem__("max_rounds", 11),
    )
    with pytest.raises(ValueError, match="max_rounds"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
            overwrite=True,
        )
    assert fixture.audit_output.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("max_rounds", 11, "max_rounds"),
        ("seeds", [0, 1], "seeds"),
        ("project_git_commit", "c" * 40, "project_git_commit"),
        ("config_sha256", "d" * 64, "config_sha256"),
        ("setting", "renamed_setting", "setting"),
    ),
)
def test_behavior_protocol_mutations_fail_closed(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools" / f"{FULL_SETTINGS[0]}.json"
    _rewrite_json(behavior, lambda payload: payload["config"].__setitem__(field, value))
    with pytest.raises(ValueError, match=match):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )
    assert not fixture.audit_output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prompt_mode", "necessary_tool"),
        ("reasoning_mode", "reasoning"),
        ("record_mode", "full"),
    ),
)
def test_each_setting_is_bound_to_its_registered_modes(
    tmp_path: Path, field: str, value: str
) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools/current_no_reasoning_fulltools.json"
    _rewrite_json(
        behavior,
        lambda payload: payload["config"].__setitem__(field, value),
    )
    with pytest.raises(ValueError, match=field):
        _audit_fixture(fixture)


def test_behavior_rows_bind_full_and_scoped_menu_hashes(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    full = fixture.root / "outputs/fulltools/current_no_reasoning_fulltools.json"
    _rewrite_json(
        full,
        lambda payload: payload["runs"][0]["rows"][0].__setitem__(
            "menu_sha256", "0" * 64
        ),
    )
    with pytest.raises(ValueError, match="provenance full menu"):
        _audit_fixture(fixture)

    fixture = _make_fixture(tmp_path / "scoped")
    scoped = fixture.root / "outputs/scoped_adapted/current_no_reasoning_scoped.json"
    _rewrite_json(
        scoped,
        lambda payload: payload["runs"][1]["rows"][0].__setitem__(
            "menu_sha256", "1" * 64
        ),
    )
    with pytest.raises(ValueError, match="rebuilt from pinned scoped task data"):
        _audit_fixture(fixture)


def test_default_scoped_menu_builder_delegates_to_production_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from when2tool_action import upstream

    calls: list[tuple[dict, str]] = []

    def fake_build_environments(task, tool_scope):
        calls.append((task, tool_scope))
        return SimpleNamespace(menu_sha256="a" * 64)

    monkeypatch.setattr(upstream, "build_environments", fake_build_environments)
    task = _task_payload(0, tool_scope="scoped")
    assert _default_scoped_menu_builder(task) == "a" * 64
    assert calls == [(task, "scoped")]


def test_uniform_but_wrong_scoped_menu_panel_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    for directory in ("scoped_adapted", "scoped_original_w2t"):
        for path in (fixture.root / "outputs" / directory).glob("*.json"):
            if path.name == "relabel_receipt.json":
                continue

            def mutate(payload):
                for run in payload["runs"]:
                    for row in run["rows"]:
                        row["menu_sha256"] = FULL_MENU_SHA

            _rewrite_json(path, mutate)

    with pytest.raises(ValueError, match="rebuilt from pinned scoped task data"):
        _audit_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("tool_calls", 1, "routed_tool_events"),
        ("pred_action", "A", "pred_action"),
        ("error_type", "over_call", "error_type"),
    ),
)
def test_every_behavior_row_must_satisfy_action_contract(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools/current_no_reasoning_fulltools.json"
    _rewrite_json(
        behavior,
        lambda payload: payload["runs"][0]["rows"][0].__setitem__(field, value),
    )
    with pytest.raises(ValueError, match=match):
        _audit_fixture(fixture)


def test_behavior_requires_three_complete_exact_id_runs(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools" / f"{FULL_SETTINGS[0]}.json"

    def mutate(payload):
        payload["runs"][2]["rows"].pop()

    _rewrite_json(behavior, mutate)
    with pytest.raises(ValueError, match="expected 2 rows"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )


def test_summary_publication_hash_and_commit_are_bound(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    published = fixture.root / "analysis/fulltools/per_run_metrics.csv"
    published.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )
    assert not fixture.audit_output.exists()

    fixture = _make_fixture(tmp_path / "second")
    summary = fixture.root / "analysis/fulltools/summary.json"
    _rewrite_json(
        summary,
        lambda payload: payload.__setitem__("statistics_code_git_commit", "c" * 40),
    )
    with pytest.raises(ValueError, match="statistics_code_git_commit"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )


def test_summary_file_panels_are_keyed_not_positionally_zipped(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    summary_path = fixture.root / "analysis/fulltools/summary.json"

    def reorder(payload):
        payload["expected_settings"].reverse()
        payload["input_artifacts"].reverse()
        payload["input_files"].reverse()
        payload["labels_files"].reverse()

    _rewrite_json(summary_path, reorder)
    payload = _audit_fixture(fixture)
    assert payload["audit_complete"] is True


def test_summary_expected_settings_reordering_cannot_change_exact_set(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    summary_path = fixture.root / "analysis/fulltools/summary.json"

    def mutate(payload):
        payload["expected_settings"][-1] = payload["expected_settings"][0]

    _rewrite_json(summary_path, mutate)
    with pytest.raises(ValueError, match="exact registered set"):
        _audit_fixture(fixture)


def test_train_label_count_is_fixed_by_formal_protocol(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    labels_path = (
        fixture.root
        / "labels"
        / fixture.formal.model_slug
        / "train_labels_no_reasoning_fulltools.json"
    )
    labels = _read(labels_path)
    extra = copy.deepcopy(labels["rows"][0])
    extra["id"] = 101
    labels["rows"].append(extra)
    labels["n"] = len(labels["rows"])
    _write_json(labels_path, labels)

    summary_path = fixture.root / "analysis/fulltools/summary.json"
    summary = _read(summary_path)
    for row in summary["labels_files"]:
        if Path(row["path"]).resolve() == labels_path.resolve():
            row["sha256"] = _sha(labels_path)
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="expected 1 train label rows"):
        _audit_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("probe_probability", 0.3, "vary across seed/threshold"),
        ("probe_logit", -0.5, "vary across seed/threshold"),
        ("probe_temperature", 3.0, "temperature differs from protocol"),
    ),
)
def test_probe_values_must_be_invariant_across_seeds(
    tmp_path: Path, field: str, value: float, match: str
) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools/probe_prefill_t0.1_fulltools.json"

    def mutate(payload):
        payload["runs"][1]["rows"][0][field] = value
        if field == "probe_probability":
            payload["runs"][1]["rows"][0]["probe_logit"] = payload["runs"][1]["rows"][
                0
            ]["probe_temperature"] * math.log(value / (1.0 - value))
        elif field == "probe_logit":
            temperature = payload["runs"][1]["rows"][0]["probe_temperature"]
            payload["runs"][1]["rows"][0]["probe_probability"] = 1.0 / (
                1.0 + math.exp(-(value / temperature))
            )

    _rewrite_json(behavior, mutate)
    _rebind_behavior_summary(fixture, behavior)
    with pytest.raises(ValueError, match=match):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )


def test_probe_values_must_be_invariant_across_thresholds(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools/probe_prefill_t0.3_fulltools.json"

    def mutate(payload):
        for run in payload["runs"]:
            run["rows"][0]["probe_probability"] = 0.35
            run["rows"][0]["probe_logit"] = 2.0 * math.log(0.35 / 0.65)
            run["rows"][0]["probe_decision"] = "use_tool"
            run["rows"][0]["probe_prefill"] = (
                "I need to use a tool for this question.\n"
            )
        decision_keys = (
            "probe_logit",
            "probe_probability",
            "probe_temperature",
            "probe_threshold",
            "probe_decision",
            "probe_prefill",
        )
        decisions = [
            {"id": row["id"], **{key: row[key] for key in decision_keys}}
            for row in payload["runs"][0]["rows"]
        ]
        payload["config"]["probe_decisions_sha256"] = canonical_json_sha256(decisions)

    _rewrite_json(behavior, mutate)
    _rebind_behavior_summary(fixture, behavior)
    with pytest.raises(ValueError, match="vary across seed/threshold"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )


def test_probe_decision_must_match_threshold(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/scoped_adapted/probe_prefill_t0.9_scoped.json"

    def mutate(payload):
        payload["runs"][1]["rows"][1]["probe_decision"] = "use_tool"
        payload["runs"][1]["rows"][1]["probe_prefill"] = (
            "I need to use a tool for this question.\n"
        )

    _rewrite_json(behavior, mutate)
    _rebind_behavior_summary(fixture, behavior)
    with pytest.raises(ValueError, match="decision/prefill violates threshold"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda config: config.__setitem__("probe_training_label_seed", 1),
            "probe_training_label_seed",
        ),
        (
            lambda config: config["probe_inputs_sha256"].pop(
                "test_hidden_no_reasoning.pt"
            ),
            "must contain exactly",
        ),
        (
            lambda config: config["probe_inputs_sha256"].__setitem__(
                "probe_no_reasoning.pt", "0" * 64
            ),
            "SHA256 mismatch",
        ),
    ),
)
def test_probe_config_binds_seed_exact_input_keys_and_real_hashes(
    tmp_path: Path, mutation, match: str
) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools/probe_prefill_t0.1_fulltools.json"
    _rewrite_json(behavior, lambda payload: mutation(payload["config"]))
    _rebind_behavior_summary(fixture, behavior)
    with pytest.raises(ValueError, match=match):
        _audit_fixture(fixture)


def test_original_probe_input_panel_requires_migration_receipt(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = (
        fixture.root
        / "outputs/scoped_original_w2t/probe_prefill_t0.1_scoped_original_w2t.json"
    )

    def mutate(payload):
        payload["config"]["probe_inputs_sha256"].pop("migration_receipt.json")

    _rewrite_json(behavior, mutate)
    _rebind_behavior_summary(fixture, behavior)
    with pytest.raises(ValueError, match="migration_receipt.json"):
        _audit_fixture(fixture)


def test_probe_probability_must_equal_clipped_temperature_sigmoid(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    behavior = fixture.root / "outputs/fulltools/probe_prefill_t0.1_fulltools.json"

    def mutate(payload):
        payload["runs"][0]["rows"][0]["probe_probability"] += 0.01

    _rewrite_json(behavior, mutate)
    _rebind_behavior_summary(fixture, behavior)
    with pytest.raises(ValueError, match="sigmoid"):
        _audit_fixture(fixture)


def test_probe_input_file_tampering_fails_even_when_behavior_is_unchanged(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    probe = fixture.root / "probes/scoped/probe_no_reasoning.pt"
    probe.write_bytes(b"tampered probe\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _audit_fixture(fixture)


def test_relabel_immutable_fields_and_receipt_hashes_are_enforced(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    target = (
        fixture.root / "outputs/scoped_original_w2t/current_no_reasoning_scoped.json"
    )

    def mutate_target(payload):
        row = payload["runs"][0]["rows"][0]
        row["final_correct"] = False
        row["error_type"] = classify_action_outcome(
            row["gold_action"], row["pred_action"], False, False
        )

    _rewrite_json(target, mutate_target)
    _rebind_behavior_summary(fixture, target)
    receipt_path = fixture.root / "outputs/scoped_original_w2t/relabel_receipt.json"
    receipt = _read(receipt_path)
    artifact = next(
        item for item in receipt["artifacts"] if item["setting"] == target.stem
    )
    artifact["output_sha256"] = _sha(target)
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="immutable behavior fields"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )

    fixture = _make_fixture(tmp_path / "receipt")
    receipt_path = fixture.root / "outputs/scoped_original_w2t/relabel_receipt.json"
    _rewrite_json(
        receipt_path,
        lambda payload: payload["artifacts"][0].__setitem__("source_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="source_sha256"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=_no_op_migration,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("gold_action", "C", "gold_action"),
        ("tool_necessary", 1, "tool_necessary"),
        ("no_tool_correct", 0, "no_tool_correct"),
        ("error_type", "over_call", "error_type"),
    ),
)
def test_relabel_target_rows_are_bound_to_formal_target_labels(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    fixture = _make_fixture(tmp_path)
    target = (
        fixture.root / "outputs/scoped_original_w2t/current_no_reasoning_scoped.json"
    )
    _rewrite_json(
        target,
        lambda payload: payload["runs"][0]["rows"][0].__setitem__(field, value),
    )
    with pytest.raises(ValueError, match=match):
        _audit_fixture(fixture)


def test_relabel_row_lengths_and_id_order_are_explicit(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    target = (
        fixture.root / "outputs/scoped_original_w2t/current_no_reasoning_scoped.json"
    )

    def mutate(payload):
        payload["runs"][0]["rows"].reverse()

    _rewrite_json(target, mutate)
    with pytest.raises(ValueError, match="task ID order"):
        _audit_fixture(fixture)


def test_default_migration_validation_reuses_protocol_metadata_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    calls: dict[str, object] = {}
    legacy_module = ModuleType("when2tool_action.legacy_scoped")
    prefill_module = ModuleType("when2tool_action.scripts.run_probe_prefill")

    def validate_metadata(receipt, labels, **kwargs):
        calls["metadata"] = (receipt, labels, kwargs)

    def validate_destination(receipt, *, output_root, model_slug, label_seed):
        calls["destination"] = (
            receipt,
            output_root,
            model_slug,
            label_seed,
        )

    legacy_module.validate_imported_scoped_destination = validate_destination
    prefill_module.validate_original_protocol_metadata = validate_metadata
    monkeypatch.setitem(sys.modules, "when2tool_action.legacy_scoped", legacy_module)
    monkeypatch.setitem(
        sys.modules,
        "when2tool_action.scripts.run_probe_prefill",
        prefill_module,
    )

    payload = _audit_fixture(fixture, migration_validator=None)
    assert payload["migration"]["original_protocol_metadata_validation_passed"]
    _, labels, kwargs = calls["metadata"]
    assert labels["protocol_id"] == RELABEL_PROTOCOL
    assert kwargs == {
        "model_slug": fixture.formal.model_slug,
        "config_sha256": _sha(fixture.config),
        "label_seed": 0,
        "task_ids": [0, 1],
        "n_layers": 37,
        "hidden_dim": 2560,
        "probe_c": 0.0001,
    }
    assert calls["destination"][1:] == (
        fixture.root.resolve(),
        fixture.formal.model_slug,
        0,
    )


def test_migration_destination_failure_propagates_without_receipt(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)

    def fail_migration(receipt, context):
        raise ValueError("destination-only migration failed")

    with pytest.raises(ValueError, match="destination-only migration failed"):
        write_formal_stage_audit(
            fixture.root,
            config_path=fixture.config,
            output=fixture.audit_output,
            behavior_commit=BEHAVIOR_COMMIT,
            statistics_commit=STATISTICS_COMMIT,
            repository_root=fixture.repository,
            formal=fixture.formal,
            migration_validator=fail_migration,
        )
    assert not fixture.audit_output.exists()
