from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from when2tool_action.constants import SCHEMA_VERSION, UPSTREAM_COMMIT
from when2tool_action.io_utils import canonical_json_sha256
from when2tool_action.runtime import EvaluationSetting
from when2tool_action.scripts import run_probe_prefill
from when2tool_action.scripts.run_probe_prefill import (
    ORIGINAL_ADAPTATION_STATUS,
    ORIGINAL_ALLOWED_CLAIM,
    ORIGINAL_PROBE_SCOPE,
    ORIGINAL_W2T_PROTOCOL,
    build_probe_prefill_artifact_template,
    resolve_probe_protocol,
    validate_original_protocol_metadata,
)


def _metadata() -> tuple[dict, dict, list[int]]:
    task_ids = [11, 12]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": ORIGINAL_W2T_PROTOCOL,
        "model": "model-slug",
        "upstream_commit": UPSTREAM_COMMIT,
        "config_sha256": "a" * 64,
        "splits": {
            "train": {},
            "test": {
                "n": 2,
                "task_ids_sha256": canonical_json_sha256(task_ids),
                "hidden_shape": [2, 37, 2560],
                "hidden_dtype": "torch.float32",
            },
        },
        "probe_validation": {
            "C": 0.0001,
            "layer": "all",
            "n_layers": 37,
            "hidden_dim": 2560,
            "official_double_standard_scaler_reconstructed": True,
            "all_saved_metrics_recomputed": True,
        },
        "compatibility": {
            "probe_scope": ORIGINAL_PROBE_SCOPE,
            "current_scoped_adapted": False,
            "allowed_claim": ORIGINAL_ALLOWED_CLAIM,
        },
    }
    rows = [
        {
            "id": task_id,
            "seed": 0,
            "tool_scope": "scoped",
            "label_protocol": ORIGINAL_W2T_PROTOCOL,
        }
        for task_id in task_ids
    ]
    labels = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "model": "model-slug",
        "split": "test",
        "seed": 0,
        "prompt_mode": "hard_no_tool",
        "reasoning_mode": "no_reasoning",
        "tool_scope": "scoped",
        "protocol_id": ORIGINAL_W2T_PROTOCOL,
        "adaptation_status": ORIGINAL_ADAPTATION_STATUS,
        "n": 2,
        "rows": rows,
    }
    return receipt, labels, task_ids


def _validate(receipt: dict, labels: dict, task_ids: list[int]) -> None:
    validate_original_protocol_metadata(
        receipt,
        labels,
        model_slug="model-slug",
        config_sha256="a" * 64,
        label_seed=0,
        task_ids=task_ids,
        n_layers=37,
        hidden_dim=2560,
        probe_c=0.0001,
    )


def test_protocol_selection_is_explicit_and_keeps_adapted_names() -> None:
    assert resolve_probe_protocol("adapted", "full").output_scope == "fulltools"
    assert resolve_probe_protocol("adapted", "scoped").output_scope == "scoped"
    original = resolve_probe_protocol(ORIGINAL_W2T_PROTOCOL, "scoped")
    assert original.output_scope == "scoped_original_w2t"
    assert original.probe_scope == ORIGINAL_PROBE_SCOPE
    with pytest.raises(ValueError, match="only compatible"):
        resolve_probe_protocol(ORIGINAL_W2T_PROTOCOL, "full")


def test_original_protocol_metadata_accepts_matching_receipt_and_seed() -> None:
    receipt, labels, task_ids = _metadata()
    _validate(receipt, labels, task_ids)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt, labels: receipt.update(protocol_id="adapted"), "protocol_id"),
        (
            lambda receipt, labels: receipt["compatibility"].update(
                current_scoped_adapted=True
            ),
            "current_scoped_adapted",
        ),
        (lambda receipt, labels: labels.update(seed=1), "labels.seed"),
    ],
)
def test_original_protocol_metadata_rejects_mismatches(mutation, message: str) -> None:
    receipt, labels, task_ids = _metadata()
    receipt = deepcopy(receipt)
    labels = deepcopy(labels)
    mutation(receipt, labels)
    with pytest.raises(ValueError, match=message):
        _validate(receipt, labels, task_ids)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        source=Path("unused.yaml"),
        model=SimpleNamespace(slug="model-slug"),
        generation=SimpleNamespace(
            seeds=(0,),
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
            max_new_tokens=2048,
            label_hidden_extraction_max_rounds=12,
            behavior_evaluation_max_rounds=10,
            max_model_len=32768,
        ),
        probe=SimpleNamespace(temperature=2.0, thresholds=(0.1,)),
    )


def test_complete_resume_target_skips_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    task_ids = [101]
    setting = EvaluationSetting(
        name="probe_prefill_t0.1_fulltools",
        tool_scope="full",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="lite",
    )
    decision = {
        "probe_logit": -1.0,
        "probe_probability": 0.2,
        "probe_temperature": 2.0,
        "probe_threshold": 0.1,
        "probe_decision": "use_tool",
        "probe_prefill": "I need to use a tool for this question.\n",
    }
    monkeypatch.setattr(run_probe_prefill, "full_menu_sha256", lambda: "f" * 64)
    template = build_probe_prefill_artifact_template(
        config=config,
        setting=setting,
        probe_protocol="adapted",
        probe_scope="full-adapted",
        threshold=0.1,
        seeds=(0,),
        task_ids=task_ids,
        smoke=False,
        config_sha256="c" * 64,
        data_sha256="d" * 64,
        labels_sha256="e" * 64,
        probe_inputs_sha256={
            "probe_no_reasoning.pt": "p" * 64,
            "test_hidden_no_reasoning.pt": "p" * 64,
            "test_labels_no_reasoning.json": "p" * 64,
        },
        probe_decisions_sha256=canonical_json_sha256([{"id": 101, **decision}]),
        runtime_provenance_sha256="r" * 64,
        project_git_commit="commit",
    )
    complete = deepcopy(template)
    complete["runs"] = [
        {
            "run_id": "run_0_seed_0",
            "seed": 0,
            "setting": setting.name,
            "rows": [
                {
                    "id": 101,
                    "schema_version": SCHEMA_VERSION,
                    "run_id": "run_0_seed_0",
                    "seed": 0,
                    "setting": setting.name,
                    "tool_scope": "full",
                    "gold_action": "NONE",
                    "routed_tool_events": [],
                    "tool_calls": 0,
                    "final_correct": True,
                    "pred_action": "NONE",
                    "termination_reason": "boxed_answer",
                    "tool_parse_failures": 0,
                    **decision,
                }
            ],
        }
    ]
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    target = output_dir / "probe_prefill_t0.1_fulltools.json"
    target.write_text(json.dumps(complete), encoding="utf-8")

    monkeypatch.setattr(run_probe_prefill, "load_config", lambda path: config)
    monkeypatch.setattr(run_probe_prefill, "require_inputs", lambda value: None)
    monkeypatch.setattr(
        run_probe_prefill,
        "validate_runtime_provenance",
        lambda value: {"sha256": "r" * 64, "git_commit": "commit"},
    )
    monkeypatch.setattr(
        run_probe_prefill,
        "sha256_file",
        lambda path: {
            "unused.yaml": "c" * 64,
            "unused.json": "d" * 64,
            "unused-labels.json": "e" * 64,
            "probe_no_reasoning.pt": "p" * 64,
            "test_hidden_no_reasoning.pt": "p" * 64,
            "test_labels_no_reasoning.json": "p" * 64,
        }[Path(path).name],
    )
    monkeypatch.setattr(
        run_probe_prefill,
        "load_task_json",
        lambda path, expected_scope: [{"id": 101}],
    )
    monkeypatch.setattr(run_probe_prefill, "_labels", lambda path: {"rows": []})
    monkeypatch.setattr(
        run_probe_prefill,
        "attach_gold_actions",
        lambda tasks, rows: [dict(task, gold_action="NONE") for task in tasks],
    )
    monkeypatch.setattr(
        run_probe_prefill,
        "compute_prefills",
        lambda probe_dir, ids, threshold, temperature: (
            {101: decision["probe_prefill"]},
            {101: decision},
        ),
    )
    monkeypatch.setattr(
        run_probe_prefill,
        "build_agent",
        lambda config: pytest.fail("completed resume target loaded the model"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_probe_prefill",
            "--config",
            "unused.yaml",
            "--data",
            "unused.json",
            "--labels",
            "unused-labels.json",
            "--probe-dir",
            "unused-probe",
            "--output-dir",
            str(output_dir),
            "--tool-scope",
            "full",
            "--thresholds",
            "0.1",
            "--seeds",
            "0",
            "--resume",
        ],
    )
    run_probe_prefill.main()


def test_template_records_generation_and_record_mode(monkeypatch) -> None:
    monkeypatch.setattr(run_probe_prefill, "full_menu_sha256", lambda: "f" * 64)
    setting = EvaluationSetting(
        name="probe_prefill_t0.1_scoped",
        tool_scope="scoped",
        prompt_mode="current",
        require_reasoning=False,
        record_mode="lite",
    )
    template = build_probe_prefill_artifact_template(
        config=_config(),
        setting=setting,
        probe_protocol="adapted",
        probe_scope="scoped-adapted",
        threshold=0.1,
        seeds=(0,),
        task_ids=[101],
        smoke=False,
        config_sha256="c" * 64,
        data_sha256="d" * 64,
        labels_sha256="e" * 64,
        probe_inputs_sha256={"probe_no_reasoning.pt": "p" * 64},
        probe_decisions_sha256="q" * 64,
        runtime_provenance_sha256="r" * 64,
        project_git_commit="commit",
    )
    protocol = template["config"]
    assert protocol["config_sha256"] == "c" * 64
    assert protocol["data_sha256"] == "d" * 64
    assert protocol["labels_sha256"] == "e" * 64
    assert protocol["probe_inputs_sha256"] == {
        "probe_no_reasoning.pt": "p" * 64
    }
    assert protocol["probe_decisions_sha256"] == "q" * 64
    assert protocol["runtime_provenance_sha256"] == "r" * 64
    assert protocol["project_git_commit"] == "commit"
    assert protocol["record_mode"] == "lite"
    assert protocol["temperature"] == 0.7
    assert protocol["top_p"] == 0.8
    assert protocol["top_k"] == 20
    assert protocol["repetition_penalty"] == 1.0
    assert protocol["max_new_tokens"] == 2048
    assert protocol["max_rounds"] == 10
    assert protocol["max_model_len"] == 32768
