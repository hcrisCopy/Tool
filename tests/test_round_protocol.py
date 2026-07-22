from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import pytest
import yaml

from when2tool_action.config import load_config
from when2tool_action.scripts import (
    extract_hidden,
    extract_tool_labels,
    run_eval,
    run_probe_prefill,
    run_scoped_baseline,
)


CONFIG_PATH = "when2tool_action/configs/qwen3_4b_instruct_2507.yaml"
LABEL_ROUNDS = "label_hidden_extraction_max_rounds"
BEHAVIOR_ROUNDS = "behavior_evaluation_max_rounds"


def _raw_config() -> dict:
    source = Path(__file__).resolve().parents[1] / CONFIG_PATH
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "round_protocol.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_registered_config_has_distinct_pinned_round_limits() -> None:
    config = load_config(CONFIG_PATH)
    assert config.generation.label_hidden_extraction_max_rounds == 12
    assert config.generation.behavior_evaluation_max_rounds == 10
    assert not hasattr(config.generation, "max_rounds")


def test_generic_max_rounds_is_not_a_compatibility_fallback(tmp_path: Path) -> None:
    raw = _raw_config()
    generation = raw["generation"]
    generation["max_rounds"] = 12

    with pytest.raises(ValueError, match="generation.max_rounds is forbidden"):
        load_config(_write_config(tmp_path, raw))

    del generation[LABEL_ROUNDS]
    del generation[BEHAVIOR_ROUNDS]
    with pytest.raises(ValueError, match="generation.max_rounds is forbidden"):
        load_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        (LABEL_ROUNDS, 10, "label/hidden extraction.*12"),
        (BEHAVIOR_ROUNDS, 12, "behavior and Probe&Prefill.*10"),
    ],
)
def test_round_limits_cannot_be_swapped(
    tmp_path: Path, field: str, bad_value: int, message: str
) -> None:
    raw = deepcopy(_raw_config())
    raw["generation"][field] = bad_value
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, raw))


def test_round_fields_stay_with_their_protocol_entrypoints() -> None:
    label_source = inspect.getsource(extract_tool_labels)
    legacy_source = (
        Path(__file__).resolve().parents[1]
        / "when2tool_action"
        / "legacy_scoped.py"
    ).read_text(encoding="utf-8")
    hidden_source = inspect.getsource(extract_hidden)
    behavior_sources = {
        module.__name__: inspect.getsource(module)
        for module in (run_eval, run_scoped_baseline, run_probe_prefill)
    }

    assert LABEL_ROUNDS in label_source
    assert BEHAVIOR_ROUNDS not in label_source
    assert LABEL_ROUNDS in legacy_source
    assert BEHAVIOR_ROUNDS not in legacy_source
    for name, source in behavior_sources.items():
        assert BEHAVIOR_ROUNDS in source, name
        assert LABEL_ROUNDS not in source, name

    # Hidden extraction performs a single forward pass over a rendered prompt;
    # its 12-round provenance comes from its hard-no-tool label artifact, not
    # from running an interaction loop of its own.
    assert LABEL_ROUNDS not in hidden_source
    assert BEHAVIOR_ROUNDS not in hidden_source
