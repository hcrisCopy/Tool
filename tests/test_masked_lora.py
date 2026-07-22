from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from when2tool_action.masked_lora import (
    LORA_RANK,
    AssistantOnlyCollator,
    _import_peft,
    audit_portable_adapter_metadata,
    attach_masked_lora,
    build_row_selection,
    parameter_accounting,
    normalize_peft_base_model_identity,
    register_lora_b_gradient_masks,
    tokenize_assistant_only,
    validate_probe_mask_metadata,
)
from when2tool_action.constants import ACTIONS
from when2tool_action.adapter_evaluation import expected_training_control
from when2tool_action.io_utils import canonical_json_sha256
from when2tool_action.neuron_ablation import NeuronMask
from when2tool_action.scripts import prepare_sft_trajectories, train_masked_lora
from when2tool_action.scripts.prepare_sft_trajectories import (
    build_parser as build_prepare_parser,
)
from when2tool_action.scripts.train_masked_lora import (
    _condition_id,
    _control_id,
    _distributed_context,
    _hyperparameters,
    _load_primary_training_mask,
    _training_protocol_fingerprint,
    _validate_sft_runtime_binding,
    build_parser as build_train_parser,
)


def _mask() -> NeuronMask:
    return NeuronMask(
        classes={
            "NONE": {0: (0, 1), 1: (2,)},
            "A": {0: (1, 2), 1: (3,)},
            "B": {0: (3,), 1: (4,)},
            "C": {0: (4,), 1: (5,)},
        },
        source_sha256="a" * 64,
    )


def _probing_payload(mask: NeuronMask) -> dict:
    classes = {}
    assignments = 0
    union = set()
    for action in ACTIONS:
        layers = {
            str(layer): [{"neuron_idx": index} for index in mask.indices(action)[layer]]
            for layer in range(2)
        }
        counts = {layer: len(entries) for layer, entries in layers.items()}
        total = sum(counts.values())
        assignments += total
        union.update(
            (int(layer), entry["neuron_idx"])
            for layer, entries in layers.items()
            for entry in entries
        )
        classes[action] = {
            "total_neurons": total,
            "layer_counts": counts,
            "layers": layers,
        }
    features = [[layer, index] for layer, index in sorted(union)]
    return {
        "schema_version": "when2tool-neuron-mask-v1",
        "class_order": list(ACTIONS),
        "total_neurons": len(features),
        "total_class_assignments": assignments,
        "union_features_sha256": canonical_json_sha256(features),
        "classes": classes,
    }


class FakeLoraLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.base_layer = torch.nn.Linear(in_features, out_features, bias=False)
        self.base_layer.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(in_features, LORA_RANK, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(LORA_RANK, out_features, bias=False)}
        )
        torch.nn.init.normal_(self.lora_A["default"].weight)
        torch.nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_layer(inputs) + self.lora_B["default"](
            self.lora_A["default"](inputs)
        )


class FakeMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = FakeLoraLinear(3, 10)
        self.up_proj = FakeLoraLinear(3, 10)


class FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = FakeMlp()


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        total = 0
        for layer in self.model.layers:
            total = total + layer.mlp.gate_proj(inputs) + layer.mlp.up_proj(inputs)
        return total


def test_random_rows_match_union_per_layer_and_exclude_all_four_classes() -> None:
    target = build_row_selection(
        _mask(), mode="target", num_hidden_layers=2, intermediate_size=10
    )
    random = build_row_selection(
        _mask(),
        mode="random",
        random_seed=1,
        num_hidden_layers=2,
        intermediate_size=10,
    )
    assert target.layers == {0: (0, 1, 2, 3, 4), 1: (2, 3, 4, 5)}
    assert random.target_union_sha256 == target.indices_sha256
    for layer in range(2):
        assert len(random.layers[layer]) == len(target.layers[layer])
        assert not (set(random.layers[layer]) & set(target.layers[layer]))
    repeated = build_row_selection(
        _mask(),
        mode="random",
        random_seed=1,
        num_hidden_layers=2,
        intermediate_size=10,
    )
    assert repeated.layers == random.layers
    assert repeated.indices_sha256 == random.indices_sha256
    snapshot = random.snapshot()
    assert snapshot["layers"]["0"]["indices"] == list(random.layers[0])
    assert snapshot["layers"]["0"]["target_union_count"] == len(target.layers[0])
    assert snapshot["layers"]["0"]["eligible_complement_count"] == 5
    assert len(snapshot["layers"]["0"]["eligible_complement_indices_sha256"]) == 64
    receipt = validate_probe_mask_metadata(_probing_payload(_mask()), random)
    assert receipt["total_union_neurons"] == 9
    tampered = _probing_payload(_mask())
    tampered["union_features_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="union_features_sha256"):
        validate_probe_mask_metadata(tampered, random)


def test_gradient_hook_and_optimizer_cannot_change_nonselected_b_rows() -> None:
    model = FakeModel()
    selection = build_row_selection(
        _mask(), mode="target", num_hidden_layers=2, intermediate_size=10
    )
    controller = register_lora_b_gradient_masks(model, selection)
    counts = parameter_accounting(model, controller)
    assert counts["raw_trainable_parameters"] > counts["effective_trainable_parameters"]
    assert counts["lora_a_parameters_fully_trainable"] > 0
    output = model(torch.ones(2, 3)).sum()
    output.backward()
    controller.assert_masked_gradients_zero()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    optimizer.step()
    controller.assert_nonselected_rows_zero()
    for entry in controller.entries:
        selected = torch.tensor(entry.selected_indices)
        assert torch.count_nonzero(entry.weight.detach()[selected]).item() > 0


def test_dense_control_is_ordinary_unmasked_mlp_lora() -> None:
    model = FakeModel()
    selection = build_row_selection(
        _mask(), mode="dense", num_hidden_layers=2, intermediate_size=10
    )
    controller = register_lora_b_gradient_masks(model, selection)
    assert all(
        entry.hook is None and entry.row_mask is None for entry in controller.entries
    )
    counts = parameter_accounting(model, controller)
    assert (
        counts["raw_trainable_parameters"] == counts["effective_trainable_parameters"]
    )


def test_nan_gradients_still_become_exact_zero_on_forbidden_rows() -> None:
    model = FakeModel()
    selection = build_row_selection(
        _mask(), mode="target", num_hidden_layers=2, intermediate_size=10
    )
    controller = register_lora_b_gradient_masks(model, selection)
    (model(torch.ones(1, 3)).sum() * torch.tensor(float("nan"))).backward()
    controller.assert_masked_gradients_zero()
    for entry in controller.entries:
        assert entry.row_mask is not None
        forbidden = entry.weight.grad.masked_select(~entry.row_mask)
        assert torch.count_nonzero(forbidden).item() == 0


def test_attach_uses_fixed_peft_configuration_with_fake_dependency(monkeypatch) -> None:
    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_task_type = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")

    def fake_get(model, config):
        captured["config_object"] = config
        return model

    monkeypatch.setattr(
        "when2tool_action.masked_lora._import_peft",
        lambda: (FakeConfig, fake_task_type, fake_get),
    )
    model, controller, _counts = attach_masked_lora(
        FakeModel(),
        build_row_selection(
            _mask(), mode="target", num_hidden_layers=2, intermediate_size=10
        ),
    )
    assert isinstance(model, FakeModel)
    assert len(controller.entries) == 4
    assert captured["r"] == 8
    assert captured["lora_alpha"] == 16
    assert captured["lora_dropout"] == 0.0
    assert captured["target_modules"] == ["gate_proj", "up_proj"]


def test_peft_identity_is_portable_and_metadata_scan_blocks_paths(tmp_path) -> None:
    posix_private = "/" + "root/private/model"
    windows_private = "D:" + r"\private\model"
    configs = {
        "default": SimpleNamespace(base_model_name_or_path=posix_private),
        "second": SimpleNamespace(base_model_name_or_path=windows_private),
    }
    model = SimpleNamespace(peft_config=configs)
    assert normalize_peft_base_model_identity(model, "qwen3-4b-instruct-2507") == (
        "default",
        "second",
    )
    assert {config.base_model_name_or_path for config in configs.values()} == {
        "qwen3-4b-instruct-2507"
    }

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config_path = adapter / "adapter_config.json"
    config_path.write_text(
        '{"base_model_name_or_path":"qwen3-4b-instruct-2507"}', encoding="utf-8"
    )
    (adapter / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "/" + "root/private/token": 0,
                        "C:" + "/token": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    receipt = audit_portable_adapter_metadata(
        adapter, forbidden_paths=(tmp_path / "private",)
    )
    assert receipt["scanned_files"] == ["adapter_config.json"]
    assert receipt["excluded_tokenizer_payload_files"] == ["tokenizer.json"]
    config_path.write_text(
        json.dumps({"base_model_name_or_path": posix_private}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="internal absolute path"):
        audit_portable_adapter_metadata(
            adapter, forbidden_paths=(tmp_path / "private",)
        )


def test_peft_runtime_version_is_exact(monkeypatch) -> None:
    monkeypatch.setattr(
        "when2tool_action.masked_lora.importlib.metadata.version",
        lambda _package: "0.17.0",
    )
    with pytest.raises(RuntimeError, match="peft==0.17.1"):
        _import_peft()


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is True
        assert enable_thinking is False
        assert len(tools) == 33
        text = "<tools>canonical33</tools>"
        for message in messages:
            role = message["role"]
            text += f"<{role}>{message['content']}</{role}>"
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode("utf-8"))


def _tools():
    return [
        {"type": "function", "function": {"name": f"tool_{index}"}}
        for index in range(33)
    ]


def test_assistant_only_labels_are_proven_by_prefix_difference() -> None:
    record = {
        "id": 7,
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "CALL"},
            {"role": "user", "content": "<tool_response>RESULT</tool_response>"},
            {"role": "assistant", "content": "\\boxed{ANSWER}"},
        ],
    }
    feature = tokenize_assistant_only(record, FakeTokenizer(), _tools())
    labeled = bytes(
        token
        for token, label in zip(
            feature["input_ids"].tolist(), feature["labels"].tolist()
        )
        if label != -100
    ).decode("utf-8")
    assert "CALL</assistant>" in labeled
    assert "\\boxed{ANSWER}</assistant>" in labeled
    assert "QUESTION" not in labeled
    assert "RESULT" not in labeled
    collated = AssistantOnlyCollator(0)([feature, feature])
    assert collated["input_ids"].shape[0] == 2


def test_overlength_raises_instead_of_truncating() -> None:
    record = {
        "id": 9,
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "QUESTION"},
            {"role": "assistant", "content": "ANSWER"},
        ],
    }
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_assistant_only(record, FakeTokenizer(), _tools(), max_length=20)


@pytest.mark.parametrize(
    ("world_size", "accumulation"), [(1, 8), (2, 4), (4, 2), (8, 1)]
)
def test_ddp_global_batch_is_exact(
    monkeypatch, world_size: int, accumulation: int
) -> None:
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert _distributed_context()["gradient_accumulation_steps"] == accumulation


def test_ddp_rejects_nondivisor_world_size(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="not divisible"):
        _distributed_context()


def test_training_control_and_evaluation_condition_id_contracts_match() -> None:
    assert _control_id("target", None) == "target_neuron_lora"
    assert _control_id("dense", None) == "dense_mlp_lora"
    assert [_control_id("random", seed) for seed in (0, 1, 2)] == [
        "random_neuron_lora",
        "random_neuron_lora",
        "random_neuron_lora",
    ]
    conditions = [_condition_id("random", seed) for seed in (0, 1, 2)]
    assert conditions == [
        "random_neuron_lora_seed0",
        "random_neuron_lora_seed1",
        "random_neuron_lora_seed2",
    ]
    assert [expected_training_control(condition) for condition in conditions] == [
        ("random_neuron_lora", 0),
        ("random_neuron_lora", 1),
        ("random_neuron_lora", 2),
    ]


def test_shared_protocol_fingerprint_is_condition_invariant_and_input_bound() -> None:
    distributed = {
        "world_size": 4,
        "rank": 0,
        "local_rank": 0,
        "gradient_accumulation_steps": 2,
    }
    common = {
        "sft_jsonl_sha256": "1" * 64,
        "sft_manifest_sha256": "2" * 64,
        "primary_mask_sha256": "3" * 64,
        "runtime_provenance_sha256": "4" * 64,
        "project_git_commit": "5" * 40,
        "base_model_slug": "qwen3-4b-instruct-2507",
        "config_sha256": "6" * 64,
        "full_menu_sha256": "7" * 64,
        "hyperparameters": _hyperparameters(distributed),
        "world_size": 4,
    }
    fingerprints = {
        _condition_id(mode, seed): _training_protocol_fingerprint(**common)
        for mode, seed in (
            ("target", None),
            ("dense", None),
            ("random", 0),
            ("random", 1),
            ("random", 2),
        )
    }
    assert len(set(fingerprints.values())) == 1
    for field, replacement in (
        ("primary_mask_sha256", "8" * 64),
        ("runtime_provenance_sha256", "9" * 64),
        ("project_git_commit", "a" * 40),
        ("base_model_slug", "different-model"),
        ("config_sha256", "b" * 64),
        ("full_menu_sha256", "c" * 64),
    ):
        changed = dict(common)
        changed[field] = replacement
        assert _training_protocol_fingerprint(**changed) not in fingerprints.values()


@pytest.mark.parametrize("parser_builder", [build_prepare_parser, build_train_parser])
def test_stage7_clis_require_explicit_runtime_provenance(parser_builder) -> None:
    required = {
        action.dest
        for action in parser_builder()._actions
        if getattr(action, "required", False)
    }
    assert "runtime_provenance" in required


@pytest.mark.parametrize("module", [prepare_sft_trajectories, train_masked_lora])
def test_stage7_runtime_helper_passes_explicit_receipt(
    monkeypatch, tmp_path, module
) -> None:
    receipt = (
        tmp_path / "stages" / "07_training" / "manifests" / "runtime_provenance.json"
    )
    seen = {}

    def fake_validate(config, path):
        seen.update(config=config, path=path)
        return {"sha256": "a" * 64, "git_commit": "b" * 40, "path": path}

    monkeypatch.setattr(module, "validate_runtime_provenance", fake_validate)
    config = object()
    result = module._load_stage_runtime_provenance(config, receipt)
    assert seen == {"config": config, "path": receipt.resolve()}
    assert result["path"] == receipt.resolve()


def test_training_requires_same_runtime_receipt_as_sft(tmp_path) -> None:
    receipt_path = tmp_path / "runtime_provenance.json"
    receipt = {
        "path": receipt_path,
        "sha256": "a" * 64,
        "git_commit": "b" * 40,
    }
    manifest = {
        "runtime_provenance": {
            "file": receipt_path.name,
            "sha256": "a" * 64,
            "project_git_commit": "b" * 40,
        }
    }
    _validate_sft_runtime_binding(manifest, receipt)
    manifest["runtime_provenance"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different Stage-07"):
        _validate_sft_runtime_binding(manifest, receipt)


def test_training_loader_requires_preregistered_train_selected_primary_mask(
    monkeypatch, tmp_path
) -> None:
    seen = {}
    sentinel = object()

    def fake_load(path, **kwargs):
        seen.update(path=path, **kwargs)
        return sentinel

    monkeypatch.setattr(train_masked_lora, "load_neuron_mask", fake_load)
    config = SimpleNamespace(
        model=SimpleNamespace(
            slug="qwen3-4b-instruct-2507",
            architecture="Qwen3ForCausalLM",
            num_hidden_layers=36,
            hidden_size=2560,
        )
    )
    mask_path = tmp_path / "tool_action_neurons.json"
    assert _load_primary_training_mask(config, mask_path) is sentinel
    assert seen["path"] == mask_path
    assert seen["expected_rho"] == 0.003
    assert seen["expected_variant"] == "signed"
    assert seen["require_train_selection"] is True
    assert seen["expected_model"]["intermediate_size"] == 9728
