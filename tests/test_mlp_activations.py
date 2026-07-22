from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from when2tool_action.mlp_activations import (
    activation_output_paths,
    capture_last_token_mlp_activations,
    down_proj_column_norms,
    last_valid_token_indices,
    preflight_activation_outputs,
    runtime_provenance_identity,
)
from when2tool_action.scripts.extract_mlp_activations import build_parser


class _FakeMLP(nn.Module):
    def __init__(self, layer_offset: float) -> None:
        super().__init__()
        self.layer_offset = layer_offset
        self.down_proj = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.down_proj.weight.zero_()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        intermediate = torch.cat(
            (hidden, hidden + self.layer_offset), dim=-1
        )
        return self.down_proj(intermediate)


class _FakeLayer(nn.Module):
    def __init__(self, layer_offset: float) -> None:
        super().__init__()
        self.mlp = _FakeMLP(layer_offset)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.mlp(hidden)
        return hidden


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(10.0), _FakeLayer(20.0)])

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> dict[str, torch.Tensor]:
        assert use_cache is False
        assert input_ids.shape == attention_mask.shape
        hidden = torch.stack(
            (input_ids.float(), input_ids.float() + 100.0), dim=-1
        )
        for layer in self.layers:
            hidden = layer(hidden)
        return {"last_hidden_state": hidden}


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeBackbone()


def test_down_proj_hook_captures_last_nonpadding_token_for_right_padded_batch() -> None:
    model = _FakeQwen()
    input_ids = torch.tensor([[1, 2, 99], [3, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)

    captured = capture_last_token_mlp_activations(
        model, input_ids, attention_mask
    )

    assert captured.dtype == torch.float16
    assert tuple(captured.shape) == (2, 2, 4)
    expected = torch.tensor(
        [
            [[2, 102, 12, 112], [2, 102, 22, 122]],
            [[5, 105, 15, 115], [5, 105, 25, 125]],
        ],
        dtype=torch.float16,
    )
    assert torch.equal(captured, expected)
    assert all(
        not layer.mlp.down_proj._forward_pre_hooks
        for layer in model.model.layers
    )


def test_right_padding_validation_rejects_left_or_internal_padding() -> None:
    assert torch.equal(
        last_valid_token_indices(torch.tensor([[1, 1, 0], [1, 1, 1]])),
        torch.tensor([1, 2]),
    )
    with pytest.raises(ValueError, match="right padding"):
        last_valid_token_indices(torch.tensor([[0, 1, 1]]))
    with pytest.raises(ValueError, match="right padding"):
        last_valid_token_indices(torch.tensor([[1, 0, 1]]))


def test_down_projection_norm_is_computed_per_intermediate_column() -> None:
    model = _FakeQwen()
    weight = torch.tensor(
        [[3.0, 0.0, 1.0, 2.0], [4.0, 5.0, 2.0, 0.0]]
    )
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.down_proj.weight.copy_(weight)
    norms = down_proj_column_norms(model)
    assert norms.dtype == torch.float32
    assert tuple(norms.shape) == (2, 4)
    expected = torch.tensor([5.0, 5.0, 5.0**0.5, 2.0])
    assert torch.allclose(norms[0], expected)
    assert torch.allclose(norms[1], expected)


def test_activation_preflight_rejects_any_split_collision(tmp_path: Path) -> None:
    output_dir = tmp_path / "activations"
    output_dir.mkdir()
    collision = activation_output_paths(output_dir)[-1]
    collision.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match=collision.name):
        preflight_activation_outputs(output_dir)


def test_extraction_cli_requires_explicit_stage5_runtime_provenance() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--data-dir",
                "data",
                "--labels-dir",
                "labels",
                "--output-dir",
                "activations",
            ]
        )
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--data-dir",
            "data",
            "--labels-dir",
            "labels",
            "--output-dir",
            "activations",
            "--runtime-provenance",
            "stage5-runtime.json",
        ]
    )
    assert args.runtime_provenance == Path("stage5-runtime.json")


def test_runtime_receipt_identity_rejects_missing_or_malformed_hash() -> None:
    expected = {
        "runtime_provenance_sha256": "a" * 64,
        "project_git_commit": "commit",
    }
    assert runtime_provenance_identity(
        {"sha256": "a" * 64, "git_commit": "commit", "path": Path("ignored")}
    ) == expected
    with pytest.raises(ValueError, match="SHA256"):
        runtime_provenance_identity({"sha256": "bad", "git_commit": "commit"})
