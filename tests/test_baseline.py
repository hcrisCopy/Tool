import torch

from experiments_2d.baseline import apply_public_final_norm


def test_public_final_norm_only_changes_last_layer() -> None:
    hidden = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6) / 10
    weight = torch.linspace(0.8, 1.2, 6, dtype=torch.bfloat16)
    public = apply_public_final_norm(hidden, weight, 1e-6, chunk_size=1)
    torch.testing.assert_close(public[:, :-1], hidden[:, :-1], rtol=0, atol=0)
    assert not torch.equal(public[:, -1], hidden[:, -1])
    assert torch.isfinite(public).all()

