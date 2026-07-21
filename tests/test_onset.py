import numpy as np
import torch

from experiments_2d.onset import (
    contrast_curve,
    select_onset,
    standardize_residual_writes,
)


def test_standardized_residual_shape_and_finiteness() -> None:
    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(12, 5, 8, generator=generator)
    residual, mean, std = standardize_residual_writes(hidden)
    assert residual.shape == (12, 4, 8)
    assert mean.shape == (4, 8)
    assert std.shape == (4, 8)
    assert torch.isfinite(residual).all()


def test_vectorized_contrast_and_onset_selection() -> None:
    generator = torch.Generator().manual_seed(9)
    values = torch.randn(24, 4, 8, generator=generator)
    positive = np.zeros(24, dtype=bool)
    for start in range(0, 24, 4):
        positive[start : start + 2] = True
    negative = ~positive
    values[torch.from_numpy(positive), 1, :] += 2.0
    strata = np.repeat(np.arange(6), 4).astype(str)
    curve = contrast_curve(
        values,
        positive,
        negative,
        n_shuffles=12,
        rng=np.random.default_rng(17),
        device="cpu",
        strata=strata,
    )
    assert curve.smoothed_z.shape == (4,)
    assert np.isfinite(curve.smoothed_z).all()
    known = np.array([0.0, 2.0, 10.0, 9.6, 1.0, 0.0])
    onset = select_onset(known, peak_fraction=0.95, max_window=5)
    assert onset["onset_layer"] == 3
    assert 3 in onset["window"]

