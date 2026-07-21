import numpy as np
import torch
from unittest.mock import patch

from experiments_2d.onset import (
    _smooth_three,
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


def test_standardization_uses_std_plus_epsilon() -> None:
    hidden = torch.tensor([[[0.0], [1.0]], [[0.0], [3.0]]])
    standardized, mean, std = standardize_residual_writes(hidden)
    expected = (torch.tensor([[[1.0]], [[3.0]]]) - mean) / (std + 1e-6)
    torch.testing.assert_close(standardized, expected, rtol=0, atol=0)


def test_three_point_smoothing_uses_available_boundary_layers() -> None:
    actual = _smooth_three(np.array([3.0, 6.0, 12.0]))
    np.testing.assert_allclose(actual, np.array([4.5, 7.0, 9.0]))


def test_permutation_z_uses_null_std_plus_epsilon() -> None:
    scores = np.array([[4.0], [1.0], [3.0], [5.0]], dtype=np.float64)
    values = torch.zeros(4, 1, 1)
    positive = np.array([True, True, False, False])
    with patch("experiments_2d.onset._layer_scores_from_masks", return_value=scores):
        curve = contrast_curve(
            values,
            positive,
            ~positive,
            n_shuffles=3,
            rng=np.random.default_rng(1),
            device="cpu",
            strata=np.array(["x", "x", "x", "x"]),
        )
    null = scores[1:, 0]
    expected = (scores[0, 0] - null.mean()) / (null.std(ddof=1) + 1e-6)
    assert curve.z_score[0] == expected


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


def test_fwhm_is_referenced_to_selected_onset_value() -> None:
    # l*=2 has value 9.5 while the global peak is 10.  Layers 1--5 at 4.8
    # belong to the l*-referenced FWHM (threshold 4.75), but not the
    # global-peak FWHM (threshold 5.0).
    known = np.array([4.8, 9.5, 10.0, 4.8, 4.8, 0.0])
    onset = select_onset(known, peak_fraction=0.95, max_window=5)
    assert onset["onset_layer"] == 2
    assert 5 in onset["window"]


def test_peak_ratio_uses_onset_and_excludes_onset_plus_minus_two() -> None:
    known = np.array([1.0, 9.5, 10.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    onset = select_onset(known, peak_fraction=0.95, max_window=5)
    assert onset["onset_layer"] == 2
    assert onset["peak_layer"] == 3
    assert onset["background_median"] == 2.0
    assert onset["peak_ratio"] == 4.75
