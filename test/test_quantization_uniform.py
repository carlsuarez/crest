# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the NumPy uniform quantizers in ``crest.quantization.uniform``."""

from __future__ import annotations

import numpy as np
import pytest

from crest.quantization import (
    ActivationObserver,
    QuantSpec,
    activation_qparams,
    compute_qparams,
    estimate_range,
    fake_quantize,
    integer_bounds,
    quantize_array,
)
from crest.quantization.uniform import channel_matrix, restore_channel_matrix, round_half_away_from_zero


@pytest.mark.parametrize(
    ("bits", "symmetric", "narrow", "expected"),
    [
        (8, True, True, (-127, 127)),
        (8, True, False, (-128, 127)),
        (8, False, True, (-128, 127)),
        (4, True, True, (-7, 7)),
        (2, True, True, (-1, 1)),
        (16, False, True, (-32768, 32767)),
    ],
)
def test_integer_bounds_follow_tflite_conventions(bits, symmetric, narrow, expected):
    """Symmetric narrow grids drop the most negative integer; asymmetric grids do not."""
    assert integer_bounds(bits, symmetric=symmetric, narrow_range=narrow) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 1},
        {"bits": 17},
        {"bits": True},
        {"bits": 4.0},
        {"bits": 4, "granularity": "per_row"},
        {"bits": 4, "range_method": "kl"},
        {"bits": 4, "percentile": 50.0},
        {"bits": 4, "percentile": 100.5},
        {"bits": 4, "symmetric": "yes"},
    ],
)
def test_quant_spec_rejects_invalid_fields(kwargs):
    """Invalid widths, granularities, methods, and percentiles fail early."""
    with pytest.raises(ValueError):
        QuantSpec(**kwargs)


def test_quant_spec_describe_is_flat_and_hides_unused_percentile():
    """describe() reports the percentile only for the percentile method."""
    assert QuantSpec(4).describe() == {
        "bits": 4,
        "symmetric": True,
        "granularity": "per_channel",
        "range_method": "minmax",
        "percentile": None,
    }
    assert QuantSpec(4, range_method="percentile", percentile=99.9).describe()["percentile"] == 99.9


def test_round_half_away_from_zero():
    """Ties round away from zero, unlike NumPy's banker's rounding."""
    values = np.array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49])
    np.testing.assert_array_equal(round_half_away_from_zero(values), [1, 2, 3, -1, -3, 0])


def test_channel_matrix_round_trip_with_trailing_and_inner_axes():
    """Moving channel axes to a matrix and back restores the array."""
    array = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    for axes in [(-1,), (2, 3), (0,)]:
        matrix, normalized, moved_shape = channel_matrix(array, axes)
        assert matrix.shape[1] == int(np.prod([array.shape[axis] for axis in normalized]))
        np.testing.assert_array_equal(restore_channel_matrix(matrix, normalized, moved_shape), array)
    with pytest.raises(ValueError):
        channel_matrix(array, ())
    with pytest.raises(ValueError):
        channel_matrix(array, (4,))


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_minmax_quantization_uses_at_most_two_to_the_bits_levels_and_bounded_error(bits, symmetric):
    """Each channel uses at most 2**bits levels and rounding error is at most half a step."""
    rng = np.random.default_rng(bits)
    kernel = rng.normal(size=(3, 3, 8, 6)) * rng.uniform(0.1, 3.0, size=6)
    spec = QuantSpec(bits, symmetric=symmetric, granularity="per_channel")
    quantized = quantize_array(kernel, spec, channel_axes=(-1,))
    assert quantized.shape == kernel.shape
    matrix, _, _ = channel_matrix(kernel, (-1,))
    lo, hi = estimate_range(matrix, spec)
    scale, _ = compute_qparams(lo, hi, spec)
    for channel in range(kernel.shape[-1]):
        levels = np.unique(quantized[..., channel])
        assert len(levels) <= 2**bits
        error = np.abs(quantized[..., channel] - kernel[..., channel])
        # Asymmetric zero-point rounding can shift the grid by half a step at the edges.
        tolerance = 0.5 if symmetric else 1.0
        assert np.max(error) <= scale[channel] * tolerance + 1e-12


def test_symmetric_minmax_quantization_is_idempotent():
    """Quantizing an already quantized kernel with the same spec changes nothing."""
    kernel = np.random.default_rng(0).normal(size=(16, 4))
    spec = QuantSpec(4)
    once = quantize_array(kernel, spec, channel_axes=(-1,))
    np.testing.assert_allclose(quantize_array(once, spec, channel_axes=(-1,)), once, rtol=0, atol=1e-12)


def test_per_channel_protects_small_channels_from_a_large_channel():
    """Per-tensor ranges crush a small channel that per-channel ranges keep."""
    rng = np.random.default_rng(1)
    kernel = rng.normal(size=(64, 2))
    kernel[:, 1] *= 100.0
    per_tensor = quantize_array(kernel, QuantSpec(4, granularity="per_tensor"))
    per_channel = quantize_array(kernel, QuantSpec(4, granularity="per_channel"), channel_axes=(-1,))
    small_error_tensor = np.mean((per_tensor[:, 0] - kernel[:, 0]) ** 2)
    small_error_channel = np.mean((per_channel[:, 0] - kernel[:, 0]) ** 2)
    # A 4-bit per-channel step on N(0, 1) data gives an MSE near 0.01; per-tensor
    # steps sized for the large channel round the small channel to zero (MSE near 1).
    assert small_error_channel < 0.05 * small_error_tensor


def test_asymmetric_ranges_include_zero_and_use_full_grid_for_nonnegative_data():
    """Zero is exact, and post-ReLU data maps 0 to qmin so all levels are positive."""
    spec = QuantSpec(8, symmetric=False, granularity="per_tensor")
    relu = np.abs(np.random.default_rng(2).normal(size=(1000, 1))) + 0.5
    lo, hi = estimate_range(relu, spec)
    assert lo[0] == 0.0
    scale, zero_point = compute_qparams(lo, hi, spec)
    assert zero_point[0] == spec.qmin
    assert fake_quantize(np.zeros(1), scale, zero_point, spec.qmin, spec.qmax)[0] == 0.0
    signed = np.random.default_rng(3).normal(size=(1000, 1))
    lo, hi = estimate_range(signed, spec)
    scale, zero_point = compute_qparams(lo, hi, spec)
    assert fake_quantize(np.zeros(1), scale, zero_point, spec.qmin, spec.qmax)[0] == 0.0


def test_all_zero_groups_stay_zero_without_nans():
    """Empty ranges get a unit scale instead of dividing by zero."""
    kernel = np.zeros((3, 3, 2))
    kernel[..., 1] = 0.25
    for spec in (QuantSpec(4), QuantSpec(4, symmetric=False), QuantSpec(4, range_method="mse")):
        quantized = quantize_array(kernel, spec, channel_axes=(-1,))
        assert np.all(np.isfinite(quantized))
        np.testing.assert_array_equal(quantized[..., 0], 0.0)


def test_percentile_and_mse_clip_outliers_and_mse_never_loses_to_minmax():
    """Clipping methods shrink an outlier-stretched range; MSE error is at most min-max error."""
    rng = np.random.default_rng(4)
    data = rng.normal(size=(5000, 1))
    data[0, 0] = 50.0
    for symmetric in (True, False):
        minmax = QuantSpec(3, symmetric=symmetric, granularity="per_tensor")
        percentile = QuantSpec(
            3, symmetric=symmetric, granularity="per_tensor", range_method="percentile", percentile=99.9
        )
        mse = QuantSpec(3, symmetric=symmetric, granularity="per_tensor", range_method="mse")
        widths = {}
        errors = {}
        for name, spec in (("minmax", minmax), ("percentile", percentile), ("mse", mse)):
            lo, hi = estimate_range(data, spec)
            widths[name] = float(hi[0] - lo[0])
            scale, zero_point = compute_qparams(lo, hi, spec)
            errors[name] = float(np.mean((fake_quantize(data, scale, zero_point, spec.qmin, spec.qmax) - data) ** 2))
        assert widths["percentile"] < 0.5 * widths["minmax"]
        assert widths["mse"] < 0.5 * widths["minmax"]
        assert errors["mse"] <= errors["minmax"]


def test_exact_extrema_bound_subsample_ranges():
    """Supplied extrema define the min-max range even when the sample misses them."""
    sample = np.linspace(-1.0, 1.0, 101).reshape(-1, 1)
    spec = QuantSpec(8, symmetric=False, granularity="per_tensor")
    lo, hi = estimate_range(sample, spec, data_min=[-3.0], data_max=[2.0])
    assert (lo[0], hi[0]) == (-3.0, 2.0)
    with pytest.raises(ValueError):
        estimate_range(np.empty((0, 1)), spec)
    with pytest.raises(ValueError):
        estimate_range(np.array([[np.nan]]), spec)


def test_activation_observer_tracks_extrema_caps_samples_and_ignores_batching():
    """Extrema are exact, the sample is capped, and batching does not change the sample."""
    data = np.random.default_rng(5).normal(size=(40, 6, 3)).astype(np.float32)
    whole = ActivationObserver(max_values=60, seed=7)
    whole.update(data)
    batched = ActivationObserver(max_values=60, seed=7)
    for start in range(0, 40, 7):
        batched.update(data[start : start + 7])
    first, second = whole.finalize(), batched.finalize()
    np.testing.assert_allclose(first.channel_min, data.reshape(-1, 3).min(axis=0))
    np.testing.assert_allclose(first.channel_max, data.reshape(-1, 3).max(axis=0))
    assert first.samples.shape == (20, 3)
    assert first.rows_seen == 240
    np.testing.assert_array_equal(np.sort(first.samples, axis=0), np.sort(second.samples, axis=0))

    observer = ActivationObserver()
    observer.update(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        observer.update(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        observer.update(np.full((1, 3), np.inf))
    with pytest.raises(ValueError):
        ActivationObserver().finalize()


def test_activation_qparams_shapes_follow_granularity():
    """Per-tensor params have one group; per-channel params have one per channel."""
    observer = ActivationObserver()
    observer.update(np.random.default_rng(6).normal(size=(50, 4)))
    stats = observer.finalize()
    per_tensor = activation_qparams(stats, QuantSpec(8, symmetric=False, granularity="per_tensor"))
    per_channel = activation_qparams(stats, QuantSpec(8, symmetric=True, granularity="per_channel"))
    assert per_tensor.scale.shape == (1,)
    assert per_channel.scale.shape == (4,)
    np.testing.assert_array_equal(per_channel.zero_point, 0.0)
    assert per_channel.apply(stats.samples).shape == stats.samples.shape
