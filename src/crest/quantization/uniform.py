# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Uniform (affine) quantizers for host-side precision simulation.

The functions in this module model ``b``-bit integer storage with NumPy and
return dequantized ("fake-quantized") float arrays.
Callers can measure the accuracy effect of a precision without producing a
deployable artifact.

Conventions follow TensorFlow Lite where it defines one.

- Symmetric quantization fixes the zero point at 0.
  By default it uses the narrow signed range ``[-(2**(b-1) - 1), 2**(b-1) - 1]``,
  as TFLite int8 weights do.
- Asymmetric quantization uses the full signed range ``[-2**(b-1), 2**(b-1) - 1]``
  with an integer zero point.
  The range is widened to include 0 so that zero is exactly representable,
  as TFLite int8 activations do.
- Rounding is half away from zero.

Range (clipping) estimation supports min-max, percentile, and MSE-optimal
clipping.
All estimators work on a two-dimensional sample matrix whose columns are the
quantization groups: one column for per-tensor quantization and one column per
channel for per-channel quantization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

GRANULARITIES = ("per_tensor", "per_channel")
RANGE_METHODS = ("minmax", "percentile", "mse")
MIN_BITS = 2
MAX_BITS = 16
# Clipping ratios tried by the MSE range search, as fractions of the min-max range.
MSE_CLIP_RATIOS = np.linspace(0.01, 1.0, 100)


@dataclass(frozen=True)
class QuantSpec:
    """Uniform quantization settings for one tensor.

    Parameters
    ----------
    bits : int
        Integer width, from ``MIN_BITS`` (2) to ``MAX_BITS`` (16).
    symmetric : bool, optional
        True fixes the zero point at 0; False uses an integer zero point.
    granularity : {"per_tensor", "per_channel"}, optional
        One range for the whole tensor, or one range per channel.
        Channel axes are supplied by the caller (for example the output-channel
        axes of a kernel, or the last axis of an activation).
    range_method : {"minmax", "percentile", "mse"}, optional
        How the clipping range is chosen from the data.
    percentile : float, optional
        Upper percentile kept by ``range_method="percentile"``, in ``(50, 100]``.
        Symmetric specs clip ``abs(x)`` at this percentile.
        Asymmetric specs clip at ``100 - percentile`` and ``percentile``.
    narrow_range : bool, optional
        Symmetric specs only.
        True drops the most negative integer so the grid is symmetric
        (TFLite int8 weights use ``[-127, 127]``).

    Raises
    ------
    ValueError
        If a field is outside its supported values.
    """

    bits: int
    symmetric: bool = True
    granularity: str = "per_channel"
    range_method: str = "minmax"
    percentile: float = 99.99
    narrow_range: bool = True

    def __post_init__(self) -> None:
        """Validate field types and values.

        Raises
        ------
        ValueError
            If a field is outside its supported values.
        """
        if isinstance(self.bits, bool) or not isinstance(self.bits, (int, np.integer)):
            raise ValueError("QuantSpec.bits must be an integer.")
        if not MIN_BITS <= int(self.bits) <= MAX_BITS:
            raise ValueError(f"QuantSpec.bits must be between {MIN_BITS} and {MAX_BITS}.")
        if not isinstance(self.symmetric, (bool, np.bool_)):
            raise ValueError("QuantSpec.symmetric must be a boolean.")
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"QuantSpec.granularity must be one of {GRANULARITIES}.")
        if self.range_method not in RANGE_METHODS:
            raise ValueError(f"QuantSpec.range_method must be one of {RANGE_METHODS}.")
        percentile = float(self.percentile)
        if not (math.isfinite(percentile) and 50.0 < percentile <= 100.0):
            raise ValueError("QuantSpec.percentile must be in (50, 100].")
        if not isinstance(self.narrow_range, (bool, np.bool_)):
            raise ValueError("QuantSpec.narrow_range must be a boolean.")

    @property
    def qmin(self) -> int:
        """Return the smallest representable integer.

        Returns
        -------
        int
            Lower integer bound of the grid.
        """
        return integer_bounds(self.bits, symmetric=self.symmetric, narrow_range=self.narrow_range)[0]

    @property
    def qmax(self) -> int:
        """Return the largest representable integer.

        Returns
        -------
        int
            Upper integer bound of the grid.
        """
        return integer_bounds(self.bits, symmetric=self.symmetric, narrow_range=self.narrow_range)[1]

    def describe(self) -> dict[str, Any]:
        """Return flat, JSON- and CSV-friendly fields for this spec.

        Returns
        -------
        dict[str, Any]
            ``bits``, ``symmetric``, ``granularity``, ``range_method``, and
            ``percentile`` (None unless the percentile method is used).
        """
        return {
            "bits": int(self.bits),
            "symmetric": bool(self.symmetric),
            "granularity": self.granularity,
            "range_method": self.range_method,
            "percentile": float(self.percentile) if self.range_method == "percentile" else None,
        }


@dataclass(frozen=True)
class UniformQuantParams:
    """Frozen scale and zero point for one quantized tensor.

    Parameters
    ----------
    scale : numpy.ndarray
        Positive step size per quantization group, shape ``(groups,)``.
    zero_point : numpy.ndarray
        Integer-valued zero point per group, shape ``(groups,)``.
    qmin : int
        Smallest representable integer.
    qmax : int
        Largest representable integer.
    """

    scale: np.ndarray
    zero_point: np.ndarray
    qmin: int
    qmax: int

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Fake-quantize values whose last axis matches the parameter groups.

        Parameters
        ----------
        values : numpy.ndarray
            Values to quantize. Per-tensor parameters broadcast to any shape.

        Returns
        -------
        numpy.ndarray
            Dequantized values in float64.
        """
        return fake_quantize(values, self.scale, self.zero_point, self.qmin, self.qmax)


@dataclass(frozen=True)
class ActivationStats:
    """Calibration statistics for one activation tensor.

    Parameters
    ----------
    channel_min : numpy.ndarray
        Exact per-channel minimum over all calibration values, shape ``(C,)``.
    channel_max : numpy.ndarray
        Exact per-channel maximum over all calibration values, shape ``(C,)``.
    samples : numpy.ndarray
        Uniform random subsample of activation rows, shape ``(m, C)``.
        Percentile and MSE ranges are estimated from these rows.
    rows_seen : int
        Number of activation rows observed during calibration.
    """

    channel_min: np.ndarray
    channel_max: np.ndarray
    samples: np.ndarray
    rows_seen: int

    @property
    def channels(self) -> int:
        """Return the channel count.

        Returns
        -------
        int
            Size of the channel (last) axis.
        """
        return int(self.channel_min.shape[0])


class ActivationObserver:
    """Accumulate calibration statistics for one activation tensor.

    The last axis of every observed array is treated as the channel axis.
    Exact extrema are tracked over all values.
    A uniform random subsample of rows (positions) is kept with bottom-k
    sampling, so the sample does not depend on how the data was batched.

    Parameters
    ----------
    max_values : int, optional
        Upper bound on stored sample values (rows times channels).
    seed : int, optional
        Seed for the row-sampling keys.
    """

    def __init__(self, *, max_values: int = 500_000, seed: int = 0) -> None:
        if int(max_values) < 1:
            raise ValueError("ActivationObserver.max_values must be positive.")
        self._max_values = int(max_values)
        self._rng = np.random.default_rng(seed)
        self._channel_min: np.ndarray | None = None
        self._channel_max: np.ndarray | None = None
        self._keys = np.empty((0,), dtype=np.float64)
        self._rows: np.ndarray | None = None
        self._rows_seen = 0

    def update(self, values: np.ndarray) -> None:
        """Add one batch of activation values.

        Parameters
        ----------
        values : numpy.ndarray
            Activation batch whose last axis is the channel axis.

        Raises
        ------
        ValueError
            If the channel count changes between batches or values are not finite.
        """
        array = np.asarray(values)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        channels = int(array.shape[-1])
        rows = array.reshape(-1, channels).astype(np.float32, copy=False)
        if rows.shape[0] == 0:
            return
        if not np.all(np.isfinite(rows)):
            raise ValueError("Activation values must be finite for calibration.")
        if self._channel_min is None:
            self._channel_min = rows.min(axis=0).astype(np.float64)
            self._channel_max = rows.max(axis=0).astype(np.float64)
            self._rows = np.empty((0, channels), dtype=np.float32)
        else:
            if channels != self._channel_min.shape[0]:
                raise ValueError("Activation channel count changed between calibration batches.")
            self._channel_min = np.minimum(self._channel_min, rows.min(axis=0))
            self._channel_max = np.maximum(self._channel_max, rows.max(axis=0))
        self._rows_seen += int(rows.shape[0])

        # Bottom-k sampling: rows with the k smallest random keys form a uniform
        # sample without replacement, independent of batch boundaries.
        capacity = max(1, self._max_values // channels)
        keys = np.concatenate([self._keys, self._rng.random(rows.shape[0])])
        stacked = np.concatenate([self._rows, rows], axis=0)
        if keys.shape[0] > capacity:
            keep = np.argpartition(keys, capacity - 1)[:capacity]
            keys = keys[keep]
            stacked = stacked[keep]
        self._keys = keys
        self._rows = stacked

    def finalize(self) -> ActivationStats:
        """Return the accumulated statistics.

        Returns
        -------
        ActivationStats
            Exact extrema plus the row subsample.

        Raises
        ------
        ValueError
            If no values were observed.
        """
        if self._channel_min is None or self._rows is None:
            raise ValueError("ActivationObserver saw no values.")
        return ActivationStats(
            channel_min=self._channel_min.copy(),
            channel_max=self._channel_max.copy(),
            samples=self._rows.copy(),
            rows_seen=self._rows_seen,
        )


def integer_bounds(bits: int, *, symmetric: bool, narrow_range: bool = True) -> tuple[int, int]:
    """Return the signed integer grid bounds for a bit width.

    Parameters
    ----------
    bits : int
        Integer width.
    symmetric : bool
        Whether the zero point is fixed at 0.
    narrow_range : bool, optional
        Symmetric only: drop the most negative integer.

    Returns
    -------
    tuple[int, int]
        ``(qmin, qmax)``.
    """
    qmax = 2 ** (int(bits) - 1) - 1
    if symmetric and narrow_range:
        return -qmax, qmax
    return -(qmax + 1), qmax


def round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Round to the nearest integer, with ties away from zero.

    Parameters
    ----------
    values : numpy.ndarray
        Values to round.

    Returns
    -------
    numpy.ndarray
        Rounded values with the input's float dtype.
    """
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def fake_quantize(
    values: np.ndarray,
    scale: np.ndarray | float,
    zero_point: np.ndarray | float,
    qmin: int,
    qmax: int,
) -> np.ndarray:
    """Quantize to integers and immediately dequantize.

    Parameters
    ----------
    values : numpy.ndarray
        Values to quantize.
    scale : numpy.ndarray | float
        Positive step size, broadcastable to ``values``.
    zero_point : numpy.ndarray | float
        Integer zero point, broadcastable to ``values``.
    qmin, qmax : int
        Integer grid bounds.

    Returns
    -------
    numpy.ndarray
        Dequantized values in float64.
    """
    data = np.asarray(values, dtype=np.float64)
    step = np.asarray(scale, dtype=np.float64)
    offset = np.asarray(zero_point, dtype=np.float64)
    integers = np.clip(round_half_away_from_zero(data / step) + offset, qmin, qmax)
    return (integers - offset) * step


def compute_qparams(lo: np.ndarray, hi: np.ndarray, spec: QuantSpec) -> tuple[np.ndarray, np.ndarray]:
    """Convert clipping ranges into scales and zero points.

    Parameters
    ----------
    lo, hi : numpy.ndarray
        Per-group clipping bounds. The range is widened to include 0.
    spec : QuantSpec
        Quantization settings.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(scale, zero_point)`` in float64. Groups with an empty range get a
        scale of 1, which maps them to exact zeros.
    """
    low = np.minimum(np.asarray(lo, dtype=np.float64), 0.0)
    high = np.maximum(np.asarray(hi, dtype=np.float64), 0.0)
    qmin, qmax = spec.qmin, spec.qmax
    if spec.symmetric:
        scale = np.maximum(np.abs(low), np.abs(high)) / qmax
        zero_point = np.zeros_like(scale)
    else:
        scale = (high - low) / (qmax - qmin)
        positive = scale > 0
        raw_zero_point = np.where(positive, qmin - low / np.where(positive, scale, 1.0), 0.0)
        zero_point = np.clip(round_half_away_from_zero(raw_zero_point), qmin, qmax)
    # A group whose values are all zero has no range; any positive scale
    # quantizes it to exact zeros.
    scale = np.where(scale > 0, scale, 1.0)
    return scale, zero_point


def _minmax_bounds(
    matrix: np.ndarray,
    spec: QuantSpec,
    data_min: Sequence[float] | np.ndarray | None,
    data_max: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-inclusive min-max bounds, symmetrized when requested.

    Parameters
    ----------
    matrix : numpy.ndarray
        Sample matrix with one column per quantization group.
    spec : QuantSpec
        Quantization settings.
    data_min, data_max : array-like | None
        Exact per-group extrema to use instead of the sample extrema.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(lo, hi)`` with ``lo <= 0 <= hi``.
    """
    lo = matrix.min(axis=0) if data_min is None else np.asarray(data_min, dtype=np.float64).reshape(-1)
    hi = matrix.max(axis=0) if data_max is None else np.asarray(data_max, dtype=np.float64).reshape(-1)
    lo = np.minimum(lo, 0.0)
    hi = np.maximum(hi, 0.0)
    if spec.symmetric:
        magnitude = np.maximum(np.abs(lo), np.abs(hi))
        return -magnitude, magnitude
    return lo, hi


def estimate_range(
    matrix: np.ndarray,
    spec: QuantSpec,
    *,
    data_min: Sequence[float] | np.ndarray | None = None,
    data_max: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one clipping range per column of a sample matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Samples with shape ``(n, groups)``.
        Use one column for per-tensor quantization.
    spec : QuantSpec
        Quantization settings; ``range_method`` selects the estimator.
    data_min, data_max : array-like, optional
        Exact per-group extrema when ``matrix`` is a subsample.
        They bound every method and define the min-max range.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(lo, hi)`` arrays of shape ``(groups,)`` with ``lo <= 0 <= hi``.

    Raises
    ------
    ValueError
        If ``matrix`` is not two-dimensional, is empty, or is not finite.

    Notes
    -----
    The MSE method shrinks the min-max range by a common ratio from
    ``MSE_CLIP_RATIOS`` and keeps, per group, the ratio with the lowest
    mean squared quantization error on ``matrix``.
    For asymmetric specs both ends shrink by the same ratio, a common
    simplification of a full two-sided search.
    """
    samples = np.asarray(matrix, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("estimate_range expects a non-empty (n, groups) matrix.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("estimate_range values must be finite.")
    lo, hi = _minmax_bounds(samples, spec, data_min, data_max)
    if spec.range_method == "minmax":
        return lo, hi

    if spec.range_method == "percentile":
        upper = float(spec.percentile)
        if spec.symmetric:
            magnitude = np.minimum(np.percentile(np.abs(samples), upper, axis=0), hi)
            return -magnitude, magnitude
        low = np.maximum(np.minimum(np.percentile(samples, 100.0 - upper, axis=0), 0.0), lo)
        high = np.minimum(np.maximum(np.percentile(samples, upper, axis=0), 0.0), hi)
        return low, high

    best_error = np.full(samples.shape[1], np.inf)
    best_ratio = np.ones(samples.shape[1])
    for ratio in MSE_CLIP_RATIOS:
        scale, zero_point = compute_qparams(lo * ratio, hi * ratio, spec)
        error = np.mean((samples - fake_quantize(samples, scale, zero_point, spec.qmin, spec.qmax)) ** 2, axis=0)
        improved = error < best_error
        best_error = np.where(improved, error, best_error)
        best_ratio = np.where(improved, ratio, best_ratio)
    return lo * best_ratio, hi * best_ratio


def channel_matrix(
    values: np.ndarray, channel_axes: Sequence[int]
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Reshape an array to ``(n, channels)`` with the channel axes last.

    Parameters
    ----------
    values : numpy.ndarray
        Array to reshape.
    channel_axes : Sequence[int]
        Axes that index channels; negative values are allowed.

    Returns
    -------
    tuple[numpy.ndarray, tuple[int, ...], tuple[int, ...]]
        The matrix, the normalized channel axes, and the moved shape needed to
        restore the original layout with :func:`restore_channel_matrix`.

    Raises
    ------
    ValueError
        If no channel axes are given or an axis is out of range or repeated.
    """
    array = np.asarray(values)
    if not channel_axes:
        raise ValueError("Per-channel quantization needs at least one channel axis.")
    axes = tuple(int(axis) % array.ndim if -array.ndim <= int(axis) < array.ndim else None for axis in channel_axes)
    if None in axes or len(set(axes)) != len(axes):
        raise ValueError(f"Invalid channel axes {tuple(channel_axes)} for an array of rank {array.ndim}.")
    destination = tuple(range(array.ndim - len(axes), array.ndim))
    moved = np.moveaxis(array, axes, destination)
    channels = int(np.prod([array.shape[axis] for axis in axes]))
    return moved.reshape(-1, channels), axes, moved.shape


def restore_channel_matrix(
    matrix: np.ndarray, channel_axes: tuple[int, ...], moved_shape: tuple[int, ...]
) -> np.ndarray:
    """Invert :func:`channel_matrix`.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix produced by :func:`channel_matrix` (possibly transformed).
    channel_axes : tuple[int, ...]
        Normalized channel axes returned by :func:`channel_matrix`.
    moved_shape : tuple[int, ...]
        Moved shape returned by :func:`channel_matrix`.

    Returns
    -------
    numpy.ndarray
        Array in the original axis order.
    """
    ndim = len(moved_shape)
    source = tuple(range(ndim - len(channel_axes), ndim))
    return np.moveaxis(np.asarray(matrix).reshape(moved_shape), source, channel_axes)


def quantize_array(values: np.ndarray, spec: QuantSpec, *, channel_axes: Sequence[int] = ()) -> np.ndarray:
    """Fake-quantize a whole array, such as a weight kernel.

    Parameters
    ----------
    values : numpy.ndarray
        Array to quantize.
    spec : QuantSpec
        Quantization settings.
    channel_axes : Sequence[int], optional
        Axes that index channels.
        Required for per-channel specs and ignored for per-tensor specs.

    Returns
    -------
    numpy.ndarray
        Dequantized array with the input's shape and float dtype.
    """
    array = np.asarray(values)
    out_dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
    if spec.granularity == "per_channel":
        matrix, axes, moved_shape = channel_matrix(array, channel_axes)
    else:
        matrix, axes, moved_shape = array.reshape(-1, 1), (), array.shape
    lo, hi = estimate_range(matrix, spec)
    scale, zero_point = compute_qparams(lo, hi, spec)
    quantized = fake_quantize(matrix, scale, zero_point, spec.qmin, spec.qmax)
    if spec.granularity == "per_channel":
        quantized = restore_channel_matrix(quantized, axes, moved_shape)
    return quantized.reshape(array.shape).astype(out_dtype)


def activation_qparams(stats: ActivationStats, spec: QuantSpec) -> UniformQuantParams:
    """Compute frozen quantization parameters for a calibrated activation.

    Parameters
    ----------
    stats : ActivationStats
        Calibration statistics whose channels are the last tensor axis.
    spec : QuantSpec
        Quantization settings.

    Returns
    -------
    UniformQuantParams
        One group for per-tensor specs, or one group per channel.
    """
    if spec.granularity == "per_tensor":
        lo, hi = estimate_range(
            stats.samples.reshape(-1, 1),
            spec,
            data_min=[float(np.min(stats.channel_min))],
            data_max=[float(np.max(stats.channel_max))],
        )
    else:
        lo, hi = estimate_range(stats.samples, spec, data_min=stats.channel_min, data_max=stats.channel_max)
    scale, zero_point = compute_qparams(lo, hi, spec)
    return UniformQuantParams(scale=scale, zero_point=zero_point, qmin=spec.qmin, qmax=spec.qmax)
