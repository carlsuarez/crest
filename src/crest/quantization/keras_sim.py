# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Simulate uniform quantization on trained Keras models.

The simulator quantizes a Keras model in place inside a context manager and
restores the float model on exit.
It finds layers nested inside subclassed layers, such as keras-tcn residual
blocks, as well as top-level functional layers.

Quantization points
-------------------
Every ``Dense``, ``Conv1D``/``Conv2D``/``Conv3D``, and
``DepthwiseConv1D``/``DepthwiseConv2D`` layer is quantizable.
For each one a :class:`LayerQuantConfig` can set two specs.

- A weight spec replaces the kernel with its fake-quantized value.
  A ``BatchNormalization`` layer that directly and solely consumes the layer's
  output is folded into the kernel first, as the TFLite converter does, so
  per-tensor ranges see the folded weights.
- An activation spec fake-quantizes the layer's input tensor with ranges
  calibrated on the float model, as TFLite post-training quantization does.

Biases stay in float.
TFLite stores them as int32 with the product of the input and weight scales,
which is close to lossless.
Tensors that do not feed a quantizable layer stay in float, for example
residual-add outputs on identity skip paths, model outputs, and pooling outputs
that feed no quantizable layer.
A TFLite integer graph requantizes those tensors as well.
Comparing the 8-bit simulation with a real TFLite int8 export bounds that
difference for a given model.

The implementation overrides ``call`` on layer instances, which Keras 3 honors
in eager and ``tf.function`` execution.
It was written against Keras 3 as bundled with TensorFlow 2.20.
"""

from __future__ import annotations

import contextlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import tensorflow as tf

from .uniform import (
    ActivationObserver,
    ActivationStats,
    QuantSpec,
    UniformQuantParams,
    activation_qparams,
    quantize_array,
)

logger = logging.getLogger(__name__)

LAYER_KINDS = ("dense", "conv", "depthwise_conv")
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_CALIBRATION_VALUES = 500_000


@dataclass(frozen=True)
class LayerQuantConfig:
    """Quantization settings for one quantizable layer.

    Parameters
    ----------
    weight : QuantSpec | None, optional
        Spec for the layer's kernel, or None to keep float weights.
    activation : QuantSpec | None, optional
        Spec for the layer's input tensor, or None to keep a float input.

    Raises
    ------
    TypeError
        If a spec is neither None nor a :class:`QuantSpec`.
    """

    weight: QuantSpec | None = None
    activation: QuantSpec | None = None

    def __post_init__(self) -> None:
        """Validate spec types.

        Raises
        ------
        TypeError
            If a spec is neither None nor a :class:`QuantSpec`.
        """
        for name in ("weight", "activation"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, QuantSpec):
                raise TypeError(f"LayerQuantConfig.{name} must be a QuantSpec or None.")

    @property
    def is_float(self) -> bool:
        """Return whether this config leaves the layer in float.

        Returns
        -------
        bool
            True when neither weights nor inputs are quantized.
        """
        return self.weight is None and self.activation is None


@dataclass(frozen=True)
class QuantizableLayer:
    """One quantizable layer found in a model.

    Parameters
    ----------
    layer_id : str
        Unique identifier, the Keras layer path when available.
    kind : {"dense", "conv", "depthwise_conv"}
        Layer category.
    kernel_shape : tuple[int, ...]
        Kernel shape.
    channel_axes : tuple[int, ...]
        Kernel axes that index output channels, used by per-channel weights.
    input_shape : tuple[int | None, ...]
        Input shape seen during tracing, without the batch axis.
    execution_index : int
        Position of the layer's first call in the forward pass.
    folded_batch_norm : str | None
        Identifier of the BatchNormalization layer folded into the kernel.
    layer : Any
        The Keras layer object.
    batch_norm : Any, optional
        The folded BatchNormalization layer object, if any.
    """

    layer_id: str
    kind: str
    kernel_shape: tuple[int, ...]
    channel_axes: tuple[int, ...]
    input_shape: tuple[int | None, ...]
    execution_index: int
    folded_batch_norm: str | None
    layer: Any = field(repr=False, compare=False)
    batch_norm: Any = field(default=None, repr=False, compare=False)

    @property
    def weight_count(self) -> int:
        """Return the number of kernel weights.

        Returns
        -------
        int
            Product of the kernel dimensions.
        """
        return int(np.prod(self.kernel_shape))

    @property
    def output_channels(self) -> int:
        """Return the number of output channels.

        Returns
        -------
        int
            Product of the kernel's output-channel dimensions.
        """
        return int(np.prod([self.kernel_shape[axis] for axis in self.channel_axes]))

    def describe(self) -> dict[str, Any]:
        """Return JSON- and CSV-friendly fields for this layer.

        Returns
        -------
        dict[str, Any]
            Identifier, kind, shapes, counts, and BatchNorm folding.
        """
        return {
            "layer_id": self.layer_id,
            "kind": self.kind,
            "execution_index": self.execution_index,
            "kernel_shape": "x".join(str(dim) for dim in self.kernel_shape),
            "weight_count": self.weight_count,
            "output_channels": self.output_channels,
            "input_shape": "x".join("?" if dim is None else str(dim) for dim in self.input_shape),
            "input_channels": self.input_shape[-1] if self.input_shape else None,
            "folded_batch_norm": self.folded_batch_norm,
        }


@dataclass
class _CallRecord:
    """One recorded leaf-layer call from the tracing pass.

    Parameters
    ----------
    layer : Any
        Called layer.
    input_ids : tuple[int, ...]
        Python ids of the tensor inputs.
    output_ids : tuple[int, ...]
        Python ids of the tensor outputs.
    input_shape : tuple[int | None, ...]
        Shape of the first tensor input without the batch axis.
    """

    layer: Any
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    input_shape: tuple[int | None, ...]


def uniform_model_config(
    layer_ids: Sequence[str],
    *,
    weight: QuantSpec | None = None,
    activation: QuantSpec | None = None,
) -> dict[str, LayerQuantConfig]:
    """Build a config that applies the same specs to every listed layer.

    Parameters
    ----------
    layer_ids : Sequence[str]
        Layers to quantize.
    weight : QuantSpec | None, optional
        Weight spec for every layer.
    activation : QuantSpec | None, optional
        Input-activation spec for every layer.

    Returns
    -------
    dict[str, LayerQuantConfig]
        Per-layer config mapping.
    """
    config = LayerQuantConfig(weight=weight, activation=activation)
    return {layer_id: config for layer_id in layer_ids}


def fake_quantize_tf(values: tf.Tensor, scale: Any, zero_point: Any, qmin: float, qmax: float) -> tf.Tensor:
    """Quantize and dequantize a tensor with frozen parameters.

    Parameters
    ----------
    values : tensorflow.Tensor
        Tensor to quantize.
    scale : array-like
        Positive step size, scalar or broadcastable along the last axis.
    zero_point : array-like
        Integer zero point, broadcastable like ``scale``.
    qmin, qmax : float
        Integer grid bounds.

    Returns
    -------
    tensorflow.Tensor
        Dequantized tensor with the input dtype.

    Notes
    -----
    Rounding is half away from zero, matching
    :func:`crest.quantization.uniform.round_half_away_from_zero`.
    """
    step = tf.cast(scale, values.dtype)
    offset = tf.cast(zero_point, values.dtype)
    scaled = values / step
    rounded = tf.sign(scaled) * tf.floor(tf.abs(scaled) + 0.5)
    integers = tf.clip_by_value(rounded + offset, qmin, qmax)
    return (integers - offset) * step


def layer_kind(layer: Any) -> str | None:
    """Classify a Keras layer for quantization.

    Parameters
    ----------
    layer : Any
        Keras layer.

    Returns
    -------
    str | None
        ``"dense"``, ``"conv"``, or ``"depthwise_conv"``; None for other layers.
    """
    layers = tf.keras.layers
    depthwise = tuple(
        cls for cls in (getattr(layers, "DepthwiseConv1D", None), getattr(layers, "DepthwiseConv2D", None)) if cls
    )
    if depthwise and isinstance(layer, depthwise):
        return "depthwise_conv"
    if isinstance(layer, (layers.Conv1D, layers.Conv2D, layers.Conv3D)):
        return "conv"
    if isinstance(layer, layers.Dense):
        return "dense"
    return None


def _kernel_channel_axes(kind: str, kernel_rank: int) -> tuple[int, ...]:
    """Return the kernel axes that index output channels.

    Parameters
    ----------
    kind : str
        Layer category from :func:`layer_kind`.
    kernel_rank : int
        Kernel rank.

    Returns
    -------
    tuple[int, ...]
        Output-channel axes. Depthwise kernels ``(..., in, multiplier)`` have
        one output channel per ``(in, multiplier)`` pair.
    """
    if kind == "depthwise_conv":
        return (kernel_rank - 2, kernel_rank - 1)
    return (kernel_rank - 1,)


def _is_container(layer: Any) -> bool:
    """Return whether a layer tracks sublayers.

    Parameters
    ----------
    layer : Any
        Keras layer.

    Returns
    -------
    bool
        True for models and subclassed layers that own other layers.
    """
    flatten = getattr(layer, "_flatten_layers", None)
    if flatten is None:
        return False
    return any(True for _ in flatten(include_self=False, recursive=False))


def _leaf_layers(model: Any) -> list[Any]:
    """Return every leaf layer reachable from a model.

    Parameters
    ----------
    model : Any
        Keras model.

    Returns
    -------
    list[Any]
        Layers without sublayers, excluding input layers.
    """
    input_layer = tf.keras.layers.InputLayer
    return [
        layer
        for layer in model._flatten_layers(include_self=False, recursive=True)
        if not isinstance(layer, input_layer) and not _is_container(layer)
    ]


def _layer_id(layer: Any) -> str:
    """Return a readable identifier for a layer.

    Parameters
    ----------
    layer : Any
        Keras layer.

    Returns
    -------
    str
        The Keras 3 layer path (for example ``tcn/residual_block_0/conv1D_0``),
        or the layer name when no path is available.
    """
    path = getattr(layer, "path", None)
    if isinstance(path, str) and path:
        return path
    return str(layer.name)


def _tensor_leaves(value: Any) -> list[Any]:
    """Return the tensors inside a possibly nested call argument.

    Parameters
    ----------
    value : Any
        Tensor or nested structure.

    Returns
    -------
    list[Any]
        Tensor leaves in flattening order.
    """
    return [item for item in tf.nest.flatten(value) if tf.is_tensor(item)]


@contextlib.contextmanager
def _patched_calls(patches: Sequence[tuple[Any, Callable[[Callable], Callable]]]) -> Iterator[None]:
    """Temporarily override ``call`` on layer instances.

    Parameters
    ----------
    patches : Sequence[tuple[Any, Callable]]
        Pairs of layer and factory; each factory receives the original bound
        ``call`` and returns its replacement.

    Yields
    ------
    None
        Control while the overrides are active.
    """
    patched: list[Any] = []
    try:
        for layer, factory in patches:
            original = layer.call
            layer.call = factory(original)
            patched.append(layer)
        yield
    finally:
        for layer in reversed(patched):
            # Deleting the instance attribute restores the class method.
            with contextlib.suppress(AttributeError):
                del layer.call


def _as_float_batch(inputs: Any) -> np.ndarray:
    """Convert model inputs to a float32 array with a sample axis.

    Parameters
    ----------
    inputs : array-like
        Inputs for a single-input model.

    Returns
    -------
    numpy.ndarray
        Float32 array.

    Raises
    ------
    ValueError
        If the inputs are empty, lack a sample axis, or are not finite.
    """
    data = np.asarray(inputs, dtype=np.float32)
    if data.ndim < 2 or data.shape[0] == 0:
        raise ValueError("Inputs must be a non-empty array with a leading sample axis.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Inputs must be finite.")
    return data


def _batches(data: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield consecutive batches.

    Parameters
    ----------
    data : numpy.ndarray
        Array with a leading sample axis.
    batch_size : int
        Batch size.

    Yields
    ------
    numpy.ndarray
        Consecutive slices of at most ``batch_size`` samples.
    """
    for start in range(0, data.shape[0], batch_size):
        yield data[start : start + batch_size]


def _trace_calls(model: Any, sample: np.ndarray) -> tuple[list[_CallRecord], list[Any]]:
    """Run one eager forward pass and record every leaf-layer call.

    Parameters
    ----------
    model : Any
        Keras model.
    sample : numpy.ndarray
        Input batch used for the trace.

    Returns
    -------
    tuple[list[_CallRecord], list[Any]]
        Call records in execution order, and the traced tensors.
        Keep the tensors alive while using the ids in the records, so Python
        cannot reuse an id for a new object.
    """
    records: list[_CallRecord] = []
    keepalive: list[Any] = []

    def make_recorder(layer: Any) -> Callable[[Callable], Callable]:
        def factory(original: Callable) -> Callable:
            def recorder(inputs: Any, *args: Any, **kwargs: Any) -> Any:
                outputs = original(inputs, *args, **kwargs)
                input_tensors = _tensor_leaves(inputs)
                output_tensors = _tensor_leaves(outputs)
                keepalive.extend(input_tensors)
                keepalive.extend(output_tensors)
                input_shape = tuple(input_tensors[0].shape[1:]) if input_tensors else ()
                records.append(
                    _CallRecord(
                        layer=layer,
                        input_ids=tuple(id(tensor) for tensor in input_tensors),
                        output_ids=tuple(id(tensor) for tensor in output_tensors),
                        input_shape=input_shape,
                    )
                )
                return outputs

            return recorder

        return factory

    with _patched_calls([(layer, make_recorder(layer)) for layer in _leaf_layers(model)]):
        model(tf.constant(sample), training=False)
    return records, keepalive


def _batch_norm_scale(batch_norm: Any) -> np.ndarray:
    """Return the per-channel multiplier of an inference-mode BatchNorm.

    Parameters
    ----------
    batch_norm : Any
        Keras ``BatchNormalization`` layer.

    Returns
    -------
    numpy.ndarray
        ``gamma / sqrt(moving_variance + epsilon)`` in float64.
    """
    variance = np.asarray(batch_norm.moving_variance.numpy(), dtype=np.float64)
    gamma_variable = getattr(batch_norm, "gamma", None)
    gamma = np.ones_like(variance) if gamma_variable is None else np.asarray(gamma_variable.numpy(), dtype=np.float64)
    return gamma / np.sqrt(variance + float(batch_norm.epsilon))


def _find_batch_norm_folds(records: Sequence[_CallRecord]) -> dict[int, Any]:
    """Find BatchNorm layers that TFLite would fold into a preceding kernel.

    Parameters
    ----------
    records : Sequence[_CallRecord]
        Leaf-layer calls from :func:`_trace_calls`.

    Returns
    -------
    dict[int, Any]
        Mapping from ``id(quantizable_layer)`` to the BatchNorm layer to fold.

    Notes
    -----
    A pair qualifies when the BatchNorm's only input is the output of a
    quantizable layer, that output has no other consumer, both layers run once
    per forward pass, and the BatchNorm normalizes the last axis.
    Pass-through calls (inference-mode dropout returns its input object) are
    ignored when counting consumers.
    """
    batch_norm_cls = tf.keras.layers.BatchNormalization
    call_counts = Counter(id(record.layer) for record in records)
    producers: dict[int, _CallRecord] = {}
    consumers: dict[int, list[_CallRecord]] = defaultdict(list)
    for record in records:
        if set(record.input_ids) & set(record.output_ids):
            continue
        for tensor_id in record.input_ids:
            consumers[tensor_id].append(record)
        for tensor_id in record.output_ids:
            producers[tensor_id] = record

    folds: dict[int, Any] = {}
    for record in records:
        if not isinstance(record.layer, batch_norm_cls) or len(record.input_ids) != 1:
            continue
        source = producers.get(record.input_ids[0])
        if source is None or layer_kind(source.layer) is None:
            continue
        if call_counts[id(source.layer)] != 1 or call_counts[id(record.layer)] != 1:
            continue
        if len(consumers[record.input_ids[0]]) != 1:
            continue
        # Functional graphs also record consumers that are not layers.
        if len(getattr(source.layer, "_outbound_nodes", ()) or ()) > 1:
            continue
        axis = getattr(record.layer, "axis", -1)
        axis = axis[0] if isinstance(axis, (list, tuple)) and len(axis) == 1 else axis
        rank = len(record.input_shape) + 1
        if axis not in (-1, rank - 1):
            continue
        folds[id(source.layer)] = record.layer
    return folds


def discover_quantizable_layers(
    model: Any,
    sample: Any,
    *,
    fold_batch_norm: bool = True,
) -> list[QuantizableLayer]:
    """List the quantizable layers of a model in execution order.

    Parameters
    ----------
    model : Any
        Keras model with a single input.
    sample : array-like
        Input batch (one sample is enough) used to trace the forward pass.
    fold_batch_norm : bool, optional
        Whether to detect BatchNorm layers that can be folded into kernels.

    Returns
    -------
    list[QuantizableLayer]
        Quantizable layers that run during the forward pass.

    Raises
    ------
    ValueError
        If a quantizable layer has no built kernel, or a foldable BatchNorm's
        channel count does not match its producer.
    """
    data = _as_float_batch(sample)
    records, keepalive = _trace_calls(model, data)
    folds = _find_batch_norm_folds(records) if fold_batch_norm else {}

    first_call: dict[int, _CallRecord] = {}
    order: dict[int, int] = {}
    for record in records:
        if layer_kind(record.layer) is not None and id(record.layer) not in order:
            order[id(record.layer)] = len(order)
            first_call[id(record.layer)] = record
    del keepalive

    found: list[QuantizableLayer] = []
    used_ids: Counter[str] = Counter()
    for layer in sorted(
        (layer for layer in _leaf_layers(model) if id(layer) in order), key=lambda item: order[id(item)]
    ):
        kind = layer_kind(layer)
        kernel = getattr(layer, "kernel", None)
        if kernel is None:
            raise ValueError(f"Quantizable layer {_layer_id(layer)!r} has no kernel.")
        if getattr(layer, "quantization_mode", None):
            raise ValueError(f"Layer {_layer_id(layer)!r} already uses Keras quantization.")
        kernel_shape = tuple(int(dim) for dim in kernel.shape)
        channel_axes = _kernel_channel_axes(kind, len(kernel_shape))
        batch_norm = folds.get(id(layer))
        if batch_norm is not None:
            out_channels = int(np.prod([kernel_shape[axis] for axis in channel_axes]))
            if _batch_norm_scale(batch_norm).shape != (out_channels,):
                raise ValueError(f"BatchNorm after {_layer_id(layer)!r} does not match its output channels.")
        base_id = _layer_id(layer)
        used_ids[base_id] += 1
        layer_id = base_id if used_ids[base_id] == 1 else f"{base_id}#{used_ids[base_id]}"
        found.append(
            QuantizableLayer(
                layer_id=layer_id,
                kind=kind,
                kernel_shape=kernel_shape,
                channel_axes=channel_axes,
                input_shape=tuple(None if dim is None else int(dim) for dim in first_call[id(layer)].input_shape),
                execution_index=order[id(layer)],
                folded_batch_norm=None if batch_norm is None else _layer_id(batch_norm),
                layer=layer,
                batch_norm=batch_norm,
            )
        )
    skipped = [
        _layer_id(layer) for layer in _leaf_layers(model) if layer_kind(layer) is not None and id(layer) not in order
    ]
    if skipped:
        logger.warning("Quantizable layers not called in the forward pass were skipped: %s", ", ".join(skipped))
    return found


def collect_activation_stats(
    model: Any,
    layers: Sequence[QuantizableLayer],
    calibration_inputs: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_values: int = DEFAULT_MAX_CALIBRATION_VALUES,
    seed: int = 0,
) -> dict[str, ActivationStats]:
    """Calibrate the input range of every quantizable layer on the float model.

    Parameters
    ----------
    model : Any
        Keras model with a single input.
    layers : Sequence[QuantizableLayer]
        Layers to observe.
    calibration_inputs : array-like
        Calibration samples with a leading sample axis.
    batch_size : int, optional
        Calibration batch size.
    max_values : int, optional
        Per-layer cap on stored sample values.
    seed : int, optional
        Base seed for row sampling.

    Returns
    -------
    dict[str, ActivationStats]
        Statistics keyed by layer identifier.
    """
    data = _as_float_batch(calibration_inputs)
    observers = {
        layer.layer_id: ActivationObserver(max_values=max_values, seed=seed + index)
        for index, layer in enumerate(layers)
    }

    def make_observer(layer_id: str) -> Callable[[Callable], Callable]:
        def factory(original: Callable) -> Callable:
            def observe(inputs: Any, *args: Any, **kwargs: Any) -> Any:
                observers[layer_id].update(np.asarray(inputs))
                return original(inputs, *args, **kwargs)

            return observe

        return factory

    with _patched_calls([(layer.layer, make_observer(layer.layer_id)) for layer in layers]):
        for batch in _batches(data, batch_size):
            # Eager calls let the observers read concrete values.
            model(tf.constant(batch), training=False)
    return {layer_id: observer.finalize() for layer_id, observer in observers.items()}


class QuantizationSimulator:
    """Apply simulated uniform quantization to a trained Keras model.

    Parameters
    ----------
    model : Any
        Trained Keras model with a single input.
        The simulator modifies it only inside :meth:`apply`.
    calibration_inputs : array-like
        Calibration samples used to trace the model and calibrate activations.
    batch_size : int, optional
        Batch size for calibration and prediction.
    max_calibration_values : int, optional
        Per-layer cap on stored calibration sample values.
    seed : int, optional
        Seed for calibration row sampling.
    fold_batch_norm : bool, optional
        Whether weight quantization folds BatchNorm into kernels first.

    Raises
    ------
    ValueError
        If the model has no quantizable layers.

    Examples
    --------
    >>> simulator = QuantizationSimulator(model, calibration_inputs)
    >>> config = uniform_model_config(
    ...     simulator.layer_ids,
    ...     weight=QuantSpec(4),
    ...     activation=QuantSpec(8, symmetric=False, granularity="per_tensor"),
    ... )
    >>> predictions = simulator.predict(test_inputs, config)
    """

    def __init__(
        self,
        model: Any,
        calibration_inputs: Any,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_calibration_values: int = DEFAULT_MAX_CALIBRATION_VALUES,
        seed: int = 0,
        fold_batch_norm: bool = True,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive.")
        data = _as_float_batch(calibration_inputs)
        self._model = model
        self._batch_size = int(batch_size)
        self._layers = tuple(discover_quantizable_layers(model, data[:1], fold_batch_norm=fold_batch_norm))
        if not self._layers:
            raise ValueError("The model has no quantizable Dense or convolution layers.")
        self._by_id = {layer.layer_id: layer for layer in self._layers}
        self._original_kernels = {
            layer.layer_id: np.array(layer.layer.kernel.numpy(), copy=True) for layer in self._layers
        }
        self._fold_scales: dict[str, np.ndarray] = {}
        for layer in self._layers:
            if layer.batch_norm is None:
                continue
            # Channel axes are trailing, so the multiplier broadcasts after a reshape.
            rank = len(layer.kernel_shape)
            trailing = tuple(layer.kernel_shape[axis] for axis in layer.channel_axes)
            scale = _batch_norm_scale(layer.batch_norm)
            self._fold_scales[layer.layer_id] = scale.reshape((1,) * (rank - len(trailing)) + trailing)
        self._stats = collect_activation_stats(
            model,
            self._layers,
            data,
            batch_size=self._batch_size,
            max_values=max_calibration_values,
            seed=seed,
        )
        self._kernel_cache: dict[tuple[str, QuantSpec], np.ndarray] = {}
        self._param_cache: dict[tuple[str, QuantSpec], UniformQuantParams] = {}
        self._active = False

    @property
    def model(self) -> Any:
        """Return the wrapped model.

        Returns
        -------
        Any
            The Keras model passed to the constructor.
        """
        return self._model

    @property
    def layers(self) -> tuple[QuantizableLayer, ...]:
        """Return the quantizable layers in execution order.

        Returns
        -------
        tuple[QuantizableLayer, ...]
            Discovered layers.
        """
        return self._layers

    @property
    def layer_ids(self) -> tuple[str, ...]:
        """Return quantizable layer identifiers in execution order.

        Returns
        -------
        tuple[str, ...]
            Layer identifiers.
        """
        return tuple(layer.layer_id for layer in self._layers)

    def activation_stats(self, layer_id: str) -> ActivationStats:
        """Return calibration statistics for a layer's input.

        Parameters
        ----------
        layer_id : str
            Layer identifier.

        Returns
        -------
        ActivationStats
            Statistics gathered on the float model.
        """
        self._require_layer(layer_id)
        return self._stats[layer_id]

    def quantized_kernel(self, layer_id: str, spec: QuantSpec) -> np.ndarray:
        """Return the fake-quantized kernel for a layer.

        Parameters
        ----------
        layer_id : str
            Layer identifier.
        spec : QuantSpec
            Weight spec.

        Returns
        -------
        numpy.ndarray
            Kernel to load into the layer.
            With a folded BatchNorm, the folded kernel is quantized and then
            divided by the BatchNorm multiplier again, so the unchanged
            BatchNorm layer reproduces the folded, quantized convolution.
        """
        layer = self._require_layer(layer_id)
        key = (layer_id, spec)
        if key not in self._kernel_cache:
            original = self._original_kernels[layer_id]
            kernel = original.astype(np.float64)
            fold = self._fold_scales.get(layer_id)
            if fold is None:
                quantized = quantize_array(kernel, spec, channel_axes=layer.channel_axes)
            else:
                folded = quantize_array(kernel * fold, spec, channel_axes=layer.channel_axes)
                # A zero multiplier makes the channel constant, so its kernel is irrelevant.
                nonzero = fold != 0
                quantized = np.where(nonzero, folded / np.where(nonzero, fold, 1.0), kernel)
            self._kernel_cache[key] = quantized.astype(original.dtype)
        return self._kernel_cache[key]

    def activation_params(self, layer_id: str, spec: QuantSpec) -> UniformQuantParams:
        """Return frozen input-quantization parameters for a layer.

        Parameters
        ----------
        layer_id : str
            Layer identifier.
        spec : QuantSpec
            Activation spec; per-channel specs use the last input axis.

        Returns
        -------
        UniformQuantParams
            Calibrated scale and zero point.
        """
        self._require_layer(layer_id)
        key = (layer_id, spec)
        if key not in self._param_cache:
            self._param_cache[key] = activation_qparams(self._stats[layer_id], spec)
        return self._param_cache[key]

    @contextlib.contextmanager
    def apply(self, config: Mapping[str, LayerQuantConfig] | None) -> Iterator[Any]:
        """Quantize the model for the duration of a ``with`` block.

        Parameters
        ----------
        config : Mapping[str, LayerQuantConfig] | None
            Per-layer settings; missing layers stay in float.

        Yields
        ------
        Any
            The quantized model.
            ``model.predict`` and ``tf.function`` traces made inside the block
            see the quantized model.

        Raises
        ------
        RuntimeError
            If called while another :meth:`apply` block is active.
        KeyError
            If the config names an unknown layer.
        """
        if self._active:
            raise RuntimeError("QuantizationSimulator.apply is not reentrant.")
        resolved = self._resolve_config(config)
        assigned: list[str] = []
        patches = []
        for layer_id, layer_config in resolved.items():
            if layer_config.activation is not None:
                params = self.activation_params(layer_id, layer_config.activation)
                patches.append((self._by_id[layer_id].layer, _input_quantizer_factory(params)))
        self._active = True
        try:
            for layer_id, layer_config in resolved.items():
                if layer_config.weight is not None:
                    self._by_id[layer_id].layer.kernel.assign(self.quantized_kernel(layer_id, layer_config.weight))
                    assigned.append(layer_id)
            self._reset_predict_function()
            with _patched_calls(patches):
                yield self._model
        finally:
            for layer_id in assigned:
                self._by_id[layer_id].layer.kernel.assign(self._original_kernels[layer_id])
            self._reset_predict_function()
            self._active = False

    def predict(
        self,
        inputs: Any,
        config: Mapping[str, LayerQuantConfig] | None = None,
        *,
        batch_size: int | None = None,
    ) -> Any:
        """Run inference with a quantization config.

        Parameters
        ----------
        inputs : array-like
            Inputs with a leading sample axis.
        config : Mapping[str, LayerQuantConfig] | None, optional
            Per-layer settings; None runs the float model.
        batch_size : int | None, optional
            Override for the simulator batch size.

        Returns
        -------
        Any
            Model outputs as NumPy arrays, in the model's output structure.
        """
        data = _as_float_batch(inputs)
        size = self._batch_size if batch_size is None else int(batch_size)
        signature = [tf.TensorSpec(shape=(None,) + data.shape[1:], dtype=tf.float32)]
        with self.apply(config) as model:
            # A fresh trace per call picks up the active kernels and input hooks.
            forward = tf.function(
                lambda batch: model(batch, training=False), input_signature=signature, autograph=False
            )
            outputs = [
                tf.nest.map_structure(lambda tensor: tensor.numpy(), forward(tf.constant(batch)))
                for batch in _batches(data, size)
            ]
        return tf.nest.map_structure(lambda *parts: np.concatenate(parts, axis=0), *outputs)

    def _require_layer(self, layer_id: str) -> QuantizableLayer:
        """Look up a layer by identifier.

        Parameters
        ----------
        layer_id : str
            Layer identifier.

        Returns
        -------
        QuantizableLayer
            The matching layer.

        Raises
        ------
        KeyError
            If the identifier is unknown.
        """
        try:
            return self._by_id[layer_id]
        except KeyError as exc:
            raise KeyError(f"Unknown quantizable layer {layer_id!r}; known layers: {list(self._by_id)}") from exc

    def _resolve_config(self, config: Mapping[str, LayerQuantConfig] | None) -> dict[str, LayerQuantConfig]:
        """Validate a config mapping.

        Parameters
        ----------
        config : Mapping[str, LayerQuantConfig] | None
            Per-layer settings.

        Returns
        -------
        dict[str, LayerQuantConfig]
            Validated non-float entries.

        Raises
        ------
        TypeError
            If an entry is not a :class:`LayerQuantConfig`.
        """
        resolved: dict[str, LayerQuantConfig] = {}
        for layer_id, layer_config in (config or {}).items():
            self._require_layer(layer_id)
            if not isinstance(layer_config, LayerQuantConfig):
                raise TypeError(f"Config for {layer_id!r} must be a LayerQuantConfig.")
            if not layer_config.is_float:
                resolved[layer_id] = layer_config
        return resolved

    def _reset_predict_function(self) -> None:
        """Drop Keras' cached predict function so it retraces.

        Returns
        -------
        None
            Clears ``model.predict_function`` when the model has one.
        """
        if hasattr(self._model, "predict_function"):
            self._model.predict_function = None


def _input_quantizer_factory(params: UniformQuantParams) -> Callable[[Callable], Callable]:
    """Build a ``call`` override that fake-quantizes the layer input.

    Parameters
    ----------
    params : UniformQuantParams
        Frozen input-quantization parameters.

    Returns
    -------
    Callable[[Callable], Callable]
        Factory for :func:`_patched_calls`.
    """
    scale = np.asarray(params.scale, dtype=np.float32)
    zero_point = np.asarray(params.zero_point, dtype=np.float32)
    # Per-tensor parameters are stored as one-element arrays; use scalars so
    # they broadcast against inputs of any rank.
    if scale.size == 1:
        scale = scale.reshape(())
        zero_point = zero_point.reshape(())
    qmin, qmax = float(params.qmin), float(params.qmax)

    def factory(original: Callable) -> Callable:
        def quantized_call(inputs: Any, *args: Any, **kwargs: Any) -> Any:
            return original(fake_quantize_tf(inputs, scale, zero_point, qmin, qmax), *args, **kwargs)

        return quantized_call

    return factory
