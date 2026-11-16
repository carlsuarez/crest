# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Rewrite a TFLite int8 model so chosen feature maps are 4-bit (INT4) tensors.

TensorFlow's converter cannot emit INT4 activation tensors, so this module
edits the flatbuffer of CREST's int8 export.
It measures each chosen feature map on calibration clips with TFLite's
built-in kernels, picks a 4-bit clipping range with the estimators in
:mod:`crest.quantization.uniform`, and stores the tensor as INT4 with that
scale and zero point.
Weights are unchanged.
Each consumer's int32 bias is rescaled to the new input scale, and the
runtime recomputes the requantization multipliers from the tensor scales.

Which tensors can be 4-bit
--------------------------
The 4-bit-activation patch in ``env_setup/patches`` supports INT4 tensors that
are outputs of ``CONV_2D`` or ``DEPTHWISE_CONV_2D`` with int8 weights, consumed
only by such operators or by a ``MEAN`` over height and width with an int8
output.
:func:`int4_feature_map_candidates` lists the tensors of a model that qualify.
No stock runtime executes the result: without the patch the model does not run
on the pinned Arduino TensorFlow Lite Micro library, and TensorFlow's own
interpreter does not run INT4 activations either.

Import this module explicitly, because it loads TensorFlow.
"""

from __future__ import annotations

import contextlib
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from .uniform import ActivationObserver, ActivationStats, QuantSpec, activation_qparams

FILE_IDENTIFIER = b"TFL3"
# The TFLite schema declares buffer data with ``force_align: 16`` and the converter honors it.
BUFFER_ALIGNMENT = 16
# TensorFlow Lite Micro aligns every planned activation buffer to 16 bytes.
ARENA_ALIGNMENT = 16
INT4_BITS = 4
PRODUCER_OPS = frozenset({"CONV_2D", "DEPTHWISE_CONV_2D"})
CONSUMER_OPS = frozenset({"CONV_2D", "DEPTHWISE_CONV_2D", "MEAN"})
_OP_NAMES = {value: key for key, value in vars(schema.BuiltinOperator).items() if not key.startswith("_")}
_TYPE_BYTES = {
    schema.TensorType.FLOAT32: 4,
    schema.TensorType.FLOAT16: 2,
    schema.TensorType.INT32: 4,
    schema.TensorType.UINT8: 1,
    schema.TensorType.INT64: 8,
    schema.TensorType.BOOL: 1,
    schema.TensorType.INT16: 2,
    schema.TensorType.INT8: 1,
    schema.TensorType.UINT32: 4,
    schema.TensorType.UINT16: 2,
}


@dataclass(frozen=True)
class Int4Params:
    """Scale and zero point of one INT4 activation tensor.

    Parameters
    ----------
    scale : float
        Step size of the 4-bit grid.
    zero_point : int
        Integer that represents real value zero, in ``[-8, 7]``.
    """

    scale: float
    zero_point: int

    @property
    def clip_range(self) -> tuple[float, float]:
        """Return the real values of the grid's end points.

        Returns
        -------
        tuple[float, float]
            ``(low, high)`` represented by integers -8 and 7.
        """
        return (-8 - self.zero_point) * self.scale, (7 - self.zero_point) * self.scale


@dataclass(frozen=True)
class _Operator:
    """One operator of the model's single subgraph."""

    index: int
    name: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    options: Any


def _load(model_bytes: bytes) -> tuple[schema.ModelT, schema.SubGraphT, list[_Operator]]:
    """Parse a TFLite flatbuffer with exactly one subgraph.

    Parameters
    ----------
    model_bytes : bytes
        Serialized model.

    Returns
    -------
    tuple[schema.ModelT, schema.SubGraphT, list[_Operator]]
        The model, its subgraph, and the operators in execution order.

    Raises
    ------
    ValueError
        If the model has more than one subgraph.
    """
    model = schema.ModelT.InitFromPackedBuf(bytes(model_bytes), 0)
    if len(model.subgraphs) != 1:
        raise ValueError("Only models with a single subgraph are supported.")
    graph = model.subgraphs[0]
    operators = []
    for index, operator in enumerate(graph.operators):
        code = model.operatorCodes[operator.opcodeIndex]
        operators.append(
            _Operator(
                index=index,
                name=_OP_NAMES[max(code.builtinCode, code.deprecatedBuiltinCode)],
                inputs=tuple(int(t) for t in operator.inputs),
                outputs=tuple(int(t) for t in operator.outputs),
                options=operator.builtinOptions,
            )
        )
    return model, graph, operators


def _constant_data(model: schema.ModelT, tensor: schema.TensorT) -> np.ndarray | None:
    """Return a constant tensor's raw bytes, or None for an activation tensor."""
    data = model.buffers[tensor.buffer].data
    if data is None or len(data) == 0:
        return None
    return np.asarray(data, dtype=np.uint8)


def _elements(tensor: schema.TensorT) -> int:
    """Return the element count of a tensor (1 for scalars)."""
    return int(np.prod(tensor.shape)) if tensor.shape is not None and len(tensor.shape) else 1


def _unsupported_reason(
    model: schema.ModelT, graph: schema.SubGraphT, operators: Sequence[_Operator], tensor: int
) -> str | None:
    """Explain why a tensor cannot be stored as INT4, or return None if it can.

    Parameters
    ----------
    model, graph, operators
        Parsed model from :func:`_load`.
    tensor : int
        Tensor index.

    Returns
    -------
    str | None
        A short reason, or None when the tensor qualifies.
    """
    if not 0 <= tensor < len(graph.tensors):
        return "no such tensor"
    entry = graph.tensors[tensor]
    if entry.type != schema.TensorType.INT8:
        return "not an int8 tensor"
    if tensor in {int(t) for t in graph.outputs}:
        return "model output"
    scale = None if entry.quantization is None else entry.quantization.scale
    if scale is None or len(scale) != 1:
        return "not per-tensor quantized"
    producers = [op for op in operators if tensor in op.outputs]
    if len(producers) != 1 or producers[0].name not in PRODUCER_OPS:
        return "not the output of CONV_2D or DEPTHWISE_CONV_2D"
    if graph.tensors[producers[0].inputs[1]].type != schema.TensorType.INT8:
        return "producer has non-int8 weights"
    consumers = [op for op in operators if tensor in op.inputs]
    if not consumers:
        return "no consumer"
    for op in consumers:
        if op.name not in CONSUMER_OPS or op.inputs[0] != tensor or op.inputs.count(tensor) != 1:
            return f"consumed by {op.name}"
        if op.name == "MEAN":
            axes = _constant_data(model, graph.tensors[op.inputs[1]])
            axes = None if axes is None else sorted(axes.view(np.int32).tolist())
            if axes not in ([1, 2], [-3, -2]) or len(entry.shape) != 4:
                return "MEAN does not reduce height and width of a 4-D tensor"
            if graph.tensors[op.outputs[0]].type != schema.TensorType.INT8:
                return "MEAN output is not int8"
        else:
            if graph.tensors[op.inputs[1]].type != schema.TensorType.INT8:
                return f"{op.name} consumer has non-int8 weights"
            if getattr(op.options, "dilationHFactor", 1) != 1:
                return f"{op.name} consumer dilates along height"
            if len(entry.shape) != 4 or int(entry.shape[0]) != 1:
                return "not a batch-1 4-D tensor"
    return None


def int4_feature_map_candidates(model_bytes: bytes) -> tuple[int, ...]:
    """List the tensors that can be stored as INT4 activations.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.

    Returns
    -------
    tuple[int, ...]
        Tensor indices in execution order.
    """
    model, graph, operators = _load(model_bytes)
    candidates = []
    for op in operators:
        for tensor in op.outputs:
            if op.name in PRODUCER_OPS and _unsupported_reason(model, graph, operators, tensor) is None:
                candidates.append(tensor)
    return tuple(candidates)


def collect_activation_stats(
    model_bytes: bytes,
    calibration_inputs: np.ndarray,
    tensors: Sequence[int],
    *,
    max_values: int = 500_000,
    seed: int = 0,
) -> dict[int, ActivationStats]:
    """Measure int8 feature maps on calibration clips with TFLite's built-in kernels.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model with a float-quantized single input.
    calibration_inputs : numpy.ndarray
        Float clips with shape ``(n, *input_shape[1:])``.
    tensors : Sequence[int]
        Tensor indices to observe.
    max_values : int, optional
        Stored sample values per tensor.
    seed : int, optional
        Seed for the row sampling.

    Returns
    -------
    dict[int, ActivationStats]
        Dequantized statistics per tensor, channels on the last axis.

    Raises
    ------
    ValueError
        If no clips are given or the model's input is not int8.
    """
    inputs = np.asarray(calibration_inputs, dtype=np.float32)
    if inputs.shape[0] == 0:
        raise ValueError("At least one calibration clip is required.")
    _, graph, _ = _load(model_bytes)
    interpreter = tf.lite.Interpreter(
        model_content=bytes(model_bytes),
        experimental_preserve_all_tensors=True,
        # XNNPACK is not bit-exact with the reference kernels, and the target runtime matches the reference.
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    detail = interpreter.get_input_details()[0]
    if detail["dtype"] != np.int8:
        raise ValueError("The model's input must be int8.")
    in_scale, in_zero = detail["quantization"]
    observers = {int(t): ActivationObserver(max_values=max_values, seed=seed + int(t)) for t in tensors}
    for clip in inputs:
        quantized = np.clip(np.round(clip / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(detail["index"], quantized[np.newaxis])
        interpreter.invoke()
        for tensor, observer in observers.items():
            quantization = graph.tensors[tensor].quantization
            scale, zero_point = float(quantization.scale[0]), int(quantization.zeroPoint[0])
            raw = interpreter.get_tensor(tensor)
            observer.update((raw.astype(np.float32) - zero_point) * scale)
    return {tensor: observer.finalize() for tensor, observer in observers.items()}


def int4_params(stats: ActivationStats, *, range_method: str = "mse", percentile: float = 99.99) -> Int4Params:
    """Choose a per-tensor asymmetric 4-bit scale and zero point.

    Parameters
    ----------
    stats : ActivationStats
        Calibration statistics of one tensor.
    range_method : {"minmax", "percentile", "mse"}, optional
        Clipping estimator.
        Min-max ranges are wide, so they leave few usable levels; MSE-optimal
        ranges keep far more accuracy at 4 bits.
    percentile : float, optional
        Upper percentile for ``range_method="percentile"``.

    Returns
    -------
    Int4Params
        Parameters on the signed grid ``[-8, 7]``.
    """
    spec = QuantSpec(
        bits=INT4_BITS, symmetric=False, granularity="per_tensor", range_method=range_method, percentile=percentile
    )
    params = activation_qparams(stats, spec)
    return Int4Params(scale=float(params.scale[0]), zero_point=int(params.zero_point[0]))


def _pack_aligned_buffer(self: schema.BufferT, builder: flatbuffers.Builder) -> int:
    """Pack a buffer with 16-byte aligned data, as the TFLite schema requires.

    The generated ``BufferT.Pack`` aligns data to 4 bytes, which breaks runtimes
    that read int8 and INT4 weights with wide loads.
    """
    data = None
    if self.data is not None:
        payload = np.asarray(self.data, dtype=np.uint8).tobytes()
        builder.StartVector(1, len(payload), BUFFER_ALIGNMENT)
        builder.head = builder.head - len(payload)
        builder.Bytes[builder.head : builder.head + len(payload)] = payload
        builder.vectorNumElems = len(payload)
        data = builder.EndVector()
    schema.BufferStart(builder)
    if data is not None:
        schema.BufferAddData(builder, data)
    schema.BufferAddOffset(builder, self.offset)
    schema.BufferAddSize(builder, self.size)
    return schema.BufferEnd(builder)


@contextlib.contextmanager
def _sixteen_byte_buffers() -> Iterator[None]:
    """Serialize buffers 16-byte aligned inside the block and restore the generated packer after."""
    original = schema.BufferT.Pack
    schema.BufferT.Pack = _pack_aligned_buffer
    try:
        yield
    finally:
        schema.BufferT.Pack = original


def buffer_data_offsets(model_bytes: bytes) -> list[int]:
    """Return the file offset of every non-empty buffer's data.

    Parameters
    ----------
    model_bytes : bytes
        Serialized TFLite model.

    Returns
    -------
    list[int]
        Offsets in buffer order; each should be a multiple of 16.
    """
    model = schema.Model.GetRootAsModel(bytes(model_bytes), 0)
    offsets = []
    for index in range(model.BuffersLength()):
        buffer = model.Buffers(index)
        if buffer.DataLength():
            offsets.append(int(buffer._tab.Vector(buffer._tab.Offset(4))))
    return offsets


def rewrite_with_int4_activations(model_bytes: bytes, params: Mapping[int, Int4Params]) -> bytes:
    """Store the given tensors as INT4 and rescale the biases of their consumers.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.
    params : Mapping[int, Int4Params]
        New scale and zero point per tensor index.

    Returns
    -------
    bytes
        The rewritten model, with 16-byte aligned buffers.

    Raises
    ------
    ValueError
        If a tensor cannot be INT4, a zero point is outside ``[-8, 7]``, or a
        bias scale is not the product of its input and weight scales.
    """
    model, graph, operators = _load(model_bytes)
    for tensor, new in params.items():
        reason = _unsupported_reason(model, graph, operators, int(tensor))
        if reason is not None:
            raise ValueError(f"Tensor {tensor} cannot be INT4: {reason}.")
        if not -8 <= new.zero_point <= 7 or not new.scale > 0:
            raise ValueError(f"Tensor {tensor} has invalid INT4 parameters {new}.")
    for tensor, new in params.items():
        entry = graph.tensors[int(tensor)]
        old_scale = float(entry.quantization.scale[0])
        entry.type = schema.TensorType.INT4
        entry.quantization.scale = [new.scale]
        entry.quantization.zeroPoint = [new.zero_point]
        entry.quantization.min = None
        entry.quantization.max = None
        for op in operators:
            if op.name not in PRODUCER_OPS or op.inputs[0] != tensor or len(op.inputs) < 3 or op.inputs[2] < 0:
                continue
            bias = graph.tensors[op.inputs[2]]
            if sum(1 for other in graph.tensors if other.buffer == bias.buffer) != 1:
                # The converter deduplicates identical constants, for example all-zero biases.
                # Rescaling a shared buffer in place would change other layers, so give this bias its own copy.
                private = schema.BufferT()
                private.data = np.array(model.buffers[bias.buffer].data, dtype=np.uint8)
                model.buffers.append(private)
                bias.buffer = len(model.buffers) - 1
            filter_scales = np.asarray(graph.tensors[op.inputs[1]].quantization.scale, dtype=np.float64)
            old_scales = np.asarray(bias.quantization.scale, dtype=np.float64)
            if not np.allclose(old_scales, old_scale * filter_scales, rtol=1e-5):
                raise ValueError(f"Bias scale of tensor {op.inputs[2]} is not input scale times weight scale.")
            buffer = model.buffers[bias.buffer]
            old_values = np.frombuffer(np.asarray(buffer.data, dtype=np.uint8).tobytes(), dtype=np.int32)
            new_scales = new.scale * filter_scales
            new_values = np.clip(np.round(old_values * old_scales / new_scales), -(2**31), 2**31 - 1)
            buffer.data = np.frombuffer(new_values.astype(np.int32).tobytes(), dtype=np.uint8)
            bias.quantization.scale = [float(s) for s in new_scales]
    with _sixteen_byte_buffers():
        builder = flatbuffers.Builder(len(model_bytes))
        builder.Finish(model.Pack(builder), file_identifier=FILE_IDENTIFIER)
        return bytes(builder.Output())


def _aligned(size: int) -> int:
    """Round a byte count up to the arena alignment."""
    return (size + ARENA_ALIGNMENT - 1) // ARENA_ALIGNMENT * ARENA_ALIGNMENT


def estimate_activation_peak_bytes(model_bytes: bytes, four_bit: Collection[int] = ()) -> int:
    """Estimate the activation memory a TFLite Micro arena needs.

    The estimate is the largest total size of simultaneously live activation
    tensors, each rounded up to 16 bytes, using TFLite Micro's lifetimes: model
    inputs live from the start, each operator's outputs are created at that
    operator, and a tensor lives until its last consumer.
    Tensors in ``four_bit`` (and any INT4 tensor) take half a byte per element.

    The arena also holds kernel scratch buffers and persistent data, which this
    estimate leaves out, and the greedy planner can fragment.
    On the medium and large audio models checked, the planner's activation
    region equalled this peak exactly; confirm with the runtime's own arena report.

    Parameters
    ----------
    model_bytes : bytes
        Serialized TFLite model.
    four_bit : Collection[int], optional
        Tensor indices to size as 4-bit.

    Returns
    -------
    int
        Peak live activation bytes.

    Raises
    ------
    ValueError
        If a live tensor has an unsupported element type.
    """
    model, graph, operators = _load(model_bytes)
    halved = {int(t) for t in four_bit}
    final = len(operators)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for tensor in graph.inputs:
        first.setdefault(int(tensor), 0)
        last[int(tensor)] = 0
    for position, op in enumerate(operators, start=1):
        for tensor in op.outputs:
            first.setdefault(tensor, position)
            last[tensor] = position
        for tensor in op.inputs:
            if tensor >= 0:
                last[tensor] = position
    for tensor in graph.outputs:
        first.setdefault(int(tensor), final)
        last[int(tensor)] = final
    sizes = {}
    for tensor, begin in first.items():
        entry = graph.tensors[tensor]
        if entry.isVariable or _constant_data(model, entry) is not None or tensor not in last:
            continue
        if entry.type == schema.TensorType.INT4 or tensor in halved:
            size = (_elements(entry) + 1) // 2
        elif entry.type in _TYPE_BYTES:
            size = _elements(entry) * _TYPE_BYTES[entry.type]
        else:
            raise ValueError(f"Tensor {tensor} has an unsupported element type.")
        sizes[tensor] = (begin, last[tensor], _aligned(size))
    return max(
        sum(size for begin, end, size in sizes.values() if begin <= scope <= end) for scope in range(final + 1)
    )


def minimal_peak_tensors(model_bytes: bytes, candidates: Sequence[int] | None = None) -> tuple[int, ...]:
    """Find a small set of 4-bit tensors that gives the full memory saving.

    Starting from every candidate at 4 bits, tensors are returned to 8 bits one
    at a time, last first, while the estimated peak does not grow.
    Only tensors live at the memory peak matter, so the result is often about
    half the candidates.
    Keeping the other feature maps at 8 bits costs less accuracy.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.
    candidates : Sequence[int], optional
        Tensors allowed to be 4-bit; defaults to :func:`int4_feature_map_candidates`.

    Returns
    -------
    tuple[int, ...]
        Tensor indices, in the order of ``candidates``.
    """
    pool = list(int4_feature_map_candidates(model_bytes) if candidates is None else candidates)
    target = estimate_activation_peak_bytes(model_bytes, pool)
    keep = list(pool)
    changed = True
    while changed:
        changed = False
        for tensor in reversed(pool):
            if tensor in keep:
                trial = [t for t in keep if t != tensor]
                if estimate_activation_peak_bytes(model_bytes, trial) <= target:
                    keep = trial
                    changed = True
    return tuple(t for t in pool if t in keep)


def export_int4_activations(
    model_bytes: bytes,
    calibration_inputs: np.ndarray,
    *,
    tensors: str | Sequence[int] = "all",
    range_method: str = "mse",
    percentile: float = 99.99,
    max_values: int = 500_000,
    seed: int = 0,
) -> tuple[bytes, dict[str, Any]]:
    """Calibrate and rewrite a model so chosen feature maps are INT4.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.
    calibration_inputs : numpy.ndarray
        Float calibration clips.
    tensors : {"all", "minimal"} | Sequence[int], optional
        ``"all"`` converts every candidate, ``"minimal"`` the smallest set with
        the same estimated memory peak, or give tensor indices explicitly.
    range_method : {"minmax", "percentile", "mse"}, optional
        4-bit clipping estimator.
    percentile : float, optional
        Upper percentile for the percentile method.
    max_values : int, optional
        Stored calibration values per tensor.
    seed : int, optional
        Calibration sampling seed.

    Returns
    -------
    tuple[bytes, dict[str, Any]]
        The rewritten model and a JSON-friendly report.

    Raises
    ------
    ValueError
        If no tensor qualifies or an explicit tensor cannot be INT4.
    """
    candidates = int4_feature_map_candidates(model_bytes)
    if isinstance(tensors, str):
        if tensors not in {"all", "minimal"}:
            raise ValueError("tensors must be 'all', 'minimal', or a sequence of tensor indices.")
        chosen = candidates if tensors == "all" else minimal_peak_tensors(model_bytes, candidates)
        selection = tensors
    else:
        chosen = tuple(int(t) for t in tensors)
        selection = "explicit"
    if not chosen:
        raise ValueError("No tensor can be stored as INT4 in this model.")
    model, graph, operators = _load(model_bytes)
    for tensor in chosen:
        reason = _unsupported_reason(model, graph, operators, tensor)
        if reason is not None:
            raise ValueError(f"Tensor {tensor} cannot be INT4: {reason}.")
    stats = collect_activation_stats(model_bytes, calibration_inputs, chosen, max_values=max_values, seed=seed)
    params = {t: int4_params(stats[t], range_method=range_method, percentile=percentile) for t in chosen}
    rewritten = rewrite_with_int4_activations(model_bytes, params)
    _, graph, _ = _load(model_bytes)
    report = {
        "selection": selection,
        "range_method": range_method,
        "calibration_clips": int(np.asarray(calibration_inputs).shape[0]),
        "candidates": list(candidates),
        "tensors": [
            {
                "tensor": int(t),
                "name": graph.tensors[t].name.decode(),
                "shape": [int(d) for d in graph.tensors[t].shape],
                "int8_scale": float(graph.tensors[t].quantization.scale[0]),
                "int8_zero_point": int(graph.tensors[t].quantization.zeroPoint[0]),
                "int4_scale": params[t].scale,
                "int4_zero_point": params[t].zero_point,
                "clip_low": params[t].clip_range[0],
                "clip_high": params[t].clip_range[1],
                "observed_max": float(stats[t].channel_max.max()),
            }
            for t in chosen
        ],
        "estimated_activation_peak_bytes": {
            "int8": estimate_activation_peak_bytes(model_bytes),
            "int4": estimate_activation_peak_bytes(model_bytes, chosen),
        },
        "buffers_16_byte_aligned": all(offset % BUFFER_ALIGNMENT == 0 for offset in buffer_data_offsets(rewritten)),
    }
    return rewritten, report


__all__ = [
    "ARENA_ALIGNMENT",
    "BUFFER_ALIGNMENT",
    "Int4Params",
    "buffer_data_offsets",
    "collect_activation_stats",
    "estimate_activation_peak_bytes",
    "export_int4_activations",
    "int4_feature_map_candidates",
    "int4_params",
    "minimal_peak_tensors",
    "rewrite_with_int4_activations",
]

