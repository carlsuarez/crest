# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Rewrite a TFLite int8 model so chosen activation tensors are 4-bit (INT4).

TensorFlow's converter cannot emit INT4 activation tensors, so this module
edits the flatbuffer of CREST's int8 export.
It measures each chosen tensor on calibration clips with TFLite's built-in
kernels, picks a 4-bit clipping range with the estimators in
:mod:`crest.quantization.uniform`, and stores the tensor as INT4 with that
scale and zero point.
Weights are unchanged.
Each convolution's int32 bias is rescaled to its new input scale, and the
runtime recomputes the requantization multipliers from the tensor scales.

Which tensors can be 4-bit
--------------------------
The analysis works on any single-subgraph model and converts only what the
4-bit-activation patch in ``env_setup/patches`` can run, so the output never
needs an unsupported kernel.

- Requantizing operators read and write each tensor with its own scale, so
  each of their activation inputs and outputs may be int8 or INT4
  independently: ``CONV_2D`` and ``DEPTHWISE_CONV_2D`` (int8 weights, batch 1),
  ``ADD`` (equal shapes, no broadcasting), and ``MEAN`` over height and width
  (4-bit input only; its output stays int8).
- Scale-preserving operators keep the scale and zero point, so their input and
  output form one *group* that must be all INT4 or all int8, with one set of
  parameters: ``PAD`` (constant paddings), ``RESHAPE``, ``EXPAND_DIMS``,
  ``MAX_POOL_2D`` and ``AVERAGE_POOL_2D`` (batch 1).
- ``SHAPE`` reads only dimensions, so it may read any of these tensors.
- A group qualifies when every operator that reads or writes any of its tensors
  is listed above at a supported position, and none of its tensors is a model
  input or output.
  Anything else, such as a group read by ``FULLY_CONNECTED`` or
  ``STRIDED_SLICE``, stays int8; :func:`explain_int4_groups` says why.

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
# Operators the patch runs with INT4 activations. Requantizing operators map to the activation input and output
# positions that may be INT4.
REQUANTIZING_OPS = {
    "CONV_2D": ((0,), (0,)),
    "DEPTHWISE_CONV_2D": ((0,), (0,)),
    "ADD": ((0, 1), (0,)),
    "MEAN": ((0,), ()),
}
# Operators whose input 0 and output 0 share a scale and zero point, so they join one group.
SCALE_PRESERVING_OPS = frozenset({"PAD", "RESHAPE", "EXPAND_DIMS", "MAX_POOL_2D", "AVERAGE_POOL_2D"})
# Operators that read only a tensor's dimensions, never its values, so any input type works.
SHAPE_ONLY_OPS = frozenset({"SHAPE"})
SUPPORTED_OPS = frozenset(REQUANTIZING_OPS) | SCALE_PRESERVING_OPS
CONVOLUTION_OPS = frozenset({"CONV_2D", "DEPTHWISE_CONV_2D"})
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


@dataclass
class _Analysis:
    """A parsed model with each tensor's producers and consumers."""

    model: schema.ModelT
    graph: schema.SubGraphT
    operators: list[_Operator]
    producers: dict[int, list[_Operator]]
    consumers: dict[int, list[tuple[_Operator, int]]]

    def is_activation(self, tensor: int) -> bool:
        """Return True for a non-constant, non-variable tensor of the subgraph."""
        if not 0 <= tensor < len(self.graph.tensors):
            return False
        entry = self.graph.tensors[tensor]
        return not entry.isVariable and _constant_data(self.model, entry) is None


def _analyze(model_bytes: bytes) -> _Analysis:
    """Parse a model and index which operators produce and consume each tensor."""
    model, graph, operators = _load(model_bytes)
    producers: dict[int, list[_Operator]] = {}
    consumers: dict[int, list[tuple[_Operator, int]]] = {}
    for op in operators:
        for tensor in op.outputs:
            producers.setdefault(tensor, []).append(op)
        for position, tensor in enumerate(op.inputs):
            if tensor >= 0:
                consumers.setdefault(tensor, []).append((op, position))
    return _Analysis(model, graph, operators, producers, consumers)


def _shape(analysis: _Analysis, tensor: int) -> tuple[int, ...]:
    shape = analysis.graph.tensors[tensor].shape
    return tuple(int(d) for d in shape) if shape is not None else ()


def _operator_reason(analysis: _Analysis, op: _Operator) -> str | None:
    """Explain why the patch cannot run ``op`` with 4-bit activations, or return None."""
    tensors = analysis.graph.tensors
    if op.name in CONVOLUTION_OPS:
        if tensors[op.inputs[1]].type != schema.TensorType.INT8:
            return f"{op.name} has non-int8 weights"
        for tensor in (op.inputs[0], op.outputs[0]):
            shape = _shape(analysis, tensor)
            if len(shape) != 4 or shape[0] != 1:
                return f"{op.name} is not a batch-1 4-D operator"
    elif op.name == "ADD":
        if len({_shape(analysis, t) for t in (op.inputs[0], op.inputs[1], op.outputs[0])}) != 1:
            return "ADD broadcasts"
    elif op.name == "MEAN":
        axes = _constant_data(analysis.model, tensors[op.inputs[1]])
        axes = None if axes is None else sorted(axes.view(np.int32).tolist())
        shape = _shape(analysis, op.inputs[0])
        if axes not in ([1, 2], [-3, -2]) or len(shape) != 4 or shape[0] != 1:
            return "MEAN does not reduce height and width of a batch-1 4-D tensor"
        if tensors[op.outputs[0]].type != schema.TensorType.INT8:
            return "MEAN output is not int8"
    elif op.name == "PAD":
        if len(op.inputs) != 2 or _constant_data(analysis.model, tensors[op.inputs[1]]) is None:
            return "PAD needs constant paddings and no fill value"
        if not 1 <= len(_shape(analysis, op.inputs[0])) <= 5:
            return "PAD has more than 5 dimensions"
    elif op.name in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
        shape = _shape(analysis, op.inputs[0])
        if len(shape) != 4 or shape[0] != 1:
            return f"{op.name} is not a batch-1 4-D operator"
    return None


def _all_groups(analysis: _Analysis) -> list[tuple[int, ...]]:
    """Group int8 activation tensors joined by scale-preserving operators, in execution order."""
    parent: dict[int, int] = {}

    def find(tensor: int) -> int:
        parent.setdefault(tensor, tensor)
        while parent[tensor] != tensor:
            parent[tensor] = parent[parent[tensor]]
            tensor = parent[tensor]
        return tensor

    first_seen: dict[int, int] = {int(t): -1 for t in analysis.graph.inputs}
    for op in analysis.operators:
        for tensor in (*op.inputs, *op.outputs):
            if analysis.is_activation(tensor):
                first_seen.setdefault(tensor, op.index)
                find(tensor)
        if op.name in SCALE_PRESERVING_OPS and op.inputs and op.outputs:
            source, target = op.inputs[0], op.outputs[0]
            if analysis.is_activation(source) and analysis.is_activation(target):
                parent[find(source)] = find(target)
    members: dict[int, list[int]] = {}
    for tensor in first_seen:
        if analysis.graph.tensors[tensor].type == schema.TensorType.INT8:
            members.setdefault(find(tensor), []).append(tensor)
    groups = [tuple(sorted(group, key=lambda t: (first_seen[t], t))) for group in members.values()]
    return sorted(groups, key=lambda group: (first_seen[group[0]], group[0]))


def _group_reason(analysis: _Analysis, group: Sequence[int]) -> str | None:
    """Explain why a group must stay int8, or return None when it can be INT4."""
    model_inputs = {int(t) for t in analysis.graph.inputs}
    model_outputs = {int(t) for t in analysis.graph.outputs}
    params = set()
    requantized = False
    for tensor in group:
        entry = analysis.graph.tensors[tensor]
        if entry.type != schema.TensorType.INT8:
            return "not int8"
        if tensor in model_inputs:
            return "includes the model input"
        if tensor in model_outputs:
            return "includes a model output"
        quantization = entry.quantization
        if quantization is None or quantization.scale is None or len(quantization.scale) != 1:
            return "not per-tensor quantized"
        params.add((float(quantization.scale[0]), int(quantization.zeroPoint[0])))
        for op in analysis.producers.get(tensor, []):
            if op.name not in SUPPORTED_OPS:
                return f"written by {op.name}"
            if op.name in REQUANTIZING_OPS:
                if op.outputs.index(tensor) not in REQUANTIZING_OPS[op.name][1]:
                    return f"{op.name} cannot write a 4-bit output"
                requantized = True
            reason = _operator_reason(analysis, op)
            if reason is not None:
                return reason
        for op, position in analysis.consumers.get(tensor, []):
            if op.name in SHAPE_ONLY_OPS:
                continue
            if op.name not in SUPPORTED_OPS:
                return f"read by {op.name}"
            if op.name in REQUANTIZING_OPS:
                if position not in REQUANTIZING_OPS[op.name][0]:
                    return f"{op.name} cannot read a 4-bit tensor at input {position}"
            elif position != 0:
                return f"{op.name} reads it at input {position}"
            reason = _operator_reason(analysis, op)
            if reason is not None:
                return reason
    if len(params) != 1:
        return "its tensors have different scales"
    if not requantized:
        return "no requantizing operator writes it"
    return None


def explain_int4_groups(model_bytes: bytes) -> list[dict[str, Any]]:
    """Describe every int8 activation group and whether it can be INT4.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.

    Returns
    -------
    list[dict[str, Any]]
        One entry per group in execution order, with ``tensors``, ``operators``
        (the operators that touch it), ``eligible`` and ``reason``.
    """
    analysis = _analyze(model_bytes)
    report = []
    for group in _all_groups(analysis):
        touching = sorted(
            {op.name for t in group for op in analysis.producers.get(t, [])}
            | {op.name for t in group for op, _ in analysis.consumers.get(t, [])}
        )
        reason = _group_reason(analysis, group)
        report.append({"tensors": list(group), "operators": touching, "eligible": reason is None, "reason": reason})
    return report


def int4_tensor_groups(model_bytes: bytes) -> tuple[tuple[int, ...], ...]:
    """List the groups of tensors that can be stored as INT4 together.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Tensor index groups in execution order. A group's tensors must share one
        set of 4-bit parameters.
    """
    analysis = _analyze(model_bytes)
    return tuple(group for group in _all_groups(analysis) if _group_reason(analysis, group) is None)


def int4_feature_map_candidates(model_bytes: bytes) -> tuple[int, ...]:
    """List every tensor that can be stored as INT4.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.

    Returns
    -------
    tuple[int, ...]
        Tensor indices of all qualifying groups, in execution order.
    """
    return tuple(tensor for group in int4_tensor_groups(model_bytes) for tensor in group)


def _expand_to_groups(model_bytes: bytes, tensors: Sequence[int]) -> list[tuple[int, ...]]:
    """Map tensor indices to the qualifying groups that contain them.

    Raises
    ------
    ValueError
        If a tensor is in no qualifying group.
    """
    analysis = _analyze(model_bytes)
    groups = _all_groups(analysis)
    owner = {tensor: group for group in groups for tensor in group}
    chosen: list[tuple[int, ...]] = []
    for tensor in tensors:
        group = owner.get(int(tensor))
        if group is None:
            raise ValueError(f"Tensor {tensor} cannot be INT4: not an int8 activation tensor.")
        reason = _group_reason(analysis, group)
        if reason is not None:
            raise ValueError(f"Tensor {tensor} cannot be INT4: {reason}.")
        if group not in chosen:
            chosen.append(group)
    order = {group: index for index, group in enumerate(groups)}
    return sorted(chosen, key=order.__getitem__)


def model_warnings(model_bytes: bytes) -> list[str]:
    """Flag export settings that make the pinned Arduino TFLite Micro library compute wrong results.

    Parameters
    ----------
    model_bytes : bytes
        Serialized TFLite model.

    Returns
    -------
    list[str]
        Human-readable warnings; empty when none apply.
    """
    analysis = _analyze(model_bytes)
    warnings = []
    names = {op.name for op in analysis.operators}
    dynamic = any(
        entry.shapeSignature is not None and any(int(d) < 0 for d in entry.shapeSignature)
        for entry in (analysis.graph.tensors[int(t)] for t in analysis.graph.inputs)
    )
    if dynamic and names & {"SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"}:
        warnings.append(
            "The model was exported with a dynamic batch, so its SPACE_TO_BATCH_ND and BATCH_TO_SPACE_ND tensors "
            "store batch 1. TensorFlow Lite Micro allocates and computes from the stored shapes and produces wrong "
            "outputs. Export from a model with a fixed batch of 1, which also turns these operators into dilated "
            "convolutions."
        )
    elif dynamic:
        warnings.append("The model has a dynamic batch dimension; TensorFlow Lite Micro uses the stored batch of 1.")
    if any(
        op.name == "FULLY_CONNECTED"
        and analysis.graph.tensors[op.inputs[1]].quantization is not None
        and analysis.graph.tensors[op.inputs[1]].quantization.scale is not None
        and len(analysis.graph.tensors[op.inputs[1]].quantization.scale) > 1
        for op in analysis.operators
    ):
        warnings.append(
            "Dense layers use per-channel scales, which the pinned Arduino TensorFlow Lite Micro library applies "
            "with the first channel's scale only. Convert with "
            "converter._experimental_disable_per_channel_quantization_for_dense_layers = True."
        )
    return warnings


def _observe(
    model_bytes: bytes,
    calibration_inputs: np.ndarray,
    keys: Mapping[Any, Sequence[int]],
    *,
    max_values: int,
    seed: int,
) -> dict[Any, ActivationStats]:
    """Run the calibration clips and accumulate the dequantized values of each key's tensors in one observer."""
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
    observers = {
        key: ActivationObserver(max_values=max_values, seed=seed + int(tensors[0])) for key, tensors in keys.items()
    }
    for clip in inputs:
        quantized = np.clip(np.round(clip / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(detail["index"], quantized[np.newaxis])
        interpreter.invoke()
        for key, tensors in keys.items():
            for tensor in tensors:
                quantization = graph.tensors[tensor].quantization
                scale, zero_point = float(quantization.scale[0]), int(quantization.zeroPoint[0])
                raw = interpreter.get_tensor(tensor)
                observers[key].update((raw.astype(np.float32) - zero_point) * scale)
    return {key: observer.finalize() for key, observer in observers.items()}


def collect_activation_stats(
    model_bytes: bytes,
    calibration_inputs: np.ndarray,
    tensors: Sequence[int],
    *,
    max_values: int = 500_000,
    seed: int = 0,
) -> dict[int, ActivationStats]:
    """Measure int8 tensors on calibration clips with TFLite's built-in kernels.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model with a single int8 input.
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
    keys = {int(t): (int(t),) for t in tensors}
    return _observe(model_bytes, calibration_inputs, keys, max_values=max_values, seed=seed)


def _group_roots(model_bytes: bytes, group: Sequence[int]) -> tuple[int, ...]:
    """Return a group's tensors written by requantizing operators (padding and copies add nothing new)."""
    analysis = _analyze(model_bytes)
    roots = tuple(t for t in group if any(op.name in REQUANTIZING_OPS for op in analysis.producers.get(t, [])))
    return roots or tuple(group)


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
        New scale and zero point per tensor index. Every tensor of a group
        (see :func:`int4_tensor_groups`) must be present with the same values.

    Returns
    -------
    bytes
        The rewritten model, with 16-byte aligned buffers.

    Raises
    ------
    ValueError
        If a tensor cannot be INT4, a group is incomplete or has mixed
        parameters, a zero point is outside ``[-8, 7]``, or a bias scale is
        not the product of its input and weight scales.
    """
    analysis = _analyze(model_bytes)
    model, graph, operators = analysis.model, analysis.graph, analysis.operators
    owner = {tensor: group for group in _all_groups(analysis) for tensor in group}
    for tensor, new in params.items():
        group = owner.get(int(tensor))
        reason = "not an int8 activation tensor" if group is None else _group_reason(analysis, group)
        if reason is not None:
            raise ValueError(f"Tensor {tensor} cannot be INT4: {reason}.")
        if not -8 <= new.zero_point <= 7 or not new.scale > 0:
            raise ValueError(f"Tensor {tensor} has invalid INT4 parameters {new}.")
        if any(params.get(member) != new for member in group):
            raise ValueError(
                f"Tensors {list(group)} share a scale through scale-preserving operators; "
                "convert them together with the same parameters."
            )
    for tensor, new in params.items():
        entry = graph.tensors[int(tensor)]
        old_scale = float(entry.quantization.scale[0])
        entry.type = schema.TensorType.INT4
        entry.quantization.scale = [new.scale]
        entry.quantization.zeroPoint = [new.zero_point]
        entry.quantization.min = None
        entry.quantization.max = None
        for op in operators:
            if op.name not in CONVOLUTION_OPS or op.inputs[0] != tensor or len(op.inputs) < 3 or op.inputs[2] < 0:
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
    It is a lower bound. On the audio DS-CNNs the planner's activation region
    equalled it exactly; on TCN models with many same-size tensors the greedy
    planner needed up to 39% more. Confirm with the runtime's own arena report.

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

    Starting from every candidate group at 4 bits, groups are returned to 8
    bits one at a time, last first, while the estimated peak does not grow.
    Only tensors live at the memory peak matter, so the result is often about
    half the candidates.
    Keeping the other tensors at 8 bits costs less accuracy.

    Parameters
    ----------
    model_bytes : bytes
        Serialized int8 TFLite model.
    candidates : Sequence[int], optional
        Tensors allowed to be 4-bit, expanded to their groups; defaults to
        every qualifying group.

    Returns
    -------
    tuple[int, ...]
        Tensor indices of the kept groups, in execution order.
    """
    pool = list(int4_tensor_groups(model_bytes)) if candidates is None else _expand_to_groups(model_bytes, candidates)

    def flatten(groups: Sequence[tuple[int, ...]]) -> list[int]:
        return [tensor for group in groups for tensor in group]

    target = estimate_activation_peak_bytes(model_bytes, flatten(pool))
    keep = list(pool)
    changed = True
    while changed:
        changed = False
        for group in reversed(pool):
            if group in keep:
                trial = [g for g in keep if g != group]
                if estimate_activation_peak_bytes(model_bytes, flatten(trial)) <= target:
                    keep = trial
                    changed = True
    return tuple(flatten([g for g in pool if g in keep]))


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
        ``"all"`` converts every qualifying group, ``"minimal"`` the smallest
        set of groups with the same estimated memory peak; tensor indices
        convert the groups that contain them.
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
        The rewritten model and a JSON-friendly report, including the groups
        that stayed int8 with their reasons, and warnings about export
        settings that break the pinned Arduino runtime.

    Raises
    ------
    ValueError
        If no tensor qualifies or an explicit tensor cannot be INT4.
    """
    all_groups = list(int4_tensor_groups(model_bytes))
    if isinstance(tensors, str):
        if tensors not in {"all", "minimal"}:
            raise ValueError("tensors must be 'all', 'minimal', or a sequence of tensor indices.")
        if tensors == "all":
            groups = all_groups
        else:
            kept = set(minimal_peak_tensors(model_bytes))
            groups = [group for group in all_groups if group[0] in kept]
        selection = tensors
    else:
        groups = _expand_to_groups(model_bytes, [int(t) for t in tensors])
        selection = "explicit"
    if not groups:
        raise ValueError("No tensor can be stored as INT4 in this model.")
    roots = {group: _group_roots(model_bytes, group) for group in groups}
    stats = _observe(model_bytes, calibration_inputs, roots, max_values=max_values, seed=seed)
    group_params = {g: int4_params(stats[g], range_method=range_method, percentile=percentile) for g in groups}
    params = {tensor: group_params[group] for group in groups for tensor in group}
    rewritten = rewrite_with_int4_activations(model_bytes, params)
    _, graph, _ = _load(model_bytes)
    chosen = [tensor for group in groups for tensor in group]
    report = {
        "selection": selection,
        "range_method": range_method,
        "calibration_clips": int(np.asarray(calibration_inputs).shape[0]),
        "candidates": [tensor for group in all_groups for tensor in group],
        "groups": [
            {
                "tensors": list(group),
                "calibrated_on": list(roots[group]),
                "int4_scale": group_params[group].scale,
                "int4_zero_point": group_params[group].zero_point,
                "clip_low": group_params[group].clip_range[0],
                "clip_high": group_params[group].clip_range[1],
                "observed_max": float(stats[group].channel_max.max()),
            }
            for group in groups
        ],
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
                "observed_max": float(stats[group].channel_max.max()),
            }
            for group in groups
            for t in group
        ],
        "kept_int8": [entry for entry in explain_int4_groups(model_bytes) if not entry["eligible"]],
        "estimated_activation_peak_bytes": {
            "int8": estimate_activation_peak_bytes(model_bytes),
            "int4": estimate_activation_peak_bytes(model_bytes, chosen),
        },
        "buffers_16_byte_aligned": all(offset % BUFFER_ALIGNMENT == 0 for offset in buffer_data_offsets(rewritten)),
        "warnings": model_warnings(model_bytes),
    }
    return rewritten, report


__all__ = [
    "ARENA_ALIGNMENT",
    "BUFFER_ALIGNMENT",
    "Int4Params",
    "buffer_data_offsets",
    "collect_activation_stats",
    "estimate_activation_peak_bytes",
    "explain_int4_groups",
    "export_int4_activations",
    "int4_feature_map_candidates",
    "int4_params",
    "int4_tensor_groups",
    "minimal_peak_tensors",
    "model_warnings",
    "rewrite_with_int4_activations",
]

