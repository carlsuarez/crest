# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for rewriting int8 TFLite models with 4-bit feature maps."""

from __future__ import annotations

import json

import flatbuffers
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tensorflow.lite.python import schema_py_generated as schema  # noqa: E402

from crest.hardware import convert_to_tflite_model  # noqa: E402
from crest.quantization import tflite_int4_activations as q4  # noqa: E402


@pytest.fixture(scope="module")
def int8_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, np.ndarray]:
    """Return a small depthwise-separable int8 model and its calibration clips."""
    keras = tf.keras
    inputs = keras.Input((24, 24, 1))
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(16, 1, activation="relu")(x)
    x = keras.layers.DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(16, 1, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    model = keras.Model(inputs, keras.layers.Dense(4)(x))
    clips = np.random.default_rng(0).normal(size=(24, 24, 24, 1)).astype(np.float32)
    path = tmp_path_factory.mktemp("int4_activations") / "model.tflite"
    convert_to_tflite_model(model, training_data=clips, quantization_mode="int8_ptq", output_name=path)
    return path.read_bytes(), clips


def _chain_model() -> bytes:
    """Build A -> B -> C with int8 tensors of 100, 200, and 50 elements, using RESHAPE operators."""
    code = schema.OperatorCodeT()
    code.builtinCode = schema.BuiltinOperator.RESHAPE
    code.deprecatedBuiltinCode = schema.BuiltinOperator.RESHAPE
    code.version = 1
    graph = schema.SubGraphT()
    graph.tensors = []
    for index, size in enumerate((100, 200, 50)):
        tensor = schema.TensorT()
        tensor.shape = [1, size]
        tensor.type = schema.TensorType.INT8
        tensor.buffer = 0
        tensor.name = f"t{index}".encode()
        graph.tensors.append(tensor)
    graph.inputs = [0]
    graph.outputs = [2]
    graph.operators = []
    for source, destination in ((0, 1), (1, 2)):
        operator = schema.OperatorT()
        operator.opcodeIndex = 0
        operator.inputs = [source]
        operator.outputs = [destination]
        graph.operators.append(operator)
    model = schema.ModelT()
    model.version = 3
    model.operatorCodes = [code]
    model.subgraphs = [graph]
    model.buffers = [schema.BufferT()]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def test_peak_estimate_follows_tflm_lifetimes_and_alignment() -> None:
    """Sizes round up to 16 bytes and a tensor lives from its producer to its last consumer."""
    data = _chain_model()
    # Scope 1 holds A and B: 112 + 208.
    assert q4.estimate_activation_peak_bytes(data) == 320
    # B at half a byte per element is 100 bytes: scope 1 holds 112 + 112.
    assert q4.estimate_activation_peak_bytes(data, {1}) == 224
    # A also halves to 64: scope 1 holds 64 + 112, scope 2 holds 112 + 64.
    assert q4.estimate_activation_peak_bytes(data, {0, 1}) == 176


def test_candidates_are_convolution_outputs_feeding_convolutions_or_mean(int8_model) -> None:
    """Only conv and depthwise outputs that the patched kernels can consume are listed."""
    data, _ = int8_model
    model, graph, operators = q4._load(data)
    candidates = q4.int4_feature_map_candidates(data)
    assert len(candidates) == 5
    for tensor in candidates:
        producer = next(op for op in operators if tensor in op.outputs)
        assert producer.name in {"CONV_2D", "DEPTHWISE_CONV_2D"}
    # No scale-preserving operators here, so every group is a single tensor.
    assert q4.int4_tensor_groups(data) == tuple((t,) for t in candidates)
    eligible = {tuple(entry["tensors"]) for entry in q4.explain_int4_groups(data) if entry["eligible"]}
    assert eligible == {(t,) for t in candidates}
    assert int(graph.inputs[0]) not in candidates
    assert int(graph.outputs[0]) not in candidates
    mean_output = next(op.outputs[0] for op in operators if op.name == "MEAN")
    assert mean_output not in candidates


def test_unsupported_tensors_are_rejected(int8_model) -> None:
    """Inputs, outputs, pooling results, and unknown indices cannot be INT4."""
    data, _ = int8_model
    _, graph, operators = q4._load(data)
    mean_output = next(op.outputs[0] for op in operators if op.name == "MEAN")
    params = q4.Int4Params(scale=0.1, zero_point=-8)
    for tensor in (int(graph.inputs[0]), int(graph.outputs[0]), mean_output, 10_000):
        with pytest.raises(ValueError, match="cannot be INT4"):
            q4.rewrite_with_int4_activations(data, {tensor: params})
    candidate = q4.int4_feature_map_candidates(data)[0]
    for bad in (q4.Int4Params(0.1, 9), q4.Int4Params(0.1, -9), q4.Int4Params(0.0, 0)):
        with pytest.raises(ValueError, match="invalid INT4 parameters"):
            q4.rewrite_with_int4_activations(data, {candidate: bad})


def test_stats_and_params_use_the_signed_four_bit_grid(int8_model) -> None:
    """Fused-ReLU outputs start at zero, so their zero point is -8 and the range is clipped."""
    data, clips = int8_model
    candidates = q4.int4_feature_map_candidates(data)
    stats = q4.collect_activation_stats(data, clips, candidates)
    assert set(stats) == set(candidates)
    _, graph, _ = q4._load(data)
    for tensor, entry in stats.items():
        assert entry.channels == int(graph.tensors[tensor].shape[-1])
        params = q4.int4_params(entry)
        assert params.scale > 0
        assert params.zero_point == -8
        low, high = params.clip_range
        assert low == pytest.approx(0.0)
        assert high <= float(entry.channel_max.max()) + params.scale
        minmax = q4.int4_params(entry, range_method="minmax")
        assert params.scale <= minmax.scale + 1e-12


def test_rewrite_changes_only_chosen_tensors_and_keeps_biases_consistent(int8_model) -> None:
    """Chosen tensors become INT4; weights and other tensors are untouched; biases follow the new scale."""
    data, clips = int8_model
    candidates = q4.int4_feature_map_candidates(data)
    chosen = candidates[1:3]
    stats = q4.collect_activation_stats(data, clips, chosen)
    params = {t: q4.int4_params(stats[t]) for t in chosen}
    rewritten = q4.rewrite_with_int4_activations(data, params)

    before_model, before, before_ops = q4._load(data)
    after_model, after, after_ops = q4._load(rewritten)
    assert [op.name for op in before_ops] == [op.name for op in after_ops]
    rescaled = {
        op.inputs[2]
        for op in before_ops
        if op.name in {"CONV_2D", "DEPTHWISE_CONV_2D"} and op.inputs[0] in params and len(op.inputs) > 2
    }
    assert rescaled
    for index, (old, new) in enumerate(zip(before.tensors, after.tensors)):
        if index in params:
            assert new.type == schema.TensorType.INT4
            assert new.quantization.scale[0] == pytest.approx(params[index].scale)
            assert new.quantization.zeroPoint[0] == params[index].zero_point
            continue
        assert new.type == old.type
        if index in rescaled:
            continue
        old_scale = np.asarray([] if old.quantization.scale is None else old.quantization.scale)
        new_scale = np.asarray([] if new.quantization.scale is None else new.quantization.scale)
        assert np.allclose(old_scale, new_scale)
        old_data, new_data = before_model.buffers[old.buffer].data, after_model.buffers[new.buffer].data
        assert (old_data is None) == (new_data is None) or len(old_data if old_data is not None else []) == 0
        if old_data is not None and len(old_data):
            assert np.array_equal(np.asarray(old_data), np.asarray(new_data))

    checked = 0
    for old_op, new_op in zip(before_ops, after_ops):
        if old_op.name in {"CONV_2D", "DEPTHWISE_CONV_2D"} and old_op.inputs[0] in params:
            bias_old, bias_new = before.tensors[old_op.inputs[2]], after.tensors[new_op.inputs[2]]
            filter_scales = np.asarray(after.tensors[new_op.inputs[1]].quantization.scale, dtype=np.float64)
            input_scale = params[old_op.inputs[0]].scale
            assert np.allclose(bias_new.quantization.scale, input_scale * filter_scales, rtol=1e-5)
            old_real = np.frombuffer(np.asarray(before_model.buffers[bias_old.buffer].data).tobytes(), np.int32) \
                * np.asarray(bias_old.quantization.scale, dtype=np.float64)
            new_real = np.frombuffer(np.asarray(after_model.buffers[bias_new.buffer].data).tobytes(), np.int32) \
                * np.asarray(bias_new.quantization.scale, dtype=np.float64)
            assert np.all(np.abs(old_real - new_real) <= 0.5 * np.asarray(bias_new.quantization.scale) + 1e-12)
            checked += 1
    assert checked >= 1
    # The converter deduplicates identical constants, so each rescaled bias must own its buffer afterward.
    assert len({after.tensors[i].buffer for i in rescaled}) == len(rescaled)
    assert all(offset % 16 == 0 for offset in q4.buffer_data_offsets(rewritten))
    assert q4.estimate_activation_peak_bytes(rewritten) == q4.estimate_activation_peak_bytes(data, chosen)


def test_minimal_set_keeps_the_peak_and_every_member_is_needed(int8_model) -> None:
    """Dropping any tensor of the minimal set raises the estimated peak above the all-candidates peak."""
    data, _ = int8_model
    candidates = q4.int4_feature_map_candidates(data)
    minimal = q4.minimal_peak_tensors(data)
    assert set(minimal) <= set(candidates)
    target = q4.estimate_activation_peak_bytes(data, candidates)
    assert target < q4.estimate_activation_peak_bytes(data)
    assert q4.estimate_activation_peak_bytes(data, minimal) == target
    for tensor in minimal:
        assert q4.estimate_activation_peak_bytes(data, [t for t in minimal if t != tensor]) > target


@pytest.mark.parametrize("selection", ["all", "minimal"])
def test_export_returns_a_model_and_a_json_report(int8_model, selection: str) -> None:
    """The one-call export reports its tensors, the estimated memory saving, and buffer alignment."""
    data, clips = int8_model
    rewritten, report = q4.export_int4_activations(data, clips, tensors=selection)
    json.dumps(report)
    expected = q4.int4_feature_map_candidates(data) if selection == "all" else q4.minimal_peak_tensors(data)
    assert [entry["tensor"] for entry in report["tensors"]] == list(expected)
    assert report["estimated_activation_peak_bytes"]["int4"] < report["estimated_activation_peak_bytes"]["int8"]
    assert report["buffers_16_byte_aligned"] is True
    assert report["calibration_clips"] == len(clips)
    _, graph, _ = q4._load(rewritten)
    assert all(graph.tensors[entry["tensor"]].type == schema.TensorType.INT4 for entry in report["tensors"])


def test_export_validates_the_selection(int8_model) -> None:
    """Bad keywords and unsupported explicit tensors raise instead of writing a broken model."""
    data, clips = int8_model
    with pytest.raises(ValueError, match="'all', 'minimal'"):
        q4.export_int4_activations(data, clips, tensors="some")
    with pytest.raises(ValueError, match="cannot be INT4"):
        q4.export_int4_activations(data, clips, tensors=[0])
    with pytest.raises(ValueError, match="No tensor"):
        q4.export_int4_activations(data, clips, tensors=[])


def _convert(model, clips: np.ndarray, path) -> bytes:
    convert_to_tflite_model(model, training_data=clips, quantization_mode="int8_ptq", output_name=path)
    return path.read_bytes()


def _tcn(batch_shape: tuple[int, ...] | None):
    """A causal, dilated Conv1D stack with a residual add and a last-step head, like keras-tcn."""
    keras = tf.keras
    inputs = keras.Input(batch_shape=batch_shape) if batch_shape else keras.Input(shape=(40, 3))
    x = keras.layers.Conv1D(8, 3, padding="causal", activation="relu")(inputs)
    y = keras.layers.Conv1D(8, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    y = keras.layers.Conv1D(8, 3, padding="causal", dilation_rate=4)(y)
    x = keras.layers.Activation("relu")(keras.layers.Add()([x, y]))
    x = keras.layers.Lambda(lambda t: t[:, -1, :])(x)
    return keras.Model(inputs, keras.layers.Dense(2)(x))


@pytest.fixture(scope="module")
def tcn_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, np.ndarray]:
    """Return a fixed-batch TCN-style int8 model and its calibration clips."""
    clips = np.random.default_rng(1).normal(size=(24, 40, 3)).astype(np.float32)
    path = tmp_path_factory.mktemp("int4_tcn") / "tcn.tflite"
    return _convert(_tcn((1, 40, 3)), clips, path), clips


@pytest.fixture(scope="module")
def pool_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, np.ndarray]:
    """Return a small CNN with max and average pooling between convolutions."""
    keras = tf.keras
    inputs = keras.Input(batch_shape=(1, 16, 16, 2))
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(8, 1, activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    model = keras.Model(inputs, keras.layers.Dense(2)(x))
    clips = np.random.default_rng(2).normal(size=(24, 16, 16, 2)).astype(np.float32)
    path = tmp_path_factory.mktemp("int4_pool") / "pool.tflite"
    return _convert(model, clips, path), clips


def test_tcn_groups_span_pad_and_reshape_and_stop_at_the_slice(tcn_model) -> None:
    """Causal padding and Conv1D reshapes join their convolution's output; the sliced head stays int8."""
    data, _ = tcn_model
    _, _, operators = q4._load(data)
    assert {op.name for op in operators} >= {"PAD", "RESHAPE", "CONV_2D", "ADD", "STRIDED_SLICE"}
    assert any(getattr(op.options, "dilationWFactor", 1) > 1 for op in operators if op.name == "CONV_2D")
    entries = q4.explain_int4_groups(data)
    assert any(e["eligible"] and {"PAD", "RESHAPE", "CONV_2D"} <= set(e["operators"]) for e in entries)
    sliced = [e for e in entries if "STRIDED_SLICE" in e["operators"] and "ADD" in e["operators"]]
    assert sliced and not sliced[0]["eligible"] and sliced[0]["reason"] == "read by STRIDED_SLICE"
    groups = q4.int4_tensor_groups(data)
    assert any(len(group) > 1 for group in groups)
    assert q4.estimate_activation_peak_bytes(data, q4.int4_feature_map_candidates(data)) < (
        q4.estimate_activation_peak_bytes(data)
    )


def test_groups_are_converted_whole_with_one_set_of_parameters(tcn_model) -> None:
    """A partial group, or a group with mixed parameters, is refused; a whole group shares one scale."""
    data, clips = tcn_model
    group = next(g for g in q4.int4_tensor_groups(data) if len(g) > 1)
    params = q4.Int4Params(scale=0.05, zero_point=-8)
    with pytest.raises(ValueError, match="convert them together"):
        q4.rewrite_with_int4_activations(data, {group[0]: params})
    mixed = {t: params for t in group}
    mixed[group[-1]] = q4.Int4Params(scale=0.06, zero_point=-8)
    with pytest.raises(ValueError, match="convert them together"):
        q4.rewrite_with_int4_activations(data, mixed)
    rewritten = q4.rewrite_with_int4_activations(data, {t: params for t in group})
    _, graph, _ = q4._load(rewritten)
    assert all(graph.tensors[t].type == schema.TensorType.INT4 for t in group)
    assert [float(graph.tensors[t].quantization.scale[0]) for t in group] == pytest.approx([0.05] * len(group))
    # Naming one tensor converts its whole group.
    _, report = q4.export_int4_activations(data, clips, tensors=[group[-1]])
    assert [entry["tensor"] for entry in report["tensors"]] == list(group)
    assert report["groups"][0]["tensors"] == list(group)
    assert any(entry["reason"] == "read by STRIDED_SLICE" for entry in report["kept_int8"])


def test_pooling_joins_groups_and_exports(pool_model) -> None:
    """Max and average pooling keep the scale, so each joins the convolution output it reads."""
    data, clips = pool_model
    entries = q4.explain_int4_groups(data)
    for pool in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
        entry = next(e for e in entries if pool in e["operators"])
        assert entry["eligible"] and len(entry["tensors"]) == 2
    rewritten, report = q4.export_int4_activations(data, clips, tensors="all")
    assert report["buffers_16_byte_aligned"] is True
    assert report["estimated_activation_peak_bytes"]["int4"] < report["estimated_activation_peak_bytes"]["int8"]
    _, graph, _ = q4._load(rewritten)
    assert all(graph.tensors[t].type == schema.TensorType.INT4 for t in q4.int4_feature_map_candidates(data))


def test_warnings_flag_exports_the_pinned_runtime_miscomputes(tcn_model, tmp_path) -> None:
    """A dynamic batch with space-to-batch operators, and per-channel dense layers, are both reported."""
    data, clips = tcn_model
    fixed = q4.model_warnings(data)
    assert not any("dynamic batch" in warning for warning in fixed)
    assert any("per-channel" in warning for warning in fixed)
    # CREST's converter now fixes the batch, so build the old dynamic-batch export with TensorFlow directly.
    dynamic = q4.model_warnings(tf.lite.TFLiteConverter.from_keras_model(_tcn(None)).convert())
    assert any("SPACE_TO_BATCH_ND" in warning for warning in dynamic)
    assert not any("dynamic batch" in w for w in q4.model_warnings(_convert(_tcn(None), clips, tmp_path / "f.tflite")))
