# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Keras quantization simulator in ``crest.quantization.keras_sim``."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from crest.quantization import QuantSpec, quantize_array  # noqa: E402
from crest.quantization.keras_sim import (  # noqa: E402
    LayerQuantConfig,
    QuantizationSimulator,
    discover_quantizable_layers,
    fake_quantize_tf,
    uniform_model_config,
)
from crest.quantization.uniform import fake_quantize  # noqa: E402

layers = tf.keras.layers
W8 = QuantSpec(8)
A8 = QuantSpec(8, symmetric=False, granularity="per_tensor")


def _randomize_batch_norms(model: tf.keras.Model, seed: int) -> None:
    """Give every BatchNorm layer non-trivial inference statistics.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Model to modify in place.
    seed : int
        Random seed.

    Returns
    -------
    None
        BatchNorm variables are reassigned.
    """
    rng = np.random.default_rng(seed)
    for layer in model._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, layers.BatchNormalization):
            channels = layer.moving_variance.shape[0]
            layer.gamma.assign(rng.uniform(0.2, 3.0, channels) * rng.choice([-1.0, 1.0], channels))
            layer.beta.assign(rng.normal(size=channels))
            layer.moving_mean.assign(rng.normal(size=channels))
            layer.moving_variance.assign(rng.uniform(0.05, 4.0, channels))


def _dscnn_like_model(seed: int = 0) -> tf.keras.Model:
    """Build a tiny DS-CNN shaped like the audio_dscnn family.

    Parameters
    ----------
    seed : int, optional
        Weight-initialization seed.

    Returns
    -------
    tensorflow.keras.Model
        Model with depthwise, pointwise, BatchNorm, ReLU, pooling, and Dense layers.
    """
    tf.keras.utils.set_random_seed(seed)
    inputs = layers.Input(shape=(12, 8))
    x = layers.Reshape((12, 8, 1))(inputs)
    x = layers.DepthwiseConv2D((3, 3), padding="same", depth_multiplier=2, use_bias=False, name="dw")(x)
    x = layers.BatchNormalization(name="dw_bn")(x)
    x = layers.ReLU(name="dw_relu")(x)
    x = layers.Conv2D(6, (1, 1), use_bias=False, name="pw")(x)
    x = layers.BatchNormalization(name="pw_bn")(x)
    x = layers.ReLU(name="pw_relu")(x)
    x = layers.Dropout(0.1, name="drop")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(4, name="logits")(x)
    model = tf.keras.Model(inputs, outputs)
    _randomize_batch_norms(model, seed)
    return model


def _inputs(count: int, seed: int = 0) -> np.ndarray:
    """Return random model inputs for the tiny DS-CNN.

    Parameters
    ----------
    count : int
        Number of samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray
        Float32 inputs of shape ``(count, 12, 8)``.
    """
    return np.random.default_rng(seed).normal(size=(count, 12, 8)).astype(np.float32)


def test_discovery_finds_layers_in_order_with_batch_norm_folds():
    """Depthwise and pointwise convs fold their BatchNorms; the head Dense does not."""
    model = _dscnn_like_model()
    found = discover_quantizable_layers(model, _inputs(1))
    assert [layer.layer_id for layer in found] == ["dw", "pw", "logits"]
    assert [layer.kind for layer in found] == ["depthwise_conv", "conv", "dense"]
    assert [layer.folded_batch_norm for layer in found] == ["dw_bn", "pw_bn", None]
    assert found[0].channel_axes == (2, 3)
    assert found[0].output_channels == 2
    assert found[1].channel_axes == (3,)
    assert found[0].input_shape == (12, 8, 1)
    assert found[2].input_shape == (6,)
    assert found[2].describe()["input_channels"] == 6


def test_float_config_reproduces_model_and_8_bit_stays_close():
    """An empty config matches the model; W8A8 is close; W2A2 is not."""
    model = _dscnn_like_model()
    calibration, test_inputs = _inputs(64, seed=1), _inputs(32, seed=2)
    simulator = QuantizationSimulator(model, calibration, batch_size=16)
    reference = model(test_inputs, training=False).numpy()
    np.testing.assert_allclose(simulator.predict(test_inputs), reference, rtol=1e-5, atol=1e-5)

    w8a8 = simulator.predict(test_inputs, uniform_model_config(simulator.layer_ids, weight=W8, activation=A8))
    w2a2 = simulator.predict(
        test_inputs,
        uniform_model_config(
            simulator.layer_ids,
            weight=QuantSpec(2),
            activation=QuantSpec(2, symmetric=False, granularity="per_tensor"),
        ),
    )
    spread = np.std(reference)
    assert np.max(np.abs(w8a8 - reference)) < 0.1 * spread
    assert np.mean(np.abs(w2a2 - reference)) > np.mean(np.abs(w8a8 - reference)) * 5


def test_folded_per_tensor_weights_match_an_explicitly_folded_model():
    """Fold-quantize-unfold equals quantizing a hand-folded conv with a folded bias."""
    tf.keras.utils.set_random_seed(3)
    inputs = layers.Input(shape=(5, 5, 3))
    x = layers.Conv2D(4, (3, 3), use_bias=True, name="conv")(inputs)
    outputs = layers.BatchNormalization(name="bn")(x)
    model = tf.keras.Model(inputs, outputs)
    _randomize_batch_norms(model, 3)
    data = np.random.default_rng(3).normal(size=(8, 5, 5, 3)).astype(np.float32)
    simulator = QuantizationSimulator(model, data)
    spec = QuantSpec(4, granularity="per_tensor")
    simulated = simulator.predict(data, {"conv": LayerQuantConfig(weight=spec)})

    conv, bn = model.get_layer("conv"), model.get_layer("bn")
    multiplier = bn.gamma.numpy() / np.sqrt(bn.moving_variance.numpy() + bn.epsilon)
    folded_kernel = quantize_array(conv.kernel.numpy().astype(np.float64) * multiplier, spec)
    folded_bias = (conv.bias.numpy() - bn.moving_mean.numpy()) * multiplier + bn.beta.numpy()
    expected = tf.nn.conv2d(data, folded_kernel.astype(np.float32), strides=1, padding="VALID") + folded_bias
    np.testing.assert_allclose(simulated, expected.numpy(), rtol=1e-4, atol=1e-4)

    unfolded = QuantizationSimulator(model, data, fold_batch_norm=False)
    assert unfolded.layers[0].folded_batch_norm is None
    assert not np.allclose(unfolded.predict(data, {"conv": LayerQuantConfig(weight=spec)}), simulated)


def test_apply_restores_kernels_calls_and_keras_predict():
    """Kernels and call methods are restored, and model.predict sees the quantized model inside apply."""
    model = _dscnn_like_model()
    data = _inputs(16)
    simulator = QuantizationSimulator(model, data)
    original = {layer.layer_id: layer.layer.kernel.numpy().copy() for layer in simulator.layers}
    reference = model.predict(data, verbose=0)
    config = uniform_model_config(
        simulator.layer_ids, weight=QuantSpec(2), activation=QuantSpec(2, symmetric=False, granularity="per_tensor")
    )
    with simulator.apply(config) as quantized_model:
        assert all("call" in layer.layer.__dict__ for layer in simulator.layers)
        inside = quantized_model.predict(data, verbose=0)
        with pytest.raises(RuntimeError):
            with simulator.apply({}):
                pass
    assert not np.allclose(inside, reference)
    for layer in simulator.layers:
        assert "call" not in layer.layer.__dict__
        np.testing.assert_array_equal(layer.layer.kernel.numpy(), original[layer.layer_id])
    np.testing.assert_allclose(model.predict(data, verbose=0), reference, rtol=1e-6, atol=1e-6)


def test_input_quantization_limits_distinct_values():
    """A 2-bit per-tensor input quantizer leaves at most four distinct input values."""
    inputs = layers.Input(shape=(6,))
    outputs = layers.Dense(6, use_bias=False, kernel_initializer="identity", name="identity")(inputs)
    model = tf.keras.Model(inputs, outputs)
    data = np.random.default_rng(4).normal(size=(200, 6)).astype(np.float32)
    simulator = QuantizationSimulator(model, data)
    spec = QuantSpec(2, symmetric=False, granularity="per_tensor")
    predictions = simulator.predict(data, {"identity": LayerQuantConfig(activation=spec)})
    assert len(np.unique(np.round(predictions, 6))) <= 4
    params = simulator.activation_params("identity", spec)
    np.testing.assert_allclose(predictions, params.apply(data), rtol=1e-5, atol=1e-6)


def test_tensorflow_and_numpy_fake_quantization_agree():
    """The graph quantizer matches the NumPy quantizer except at float32 rounding ties."""
    values = np.random.default_rng(5).normal(size=(4000, 3)).astype(np.float32)
    scale = np.array([0.05, 0.1, 0.2], dtype=np.float32)
    zero_point = np.array([0.0, 3.0, -5.0], dtype=np.float32)
    graph = fake_quantize_tf(tf.constant(values), scale, zero_point, -128.0, 127.0).numpy()
    reference = fake_quantize(values, scale, zero_point, -128, 127)
    mismatched = ~np.isclose(graph, reference, rtol=0, atol=1e-6)
    assert mismatched.mean() < 1e-3
    assert np.all(np.abs(graph - reference)[mismatched] <= np.broadcast_to(scale, values.shape)[mismatched] * 1.0001)


def test_unknown_layers_and_invalid_configs_are_rejected():
    """Configs must name discovered layers and use LayerQuantConfig values."""
    model = _dscnn_like_model()
    simulator = QuantizationSimulator(model, _inputs(4))
    with pytest.raises(KeyError):
        simulator.predict(_inputs(2), {"missing": LayerQuantConfig(weight=W8)})
    with pytest.raises(TypeError):
        simulator.predict(_inputs(2), {"dw": W8})
    with pytest.raises(TypeError):
        LayerQuantConfig(weight="int8")
    with pytest.raises(ValueError):
        QuantizationSimulator(tf.keras.Sequential([layers.Input((3,)), layers.ReLU()]), np.zeros((2, 3)))


def test_tcn_layers_nested_in_residual_blocks_are_found_and_folded():
    """keras-tcn convolutions inside residual blocks are quantizable and fold their BatchNorms."""
    tcn = pytest.importorskip("tcn")
    tf.keras.utils.set_random_seed(6)
    inputs = layers.Input(shape=(24, 6))
    x = tcn.TCN(nb_filters=8, kernel_size=3, dilations=(1, 2), use_batch_norm=True, return_sequences=False, name="tcn")(
        inputs
    )
    outputs = layers.Dense(3, name="head")(x)
    model = tf.keras.Model(inputs, outputs)
    _randomize_batch_norms(model, 6)
    data = np.random.default_rng(6).normal(size=(32, 24, 6)).astype(np.float32)
    simulator = QuantizationSimulator(model, data)
    ids = simulator.layer_ids
    assert ids[-1] == "head"
    convs = [layer for layer in simulator.layers if layer.kind == "conv"]
    assert len(convs) == 5
    assert sum(layer.folded_batch_norm is not None for layer in convs) == 4
    assert any("matching_conv1D" in layer.layer_id for layer in convs)
    reference = model(data, training=False).numpy()
    w8a8 = simulator.predict(data, uniform_model_config(ids, weight=W8, activation=A8))
    assert np.max(np.abs(w8a8 - reference)) < 0.1 * np.std(reference)
