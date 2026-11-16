# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Simulated quantization for host-side precision studies.

``crest.quantization.uniform`` holds NumPy uniform quantizers and range
estimators.
``crest.quantization.keras_sim`` applies them to trained Keras models.
``crest.quantization.tflite_int4_activations`` rewrites int8 TFLite models so feature maps are 4-bit tensors.
Import those two explicitly, because they load TensorFlow.

Nothing in this package changes CREST's deployment exports or the
``training.quantization`` modes used by search.
"""

from .uniform import (
    GRANULARITIES,
    MAX_BITS,
    MIN_BITS,
    RANGE_METHODS,
    ActivationObserver,
    ActivationStats,
    QuantSpec,
    UniformQuantParams,
    activation_qparams,
    compute_qparams,
    estimate_range,
    fake_quantize,
    integer_bounds,
    quantize_array,
)

__all__ = [
    "GRANULARITIES",
    "MAX_BITS",
    "MIN_BITS",
    "RANGE_METHODS",
    "ActivationObserver",
    "ActivationStats",
    "QuantSpec",
    "UniformQuantParams",
    "activation_qparams",
    "compute_qparams",
    "estimate_range",
    "fake_quantize",
    "integer_bounds",
    "quantize_array",
]
