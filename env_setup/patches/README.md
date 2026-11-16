<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Arduino TensorFlow Lite Micro patches

## 4-bit activations

`arduino_tflite_int4_activations.patch` lets the pinned Arduino TensorFlow Lite Micro library run models whose feature maps are stored at 4 bits.
The pinned library (`tools/arduino-user/libraries/Arduino_TensorFlowLite`, Chirale_TensorFlowLite `46da2f9`) accepts INT4 only for weights.
Its convolution and mean kernels reject INT4 activation tensors, and its memory planner counts an INT4 element as one byte.

The patch is opt-in.
`make arduino-setup` does not apply it, so default builds use the stock library.

```bash
make arduino-tflite-patch                        # apply
make arduino-tflite-patch PATCH_ACTION=status    # prints unpatched, patched, or conflict
make arduino-tflite-patch PATCH_ACTION=revert    # restore the stock library
```

The patch leaves the submodule pointer unchanged, so `git status` shows the submodule as modified while it is applied.

### What it changes

- The memory planner reserves half a byte per INT4 activation element.
- `CONV_2D` and `DEPTHWISE_CONV_2D` unpack the input rows a tile of output rows needs into a small scratch buffer, call the unchanged CMSIS-NN int8 kernel, and pack the output rows.
  The arithmetic is the int8 kernel's, and the activation clamp is -8 to 7.
- `MEAN` over axes 1 and 2 reads packed 4-bit inputs and produces int8 outputs with the int8 kernel's rounding.
- One guard in `arm_nn_mat_mult_nt_t_s8` skips the weight-sum loop when the input offset is zero.
  Int8 results are unchanged.
- `crest_act4.h` and `crest_act4.cpp` hold the packing helpers.
  `CREST_ACT4_CONTAINER_DEFAULT` selects one-value-per-byte storage, which runs the stock int8 kernels and serves as a bit-exact reference.

### Limits

- Feature maps must be INT4 tensors with int8 weights, and the 4-bit tensors must be outputs of convolutions consumed by convolutions or by a global average pool.
  Other operators, dilation along the height, and batch sizes above one are not supported.
- TensorFlow's converter cannot emit INT4 activation tensors.
  Create them from an int8 export with the [`export-int4-activations`](../../analysis_scripts/quantization_study/README.md#export-a-model-with-4-bit-feature-maps) command of the quantization study.
  CREST's search and deployment paths do not use them yet, so the patch alone does not change what CREST builds.
- It applies to the Arduino TensorFlow Lite Micro path only.
  The STM32 path uses ST's runtime, which has no 4-bit activations.

### Verification status

- Packed and one-byte-per-value storage give identical outputs on every validation and test clip of the UrbanSound8K audio models.
- The same outputs come from a Cortex-M4 build of the library run in an instruction-set emulator.
- The smallest working arena is exact, and AddressSanitizer is clean at that size.
- CREST's Arduino sketch compiles for the Nano 33 BLE with the patch applied.
- Nothing has run on a board, so latency and energy are unmeasured.
  The emulator counts instructions, not cycles.
