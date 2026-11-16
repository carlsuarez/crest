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
- Packed values use a layout that depends only on an element's flat index, so a reshape is a byte copy.
  Each group of 8 values fills one 32-bit word, which unpacks with a few masks.
- `CONV_2D`, `DEPTHWISE_CONV_2D`, `MAX_POOL_2D` and `AVERAGE_POOL_2D` work in tiles.
  A tile's input lines are unpacked into a small scratch buffer, the unchanged CMSIS-NN int8 kernel runs on the tile, and the output lines are packed back.
  The arithmetic is the int8 kernel's, and the activation clamp is -8 to 7.
- Tiles run along height, or along width when a tensor has height 1, as the Conv1D layers of a TCN do.
  With a packed input and dilation d at stride 1, a tile gathers every d-th line, so the kernel sees a dense convolution.
  Without this, one output step of a dilation-64 layer would read over 700 input steps.
- `ADD` unpacks 64 values of each input at a time and calls the int8 kernel.
- `PAD` writes the zero point and copies input runs.
  `RESHAPE` and `EXPAND_DIMS` copy the packed bytes.
- `MEAN` over axes 1 and 2 reads packed 4-bit inputs and produces int8 outputs with the int8 kernel's rounding.
- One guard in `arm_nn_mat_mult_nt_t_s8` skips the weight-sum loop when the input offset is zero.
  Int8 results are unchanged.
- `crest_act4.h` and `crest_act4.cpp` hold the packing and tiling helpers.
  `CREST_ACT4_CONTAINER_DEFAULT` selects one-value-per-byte storage, which runs the stock int8 kernels and serves as a bit-exact reference.

### Limits

- Supported operators with 4-bit tensors: `CONV_2D` and `DEPTHWISE_CONV_2D` (int8 weights, batch 1), `ADD` (no broadcasting), `MEAN` (4-bit input only), `PAD`, `RESHAPE`, `EXPAND_DIMS`, `MAX_POOL_2D` and `AVERAGE_POOL_2D`.
  `SHAPE` may read a 4-bit tensor because it reads only dimensions.
  Every other operator, including `FULLY_CONNECTED`, `STRIDED_SLICE` and `SQUEEZE`, needs int8 tensors.
- TensorFlow's converter cannot emit INT4 activation tensors.
  Create them from an int8 export with the [`export-int4-activations`](../../analysis_scripts/quantization_study/README.md#export-a-model-with-4-bit-feature-maps) command of the quantization study, which converts only tensors these operators can handle.
  CREST's search and deployment paths do not use them yet, so the patch alone does not change what CREST builds.
- Only the activation part of the arena halves.
  The persistent part (operator data, tensor records) stays the same, so the whole arena shrinks by 24 to 48% on the models tested.
- It applies to the Arduino TensorFlow Lite Micro path only.
  The STM32 path uses ST's runtime, which has no 4-bit activations.

### Verification status

- Models tested: the UrbanSound8K medium and large DS-CNNs, three `odom_tcn` variants with random weights (no OxIOD data was available), and a synthetic CNN with pooling, a residual add and `EXPAND_DIMS`.
- Packed and one-byte-per-value storage give identical outputs on every input tried.
- A Cortex-M4 build of the library, run in an instruction-set emulator, gives the same outputs as the host build.
- The smallest working arena is exact, and AddressSanitizer is clean at that size.
- CREST's Arduino sketch compiles for the Nano 33 BLE with the patch applied.
- Nothing has run on a board, so latency and energy are unmeasured.
  The emulator counts instructions, not cycles.
