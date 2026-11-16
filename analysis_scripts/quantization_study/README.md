<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Uniform quantization study

No hardware required.
This study measures how simulated uniform quantization of weights and activations changes the task metrics of trained CREST models.
It runs on the host with the dataset, task, and model family of a CREST config.
It does not run search or change `training.quantization`.
Its deployable artifacts are a reference TFLite int8 export and, through `export-int4-activations`, int8 models rewritten with 4-bit feature maps for the opt-in Arduino patch.

Follow the [common command setup](../README.md) and run commands from the repository root, because CREST configs use repository-relative dataset paths.

## What is simulated

The simulator in [`crest.quantization`](../../src/crest/quantization/) quantizes every `Dense`, `Conv1D`/`Conv2D`/`Conv3D`, and `DepthwiseConv1D`/`DepthwiseConv2D` layer.
It also finds layers nested inside subclassed layers, such as keras-tcn residual blocks.

- Weights: each kernel is replaced by its quantized-then-dequantized value.
  A BatchNorm layer that directly follows the layer is folded into the kernel first, as the TFLite converter does.
- Activations: the input of each quantized layer is quantized with a range calibrated on the float model.
  Calibration uses the split and the evenly spaced subset (at most 1,000 samples) that CREST's `int8_ptq` export uses.
- Biases stay in float.
  Tensors that feed no quantized layer also stay in float, for example model outputs and residual-add outputs on identity skip paths.
  A TFLite integer graph quantizes those tensors as well, so the sweep also exports a real TFLite int8 model to compare against the 8-bit simulation.

Defaults follow TFLite int8.
Weights are symmetric per-channel with the narrow range `[-(2^(b-1) - 1), 2^(b-1) - 1]`.
Activations are asymmetric per-tensor, with the range widened to include zero.
Rounding is half away from zero.

## Experiments

| Experiment | Configurations |
| --- | --- |
| `float` | The float Keras model |
| `tflite_int8` | CREST's `int8_ptq` export, run with the host TFLite interpreter |
| `bits` | Every weight width (2–8 bits or float) crossed with every activation width (2–8 or 16 bits, or float), with default specs |
| `granularity` | Per-tensor or per-channel, symmetric or asymmetric, for weights alone and for activations alone |
| `clipping` | Min-max, percentile, or MSE-optimal ranges, for weights alone and for activations alone |
| `layer_sensitivity` | One layer's weights, or one layer's input, quantized while everything else stays in float |

The 8-bit-weight, 16-bit-activation rows approximate TFLite's 16x8 mode.
Use `--experiments` and the `--weight-bits`, `--activation-bits`, `--ablation-bits`, and `--layer-bits` options to shrink the sweep.
Run `quantization_study.py <command> --help` for every option.

## Commands

Train a fixed candidate panel.
Without `--execute`, the command only validates each candidate's hyperparameters and logs its parameter count.

```sh
python analysis_scripts/quantization_study/quantization_study.py train \
  --config src/config/nas_config_audio_stm32.yaml \
  --panel analysis_scripts/quantization_study/panels/audio_dscnn.yaml \
  --output-dir /path/to/quant_panel --execute
```

Training follows CREST's final-training recipe: the task's compile and fit plan, batch size 256, early stopping on validation loss with the best checkpoint restored, and at most `training.model_epochs` epochs.
Override the budget with `--epochs` and `--patience`.
Each candidate starts from `--seed` (default 17).
The [audio panel](panels/audio_dscnn.yaml) spans small to large DS-CNNs and adds two variants of the medium model, one with ReLU6 and one without BatchNorm.

Run the sweep on the trained panel.

```sh
python analysis_scripts/quantization_study/quantization_study.py sweep \
  --config src/config/nas_config_audio_stm32.yaml \
  --panel-dir /path/to/quant_panel \
  --output-dir /path/to/quant_results
```

Repeat `--panel-dir` to combine panels trained by separate jobs, for example one `train --candidates` run per candidate.
Add other trained models, such as a search checkpoint, with `--model NAME=/path/to/model.keras`.
They must load with the config's model family.

Summarize and plot the results.

```sh
python analysis_scripts/quantization_study/quantization_study.py plot \
  --results-dir /path/to/quant_results
```

## Export a model with 4-bit feature maps

TensorFlow's converter cannot emit 4-bit activation tensors, so this command rewrites an existing int8 export.
It stores chosen activation tensors as INT4 and leaves weights and every other tensor unchanged.
It works on any single-subgraph model and converts only tensors the patched runtime can run; everything else stays int8.

```sh
python analysis_scripts/quantization_study/quantization_study.py export-int4-activations \
  --config src/config/nas_config_audio_stm32.yaml \
  --tflite /path/to/quant_results/tflite/medium_int8_ptq.tflite \
  --output /path/to/int4/medium.tflite --tensors minimal
```

- The command measures each chosen tensor on the same calibration clips as the int8 export, with TFLite's built-in kernels.
  It picks a 4-bit range per tensor with `--range-method` (default `mse`), then rescales the biases of the layers that read the tensor.
  Min-max ranges lose far more accuracy at 4 bits than MSE ranges.
- Tensors joined by operators that keep the scale (`PAD`, `RESHAPE`, `EXPAND_DIMS`, max and average pooling) form a group that is converted whole with one set of parameters.
  For example, a TCN's causal padding and Conv1D reshapes join the convolution output they copy.
  A group qualifies only if every operator touching it is supported (see the [patch README](../../env_setup/patches/README.md#limits)).
- `--tensors all` converts every qualifying group.
  `--tensors minimal` converts the smallest set of groups with the same estimated memory peak, which keeps the rest at 8 bits and loses less accuracy.
  Explicit tensor indices (comma-separated) convert the groups that contain them.
- The memory estimate is the largest total of simultaneously live activation tensors, in 16-byte units, with TFLite Micro's tensor lifetimes.
  It is a lower bound: it excludes kernel scratch and persistent data, and the greedy planner can fragment.
  It equalled the planner's activation region on the audio DS-CNNs, but the planner needed up to 39% more on `odom_tcn` models, so confirm with the runtime's arena report.
- The output next to the model, `<output>.json`, lists each group and tensor with its int8 and int4 scale and zero point and clipping range, every group kept at int8 with the reason, the estimated peak for int8 and int4, warnings, and the input and output hashes.
- Warnings flag exports that the pinned Arduino runtime computes wrongly even at int8.
  A model exported with a dynamic batch stores batch 1 in its `SPACE_TO_BATCH_ND` and `BATCH_TO_SPACE_ND` tensors (dilated Conv1D layers), and TensorFlow Lite Micro's outputs are then unrelated to TFLite's.
  CREST's `convert_to_tflite_model` now exports with a fixed batch of 1, which also turns those operators into dilated convolutions; files exported before that change, or elsewhere, can still trigger the warning.
- Accuracy depends on the calibration sample.
  Across four `--seed` values, the all-feature-map models' test macro-F1 moved by up to 4 points, so compare seeds before drawing conclusions.
- No stock runtime executes the output, and TensorFlow's interpreter does not either.
  Apply the [4-bit activation patch](../../env_setup/patches/README.md) to the Arduino TensorFlow Lite Micro library first.
- The pinned Arduino library ignores all but the first per-channel scale of dense layers in TensorFlow 2.20 exports.
  This was reproduced on a host build and not on a board.
  Convert the base model with `converter._experimental_disable_per_channel_quantization_for_dense_layers = True` before comparing on-device outputs.

## Outputs

| File | Contents |
| --- | --- |
| `manifest.json` (train) | Config and panel hashes, versions, training settings, and per-candidate checkpoints and float metrics |
| `results.csv` | One row per model, configuration, and split, with the task's metrics |
| `layers.csv` | Quantizable layers per model in execution order, with shapes and folded BatchNorm layers |
| `metadata.json` | Inputs, hashes, versions, calibration details, and simulation conventions |
| `tflite/<model>_int8_ptq.tflite` | The reference int8 export |
| `<output>.tflite` and `<output>.json` (`export-int4-activations`) | The model with 4-bit feature maps and its report |
| `figures/summary.csv` | Float, TFLite int8, and simulated W8A8 metrics, and the smallest widths within `--tolerance` of float |
| `figures/*.png` | Bit-width curves, weight-by-activation grids, granularity and clipping comparisons, and layer sensitivity |

Each figure has a CSV sidecar with the plotted values.
In the layer-sensitivity figures, a tick at zero marks a layer whose quantization left the metric unchanged.
For example, a layer with one weight per output channel is represented exactly by per-channel weight quantization.
The smallest width within tolerance also requires every wider width to stay within tolerance.
`plot` reports the `test` split and the `macro_f1` metric by default; use `--split` and `--metric` to change them.
Keep outputs outside the checkout.

## Interpreting results

- CREST currently exports float32 and int8 models only, so widths below 8 bits change accuracy but not measured flash, RAM, latency, or energy.
  The one exception is `export-int4-activations`, which needs the 4-bit activation patch and has not been run on a board.
- These are post-training results.
  Quantization-aware training can recover much of the loss at low widths.
- `sim_w8a8_minus_tflite` in the summary shows how far the simulation's quantization points are from a real int8 graph for each model.
  For classifiers, much of the gap can come from the output layer.
  TFLite rounds logits to int8, so two close logits can tie, and argmax then picks the lower class index.
  The simulation keeps logits in float.
- Per-channel activation ranges are a simulation-only variant; TFLite activations are per-tensor.

## Tests

```sh
python -m pytest -q test/test_quantization_uniform.py test/test_quantization_keras_sim.py test/test_quantization_tflite_int4.py
RUN_ANALYSIS_SCRIPT_TESTS=1 python -m pytest -q test/test_quantization_study.py
```

The second command trains a one-epoch model on a synthetic cache and runs every subcommand.
