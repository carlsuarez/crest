<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Uniform quantization study

No hardware required.
This study measures how simulated uniform quantization of weights and activations changes the task metrics of trained CREST models.
It runs on the host with the dataset, task, and model family of a CREST config.
It does not run search or change `training.quantization`.
Its only deployable artifact is a reference TFLite int8 export.

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

## Outputs

| File | Contents |
| --- | --- |
| `manifest.json` (train) | Config and panel hashes, versions, training settings, and per-candidate checkpoints and float metrics |
| `results.csv` | One row per model, configuration, and split, with the task's metrics |
| `layers.csv` | Quantizable layers per model in execution order, with shapes and folded BatchNorm layers |
| `metadata.json` | Inputs, hashes, versions, calibration details, and simulation conventions |
| `tflite/<model>_int8_ptq.tflite` | The reference int8 export |
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
- These are post-training results.
  Quantization-aware training can recover much of the loss at low widths.
- `sim_w8a8_minus_tflite` in the summary shows how far the simulation's quantization points are from a real int8 graph for each model.
  For classifiers, much of the gap can come from the output layer.
  TFLite rounds logits to int8, so two close logits can tie, and argmax then picks the lower class index.
  The simulation keeps logits in float.
- Per-channel activation ranges are a simulation-only variant; TFLite activations are per-tensor.

## Tests

```sh
python -m pytest -q test/test_quantization_uniform.py test/test_quantization_keras_sim.py
RUN_ANALYSIS_SCRIPT_TESTS=1 python -m pytest -q test/test_quantization_study.py
```

The second command trains a one-epoch model on a synthetic cache and runs every subcommand.
