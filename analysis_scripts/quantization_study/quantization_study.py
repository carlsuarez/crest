#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Host-only study of simulated uniform quantization on trained CREST models.

Subcommands
-----------
train
    Train a fixed panel of model-family candidates with CREST's final-training
    recipe and save each as a ``.keras`` model. Requires ``--execute``.
sweep
    Simulate weight and activation quantization on trained models, export a
    real TFLite int8 model as a reference, and write task metrics for every
    configuration to ``results.csv``.
plot
    Summarize ``results.csv`` and plot it, with a numeric sidecar per figure.
export-int4-activations
    Rewrite an int8 TFLite model so chosen feature maps are 4-bit tensors, for
    the opt-in 4-bit activation patch of the Arduino TensorFlow Lite Micro
    library.

No subcommand runs search or accesses hardware.
Run from the repository root with the ``crest`` environment active, because
CREST configs use repository-relative dataset paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crest.quantization import GRANULARITIES, RANGE_METHODS, QuantSpec  # noqa: E402

logger = logging.getLogger("quantization_study")

RESULTS_FILE = "results.csv"
LAYERS_FILE = "layers.csv"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.csv"
MODEL_FILENAME = "model.keras"
EXPERIMENTS = ("float", "tflite_int8", "bits", "granularity", "clipping", "layer_sensitivity")
DEFAULT_WEIGHT_BITS = (2, 3, 4, 5, 6, 7, 8)
DEFAULT_ACTIVATION_BITS = (2, 3, 4, 5, 6, 7, 8, 16)
DEFAULT_ABLATION_BITS = (2, 3, 4, 6, 8)
DEFAULT_LAYER_BITS = (2, 4)
DEFAULT_WEIGHT_PERCENTILE = 99.9
DEFAULT_ACTIVATION_PERCENTILE = 99.99
# NASModelClient._train_with_decoded_hparams uses this batch size for final training.
FINAL_TRAINING_BATCH_SIZE = 256
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LOWER_IS_BETTER_HINTS = ("loss", "rmse", "mae", "mse", "error")
RESULT_BASE_COLUMNS = (
    "model",
    "split",
    "examples",
    "experiment",
    "backend",
    "scope",
    "layer_id",
    "layer_index",
    "quantized",
    "w_bits",
    "w_symmetric",
    "w_granularity",
    "w_range_method",
    "w_percentile",
    "a_bits",
    "a_symmetric",
    "a_granularity",
    "a_range_method",
    "a_percentile",
)
SPEC_FIELDS = ("bits", "symmetric", "granularity", "range_method", "percentile")

# Light-surface palette: neutral ink for text and categorical slots assigned in a fixed order.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
# Marker shapes are a second identity channel for print and color-vision deficiency.
SERIES_MARKERS = ("o", "s", "^", "D")
# Space above the panels for the figure legend, and the title's height above the figure, in inches.
LEGEND_BAND_INCHES = 0.3
TITLE_OFFSET_INCHES = 0.3
SEQUENTIAL_BLUE = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def parse_int_list(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers.

    Parameters
    ----------
    raw : str
        Text such as ``"2,3,4"``.

    Returns
    -------
    tuple[int, ...]
        Sorted unique integers.

    Raises
    ------
    argparse.ArgumentTypeError
        If the list is empty or contains a non-integer.
    """
    try:
        values = sorted({int(item) for item in raw.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers, got {raw!r}.") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return tuple(values)


def parse_name_list(raw: str, allowed: Sequence[str] | None = None) -> tuple[str, ...]:
    """Parse a comma-separated list of names, keeping order and dropping repeats.

    Parameters
    ----------
    raw : str
        Text such as ``"val,test"``.
    allowed : Sequence[str] | None, optional
        Permitted names.

    Returns
    -------
    tuple[str, ...]
        Parsed names.

    Raises
    ------
    argparse.ArgumentTypeError
        If the list is empty or contains a name outside ``allowed``.
    """
    names = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not names:
        raise argparse.ArgumentTypeError("Expected at least one name.")
    if allowed is not None:
        unknown = [name for name in names if name not in allowed]
        if unknown:
            raise argparse.ArgumentTypeError(f"Unknown names {unknown}; choose from {list(allowed)}.")
    return names


def validate_name(name: str, *, kind: str) -> str:
    """Check that a model or candidate name is safe to use as a file name.

    Parameters
    ----------
    name : str
        Proposed name.
    kind : str
        Description used in the error message.

    Returns
    -------
    str
        The name.

    Raises
    ------
    ValueError
        If the name is empty or contains characters outside ``[A-Za-z0-9_.-]``.
    """
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"Invalid {kind} name {name!r}; use letters, digits, '_', '.', or '-'.")
    return name


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    """Return the checkout's commit and whether tracked files changed.

    Returns
    -------
    dict[str, Any]
        ``commit`` and ``dirty``; values are None when git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def runtime_versions() -> dict[str, str | None]:
    """Return interpreter and library versions relevant to reproduction.

    Returns
    -------
    dict[str, str | None]
        Python, NumPy, TensorFlow, and Keras versions.
    """
    versions: dict[str, str | None] = {"python": platform.python_version(), "numpy": np.__version__}
    for module_name in ("tensorflow", "keras"):
        module = sys.modules.get(module_name)
        versions[module_name] = getattr(module, "__version__", None) if module else None
    return versions


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 format.

    Returns
    -------
    str
        Timestamp with second precision.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    """Write indented JSON, converting unknown objects to strings.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    payload : Any
        JSON-like payload.

    Returns
    -------
    None
        The file is written.
    """
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    """Normalize a value for CSV output.

    Parameters
    ----------
    value : Any
        Cell value.

    Returns
    -------
    Any
        ``""`` for None, ``"true"``/``"false"`` for booleans, else the value.
    """
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return value


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], leading: Sequence[str] = ()) -> None:
    """Write dictionaries to CSV with a stable column order.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV.
    rows : Sequence[Mapping[str, Any]]
        Rows to write.
    leading : Sequence[str], optional
        Columns placed first; other columns follow in first-seen order.

    Returns
    -------
    None
        The file is written.
    """
    columns = list(leading)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries.

    Parameters
    ----------
    path : pathlib.Path
        CSV file.

    Returns
    -------
    list[dict[str, str]]
        Rows as strings.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# CREST pipeline access
# --------------------------------------------------------------------------
def load_pipeline(config_path: Path) -> tuple[Any, Any]:
    """Load a CREST config and bootstrap its dataset, task, and model family.

    Parameters
    ----------
    config_path : pathlib.Path
        CREST YAML config.

    Returns
    -------
    tuple[Any, Any]
        The loaded config and the bootstrapped pipeline.
    """
    from crest.builtin_components import ensure_builtin_components_registered
    from crest.model import load_config
    from crest.runtime_bootstrap import bootstrap_pipeline

    ensure_builtin_components_registered()
    config = load_config(config_path)
    return config, bootstrap_pipeline(config)


def select_calibration(pipeline: Any) -> tuple[np.ndarray, np.ndarray, str]:
    """Pick calibration data the way CREST's int8 export does.

    Parameters
    ----------
    pipeline : Any
        Bootstrapped CREST pipeline.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, str]
        The full representative split, the evenly spaced subset the TFLite
        converter would use, and the split name.
    """
    from crest.hardware import _representative_calibration_samples

    bundle = pipeline.bundle
    source = "calibration" if bundle.calibration is not None else "train"
    split = bundle.calibration if bundle.calibration is not None else bundle.train
    inputs = np.asarray(split.inputs, dtype=np.float32)
    return inputs, _representative_calibration_samples(inputs), source


def evaluate_split(pipeline: Any, predictions: Any, split: Any) -> dict[str, float]:
    """Score predictions with the configured task.

    Parameters
    ----------
    pipeline : Any
        Bootstrapped CREST pipeline.
    predictions : Any
        Model outputs for ``split``.
    split : Any
        Dataset split with targets.

    Returns
    -------
    dict[str, float]
        Numeric task metrics.
    """
    result = pipeline.task.evaluate_predictions(
        predictions,
        split,
        pipeline.selection["task_config"],
        pipeline.target_spec,
    )
    return {
        str(key): float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    }


def slice_outputs(outputs: Any, start: int, stop: int) -> Any:
    """Slice model outputs along the sample axis.

    Parameters
    ----------
    outputs : Any
        Array or nested list, tuple, or dict of arrays.
    start, stop : int
        Sample range.

    Returns
    -------
    Any
        Outputs with the same structure.

    Raises
    ------
    TypeError
        If the structure contains something other than arrays.
    """
    if isinstance(outputs, np.ndarray):
        return outputs[start:stop]
    if isinstance(outputs, (list, tuple)):
        return type(outputs)(slice_outputs(item, start, stop) for item in outputs)
    if isinstance(outputs, dict):
        return {key: slice_outputs(value, start, stop) for key, value in outputs.items()}
    raise TypeError(f"Unsupported model output type {type(outputs).__name__}.")


# --------------------------------------------------------------------------
# Experiment definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepOptions:
    """Bit-widths and clipping settings for the sweep.

    Parameters
    ----------
    weight_bits : tuple[int, ...]
        Weight widths in the bit-width grid.
    activation_bits : tuple[int, ...]
        Activation widths in the bit-width grid.
    ablation_bits : tuple[int, ...]
        Widths used by the granularity and clipping experiments.
    layer_bits : tuple[int, ...]
        Widths used by the one-layer-at-a-time sensitivity experiment.
    weight_percentile : float
        Percentile kept by percentile clipping of weights.
    activation_percentile : float
        Percentile kept by percentile clipping of activations.
    """

    weight_bits: tuple[int, ...] = DEFAULT_WEIGHT_BITS
    activation_bits: tuple[int, ...] = DEFAULT_ACTIVATION_BITS
    ablation_bits: tuple[int, ...] = DEFAULT_ABLATION_BITS
    layer_bits: tuple[int, ...] = DEFAULT_LAYER_BITS
    weight_percentile: float = DEFAULT_WEIGHT_PERCENTILE
    activation_percentile: float = DEFAULT_ACTIVATION_PERCENTILE


@dataclass(frozen=True)
class SimulatedRun:
    """One simulated quantization configuration.

    Parameters
    ----------
    experiment : str
        Experiment family from ``EXPERIMENTS``.
    scope : {"model", "layer"}
        Whether the specs apply to every quantizable layer or to one layer.
    layer_id : str | None
        Target layer when ``scope == "layer"``.
    weight : QuantSpec | None
        Weight spec, or None for float weights.
    activation : QuantSpec | None
        Input-activation spec, or None for float inputs.
    """

    experiment: str
    scope: str
    layer_id: str | None
    weight: QuantSpec | None
    activation: QuantSpec | None

    @property
    def quantized(self) -> str:
        """Return which tensors this run quantizes.

        Returns
        -------
        str
            ``"none"``, ``"weights"``, ``"activations"``, or ``"both"``.
        """
        if self.weight is not None and self.activation is not None:
            return "both"
        if self.weight is not None:
            return "weights"
        if self.activation is not None:
            return "activations"
        return "none"

    @property
    def cache_key(self) -> tuple[Any, ...]:
        """Return a key shared by runs that produce identical predictions.

        Returns
        -------
        tuple[Any, ...]
            Scope, layer, and specs; the experiment label is excluded.
        """
        return (self.scope, self.layer_id, self.weight, self.activation)

    def row_fields(self, layer_index: Mapping[str, int]) -> dict[str, Any]:
        """Return the configuration columns for a results row.

        Parameters
        ----------
        layer_index : Mapping[str, int]
            Execution index per layer identifier.

        Returns
        -------
        dict[str, Any]
            Experiment, scope, layer, and spec columns.
        """
        fields: dict[str, Any] = {
            "experiment": self.experiment,
            "backend": "keras",
            "scope": self.scope,
            "layer_id": self.layer_id,
            "layer_index": None if self.layer_id is None else layer_index[self.layer_id],
            "quantized": self.quantized,
        }
        fields.update(spec_fields("w", self.weight))
        fields.update(spec_fields("a", self.activation))
        return fields


def spec_fields(prefix: str, spec: QuantSpec | None) -> dict[str, Any]:
    """Flatten a spec into prefixed result columns.

    Parameters
    ----------
    prefix : str
        ``"w"`` or ``"a"``.
    spec : QuantSpec | None
        Spec to describe.

    Returns
    -------
    dict[str, Any]
        Columns such as ``w_bits``; values are None for a float tensor.
    """
    described = spec.describe() if spec is not None else {}
    return {f"{prefix}_{name}": described.get(name) for name in SPEC_FIELDS}


def default_weight_spec(bits: int, *, range_method: str = "minmax", percentile: float | None = None) -> QuantSpec:
    """Return the TFLite-style weight spec: symmetric, per-channel.

    Parameters
    ----------
    bits : int
        Weight width.
    range_method : str, optional
        Clipping method.
    percentile : float | None, optional
        Percentile for ``range_method="percentile"``.

    Returns
    -------
    QuantSpec
        Weight spec.
    """
    extra = {} if percentile is None else {"percentile": percentile}
    return QuantSpec(bits, symmetric=True, granularity="per_channel", range_method=range_method, **extra)


def default_activation_spec(bits: int, *, range_method: str = "minmax", percentile: float | None = None) -> QuantSpec:
    """Return the TFLite-style activation spec: asymmetric, per-tensor.

    Parameters
    ----------
    bits : int
        Activation width.
    range_method : str, optional
        Clipping method.
    percentile : float | None, optional
        Percentile for ``range_method="percentile"``.

    Returns
    -------
    QuantSpec
        Activation spec.
    """
    extra = {} if percentile is None else {"percentile": percentile}
    return QuantSpec(bits, symmetric=False, granularity="per_tensor", range_method=range_method, **extra)


def build_simulated_runs(
    layer_ids: Sequence[str],
    experiments: Iterable[str],
    options: SweepOptions,
) -> list[SimulatedRun]:
    """Enumerate the simulated configurations for one model.

    Parameters
    ----------
    layer_ids : Sequence[str]
        Quantizable layers in execution order.
    experiments : Iterable[str]
        Experiments to include; ``"tflite_int8"`` is not simulated.
    options : SweepOptions
        Bit-widths and clipping settings.

    Returns
    -------
    list[SimulatedRun]
        Runs in a stable order.

    Notes
    -----
    ``bits`` crosses every weight width (or float) with every activation width
    (or float) using the TFLite-style default specs.
    ``granularity`` varies granularity and symmetry for weights alone and for
    activations alone, with min-max ranges.
    ``clipping`` varies the range method for weights alone and for
    activations alone, with the default granularity and symmetry.
    ``layer_sensitivity`` quantizes one layer's weights, or one layer's input,
    with every other tensor in float.
    """
    selected = set(experiments)
    runs: list[SimulatedRun] = []
    if "float" in selected:
        runs.append(SimulatedRun("float", "model", None, None, None))
    if "bits" in selected:
        for weight_bits in (None, *options.weight_bits):
            for activation_bits in (None, *options.activation_bits):
                if weight_bits is None and activation_bits is None:
                    continue
                runs.append(
                    SimulatedRun(
                        "bits",
                        "model",
                        None,
                        None if weight_bits is None else default_weight_spec(weight_bits),
                        None if activation_bits is None else default_activation_spec(activation_bits),
                    )
                )
    if "granularity" in selected:
        for bits in options.ablation_bits:
            for granularity in GRANULARITIES:
                for symmetric in (True, False):
                    spec = QuantSpec(bits, symmetric=symmetric, granularity=granularity)
                    runs.append(SimulatedRun("granularity", "model", None, spec, None))
                    runs.append(SimulatedRun("granularity", "model", None, None, spec))
    if "clipping" in selected:
        for bits in options.ablation_bits:
            for method in RANGE_METHODS:
                weight_percentile = options.weight_percentile if method == "percentile" else None
                activation_percentile = options.activation_percentile if method == "percentile" else None
                runs.append(
                    SimulatedRun(
                        "clipping",
                        "model",
                        None,
                        default_weight_spec(bits, range_method=method, percentile=weight_percentile),
                        None,
                    )
                )
                runs.append(
                    SimulatedRun(
                        "clipping",
                        "model",
                        None,
                        None,
                        default_activation_spec(bits, range_method=method, percentile=activation_percentile),
                    )
                )
    if "layer_sensitivity" in selected:
        for layer_id in layer_ids:
            for bits in options.layer_bits:
                runs.append(SimulatedRun("layer_sensitivity", "layer", layer_id, default_weight_spec(bits), None))
                runs.append(SimulatedRun("layer_sensitivity", "layer", layer_id, None, default_activation_spec(bits)))
    return runs


def run_config(run: SimulatedRun, layer_ids: Sequence[str]) -> dict[str, Any]:
    """Convert a run into a simulator config.

    Parameters
    ----------
    run : SimulatedRun
        Run to convert.
    layer_ids : Sequence[str]
        All quantizable layers.

    Returns
    -------
    dict[str, Any]
        Mapping from layer identifier to ``LayerQuantConfig``.
    """
    from crest.quantization.keras_sim import LayerQuantConfig

    if run.quantized == "none":
        return {}
    targets = tuple(layer_ids) if run.scope == "model" else (run.layer_id,)
    layer_config = LayerQuantConfig(weight=run.weight, activation=run.activation)
    return {layer_id: layer_config for layer_id in targets}


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def load_panel(path: Path) -> tuple[str | None, dict[str, dict[str, Any]]]:
    """Read a candidate panel YAML.

    Parameters
    ----------
    path : pathlib.Path
        Panel file with an optional ``family`` and a ``candidates`` mapping
        from name to model-family hyperparameters.

    Returns
    -------
    tuple[str | None, dict[str, dict[str, Any]]]
        Declared family name and candidates in file order.

    Raises
    ------
    ValueError
        If the file is malformed.
    """
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), dict) or not payload["candidates"]:
        raise ValueError(f"{path} needs a non-empty 'candidates' mapping.")
    family = payload.get("family")
    if family is not None and not isinstance(family, str):
        raise ValueError(f"{path}: 'family' must be a string.")
    candidates: dict[str, dict[str, Any]] = {}
    for name, hparams in payload["candidates"].items():
        validate_name(name, kind="candidate")
        if not isinstance(hparams, dict) or not hparams:
            raise ValueError(f"{path}: candidate {name!r} needs a hyperparameter mapping.")
        candidates[name] = dict(hparams)
    return family, candidates


def run_train(args: argparse.Namespace) -> int:
    """Train a candidate panel.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``train`` arguments.

    Returns
    -------
    int
        Process exit code.

    Raises
    ------
    ValueError
        If the panel does not match the config or the output already exists.
    """
    family_name, candidates = load_panel(args.panel)
    if args.candidates:
        missing = [name for name in args.candidates if name not in candidates]
        if missing:
            raise ValueError(f"Unknown candidates {missing}; the panel defines {list(candidates)}.")
        candidates = {name: candidates[name] for name in args.candidates}
    output_dir = Path(args.output_dir)
    if (output_dir / MANIFEST_FILE).exists() and not args.overwrite:
        raise ValueError(f"{output_dir / MANIFEST_FILE} exists; pass --overwrite or choose a new directory.")

    config, pipeline = load_pipeline(args.config)
    import tensorflow as tf

    from crest.runtime_bootstrap import instantiate_task_component

    family = pipeline.model_family
    configured_family = pipeline.selection["model_family_name"]
    if family_name is not None and family_name != configured_family:
        raise ValueError(f"Panel family {family_name!r} does not match config family {configured_family!r}.")
    context = pipeline.model_build_context
    model_config = pipeline.selection["model_config"]
    task_config = pipeline.selection["task_config"]

    for name, hparams in candidates.items():
        family.validate_hparams(hparams, context, model_config)
        model = family.build_model(hparams, context, model_config)
        logger.info("Candidate %-16s %9d parameters", name, model.count_params())
        del model
    tf.keras.backend.clear_session()
    if not args.execute:
        logger.info("Dry run complete; pass --execute to train %d candidate(s).", len(candidates))
        return 0

    epochs = int(args.epochs) if args.epochs else int(config.training.model_epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    patience_used = None
    for name, hparams in candidates.items():
        candidate_dir = output_dir / name
        candidate_dir.mkdir(exist_ok=True)
        checkpoint = candidate_dir / MODEL_FILENAME
        tf.keras.utils.set_random_seed(args.seed)
        task = instantiate_task_component(
            pipeline.selection["task_name"],
            config,
            task_config,
            checkpoint_path=checkpoint,
            early_stopping_patience=args.patience,
        )
        patience_used = getattr(task, "early_stopping_patience", args.patience)
        model = family.build_model(hparams, context, model_config)
        task.validate_model_outputs(model, pipeline.target_spec)
        task.compile_model(model, task_config, pipeline.target_spec)
        plan = task.build_fit_plan(
            pipeline.bundle,
            task_config,
            pipeline.target_spec,
            mode="final",
            combine_train_val=False,
        )
        fit_kwargs = dict(plan.fit_kwargs)
        fit_kwargs.update(
            callbacks=list(plan.callbacks),
            epochs=epochs,
            batch_size=FINAL_TRAINING_BATCH_SIZE,
            verbose=args.fit_verbose,
        )
        logger.info("Training %s for up to %d epochs (patience %s).", name, epochs, patience_used)
        started = time.perf_counter()
        history = model.fit(**fit_kwargs)
        train_seconds = time.perf_counter() - started
        if not checkpoint.exists():
            # The checkpoint callback writes only on improvement; keep the restored weights.
            model.save(checkpoint)
        trained = family.load_model(checkpoint, context, model_config)
        float_metrics = {}
        for split_name in ("val", "test"):
            split = getattr(pipeline.bundle, split_name, None)
            if split is not None:
                predictions = trained.predict(np.asarray(split.inputs, dtype=np.float32), verbose=0)
                float_metrics[split_name] = evaluate_split(pipeline, predictions, split)
        history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
        write_json(candidate_dir / "history.json", history_dict)
        entries[name] = {
            "hparams": hparams,
            "model_path": f"{name}/{MODEL_FILENAME}",
            "model_sha256": sha256_file(checkpoint),
            "weight_count": int(trained.count_params()),
            "epochs_run": len(next(iter(history_dict.values()), [])),
            "train_seconds": round(train_seconds, 1),
            "float_metrics": float_metrics,
        }
        logger.info("Trained %s in %.0f s: %s", name, train_seconds, float_metrics.get("test", float_metrics))
        del model, trained
        tf.keras.backend.clear_session()

    manifest = {
        "created_utc": utc_now(),
        "command": "train",
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "panel": str(args.panel),
        "panel_sha256": sha256_file(args.panel),
        "family": configured_family,
        "seed": args.seed,
        "max_epochs": epochs,
        "early_stopping_patience": patience_used,
        "batch_size": FINAL_TRAINING_BATCH_SIZE,
        "git": git_state(),
        "versions": runtime_versions(),
        "dataset": dict(pipeline.bundle.metadata),
        "candidates": entries,
    }
    write_json(output_dir / MANIFEST_FILE, manifest)
    logger.info("Wrote %s", output_dir / MANIFEST_FILE)
    return 0


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def resolve_models(panel_dirs: Sequence[Path], model_args: Sequence[str]) -> list[tuple[str, Path]]:
    """Collect model names and paths from panel directories and ``--model`` flags.

    Parameters
    ----------
    panel_dirs : Sequence[pathlib.Path]
        Directories written by ``train``, for example one per parallel training job.
    model_args : Sequence[str]
        ``NAME=PATH`` entries.

    Returns
    -------
    list[tuple[str, pathlib.Path]]
        Unique model names with existing model paths, in argument order.

    Raises
    ------
    ValueError
        If no model is given, a name repeats, or a path does not exist.
    """
    models: list[tuple[str, Path]] = []
    for panel_dir in panel_dirs:
        manifest = json.loads((Path(panel_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
        for name, entry in manifest["candidates"].items():
            models.append((validate_name(name, kind="model"), Path(panel_dir) / entry["model_path"]))
    for item in model_args:
        name, separator, raw_path = item.partition("=")
        if not separator or not raw_path:
            raise ValueError(f"--model expects NAME=PATH, got {item!r}.")
        models.append((validate_name(name, kind="model"), Path(raw_path)))
    if not models:
        raise ValueError("Pass --panel-dir or at least one --model NAME=PATH.")
    names = [name for name, _ in models]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Model names must be unique; repeated: {duplicates}.")
    missing = [str(path) for _, path in models if not path.is_file()]
    if missing:
        raise ValueError(f"Model files not found: {missing}.")
    return models


def export_and_predict_tflite(model: Any, calibration_inputs: np.ndarray, inputs: np.ndarray, path: Path) -> Any:
    """Export CREST's int8 PTQ TFLite model and run it on the host.

    Parameters
    ----------
    model : Any
        Float Keras model.
    calibration_inputs : numpy.ndarray
        Representative split passed to CREST's converter.
    inputs : numpy.ndarray
        Inputs to predict.
    path : pathlib.Path
        Destination ``.tflite`` file.

    Returns
    -------
    Any
        Dequantized predictions.
    """
    from crest.hardware import convert_to_tflite_model, predict_tflite_model_subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    convert_to_tflite_model(model, training_data=calibration_inputs, quantization_mode="int8_ptq", output_name=path)
    return predict_tflite_model_subprocess(path, inputs)


def run_sweep(args: argparse.Namespace) -> int:
    """Run the simulated quantization sweep.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``sweep`` arguments.

    Returns
    -------
    int
        Process exit code.

    Raises
    ------
    ValueError
        If outputs exist, models are missing, or a requested split is absent.
    """
    output_dir = Path(args.output_dir)
    if (output_dir / RESULTS_FILE).exists() and not args.overwrite:
        raise ValueError(f"{output_dir / RESULTS_FILE} exists; pass --overwrite or choose a new directory.")
    models = resolve_models(args.panel_dir, args.model)
    options = SweepOptions(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        ablation_bits=args.ablation_bits,
        layer_bits=args.layer_bits,
        weight_percentile=args.weight_percentile,
        activation_percentile=args.activation_percentile,
    )
    # Validate every width before loading data or models.
    for bits in {*options.weight_bits, *options.activation_bits, *options.ablation_bits, *options.layer_bits}:
        QuantSpec(bits)

    config, pipeline = load_pipeline(args.config)
    import tensorflow as tf

    from crest.quantization.keras_sim import QuantizationSimulator

    splits = {}
    for split_name in args.splits:
        split = getattr(pipeline.bundle, split_name, None)
        if split is None:
            raise ValueError(f"The dataset has no {split_name!r} split.")
        splits[split_name] = split
    split_inputs = [np.asarray(splits[name].inputs, dtype=np.float32) for name in args.splits]
    eval_inputs = np.concatenate(split_inputs, axis=0)
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for name, array in zip(args.splits, split_inputs):
        offsets[name] = (start, start + array.shape[0])
        start += array.shape[0]
    calibration_split, calibration_samples, calibration_source = select_calibration(pipeline)
    context = pipeline.model_build_context
    model_config = pipeline.selection["model_config"]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {}
    for name, path in models:
        logger.info("Model %s: loading %s", name, path)
        model = pipeline.model_family.load_model(path, context, model_config)
        simulator = QuantizationSimulator(
            model,
            calibration_samples,
            batch_size=args.batch_size,
            max_calibration_values=args.max_calibration_values,
            seed=args.seed,
        )
        layer_rows.extend({"model": name, **layer.describe()} for layer in simulator.layers)
        layer_index = {layer.layer_id: layer.execution_index for layer in simulator.layers}
        runs = build_simulated_runs(simulator.layer_ids, args.experiments, options)
        predictions: dict[tuple[Any, ...], Any] = {}
        started = time.perf_counter()
        for position, run in enumerate(runs, start=1):
            if run.cache_key not in predictions:
                predictions[run.cache_key] = simulator.predict(eval_inputs, run_config(run, simulator.layer_ids))
            fields = run.row_fields(layer_index)
            for split_name, (first, last) in offsets.items():
                metrics = evaluate_split(
                    pipeline, slice_outputs(predictions[run.cache_key], first, last), splits[split_name]
                )
                rows.append({"model": name, "split": split_name, "examples": last - first, **fields, **metrics})
            if position % 25 == 0 or position == len(runs):
                logger.info("Model %s: %d/%d runs (%.0f s)", name, position, len(runs), time.perf_counter() - started)

        entry: dict[str, Any] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "weight_count": int(model.count_params()),
            "quantizable_layers": len(simulator.layers),
            "folded_batch_norms": sum(layer.folded_batch_norm is not None for layer in simulator.layers),
            "simulated_configurations": len(predictions),
        }
        if "tflite_int8" in args.experiments:
            tflite_path = output_dir / "tflite" / f"{name}_int8_ptq.tflite"
            outputs = export_and_predict_tflite(model, calibration_split, eval_inputs, tflite_path)
            fields = {
                "experiment": "tflite_int8",
                "backend": "tflite",
                "scope": "model",
                "layer_id": None,
                "layer_index": None,
                "quantized": "both",
                **spec_fields("w", default_weight_spec(8)),
                **spec_fields("a", default_activation_spec(8)),
            }
            for split_name, (first, last) in offsets.items():
                metrics = evaluate_split(pipeline, slice_outputs(outputs, first, last), splits[split_name])
                rows.append({"model": name, "split": split_name, "examples": last - first, **fields, **metrics})
            entry["tflite_int8_path"] = str(tflite_path)
            entry["tflite_int8_bytes"] = tflite_path.stat().st_size
        model_metadata[name] = entry
        write_rows(output_dir / RESULTS_FILE, rows, RESULT_BASE_COLUMNS)
        write_rows(output_dir / LAYERS_FILE, layer_rows, ("model",))
        del simulator, model
        tf.keras.backend.clear_session()

    metadata = {
        "created_utc": utc_now(),
        "command": "sweep",
        "arguments": {key: value for key, value in vars(args).items() if key != "handler"},
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "git": git_state(),
        "versions": runtime_versions(),
        "dataset": dict(pipeline.bundle.metadata),
        "splits": {name: last - first for name, (first, last) in offsets.items()},
        "calibration": {
            "source_split": calibration_source,
            "source_examples": int(calibration_split.shape[0]),
            "samples": int(calibration_samples.shape[0]),
            "max_values_per_layer": args.max_calibration_values,
            "seed": args.seed,
        },
        "experiments": list(args.experiments),
        "options": asdict(options),
        "conventions": {
            "weights": "fake-quantized kernels; symmetric per-channel min-max unless the experiment varies it",
            "activations": "input of each quantized layer, calibrated on the float model; "
            "asymmetric per-tensor min-max unless the experiment varies it",
            "batch_norm": "folded into the preceding kernel before weight quantization",
            "bias": "float",
            "unquantized_tensors": "tensors that feed no Dense or convolution layer, including model outputs",
        },
        "models": model_metadata,
    }
    write_json(output_dir / METADATA_FILE, metadata)
    logger.info("Wrote %d rows to %s", len(rows), output_dir / RESULTS_FILE)
    return 0


# --------------------------------------------------------------------------
# export-int4-activations
# --------------------------------------------------------------------------
def parse_tensor_selection(raw: str) -> str | tuple[int, ...]:
    """Parse ``--tensors``: ``all``, ``minimal``, or comma-separated tensor indices.

    Parameters
    ----------
    raw : str
        Option text.

    Returns
    -------
    str | tuple[int, ...]
        The keyword, or sorted unique tensor indices.

    Raises
    ------
    argparse.ArgumentTypeError
        If the text is neither a keyword nor an integer list.
    """
    text = raw.strip().lower()
    return text if text in {"all", "minimal"} else parse_int_list(text)


def run_export_int4_activations(args: argparse.Namespace) -> int:
    """Rewrite an int8 TFLite model with 4-bit feature maps and write a report.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``export-int4-activations`` options.

    Returns
    -------
    int
        Process exit code.

    Raises
    ------
    ValueError
        If the output exists without ``--overwrite`` or the input is missing.
    """
    from crest.quantization.tflite_int4_activations import export_int4_activations

    source, output = Path(args.tflite), Path(args.output)
    report_path = output.with_suffix(".json")
    if not source.is_file():
        raise ValueError(f"TFLite model not found: {source}.")
    existing = [path for path in (output, report_path) if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(f"{existing[0]} already exists; pass --overwrite to replace it.")

    _, pipeline = load_pipeline(args.config)
    _, calibration, calibration_source = select_calibration(pipeline)
    rewritten, report = export_int4_activations(
        source.read_bytes(),
        calibration,
        tensors=args.tensors,
        range_method=args.range_method,
        percentile=args.percentile,
        max_values=args.max_calibration_values,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rewritten)
    report.update(
        {
            "created_utc": utc_now(),
            "config": str(args.config),
            "source_model": str(source),
            "source_sha256": sha256_file(source),
            "output_model": str(output),
            "output_sha256": sha256_file(output),
            "output_bytes": output.stat().st_size,
            "calibration_source_split": calibration_source,
            "versions": runtime_versions(),
        }
    )
    write_json(report_path, report)
    peak = report["estimated_activation_peak_bytes"]
    logger.info(
        "Stored %d tensors as INT4; estimated activation peak %d -> %d bytes (%.0f%%). Wrote %s and %s.",
        len(report["tensors"]),
        peak["int8"],
        peak["int4"],
        100.0 * peak["int4"] / peak["int8"],
        output,
        report_path,
    )
    return 0


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
def _bits(value: str) -> int | None:
    """Parse a bit-width cell.

    Parameters
    ----------
    value : str
        CSV cell.

    Returns
    -------
    int | None
        Width, or None for a float tensor.
    """
    return int(value) if value not in ("", None) else None


def _flag(value: str) -> bool | None:
    """Parse a boolean cell.

    Parameters
    ----------
    value : str
        CSV cell.

    Returns
    -------
    bool | None
        Parsed flag, or None when empty.
    """
    return {"true": True, "false": False}.get(str(value).lower())


def metric_columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Return metric columns in file order.

    Parameters
    ----------
    rows : Sequence[Mapping[str, str]]
        Results rows.

    Returns
    -------
    list[str]
        Columns that are not configuration columns.
    """
    return [column for column in (rows[0] if rows else {}) if column not in RESULT_BASE_COLUMNS]


def choose_metric(rows: Sequence[Mapping[str, str]], requested: str | None) -> str:
    """Choose the metric to report.

    Parameters
    ----------
    rows : Sequence[Mapping[str, str]]
        Results rows.
    requested : str | None
        User choice.

    Returns
    -------
    str
        ``requested`` if given, else ``macro_f1``, ``accuracy``, or the first metric.

    Raises
    ------
    ValueError
        If the requested metric is absent or no metric exists.
    """
    available = metric_columns(rows)
    if requested:
        if requested not in available:
            raise ValueError(f"Metric {requested!r} not in results; available: {available}.")
        return requested
    for preferred in ("macro_f1", "accuracy"):
        if preferred in available:
            return preferred
    if not available:
        raise ValueError("The results contain no metric columns.")
    return available[0]


def metric_is_lower_better(metric: str) -> bool:
    """Guess the direction of a metric from its name.

    Parameters
    ----------
    metric : str
        Metric name.

    Returns
    -------
    bool
        True for loss- and error-like names.
    """
    lowered = metric.lower()
    return any(hint in lowered for hint in LOWER_IS_BETTER_HINTS)


def _value(rows: Iterable[Mapping[str, str]], metric: str, **conditions: Any) -> float | None:
    """Return the metric of the single row matching the conditions.

    Parameters
    ----------
    rows : Iterable[Mapping[str, str]]
        Results rows.
    metric : str
        Metric column.
    **conditions : Any
        Column filters; callables receive the cell, other values must equal it.

    Returns
    -------
    float | None
        Metric of the first match, or None.
    """
    for row in rows:
        matched = True
        for column, expected in conditions.items():
            cell = row.get(column, "")
            if callable(expected):
                if not expected(cell):
                    matched = False
                    break
            elif cell != expected:
                matched = False
                break
        if matched and row.get(metric, "") != "":
            return float(row[metric])
    return None


def bit_series(rows: Sequence[Mapping[str, str]], metric: str) -> dict[str, dict[int, float]]:
    """Extract whole-model bit-width curves from the ``bits`` experiment.

    Parameters
    ----------
    rows : Sequence[Mapping[str, str]]
        Rows for one model and split.
    metric : str
        Metric column.

    Returns
    -------
    dict[str, dict[int, float]]
        ``weights``, ``activations``, and ``both`` (equal widths) curves.
    """
    series: dict[str, dict[int, float]] = {"weights": {}, "activations": {}, "both": {}}
    for row in rows:
        if row["experiment"] != "bits" or row.get(metric, "") == "":
            continue
        weight_bits, activation_bits = _bits(row["w_bits"]), _bits(row["a_bits"])
        value = float(row[metric])
        if row["quantized"] == "weights":
            series["weights"][weight_bits] = value
        elif row["quantized"] == "activations":
            series["activations"][activation_bits] = value
        elif row["quantized"] == "both" and weight_bits == activation_bits:
            series["both"][weight_bits] = value
    return series


def min_bits_within(
    curve: Mapping[int, float], reference: float | None, tolerance: float, lower_is_better: bool
) -> int | None:
    """Return the smallest width from which every wider width stays within tolerance.

    Parameters
    ----------
    curve : Mapping[int, float]
        Metric per bit-width.
    reference : float | None
        Float-model metric.
    tolerance : float
        Allowed degradation in metric units.
    lower_is_better : bool
        Metric direction.

    Returns
    -------
    int | None
        Smallest qualifying width, or None if the widest width already fails.
    """
    if reference is None or not curve:
        return None
    best = None
    for bits in sorted(curve, reverse=True):
        degradation = curve[bits] - reference if lower_is_better else reference - curve[bits]
        if degradation > tolerance:
            break
        best = bits
    return best


def summarize(
    rows: Sequence[Mapping[str, str]],
    models: Sequence[str],
    metric: str,
    tolerance: float,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    """Build one summary row per model.

    Parameters
    ----------
    rows : Sequence[Mapping[str, str]]
        Rows for one split.
    models : Sequence[str]
        Model names in report order.
    metric : str
        Metric column.
    tolerance : float
        Allowed degradation for the minimum-width columns.
    lower_is_better : bool
        Metric direction.

    Returns
    -------
    list[dict[str, Any]]
        Float, TFLite int8, simulated W8A8, the simulation's deviation from
        TFLite, and minimum widths within tolerance.
    """
    summary = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        reference = _value(model_rows, metric, experiment="float")
        tflite = _value(model_rows, metric, experiment="tflite_int8")
        w8a8 = _value(model_rows, metric, experiment="bits", w_bits="8", a_bits="8")
        curves = bit_series(model_rows, metric)
        summary.append(
            {
                "model": model,
                "split": model_rows[0]["split"] if model_rows else "",
                "metric": metric,
                "float": reference,
                "tflite_int8": tflite,
                "sim_w8a8": w8a8,
                "sim_w8a8_minus_tflite": None if tflite is None or w8a8 is None else w8a8 - tflite,
                "tolerance": tolerance,
                "min_weight_bits": min_bits_within(curves["weights"], reference, tolerance, lower_is_better),
                "min_activation_bits": min_bits_within(curves["activations"], reference, tolerance, lower_is_better),
                "min_joint_bits": min_bits_within(curves["both"], reference, tolerance, lower_is_better),
            }
        )
    return summary


def _pyplot() -> Any:
    """Import pyplot with a file-only backend and the shared style.

    Returns
    -------
    Any
        ``matplotlib.pyplot``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )
    return plt


def _style_axes(ax: Any, *, grid_axis: str = "y") -> None:
    """Apply recessive hairline axes and gridlines.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    grid_axis : str, optional
        Axis that carries gridlines.

    Returns
    -------
    None
        The axes are styled in place.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(length=3, width=0.75, color=AXIS)
    ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.75, linestyle="-")
    ax.set_axisbelow(True)


def _metric_label(metric: str) -> str:
    """Return a readable metric label.

    Parameters
    ----------
    metric : str
        Metric column.

    Returns
    -------
    str
        Label for axes and titles.
    """
    return {"macro_f1": "macro-F1", "accuracy": "accuracy", "loss": "loss"}.get(metric, metric.replace("_", " "))


def _grid_shape(count: int, max_columns: int = 5) -> tuple[int, int]:
    """Return balanced rows and columns for small multiples.

    Parameters
    ----------
    count : int
        Number of panels.
    max_columns : int, optional
        Column cap.
        The default matches the one-column-per-model method figures.

    Returns
    -------
    tuple[int, int]
        ``(rows, columns)``, using as few rows as the cap allows and then as
        few columns as those rows need, so 6 panels become 2 x 3, not 2 x 5.
    """
    rows = max(1, math.ceil(count / max_columns))
    return rows, max(1, math.ceil(count / rows))


def _line(ax: Any, xs: Sequence[float], ys: Sequence[float], slot: int, label: str) -> Any:
    """Draw one categorical series as a 2 px line with ringed markers.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    xs, ys : Sequence[float]
        Coordinates.
    slot : int
        Categorical slot (0-based).
    label : str
        Legend label.

    Returns
    -------
    Any
        The line artist.
    """
    (artist,) = ax.plot(
        xs,
        ys,
        color=SERIES_COLORS[slot],
        linewidth=1.5,
        marker=SERIES_MARKERS[slot],
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        solid_joinstyle="round",
        solid_capstyle="round",
        label=label,
        zorder=3,
    )
    return artist


def _reference_line(ax: Any, value: float | None) -> Any:
    """Draw the float-model reference as a muted hairline.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    value : float | None
        Float-model metric.

    Returns
    -------
    Any
        The line artist, or None.
    """
    if value is None:
        return None
    return ax.axhline(value, color=MUTED, linewidth=1.0, zorder=2, label="float model")


REFERENCE_LABELS = ("float model", "TFLite int8 export")


def _figure_legend(fig: Any, handles: Mapping[str, Any], panel_columns: int) -> None:
    """Add one figure-level legend: data series first, references last.

    Parameters
    ----------
    fig : Any
        Matplotlib figure.
    handles : Mapping[str, Any]
        Legend label to artist.
    panel_columns : int
        Number of panel columns, used to keep the legend about as wide as the panels.

    Returns
    -------
    None
        The legend is added to ``fig``.
    """
    ordered = [label for label in handles if label not in REFERENCE_LABELS]
    ordered += [label for label in REFERENCE_LABELS if label in handles]
    fig.legend(
        [handles[label] for label in ordered],
        ordered,
        loc="upper center",
        ncol=min(len(ordered), max(2, 2 * panel_columns)),
        bbox_to_anchor=(0.5, 1.0),
    )


def _finish_layout(fig: Any, title: str) -> None:
    """Fit the panels below the legend band and put the title above the legend.

    Parameters
    ----------
    fig : Any
        Matplotlib figure with a legend from :func:`_figure_legend`.
    title : str
        Figure title.

    Returns
    -------
    None
        The figure is laid out in place.

    Notes
    -----
    The band and the title offset are fixed in inches, so one-row and
    two-row figures get the same spacing.
    The title is figure text above the figure, where ``bbox_inches="tight"``
    keeps it when saving.
    It is not a suptitle, because ``tight_layout`` reserves a suptitle's
    height inside the layout rectangle, which would leave a gap below the legend.
    """
    height = fig.get_figheight()
    fig.text(
        0.01,
        1.0 + TITLE_OFFSET_INCHES / height,
        title,
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 1.0 - LEGEND_BAND_INCHES / height))


def plot_bit_width(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    models: Sequence[str],
    metric: str,
    split: str,
    output_dir: Path,
) -> list[Path]:
    """Plot whole-model metric against bit-width, one panel per model.

    Parameters
    ----------
    plt : Any
        ``matplotlib.pyplot``.
    rows : Sequence[Mapping[str, str]]
        Rows for one split.
    models : Sequence[str]
        Models in panel order.
    metric : str
        Metric column.
    split : str
        Split name for the title.
    output_dir : pathlib.Path
        Destination directory.

    Returns
    -------
    list[pathlib.Path]
        Written figure and sidecar paths.
    """
    labels = {"weights": "weights only", "activations": "activations only", "both": "weights and activations"}
    curves = {model: bit_series([row for row in rows if row["model"] == model], metric) for model in models}
    widths = sorted({bits for model in models for curve in curves[model].values() for bits in curve})
    if not widths:
        return []
    position = {bits: index for index, bits in enumerate(widths)}
    nrows, ncols = _grid_shape(len(models))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows + 0.6), squeeze=False, sharey=True)
    sidecar = []
    handles = {}
    for index, model in enumerate(models):
        ax = axes[index // ncols][index % ncols]
        _style_axes(ax)
        model_rows = [row for row in rows if row["model"] == model]
        reference = _value(model_rows, metric, experiment="float")
        artist = _reference_line(ax, reference)
        if artist is not None:
            handles["float model"] = artist
            sidecar.append({"model": model, "series": "float model", "bits": None, metric: reference})
        for slot, key in enumerate(("weights", "activations", "both")):
            curve = curves[model][key]
            if not curve:
                continue
            ordered = sorted(curve)
            handles[labels[key]] = _line(
                ax, [position[bits] for bits in ordered], [curve[bits] for bits in ordered], slot, labels[key]
            )
            sidecar.extend(
                {"model": model, "series": labels[key], "bits": bits, metric: curve[bits]} for bits in ordered
            )
        tflite = _value(model_rows, metric, experiment="tflite_int8")
        if tflite is not None and 8 in position:
            handles["TFLite int8 export"] = ax.scatter(
                [position[8]],
                [tflite],
                marker="X",
                s=48,
                color=TEXT_PRIMARY,
                edgecolors=SURFACE,
                linewidths=1.0,
                zorder=4,
                label="TFLite int8 export",
            )
            sidecar.append({"model": model, "series": "TFLite int8 export", "bits": 8, metric: tflite})
        ax.set_xticks(range(len(widths)), [str(bits) for bits in widths])
        ax.set_title(model, loc="left")
        ax.set_xlabel("bits")
        if index % ncols == 0:
            ax.set_ylabel(_metric_label(metric))
    for index in range(len(models), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)
    _figure_legend(fig, handles, ncols)
    _finish_layout(fig, f"{_metric_label(metric)} against bit-width, {split} split")
    figure_path = output_dir / "bit_width.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    sidecar_path = output_dir / "bit_width.csv"
    write_rows(sidecar_path, sidecar)
    return [figure_path, sidecar_path]


def plot_bit_grids(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    models: Sequence[str],
    metric: str,
    split: str,
    output_dir: Path,
) -> list[Path]:
    """Plot the weight-by-activation bit-width grid for each model.

    Parameters
    ----------
    plt : Any
        ``matplotlib.pyplot``.
    rows : Sequence[Mapping[str, str]]
        Rows for one split.
    models : Sequence[str]
        Models to plot.
    metric : str
        Metric column.
    split : str
        Split name for the title.
    output_dir : pathlib.Path
        Destination directory.

    Returns
    -------
    list[pathlib.Path]
        Written figure and sidecar paths.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Missing cells render as the surface, so they read as "not run" rather than a value.
    colormap = LinearSegmentedColormap.from_list("crest_sequential_blue", SEQUENTIAL_BLUE).with_extremes(bad=SURFACE)
    written = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        cells = {}
        for row in model_rows:
            if row["experiment"] == "bits" and row.get(metric, "") != "":
                cells[(_bits(row["w_bits"]), _bits(row["a_bits"]))] = float(row[metric])
        if not cells:
            continue
        reference = _value(model_rows, metric, experiment="float")
        if reference is not None:
            cells[(None, None)] = reference
        weight_axis = [None] + sorted({w for w, _ in cells if w is not None}, reverse=True)
        activation_axis = [None] + sorted({a for _, a in cells if a is not None}, reverse=True)
        grid = np.full((len(weight_axis), len(activation_axis)), np.nan)
        for (weight_bits, activation_bits), value in cells.items():
            grid[weight_axis.index(weight_bits), activation_axis.index(activation_bits)] = value
        fig, ax = plt.subplots(figsize=(0.42 * len(activation_axis) + 2.2, 0.36 * len(weight_axis) + 1.3))
        mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=colormap, edgecolors=SURFACE, linewidth=1.5)
        ax.set_xticks(
            np.arange(len(activation_axis)) + 0.5, ["float" if a is None else str(a) for a in activation_axis]
        )
        ax.set_yticks(np.arange(len(weight_axis)) + 0.5, ["float" if w is None else str(w) for w in weight_axis])
        ax.invert_yaxis()
        ax.set_xlabel("activation bits")
        ax.set_ylabel("weight bits")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03)
        colorbar.set_label(_metric_label(metric))
        colorbar.outline.set_visible(False)
        ax.set_title(f"{model}: {_metric_label(metric)} by weight and activation bits, {split} split", loc="left")
        figure_path = output_dir / f"bit_grid_{model}.png"
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        sidecar_path = output_dir / f"bit_grid_{model}.csv"
        write_rows(
            sidecar_path,
            [
                {"model": model, "w_bits": weight_bits, "a_bits": activation_bits, metric: value}
                for (weight_bits, activation_bits), value in sorted(
                    cells.items(), key=lambda item: (item[0][0] or 99, item[0][1] or 99), reverse=True
                )
            ],
        )
        written += [figure_path, sidecar_path]
    return written


def _variant_key(row: Mapping[str, str], prefix: str, experiment: str) -> str:
    """Return the series label for a granularity or clipping row.

    Parameters
    ----------
    row : Mapping[str, str]
        Results row.
    prefix : str
        ``"w"`` or ``"a"``.
    experiment : str
        ``"granularity"`` or ``"clipping"``.

    Returns
    -------
    str
        Series label; one label per method or variant, so a series keeps
        one color in every panel.
    """
    if experiment == "clipping":
        return {"minmax": "min-max", "percentile": "percentile", "mse": "MSE-optimal"}[row[f"{prefix}_range_method"]]
    granularity = row[f"{prefix}_granularity"].replace("_", "-")
    symmetry = "symmetric" if _flag(row[f"{prefix}_symmetric"]) else "asymmetric"
    return f"{granularity}, {symmetry}"


def _percentile_label(rows: Sequence[Mapping[str, str]]) -> str:
    """Return the legend label for percentile clipping, naming the percentiles used.

    Parameters
    ----------
    rows : Sequence[Mapping[str, str]]
        Clipping rows.

    Returns
    -------
    str
        For example ``"percentile (weights 99.9, activations 99.99)"``.
    """
    parts = []
    for prefix, name in (("w", "weights"), ("a", "activations")):
        values = sorted({row[f"{prefix}_percentile"] for row in rows if row[f"{prefix}_range_method"] == "percentile"})
        if values:
            parts.append(f"{name} {'/'.join(values)}")
    return f"percentile ({', '.join(parts)})" if parts else "percentile"


def plot_variants(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    models: Sequence[str],
    metric: str,
    split: str,
    experiment: str,
    output_dir: Path,
) -> list[Path]:
    """Plot granularity or clipping variants for weights and activations.

    Parameters
    ----------
    plt : Any
        ``matplotlib.pyplot``.
    rows : Sequence[Mapping[str, str]]
        Rows for one split.
    models : Sequence[str]
        Models, one column each.
    metric : str
        Metric column.
    split : str
        Split name for the title.
    experiment : {"granularity", "clipping"}
        Experiment to plot.
    output_dir : pathlib.Path
        Destination directory.

    Returns
    -------
    list[pathlib.Path]
        Written figure and sidecar paths.
    """
    selected = [row for row in rows if row["experiment"] == experiment and row.get(metric, "") != ""]
    if not selected:
        return []
    # Fixed series order keeps each variant's color stable across panels.
    if experiment == "granularity":
        order = ["per-channel, symmetric", "per-tensor, symmetric", "per-channel, asymmetric", "per-tensor, asymmetric"]
        display = {label: label for label in order}
    else:
        order = ["min-max", "percentile", "MSE-optimal"]
        display = {"min-max": "min-max", "percentile": _percentile_label(selected), "MSE-optimal": "MSE-optimal"}
    targets = (("weights", "w"), ("activations", "a"))
    fig, axes = plt.subplots(2, len(models), figsize=(3.0 * len(models), 5.4), squeeze=False, sharey="row")
    sidecar = []
    handles: dict[str, Any] = {}
    for column, model in enumerate(models):
        model_rows = [row for row in rows if row["model"] == model]
        reference = _value(model_rows, metric, experiment="float")
        for row_index, (target, prefix) in enumerate(targets):
            ax = axes[row_index][column]
            _style_axes(ax)
            artist = _reference_line(ax, reference)
            if artist is not None:
                handles["float model"] = artist
            curves: dict[str, dict[int, float]] = {}
            for row in selected:
                if row["model"] == model and row["quantized"] == target:
                    curves.setdefault(_variant_key(row, prefix, experiment), {})[_bits(row[f"{prefix}_bits"])] = float(
                        row[metric]
                    )
            widths = sorted({bits for curve in curves.values() for bits in curve})
            position = {bits: index for index, bits in enumerate(widths)}
            for slot, label in enumerate(order):
                curve = curves.get(label)
                if not curve:
                    continue
                ordered = sorted(curve)
                handles[display[label]] = _line(
                    ax, [position[bits] for bits in ordered], [curve[bits] for bits in ordered], slot, display[label]
                )
                sidecar.extend(
                    {"model": model, "tensor": target, "series": label, "bits": bits, metric: curve[bits]}
                    for bits in ordered
                )
            ax.set_xticks(range(len(widths)), [str(bits) for bits in widths])
            if row_index == 0:
                ax.set_title(model, loc="left")
            if row_index == 1:
                ax.set_xlabel("bits")
            if column == 0:
                ax.set_ylabel(f"{target}: {_metric_label(metric)}")
    _figure_legend(fig, handles, len(models))
    title = "granularity and symmetry" if experiment == "granularity" else "clipping range"
    _finish_layout(fig, f"{_metric_label(metric)} by {title}, {split} split")
    figure_path = output_dir / f"{experiment}.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    sidecar_path = output_dir / f"{experiment}.csv"
    write_rows(sidecar_path, sidecar)
    return [figure_path, sidecar_path]


def plot_layer_sensitivity(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    models: Sequence[str],
    metric: str,
    split: str,
    output_dir: Path,
) -> list[Path]:
    """Plot the metric change from quantizing one layer at a time.

    Parameters
    ----------
    plt : Any
        ``matplotlib.pyplot``.
    rows : Sequence[Mapping[str, str]]
        Rows for one split.
    models : Sequence[str]
        Models to plot, one figure each.
    metric : str
        Metric column.
    split : str
        Split name for the title.
    output_dir : pathlib.Path
        Destination directory.

    Returns
    -------
    list[pathlib.Path]
        Written figure and sidecar paths.
    """
    written = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        selected = [row for row in model_rows if row["experiment"] == "layer_sensitivity" and row.get(metric, "") != ""]
        reference = _value(model_rows, metric, experiment="float")
        if not selected or reference is None:
            continue
        layers = sorted({(int(row["layer_index"]), row["layer_id"]) for row in selected})
        layer_names = [name for _, name in layers]
        widths = sorted({_bits(row["w_bits"]) or _bits(row["a_bits"]) for row in selected})[: len(SERIES_COLORS)]
        fig, axes = plt.subplots(
            1, 2, figsize=(9.0, 0.32 * len(layers) * len(widths) / 2 + 1.6), sharey=True, squeeze=False
        )
        sidecar = []
        handles: dict[str, Any] = {}
        bar_height = 0.8 / max(1, len(widths))
        for column, (target, prefix) in enumerate((("weights", "w"), ("activations", "a"))):
            ax = axes[0][column]
            _style_axes(ax, grid_axis="x")
            ax.axvline(0.0, color=AXIS, linewidth=0.75, zorder=2)
            for slot, bits in enumerate(widths):
                deltas = []
                for name in layer_names:
                    value = _value(selected, metric, layer_id=name, quantized=target, **{f"{prefix}_bits": str(bits)})
                    deltas.append(np.nan if value is None else value - reference)
                    sidecar.append(
                        {
                            "model": model,
                            "tensor": target,
                            "layer_id": name,
                            "bits": bits,
                            f"delta_{metric}": None if value is None else value - reference,
                        }
                    )
                offsets = np.arange(len(layer_names)) + (slot - (len(widths) - 1) / 2) * bar_height
                handles[f"{bits}-bit"] = ax.barh(
                    offsets,
                    deltas,
                    height=bar_height * 0.85,
                    color=SERIES_COLORS[slot],
                    edgecolor=SURFACE,
                    linewidth=0.0,
                    label=f"{bits}-bit",
                    zorder=3,
                )
                # A zero-length bar is invisible, so a tick marks a measured change of exactly zero.
                unchanged = np.isclose(np.asarray(deltas, dtype=float), 0.0, rtol=0.0, atol=1e-12)
                if unchanged.any():
                    ax.plot(
                        np.zeros(int(unchanged.sum())),
                        offsets[unchanged],
                        linestyle="none",
                        marker="|",
                        markersize=7,
                        markeredgewidth=2.0,
                        color=SERIES_COLORS[slot],
                        zorder=4,
                    )
            ax.set_yticks(range(len(layer_names)), layer_names)
            if column == 0:
                # The panels share one y-axis; inverting it twice would undo the flip.
                ax.invert_yaxis()
            ax.set_title(f"{target} of one layer", loc="left")
            ax.set_xlabel(f"change in {_metric_label(metric)} vs float")
        _figure_legend(fig, handles, 2)
        _finish_layout(fig, f"{model}: one layer quantized at a time, {split} split")
        figure_path = output_dir / f"layer_sensitivity_{model}.png"
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        sidecar_path = output_dir / f"layer_sensitivity_{model}.csv"
        write_rows(sidecar_path, sidecar)
        written += [figure_path, sidecar_path]
    return written


def run_plot(args: argparse.Namespace) -> int:
    """Summarize and plot sweep results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``plot`` arguments.

    Returns
    -------
    int
        Process exit code.

    Raises
    ------
    ValueError
        If the results lack the requested split or metric.
    """
    results_dir = Path(args.results_dir)
    all_rows = read_rows(results_dir / RESULTS_FILE)
    metric = choose_metric(all_rows, args.metric)
    rows = [row for row in all_rows if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No results for split {args.split!r}.")
    models = list(dict.fromkeys(row["model"] for row in rows))
    if args.models:
        unknown = [model for model in args.models if model not in models]
        if unknown:
            raise ValueError(f"Unknown models {unknown}; results contain {models}.")
        models = list(args.models)
    lower_is_better = args.lower_is_better or metric_is_lower_better(metric)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / SUMMARY_FILE
    write_rows(summary_path, summarize(rows, models, metric, args.tolerance, lower_is_better))
    plt = _pyplot()
    written = [summary_path]
    written += plot_bit_width(plt, rows, models, metric, args.split, output_dir)
    written += plot_bit_grids(plt, rows, models, metric, args.split, output_dir)
    written += plot_variants(plt, rows, models, metric, args.split, "granularity", output_dir)
    written += plot_variants(plt, rows, models, metric, args.split, "clipping", output_dir)
    written += plot_layer_sensitivity(plt, rows, models, metric, args.split, output_dir)
    for path in written:
        logger.info("Wrote %s", path)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``train``, ``sweep``, and ``plot`` subcommands.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: %(default)s).")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a fixed candidate panel.")
    train.add_argument("--config", type=Path, required=True, help="CREST YAML config (dataset, task, family).")
    train.add_argument("--panel", type=Path, required=True, help="Panel YAML with named hyperparameter sets.")
    train.add_argument("--output-dir", type=Path, required=True, help="Directory for models and manifest.json.")
    train.add_argument("--candidates", type=parse_name_list, help="Comma-separated subset of panel candidates.")
    train.add_argument("--epochs", type=int, help="Maximum epochs (default: training.model_epochs).")
    train.add_argument("--patience", type=int, help="Early-stopping patience (default: the task config's value).")
    train.add_argument(
        "--seed", type=int, default=17, help="Seed set before building each candidate (default: %(default)s)."
    )
    train.add_argument(
        "--fit-verbose", type=int, default=2, choices=(0, 1, 2), help="Keras fit verbosity (default: %(default)s)."
    )
    train.add_argument("--overwrite", action="store_true", help="Allow an existing manifest to be replaced.")
    train.add_argument(
        "--execute", action="store_true", help="Train; without it the command only validates and sizes the panel."
    )
    train.set_defaults(handler=run_train)

    sweep = commands.add_parser("sweep", help="Simulate quantization on trained models.")
    sweep.add_argument("--config", type=Path, required=True, help="CREST YAML config (dataset, task, family).")
    sweep.add_argument(
        "--panel-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory written by the train subcommand; repeatable.",
    )
    sweep.add_argument("--model", action="append", default=[], help="Extra model as NAME=PATH; repeatable.")
    sweep.add_argument("--output-dir", type=Path, required=True, help="Directory for results.csv and metadata.json.")
    sweep.add_argument(
        "--experiments",
        type=lambda raw: parse_name_list(raw, EXPERIMENTS),
        default=EXPERIMENTS,
        help=f"Comma-separated subset of {','.join(EXPERIMENTS)} (default: all).",
    )
    sweep.add_argument(
        "--splits",
        type=lambda raw: parse_name_list(raw, ("train", "val", "test")),
        default=("val", "test"),
        help="Splits to evaluate (default: val,test).",
    )
    sweep.add_argument(
        "--weight-bits", type=parse_int_list, default=DEFAULT_WEIGHT_BITS, help="Weight widths for the bits grid."
    )
    sweep.add_argument(
        "--activation-bits",
        type=parse_int_list,
        default=DEFAULT_ACTIVATION_BITS,
        help="Activation widths for the bits grid.",
    )
    sweep.add_argument(
        "--ablation-bits",
        type=parse_int_list,
        default=DEFAULT_ABLATION_BITS,
        help="Widths for granularity and clipping.",
    )
    sweep.add_argument(
        "--layer-bits", type=parse_int_list, default=DEFAULT_LAYER_BITS, help="Widths for layer sensitivity."
    )
    sweep.add_argument(
        "--weight-percentile", type=float, default=DEFAULT_WEIGHT_PERCENTILE, help="Percentile clipping for weights."
    )
    sweep.add_argument(
        "--activation-percentile",
        type=float,
        default=DEFAULT_ACTIVATION_PERCENTILE,
        help="Percentile clipping for activations.",
    )
    sweep.add_argument(
        "--batch-size", type=int, default=256, help="Inference and calibration batch size (default: %(default)s)."
    )
    sweep.add_argument(
        "--max-calibration-values",
        type=int,
        default=500_000,
        help="Stored calibration values per layer (default: %(default)s).",
    )
    sweep.add_argument("--seed", type=int, default=0, help="Calibration sampling seed (default: %(default)s).")
    sweep.add_argument("--overwrite", action="store_true", help="Allow an existing results.csv to be replaced.")
    sweep.set_defaults(handler=run_sweep)

    export = commands.add_parser(
        "export-int4-activations",
        help="Store feature maps of an int8 TFLite model as 4-bit tensors (needs the 4-bit activation patch to run).",
    )
    export.add_argument("--config", type=Path, required=True, help="CREST YAML config (calibration data).")
    export.add_argument("--tflite", type=Path, required=True, help="Input int8 .tflite, for example from sweep.")
    export.add_argument(
        "--output", type=Path, required=True, help="Output .tflite; the report goes next to it as .json."
    )
    export.add_argument(
        "--tensors",
        type=parse_tensor_selection,
        default="all",
        help="'all' feature maps, the 'minimal' set with the same estimated memory peak, or tensor indices "
        "(default: %(default)s).",
    )
    export.add_argument(
        "--range-method",
        choices=RANGE_METHODS,
        default="mse",
        help="4-bit clipping estimator (default: %(default)s).",
    )
    export.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_ACTIVATION_PERCENTILE,
        help="Percentile for --range-method percentile.",
    )
    export.add_argument(
        "--max-calibration-values",
        type=int,
        default=500_000,
        help="Stored calibration values per tensor (default: %(default)s).",
    )
    export.add_argument("--seed", type=int, default=0, help="Calibration sampling seed (default: %(default)s).")
    export.add_argument("--overwrite", action="store_true", help="Allow an existing output to be replaced.")
    export.set_defaults(handler=run_export_int4_activations)

    plot = commands.add_parser("plot", help="Summarize and plot sweep results.")
    plot.add_argument("--results-dir", type=Path, required=True, help="Directory written by the sweep subcommand.")
    plot.add_argument("--output-dir", type=Path, help="Figure directory (default: RESULTS_DIR/figures).")
    plot.add_argument("--split", default="test", help="Split to report (default: %(default)s).")
    plot.add_argument("--metric", help="Metric column (default: macro_f1, accuracy, or the first metric).")
    plot.add_argument("--models", type=parse_name_list, help="Comma-separated models to include, in order.")
    plot.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Allowed metric drop for minimum-width columns (default: %(default)s).",
    )
    plot.add_argument("--lower-is-better", action="store_true", help="Treat the metric as lower-is-better.")
    plot.set_defaults(handler=run_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s:%(name)s:%(message)s"
    )
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
