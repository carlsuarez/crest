# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the host-only quantization study script."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "analysis_scripts" / "quantization_study" / "quantization_study.py"
PANEL_PATH = ROOT_DIR / "analysis_scripts" / "quantization_study" / "panels" / "audio_dscnn.yaml"
AUDIO_CONFIG_PATH = ROOT_DIR / "src" / "config" / "nas_config_audio_stm32.yaml"


def load_study_module() -> Any:
    """Load the study script as an importable module.

    Returns
    -------
    Any
        Imported module object.

    Raises
    ------
    RuntimeError
        If the script cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location("quantization_study_for_tests", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["quantization_study_for_tests"] = module
    spec.loader.exec_module(module)
    return module


study = load_study_module()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.

    Returns
    -------
    list[dict[str, str]]
        Rows.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_simulated_runs_cover_experiments_and_share_cache_keys():
    """Run counts follow the options, and default-spec runs share predictions across experiments."""
    options = study.SweepOptions(weight_bits=(4, 8), activation_bits=(8, 16), ablation_bits=(4,), layer_bits=(2, 4))
    runs = study.build_simulated_runs(["a", "b", "c"], study.EXPERIMENTS, options)
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.experiment] = counts.get(run.experiment, 0) + 1
    assert counts == {"float": 1, "bits": 3 * 3 - 1, "granularity": 8, "clipping": 6, "layer_sensitivity": 3 * 2 * 2}
    keys = {run.cache_key for run in runs}
    # Granularity and clipping repeat the bits experiment's default W4 and A-default specs.
    assert len(keys) < len(runs)
    bits_w4 = next(
        run for run in runs if run.experiment == "bits" and run.quantized == "weights" and run.weight.bits == 4
    )
    assert any(run.experiment == "granularity" and run.cache_key == bits_w4.cache_key for run in runs)
    assert any(run.experiment == "clipping" and run.cache_key == bits_w4.cache_key for run in runs)
    layer_run = next(run for run in runs if run.scope == "layer")
    assert layer_run.row_fields({"a": 0, "b": 1, "c": 2})["layer_index"] == 0
    assert study.build_simulated_runs(["a"], ("tflite_int8",), options) == []


def test_parsers_and_names_reject_bad_input():
    """CLI list parsers and name checks fail early."""
    assert study.parse_int_list("8,2,4,2") == (2, 4, 8)
    assert study.parse_name_list("test,val,test") == ("test", "val")
    with pytest.raises(Exception):
        study.parse_int_list("2,x")
    with pytest.raises(Exception):
        study.parse_name_list("bits,unknown", study.EXPERIMENTS)
    assert study.parse_tensor_selection(" Minimal ") == "minimal"
    assert study.parse_tensor_selection("12,3,12") == (3, 12)
    with pytest.raises(Exception):
        study.parse_tensor_selection("every")
    with pytest.raises(ValueError):
        study.validate_name("../escape", kind="model")
    with pytest.raises(ValueError):
        study.resolve_models([], [])
    with pytest.raises(ValueError):
        study.resolve_models([], ["missing_separator"])


def test_load_panel_validates_structure(tmp_path):
    """Panels need named hyperparameter mappings."""
    path = tmp_path / "panel.yaml"
    path.write_text(
        yaml.safe_dump({"family": "audio_dscnn", "candidates": {"ok": {"num_blocks": 2}}}), encoding="utf-8"
    )
    assert study.load_panel(path) == ("audio_dscnn", {"ok": {"num_blocks": 2}})
    for payload in ({"candidates": {}}, {"candidates": {"bad name": {"a": 1}}}, {"candidates": {"x": []}}):
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            study.load_panel(path)


def test_shipped_audio_panel_uses_legal_family_choices():
    """Every shipped candidate sets each audio_dscnn hyperparameter to a searchable value."""
    from crest.model_families.audio_dscnn.decode import AUDIO_DSCNN_SEARCH_CHOICES

    family, candidates = study.load_panel(PANEL_PATH)
    assert family == "audio_dscnn"
    for name, hparams in candidates.items():
        assert set(hparams) == set(AUDIO_DSCNN_SEARCH_CHOICES), name
        for key, value in hparams.items():
            assert value in AUDIO_DSCNN_SEARCH_CHOICES[key], (name, key, value)


@pytest.mark.parametrize(
    ("count", "shape"),
    [(1, (1, 1)), (3, (1, 3)), (5, (1, 5)), (6, (2, 3)), (7, (2, 4)), (11, (3, 4))],
)
def test_grid_shape_balances_small_multiples(count, shape):
    """Panels fill as few rows as the column cap allows, without a mostly empty last row."""
    assert study._grid_shape(count) == shape


def test_min_bits_and_summary_follow_metric_direction():
    """The minimum width stops at the first width outside tolerance, from wide to narrow."""
    curve = {2: 0.40, 3: 0.79, 4: 0.80, 8: 0.81}
    assert study.min_bits_within(curve, 0.81, 0.025, lower_is_better=False) == 3
    assert study.min_bits_within(curve, 0.81, 0.015, lower_is_better=False) == 4
    assert study.min_bits_within({2: 0.81, 4: 0.5, 8: 0.81}, 0.81, 0.01, lower_is_better=False) == 8
    assert study.min_bits_within({4: 1.30, 8: 1.02}, 1.0, 0.05, lower_is_better=True) == 8
    assert study.min_bits_within({}, 1.0, 0.05, lower_is_better=True) is None
    assert study.metric_is_lower_better("rmse_total") and not study.metric_is_lower_better("macro_f1")


def _cache_metadata() -> dict[str, Any]:
    """Build metadata for a fixed-split UrbanSound8K cache fixture.

    Returns
    -------
    dict[str, Any]
        Metadata accepted by the dataset loader.
    """
    from crest.datasets import urbansound8k_common as common

    return {
        "schema_version": common.CACHE_SCHEMA_VERSION,
        "dataset_name": common.DATASET_NAME,
        "component_name": common.COMPONENT_NAME,
        "cache_version": common.CACHE_VERSION,
        "feature_kind": common.FEATURE_KIND,
        "sample_rate_hz": common.SAMPLE_RATE_HZ,
        "clip_duration_s": common.CLIP_DURATION_S,
        "target_num_samples": common.TARGET_NUM_SAMPLES,
        "mel_bins": common.MEL_BINS,
        "window_ms": common.WINDOW_MS,
        "window_length_samples": common.WINDOW_LENGTH_SAMPLES,
        "hop_ms": common.HOP_MS,
        "hop_length_samples": common.HOP_LENGTH_SAMPLES,
        "n_fft": common.N_FFT,
        "center": common.CENTER,
        "power": common.POWER,
        "log_floor": common.LOG_FLOOR,
        "log_scale": "natural_log",
        "expected_frames": common.EXPECTED_FRAMES,
        "feature_dtype": common.FEATURE_DTYPE,
        "normalization": "per_mel_bin_standardization",
        "normalization_mean_shape": [common.MEL_BINS],
        "normalization_std_shape": [common.MEL_BINS],
        "normalization_mean": [0.0] * common.MEL_BINS,
        "normalization_std": [1.0] * common.MEL_BINS,
        "normalization_epsilon": common.NORMALIZATION_EPSILON,
        "evaluation_protocol": "fixed_split",
        "fold_split": {key: list(value) for key, value in common.FOLD_SPLIT.items()},
        "normalization_scope": "train_split",
        "batch_period_ms": common.BATCH_PERIOD_MS,
        "train_crop_variants": common.TRAIN_CROP_VARIANTS,
        "crop_seed": common.CROP_SEED,
        "calibration_examples": common.CALIBRATION_EXAMPLES,
        "class_names": list(common.CLASS_NAMES),
        "label_encoding": common.LABEL_ENCODING,
    }


def _write_cache(cache_dir: Path, rows_per_split: dict[str, int]) -> None:
    """Write a small synthetic UrbanSound8K log-mel cache.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Destination cache directory.
    rows_per_split : dict[str, int]
        Row count for each split.

    Returns
    -------
    None
        Metadata and split archives are written.
    """
    from crest.datasets import urbansound8k_common as common

    cache_dir.mkdir(parents=True)
    (cache_dir / "metadata.json").write_text(json.dumps(_cache_metadata()), encoding="utf-8")
    rng = np.random.default_rng(0)
    for split_name, count in rows_per_split.items():
        labels = np.arange(count, dtype=np.int64) % len(common.CLASS_NAMES)
        # A label-dependent offset gives the classifier something to learn.
        inputs = rng.normal(size=(count, common.EXPECTED_FRAMES, common.MEL_BINS)).astype(np.float32)
        inputs += (labels[:, None, None] / 5.0).astype(np.float32)
        folds = common.FOLD_SPLIT["train" if split_name == "calibration" else split_name]
        np.savez_compressed(
            cache_dir / f"{split_name}.npz",
            inputs=inputs,
            labels=labels,
            folds=np.full(count, folds[0], dtype=np.int64),
            clip_ids=np.asarray([f"{split_name}_{index}" for index in range(count)], dtype=str),
            crop_start_sample=np.zeros(count, dtype=np.int64),
            crop_variant_index=np.zeros(count, dtype=np.int64),
            pad_before_sample=np.zeros(count, dtype=np.int64),
            pad_after_sample=np.zeros(count, dtype=np.int64),
        )


def _write_config(tmp_path: Path, cache_dir: Path) -> Path:
    """Copy the shipped audio config and point it at temporary paths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test directory.
    cache_dir : pathlib.Path
        Synthetic cache directory.

    Returns
    -------
    pathlib.Path
        Config path.
    """
    config = yaml.safe_load(AUDIO_CONFIG_PATH.read_text(encoding="utf-8"))
    config["dataset"]["params"]["cache_dir"] = str(cache_dir)
    config["dataset"]["params"]["fold_rotation_cache_dir"] = str(cache_dir / "fold_rotation")
    config["outputs"]["models_dir"] = str(tmp_path / "models")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_train_sweep_and_plot_end_to_end(tmp_path):
    """The three subcommands run on a synthetic cache and write consistent outputs."""
    pytest.importorskip("tensorflow")
    from crest.datasets import urbansound8k_common as common

    cache_dir = tmp_path / "cache" / common.CACHE_VERSION
    _write_cache(cache_dir, {"train": 30, "val": 20, "test": 20, "calibration": 20})
    config_path = _write_config(tmp_path, cache_dir)
    _, candidates = study.load_panel(PANEL_PATH)
    panel_path = tmp_path / "panel.yaml"
    panel_path.write_text(
        yaml.safe_dump({"family": "audio_dscnn", "candidates": {"small": candidates["small"]}}), encoding="utf-8"
    )

    panel_dir = tmp_path / "panel"
    common_train = ["train", "--config", str(config_path), "--panel", str(panel_path), "--output-dir", str(panel_dir)]
    assert study.main(common_train) == 0
    assert not (panel_dir / study.MANIFEST_FILE).exists()
    assert study.main(common_train + ["--epochs", "1", "--fit-verbose", "0", "--execute"]) == 0
    manifest = json.loads((panel_dir / study.MANIFEST_FILE).read_text(encoding="utf-8"))
    entry = manifest["candidates"]["small"]
    assert (panel_dir / entry["model_path"]).is_file()
    assert set(entry["float_metrics"]) == {"val", "test"}
    with pytest.raises(ValueError):
        study.main(common_train + ["--execute"])

    results_dir = tmp_path / "results"
    sweep_args = [
        "sweep",
        "--config",
        str(config_path),
        "--panel-dir",
        str(panel_dir),
        "--model",
        f"copy={panel_dir / entry['model_path']}",
        "--output-dir",
        str(results_dir),
        "--weight-bits",
        "4,8",
        "--activation-bits",
        "8,16",
        "--ablation-bits",
        "4",
        "--layer-bits",
        "4",
        "--batch-size",
        "16",
    ]
    assert study.main(sweep_args) == 0
    rows = read_csv(results_dir / study.RESULTS_FILE)
    layers = read_csv(results_dir / study.LAYERS_FILE)
    metadata = json.loads((results_dir / study.METADATA_FILE).read_text(encoding="utf-8"))
    layer_count = sum(1 for layer in layers if layer["model"] == "small")
    assert layer_count == 7  # three depthwise/pointwise blocks plus the logits Dense
    assert sum(layer["folded_batch_norm"] != "" for layer in layers if layer["model"] == "small") == 6
    simulated = 1 + 8 + 8 + 6 + layer_count * 2
    assert len(rows) == 2 * 2 * (simulated + 1)
    assert {row["split"] for row in rows} == {"val", "test"}
    assert sum(row["experiment"] == "tflite_int8" for row in rows) == 4
    assert all(0.0 <= float(row["accuracy"]) <= 1.0 for row in rows)
    float_test = next(
        float(row["accuracy"])
        for row in rows
        if row["model"] == "small" and row["experiment"] == "float" and row["split"] == "test"
    )
    assert float_test == pytest.approx(entry["float_metrics"]["test"]["accuracy"])
    assert metadata["models"]["small"]["tflite_int8_bytes"] > 0
    assert metadata["calibration"]["samples"] == 20
    with pytest.raises(ValueError):
        study.main(sweep_args)

    source_model = results_dir / "tflite" / "small_int8_ptq.tflite"
    export_output = tmp_path / "int4_activations" / "small.tflite"
    export_args = [
        "export-int4-activations",
        "--config",
        str(config_path),
        "--tflite",
        str(source_model),
        "--output",
        str(export_output),
    ]
    assert study.main(export_args + ["--tensors", "minimal"]) == 0
    export_report = json.loads(export_output.with_suffix(".json").read_text(encoding="utf-8"))
    assert export_report["selection"] == "minimal" and export_report["tensors"]
    peak = export_report["estimated_activation_peak_bytes"]
    assert peak["int4"] <= peak["int8"]
    assert export_report["source_sha256"] == study.sha256_file(source_model)
    assert export_report["output_sha256"] == study.sha256_file(export_output)
    assert export_report["buffers_16_byte_aligned"] is True
    with pytest.raises(ValueError):
        study.main(export_args)
    assert study.main(export_args + ["--tensors", "all", "--overwrite"]) == 0
    all_report = json.loads(export_output.with_suffix(".json").read_text(encoding="utf-8"))
    assert all_report["selection"] == "all"
    assert len(all_report["tensors"]) >= len(export_report["tensors"])

    assert study.main(["plot", "--results-dir", str(results_dir), "--metric", "accuracy"]) == 0
    figures = results_dir / "figures"
    for name in (
        "summary.csv",
        "bit_width.png",
        "bit_width.csv",
        "bit_grid_small.png",
        "granularity.png",
        "clipping.png",
        "layer_sensitivity_copy.png",
    ):
        assert (figures / name).is_file(), name
    summary = read_csv(figures / study.SUMMARY_FILE)
    assert [row["model"] for row in summary] == ["small", "copy"]
    assert summary[0]["float"] == summary[1]["float"]
