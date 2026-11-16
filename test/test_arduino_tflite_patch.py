# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the opt-in 4-bit activation patch to the pinned Arduino TensorFlow Lite Micro submodule."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY = REPO_ROOT / "tools" / "arduino-user" / "libraries" / "Arduino_TensorFlowLite"
PATCH = REPO_ROOT / "env_setup" / "patches" / "arduino_tflite_int4_activations.patch"
SCRIPT = REPO_ROOT / "env_setup" / "apply_arduino_tflite_patch.sh"

needs_submodule = pytest.mark.skipif(
    not (LIBRARY / ".git").exists(), reason="Arduino_TensorFlowLite submodule is not initialized"
)


def test_patch_has_license_header_and_only_touches_library_sources() -> None:
    """The patch carries the repository header and edits only files under ``src``."""
    lines = PATCH.read_text().splitlines()
    assert lines[1] == "# SPDX-License-Identifier: BSD-3-Clause"
    changed = [line.split()[-1] for line in lines if line.startswith("diff ")]
    assert changed, "patch has no file diffs"
    assert all(path.startswith("b/src/") for path in changed)


@needs_submodule
def test_patch_applies_to_pristine_pinned_library(tmp_path: Path) -> None:
    """Applying to a clean export of the pinned commit succeeds, whatever state the working tree is in."""
    archive = tmp_path / "pinned.tar"
    subprocess.run(["git", "-C", str(LIBRARY), "archive", "--format=tar", "-o", str(archive), "HEAD", "src"], check=True)
    pristine = tmp_path / "pristine"
    pristine.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(pristine, filter="data")
    subprocess.run(["git", "apply", "--check", "-p1", str(PATCH)], cwd=pristine, check=True)
    subprocess.run(["git", "apply", "-p1", str(PATCH)], cwd=pristine, check=True)
    assert (pristine / "src" / "tensorflow" / "lite" / "micro" / "kernels" / "crest_act4.cpp").is_file()
    subprocess.run(["git", "apply", "--reverse", "--check", "-p1", str(PATCH)], cwd=pristine, check=True)


@needs_submodule
def test_script_reports_a_consistent_state() -> None:
    """The submodule is either stock or patched, never in conflict with the patch."""
    done = subprocess.run(["bash", str(SCRIPT), "status"], capture_output=True, text=True, check=True)
    assert done.stdout.strip() in {"unpatched", "patched"}
