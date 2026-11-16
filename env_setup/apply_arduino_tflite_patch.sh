#!/usr/bin/env bash
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
# Applies, reverts, or reports the 4-bit activation patch on the pinned Arduino TensorFlow Lite Micro submodule.
# Usage: env_setup/apply_arduino_tflite_patch.sh [apply|revert|status]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$PROJECT_ROOT/tools/arduino-user/libraries/Arduino_TensorFlowLite"
PATCH="$PROJECT_ROOT/env_setup/patches/arduino_tflite_int4_activations.patch"
ACTION="${1:-apply}"

if [[ ! -e "$LIB/.git" ]]; then
  echo "Submodule missing: run 'git submodule update --init' first." >&2
  exit 1
fi

# Prints unpatched, patched, or conflict, by asking git whether the patch applies forward or in reverse.
patch_state() {
  if git -C "$LIB" apply --check -p1 "$PATCH" 2>/dev/null; then
    echo unpatched
  elif git -C "$LIB" apply --check --reverse -p1 "$PATCH" 2>/dev/null; then
    echo patched
  else
    echo conflict
  fi
}

STATE="$(patch_state)"
case "$ACTION" in
  status)
    echo "$STATE"
    ;;
  apply)
    case "$STATE" in
      unpatched)
        git -C "$LIB" apply -p1 "$PATCH"
        echo "Applied the 4-bit activation patch to $LIB."
        ;;
      patched) echo "Already applied; nothing to do." ;;
      *)
        echo "The submodule has other local changes that conflict with the patch." >&2
        echo "Restore it with: git -C $LIB checkout . && git -C $LIB clean -fd src" >&2
        exit 1
        ;;
    esac
    ;;
  revert)
    case "$STATE" in
      patched)
        git -C "$LIB" apply --reverse -p1 "$PATCH"
        echo "Reverted the 4-bit activation patch."
        ;;
      unpatched) echo "Not applied; nothing to do." ;;
      *)
        echo "The submodule has other local changes, so the patch cannot be reversed cleanly." >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Usage: $0 [apply|revert|status]" >&2
    exit 2
    ;;
esac
