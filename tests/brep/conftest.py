"""Pure-Python tests for fabex/brep/brepcam (no Blender needed).

Run with BRepKit's test venv (Blender's Python + numpy + pytest):

    FABEX_BREPKIT_PATH=~/Documents/Blender \
        ~/Documents/Blender/.venv/bin/python -m pytest tests/brep -q

`brepcam` is imported as a top-level package (fabex/brep on sys.path), so nothing that
imports bpy is ever loaded. Tests needing the kernel skip if FABEX_BREPKIT_PATH is unset.
"""

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BREP_DIR = REPO / "fabex" / "brep"
sys.path.insert(0, str(BREP_DIR))

BREPKIT_PATH = os.environ.get("FABEX_BREPKIT_PATH")


@pytest.fixture(scope="session")
def brepkit_path():
    if not BREPKIT_PATH:
        pytest.skip("Set FABEX_BREPKIT_PATH to a BRepKit checkout")
    return Path(BREPKIT_PATH).expanduser()


@pytest.fixture(scope="session")
def kernel(brepkit_path):
    from brepcam.kernel import Kernel, load_kernel_from_path

    return Kernel(load_kernel_from_path(brepkit_path), "path")
