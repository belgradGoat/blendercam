"""Pure-Python tests for fabex/autocam/camcore (no Blender needed).

Run with any Python that has numpy and pytest (e.g. BRepKit's test venv):

    ~/Documents/Blender/.venv/bin/python -m pytest tests/autocam -q

`camcore` is imported as a top-level package (fabex/autocam on sys.path), so nothing that
imports bpy is ever loaded. FABEX_UPDATE_REFERENCE=1 rewrites the golden programs.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "fabex" / "autocam"))
