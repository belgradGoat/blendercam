"""camcore must stay importable without Blender."""

import re
import subprocess
import sys
from pathlib import Path

AUTOCAM = Path(__file__).resolve().parents[2] / "fabex" / "autocam"
CAMCORE = AUTOCAM / "camcore"
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(bpy|mathutils|bmesh)\b", re.MULTILINE)


def test_no_blender_imports_in_source():
    offenders = [p.name for p in CAMCORE.glob("*.py") if FORBIDDEN.search(p.read_text())]
    assert offenders == []


def test_modules_import_without_bpy():
    modules = [p.stem for p in CAMCORE.glob("*.py") if p.stem != "__init__"]
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib\n"
        "for m in %r: importlib.import_module('camcore.' + m)\n"
        "assert 'bpy' not in sys.modules, 'bpy got imported'\n"
        "print('OK')\n"
    ) % (str(AUTOCAM), modules)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout
