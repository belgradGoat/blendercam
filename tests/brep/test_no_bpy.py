"""brepcam must stay importable without Blender."""

import re
import subprocess
import sys

from conftest import BREP_DIR

BREPCAM = BREP_DIR / "brepcam"
FORBIDDEN = re.compile(r"^\s*(import|from)\s+(bpy|mathutils|bmesh)\b", re.MULTILINE)


def test_no_blender_imports_in_source():
    offenders = [p.name for p in BREPCAM.glob("*.py") if FORBIDDEN.search(p.read_text())]
    assert offenders == []


def test_modules_import_without_bpy():
    modules = [p.stem for p in BREPCAM.glob("*.py") if p.stem != "__init__"]
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib\n"
        "for m in %r: importlib.import_module('brepcam.' + m)\n"
        "assert 'bpy' not in sys.modules, 'bpy got imported'\n"
        "print('OK')\n"
    ) % (str(BREP_DIR), modules)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout
