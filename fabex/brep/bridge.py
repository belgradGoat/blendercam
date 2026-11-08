"""Fabex 'bridge.py'

Locating the BRepKit kernel and reading B-rep data from Blender objects.

A BRepKit object is an ordinary MESH object (an exact-normal tessellation) with the
B-rep stored as JSON in obj["brepkit_data"]. Fabex can always machine it as a mesh;
the exact B-rep is only needed for B-rep specific features, and for those the
kernel is looked up in this order:

1. an enabled BRepKit extension (bl_ext.<repo>.brepkit) - uses its cached reader
2. an importable top-level `brepkit` package
3. an installed but disabled BRepKit extension, loaded from disk
4. a checkout path in the FABEX_BREPKIT_PATH environment variable
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import re
from pathlib import Path

import bpy
import numpy as np

from ..exception import BrepError
from ..utilities.logging_utils import log
from .brepcam.kernel import Kernel, KernelError, load_kernel_from_path

DATA_PROP = "brepkit_data"
FACE_ID_ATTR = "brep_face_id"  # BRepKit's per-polygon face id on the display mesh
EXTENSION_ID = "brepkit"
ENV_PATH = "FABEX_BREPKIT_PATH"

# (addon module name, Kernel, sync module or None) for an enabled extension
_extension: tuple[str, Kernel, object] | None = None
# Kernel found by any other route
_fallback: Kernel | None = None
_last_error: str = ""
# session_uid -> (data hash, kernel, Body) for bodies read without the extension's cache
_body_cache: dict[int, tuple[str, Kernel, object]] = {}


# --- object level (no kernel needed) --------------------------------------------------------


def is_brep(obj) -> bool:
    """True if `obj` carries BRepKit B-rep data."""
    return obj is not None and obj.type == "MESH" and bool(obj.get(DATA_PROP))


def data_hash(text: str) -> str:
    """Same hash BRepKit uses (sync.data_hash), so results can be compared."""
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def body_hash(obj) -> str:
    """Hash of the object's B-rep data ("" for non B-rep objects)."""
    return data_hash(obj.get(DATA_PROP, "")) if is_brep(obj) else ""


def world_matrix(obj) -> np.ndarray:
    """The object's world matrix as a (4, 4) numpy array. B-rep data is in local space."""
    return np.array(obj.matrix_world, dtype=float)


def brep_objects(objects) -> list:
    """The B-rep objects among `objects` (e.g. an operation's `o.objects`)."""
    return [ob for ob in (objects or []) if is_brep(ob)]


def operation_brep_objects(o) -> list:
    """B-rep objects among an operation's geometry sources (without touching o.objects)."""
    if o.geometry_source == "OBJECT":
        ob = bpy.data.objects.get(o.object_name)
        return brep_objects([ob] if ob else [])
    if o.geometry_source == "COLLECTION":
        col = bpy.data.collections.get(o.collection_name)
        return brep_objects(col.objects if col else [])
    return []


# --- kernel lookup --------------------------------------------------------------------------


def _enabled_extension_module() -> str | None:
    for name in bpy.context.preferences.addons.keys():
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "bl_ext" and parts[2] == EXTENSION_ID:
            return name
    return None


def _installed_extension_dirs():
    """Directories of installed (possibly disabled) BRepKit extensions."""
    pattern = re.compile(r'^\s*id\s*=\s*"' + EXTENSION_ID + r'"\s*$', re.MULTILINE)
    for repo in bpy.context.preferences.extensions.repos:
        if not repo.enabled or not repo.directory:
            continue
        root = Path(repo.directory)
        if not root.is_dir():
            continue
        for child in root.iterdir():
            manifest = child / "blender_manifest.toml"
            try:
                if manifest.is_file() and pattern.search(manifest.read_text()):
                    yield child
            except OSError:
                continue


def _find_fallback() -> Kernel | None:
    global _last_error
    errors = []

    # 2. importable top-level package
    if importlib.util.find_spec("brepkit") is not None:
        try:
            return Kernel(importlib.import_module("brepkit"), "module")
        except Exception as e:
            errors.append(f"brepkit module: {e}")

    # 3. installed but disabled extension, then 4. env var checkout
    candidates = [d / "brepkit" for d in _installed_extension_dirs()]
    if os.environ.get(ENV_PATH):
        candidates.append(Path(os.environ[ENV_PATH]))
    for path in candidates:
        try:
            return Kernel(load_kernel_from_path(path), "path")
        except (KernelError, Exception) as e:
            errors.append(f"{path}: {e}")

    _last_error = "; ".join(errors) if errors else "BRepKit is not installed"
    return None


def find_kernel(refresh: bool = False) -> Kernel | None:
    """The BRepKit kernel, or None if BRepKit can't be found."""
    global _extension, _fallback, _last_error

    ext = _enabled_extension_module()
    if ext:
        if refresh or _extension is None or _extension[0] != ext:
            try:
                package = importlib.import_module(ext + ".brepkit")
                try:
                    sync = importlib.import_module(ext + ".sync")
                except ImportError:
                    sync = None
                _extension = (ext, Kernel(package, "extension"), sync)
                _last_error = ""
            except Exception as e:
                log.warning(f"BRepKit extension {ext} found but its kernel failed to load: {e}")
                _extension = None
        if _extension is not None:
            return _extension[1]
    else:
        _extension = None

    if refresh or _fallback is None:
        _fallback = _find_fallback()
    return _fallback


def require_kernel() -> Kernel:
    kernel = find_kernel()
    if kernel is None:
        raise BrepError(
            "BREPKIT_UNAVAILABLE",
            "BRepKit not found: enable the BRepKit extension, or set "
            f"{ENV_PATH} to a BRepKit checkout",
            {"reason": _last_error},
        )
    return kernel


def kernel_status() -> dict:
    kernel = find_kernel()
    return {
        "available": kernel is not None,
        "kernel": kernel.info() if kernel else None,
        "uses_extension_cache": bool(_extension and _extension[1] is kernel and _extension[2]),
        "error": None if kernel else _last_error,
    }


# --- reading bodies -------------------------------------------------------------------------


def read_body(obj):
    """The object's B-rep Body (shared cached instance - treat as read-only)."""
    if not is_brep(obj):
        raise BrepError("NOT_A_BREP", f"Object '{getattr(obj, 'name', obj)}' has no B-rep data", {})
    kernel = require_kernel()

    sync = _extension[2] if (_extension and _extension[1] is kernel) else None
    if sync is not None and hasattr(sync, "read_body"):
        return sync.read_body(obj)

    text = obj.get(DATA_PROP, "")
    h = data_hash(text)
    cached = _body_cache.get(obj.session_uid)
    if cached and cached[0] == h and cached[1] is kernel:
        return cached[2]
    try:
        body = kernel.loads_body(text)
    except Exception as e:
        raise BrepError(
            "BREP_PARSE_FAILED",
            f"Could not read the B-rep data of '{obj.name}': {e}",
            {"object": obj.name, "kernel": kernel.info()},
        ) from e
    _body_cache[obj.session_uid] = (h, kernel, body)
    return body
