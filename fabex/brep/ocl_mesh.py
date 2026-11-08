"""Fabex 'ocl_mesh.py'

Exact-tolerance meshes of B-rep objects for OpenCAMLib (drop-cutter and waterline).

A BRepKit object's display mesh is tessellated for looks (and coarser while editing).
For sampling, OpenCAMLib instead gets triangles generated from the exact surfaces and
verified against them to the operation's 'B-rep Mesh Tolerance' (o.brep.mesh_tolerance).
Results are cached per object by (B-rep data hash, world matrix, tolerance).
"""

from __future__ import annotations

import numpy as np

from ..exception import BrepError
from ..utilities.logging_utils import log
from . import bridge
from .brepcam.tessellate import cam_mesh

_CACHE_SIZE = 8
# session_uid -> (key, CamMesh), oldest first
_cache: dict[int, tuple[tuple, object]] = {}
# operation name -> {object name: report}, filled while an operation computes
_reports: dict[str, dict[str, dict]] = {}


def _key(obj, kernel, tolerance: float) -> tuple:
    M = np.round(bridge.world_matrix(obj), 12)
    return (bridge.body_hash(obj), M.tobytes(), float(tolerance), id(kernel))


def cam_mesh_for(obj, tolerance: float):
    """(CamMesh in world space, cached?) for a B-rep object. Raises BrepError."""
    kernel = bridge.require_kernel()
    key = _key(obj, kernel, tolerance)
    hit = _cache.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1], True
    body = bridge.read_body(obj)
    try:
        mesh = cam_mesh(kernel, body, bridge.world_matrix(obj), tolerance)
    except Exception as e:
        raise BrepError(
            "TESSELLATION_FAILED",
            f"Could not tessellate the B-rep of '{obj.name}': {e}",
            {"object": obj.name, "tolerance": tolerance},
        ) from e
    _cache.pop(obj.session_uid, None)
    _cache[obj.session_uid] = (key, mesh)
    while len(_cache) > _CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    return mesh, False


def clear_cache() -> None:
    _cache.clear()


def begin_report(operation) -> None:
    _reports[operation.name] = {}


def report(operation) -> list[dict]:
    """Which mesh OpenCAMLib used for each B-rep source in the operation's last compute."""
    return list(_reports.get(operation.name, {}).values())


def _warn(operation, text: str) -> None:
    log.warning(text)
    line = text + "\n"
    if line not in operation.info.warnings:
        operation.info.warnings += line


def _record(operation, obj, entry: dict) -> None:
    _reports.setdefault(operation.name, {})[obj.name] = {"object": obj.name, **entry}


def _has_active_modifiers(obj) -> bool:
    return any(m.show_viewport for m in obj.modifiers)


def exact_triangles(operation, obj):
    """World-space triangles (n, 3, 3) for OpenCAMLib, or None to use the display mesh.

    None (with a warning) when the object is a B-rep but the exact mesh can't be used.
    """
    if not bridge.is_brep(obj):
        return None
    settings = getattr(operation, "brep", None)
    if settings is None or not settings.use_exact_mesh:
        _record(operation, obj, {"mesh": "display", "reason": "exact B-rep mesh is off"})
        return None
    if operation.use_modifiers and _has_active_modifiers(obj):
        _warn(operation, f"{obj.name}: has modifiers - sampling its display mesh, not the B-rep")
        _record(operation, obj, {"mesh": "display", "reason": "object has modifiers"})
        return None
    try:
        mesh, cached = cam_mesh_for(obj, settings.mesh_tolerance)
    except BrepError as e:
        _warn(operation, f"{obj.name}: {e.message} - sampling its display mesh")
        _record(operation, obj, {"mesh": "display", "reason": e.code})
        return None
    for w in mesh.warnings:
        _warn(operation, f"{obj.name}: {w}")
    stats = mesh.stats()
    log.info(
        f"B-rep mesh for {obj.name}: {stats['triangles']} triangles, max deviation "
        f"{stats['max_deviation']} (tolerance {stats['tolerance']}), "
        f"{'cached' if cached else str(stats['seconds']) + ' s'}"
    )
    _record(
        operation,
        obj,
        {
            "mesh": "exact",
            "cached": cached,
            "triangles": stats["triangles"],
            "tolerance": stats["tolerance"],
            "max_deviation": stats["max_deviation"],
            "within_tolerance": stats["within_tolerance"],
            "seconds": stats["seconds"],
        },
    )
    return mesh.triangles
