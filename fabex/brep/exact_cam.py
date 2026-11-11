"""Fabex 'brep/exact_cam.py'

Exact sampling of BRepKit B-rep objects: the cutter dropped onto, and waterlines traced
around, the exact surfaces with the kernel's `brepkit.cam` instead of OpenCAMLib on a
mesh. No tessellation tolerance: heights are those of real contacts with the surfaces.

`ocl_utils.ocl_sample` and `waterline_utils.oclGetWaterline` ask here first. Exact
sampling applies when the operation's 'Exact B-rep Sampling' is on, the kernel has
`brepkit.cam` (DropCutter, and waterlines for the waterline strategy), the cutter is END,
BALLNOSE or BULLNOSE, and the source has B-rep objects (for waterlines: only B-rep
objects). Other sources of a drop-cutter operation are still sampled by OpenCAMLib, and
each point takes the higher of the two heights.

Stock to leave (skin) is exact: a cutter grown by the skin all round (radius R + s,
corner r + s; a flat end mill becomes a bull nose with corner radius s) touches the
surface where the cutter itself touches the surface offset by s, with its tip s lower.
"""

from __future__ import annotations

import time

import numpy as np

from . import bridge, ocl_mesh

CUTTERS = ("END", "BALLNOSE", "BULLNOSE")
_CACHE_SIZE = 4
_cache: dict[tuple, object] = {}       # (sources, cutter, kernel) -> prepared DropCutter


def _has_active_modifiers(obj) -> bool:
    return any(m.show_viewport for m in obj.modifiers)


def sources(o) -> tuple[list, list]:
    """(B-rep objects sampled exactly, the operation's other objects)."""
    exact, rest = [], []
    for ob in o.objects:
        usable = bridge.is_brep(ob) and not (o.use_modifiers and _has_active_modifiers(ob))
        (exact if usable else rest).append(ob)
    return exact, rest


def _kernel(o, waterline: bool = False):
    """The kernel when exact sampling applies to the operation's settings, else None."""
    settings = getattr(o, "brep", None)
    if settings is None or not getattr(settings, "exact_sampling", False):
        return None
    if o.cutter_type not in CUTTERS or o.skin < 0:
        return None
    kernel = bridge.find_kernel()
    if kernel is None or not kernel.has("cam", "DropCutter"):
        return None
    if waterline and not kernel.has("cam", "waterlines"):
        return None
    return kernel


def _cutter(kernel, o):
    """(the cutter grown by the skin, skin)."""
    cam = kernel.module("cam")
    corner = o.bull_corner_radius if o.cutter_type == "BULLNOSE" else None
    c = cam.Cutter.from_fabex(o.cutter_type, o.cutter_diameter, corner)
    s = float(o.skin)
    return cam.Cutter(c.radius + s, c.corner + s), s


def _body(kernel, objs):
    """The objects' bodies placed in world space, as one body."""
    edit = kernel.module("topology.edit")
    bodies = [edit.transform_body(bridge.read_body(ob), bridge.world_matrix(ob)) for ob in objs]
    if len(bodies) == 1:
        return bodies[0]
    if not hasattr(edit, "merge_bodies"):
        raise RuntimeError("this BRepKit cannot merge bodies (several B-rep sources)")
    return edit.merge_bodies(bodies)


def _key(kernel, objs, cutter) -> tuple:
    placed = tuple((bridge.body_hash(ob), np.round(bridge.world_matrix(ob), 12).tobytes())
                   for ob in objs)
    return placed, cutter.radius, cutter.corner, id(kernel)


def _drop_cutter(kernel, objs, cutter):
    """(prepared DropCutter, cached?)"""
    key = _key(kernel, objs, cutter)
    hit = _cache.get(key)
    if hit is not None:
        return hit, True
    dc = kernel.module("cam").DropCutter(_body(kernel, objs), cutter)
    _cache[key] = dc
    while len(_cache) > _CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    return dc, False


def clear_cache() -> None:
    _cache.clear()


def _record(o, objs, entry: dict) -> None:
    for ob in objs:
        ocl_mesh._record(o, ob, {"mesh": "surfaces", **entry})


def drop_heights(o, xy):
    """Exact tip heights (world z, -inf where the cutter misses) at the points xy (n, 2)
    over the operation's B-rep sources, and the sources left for OpenCAMLib; None when
    exact sampling does not apply (OpenCAMLib then samples everything)."""
    kernel = _kernel(o)
    if kernel is None:
        return None
    exact, rest = sources(o)
    if not exact:
        return None
    cutter, skin = _cutter(kernel, o)
    t0 = time.time()
    try:
        dc, cached = _drop_cutter(kernel, exact, cutter)
        z = dc(np.asarray(xy, dtype=float).reshape(-1, 2)).z + skin
    except Exception as e:                      # never worse than before: OpenCAMLib then
        ocl_mesh._warn(o, f"exact B-rep sampling failed ({e}) - sampling a mesh")
        return None
    _record(o, exact, {"strategy": "drop", "cached": cached, "points": int(len(z)),
                       "seconds": round(time.time() - t0, 3)})
    return z, rest


def waterline_loops(o, levels):
    """Exact waterline loops for each tip height in levels (lists of closed (k, 2) arrays,
    first point not repeated, running like OpenCAMLib's: clockwise around islands), or
    None when exact sampling does not apply (OpenCAMLib then): every source must be a
    B-rep object."""
    kernel = _kernel(o, waterline=True)
    if kernel is None:
        return None
    exact, rest = sources(o)
    if not exact or rest:
        return None
    cutter, skin = _cutter(kernel, o)
    t0 = time.time()
    try:
        dc, cached = _drop_cutter(kernel, exact, cutter)
        loops = kernel.module("cam").waterlines(dc, [float(z) - skin for z in levels], cutter)
    except Exception as e:
        ocl_mesh._warn(o, f"exact B-rep waterline failed ({e}) - using a mesh")
        return None
    _record(o, exact, {"strategy": "waterline", "cached": cached, "levels": len(levels),
                       "seconds": round(time.time() - t0, 3)})
    # OpenCAMLib's direction (clockwise around islands), which the strategy's climb /
    # conventional handling expects
    return [[L[::-1] for L in level] for level in loops]
