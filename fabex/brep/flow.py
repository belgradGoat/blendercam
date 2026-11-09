"""Fabex 'brep/flow.py'

Flowline patterns for B-rep objects (see brepcam/flowline.py): settings from an operation
or from arguments, a small cache, and preview curves.

The preview is a curve object with one POLY spline per pass, in cutting order: the contact
points on the surface ("contact"), or where the tip of the cutter goes ("tool": a round tip
touching the contact points, before Fabex drops the cutter onto the whole model). It is
tagged like extracted edges (obj["fabex_brep_source"], kind "flowline_preview") and
replaced when previewed again under the same name.
"""

from __future__ import annotations

from types import SimpleNamespace

import bpy
import numpy as np

from ..exception import FabexAPIError
from . import bridge, curves
from .brepcam import flowline

SHOW = ("contact", "tool")
_CACHE_SIZE = 2  # patterns can be big: ~80 bytes a point
_CACHE_MAX_POINTS = 1_000_000  # bigger patterns aren't kept
# session_uid -> (key, Flowline), oldest first
_cache: dict[int, tuple[tuple, object]] = {}


def clear_cache() -> None:
    _cache.clear()


def order_of(o) -> str:
    """Fabex movement type -> pass order (meander: zigzag; climb / conventional: one way)."""
    return "ZIGZAG" if o.movement.type == "MEANDER" else "ONEWAY"


def params_for(o=None, cutter=None, **given) -> flowline.FlowlineParams:
    """Flowline settings: the given values, else the operation's, else defaults.

    o: a Fabex operation (stepover = distance between paths, step along = distance along
    paths, order from the movement type, tolerance = brep.mesh_tolerance). cutter: a
    fitcheck.Cutter; its round tip places the tool centres."""
    values = {k: v for k, v in given.items() if v is not None}
    if o is not None:
        values.setdefault("stepover", o.distance_between_paths)
        values.setdefault("step_along", o.distance_along_paths)
        values.setdefault("order", order_of(o))
        values.setdefault("tolerance", o.brep.mesh_tolerance)
    if "stepover" not in values:
        raise FabexAPIError(
            "INVALID_VALUE", "Give a stepover (or an op to take it from)", {"stepover": None}
        )
    values.setdefault("step_along", float(values["stepover"]) / 2)
    if cutter is not None:
        values.setdefault("tip_radius", cutter.tip_radius or 0.0)
    try:
        return flowline.FlowlineParams(**values).validated()
    except (TypeError, ValueError) as e:
        raise FabexAPIError(
            "INVALID_VALUE",
            str(e),
            {"directions": list(flowline.DIRECTIONS), "orders": list(flowline.ORDERS)},
        ) from e


def pattern(ob, params: flowline.FlowlineParams, faces=None):
    """(Flowline in world space, cached?) for a B-rep object."""
    faces_key = None if faces is None else tuple(sorted({int(f) for f in faces}))
    kernel = bridge.require_kernel()
    M = bridge.world_matrix(ob)
    key = (
        bridge.body_hash(ob),
        np.round(M, 12).tobytes(),
        tuple(sorted(params.to_dict().items())),
        faces_key,
        id(kernel),
    )
    hit = _cache.get(ob.session_uid)
    if hit and hit[0] == key:
        return hit[1], True
    body = bridge.read_body(ob)
    try:
        result = flowline.flowline(kernel, body, params, M, faces_key)
    except flowline.PatternTooLarge as e:
        raise FabexAPIError(
            "TOO_LARGE",
            str(e),
            {"estimated_points": e.estimate, "max_points": e.limit},
        ) from e
    except (ValueError, KeyError) as e:
        raise FabexAPIError(
            "INVALID_VALUE",
            str(e.args[0]) if e.args else str(e),
            {"faces": sorted(body.faces) if isinstance(e, KeyError) else None},
        ) from e
    _cache.pop(ob.session_uid, None)
    if result.points <= _CACHE_MAX_POINTS:
        _cache[ob.session_uid] = (key, result)
    while len(_cache) > _CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    return result, False


def preview_object(
    ob, fl, show: str = "contact", name: str | None = None, replace: bool = True, faces=None
):
    """A curve object showing the passes (None when there are none). faces: the face ids
    the pattern was asked for (None: all), for the provenance."""
    if show not in SHOW:
        raise FabexAPIError("INVALID_VALUE", f"show must be one of {SHOW}", {"show": show})
    base = name or f"{ob.name}_flowline"
    tip = fl.params.tip_radius
    lines = [
        SimpleNamespace(points=p.points if show == "contact" else p.tool_tips(tip), closed=p.closed)
        for p in fl.passes
    ]
    old = bpy.data.objects.get(base)
    if not lines:
        if old is not None and replace and curves.source_info(old) is not None:
            data = old.data
            bpy.data.objects.remove(old)
            if data is not None and data.users == 0:
                bpy.data.curves.remove(data)
        return None
    tag = {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        "kind": "flowline_preview",
        "show": show,
        "group": base,
        "faces": None if faces is None else sorted({int(f) for f in faces}),
        "params": fl.params.to_dict(),
        "passes": len(fl.passes),
    }
    collection = ob.users_collection[0] if ob.users_collection else bpy.context.scene.collection
    out = curves.write_object(base, lines, tag, collection, replace)
    out.show_in_front = True  # the contact lines lie on the surface
    return out
