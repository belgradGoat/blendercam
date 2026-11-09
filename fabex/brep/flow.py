"""Fabex 'brep/flow.py'

Flowline patterns for B-rep objects (see brepcam/flowline.py): settings from an operation
or from arguments, a small cache, the FLOWLINE strategy's pattern and preview curves.

The strategy (strategies/flowline.py) turns each pass into a Fabex path chunk at the tool
centres' XY; Fabex then drops the cutter there onto the whole model (OpenCAMLib, Bullet
or the depth image), so Z never comes from the pattern.

The preview is a curve object with one POLY spline per pass, in cutting order: the contact
points on the surface ("contact"), or where the tip of the cutter goes ("tool": a round tip
touching the contact points, before Fabex drops the cutter onto the whole model). It is
tagged like extracted edges (obj["fabex_brep_source"], kind "flowline_preview") and
replaced when previewed again under the same name.
"""

from __future__ import annotations

import re
from math import degrees
from types import SimpleNamespace

import bpy
import numpy as np

from ..chunk_builder import CamPathChunk
from ..exception import CamException, FabexAPIError
from ..utilities.operation_utils import get_move_and_spin
from . import bridge, curves, fit
from .brepcam import flowline

SHOW = ("contact", "tool")
_CACHE_SIZE = 2  # patterns can be big: ~80 bytes a point
_CACHE_MAX_POINTS = 1_000_000  # bigger patterns aren't kept
# session_uid -> (key, Flowline), oldest first
_cache: dict[int, tuple[tuple, object]] = {}
# operation name -> {object name: report}, from the operation's last pattern
_reports: dict[str, dict[str, dict]] = {}


def clear_cache() -> None:
    _cache.clear()


def order_of(o) -> str:
    """Fabex movement type -> pass order (meander: zigzag; climb / conventional: one way)."""
    return "ZIGZAG" if o.movement.type == "MEANDER" else "ONEWAY"


def reverse_of(o) -> bool:
    """Run the passes the other way? One-way passes are reversed for conventional milling
    with a CW spindle and climb milling with a CCW one (as Parallel does), then flipped
    again by brep.flow_reverse."""
    _, climb_ccw, conventional_cw, _ = get_move_and_spin(o)
    flip = o.movement.type != "MEANDER" and (conventional_cw or climb_ccw)
    return bool(flip) != bool(o.brep.flow_reverse)


def parse_faces(text) -> list | None:
    """'12, 15 31' -> [12, 15, 31]; empty -> None (every face)."""
    if text is None:
        return None
    if not isinstance(text, str):
        return sorted({int(f) for f in text})
    parts = [p for p in re.split(r"[\s,;]+", text.strip()) if p]
    if not parts:
        return None
    try:
        return sorted({int(p) for p in parts})
    except ValueError:
        raise FabexAPIError(
            "INVALID_VALUE",
            f"Faces must be B-rep face ids separated by commas, not '{text}'",
            {"faces": text},
        ) from None


def params_for(o=None, cutter=None, **given) -> flowline.FlowlineParams:
    """Flowline settings: the given values, else the operation's, else defaults.

    o: a Fabex operation - stepover = distance between paths, step along = distance along
    paths, order and reverse from the movement settings, brep.flow_direction /
    flow_max_slope / flow_reverse, tolerance = brep.mesh_tolerance, and its skin. cutter: a
    fitcheck.Cutter; its round tip (plus the skin) places the tool centres."""
    values = {k: v for k, v in given.items() if v is not None}
    skin = 0.0
    if o is not None:
        values.setdefault("stepover", o.distance_between_paths)
        values.setdefault("step_along", o.distance_along_paths)
        values.setdefault("order", order_of(o))
        values.setdefault("reverse", reverse_of(o))
        values.setdefault("direction", o.brep.flow_direction)
        values.setdefault("max_slope", round(degrees(o.brep.flow_max_slope), 6))
        values.setdefault("tolerance", o.brep.mesh_tolerance)
        skin = fit.setting(o.skin)
    if "stepover" not in values:
        raise FabexAPIError(
            "INVALID_VALUE", "Give a stepover (or an op to take it from)", {"stepover": None}
        )
    values.setdefault("step_along", float(values["stepover"]) / 2)
    if cutter is not None:
        values.setdefault("tip_radius", (cutter.tip_radius or 0.0) + skin)
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
        details = {"estimated_points": e.estimate, "max_points": e.limit}
        message = str(e)
        if e.estimate > e.limit:
            factor = round(1.05 * (e.estimate / e.limit) ** 0.5, 2)
            details["scale_steps_by"] = factor
            message += f" (stepover and step_along x {factor} would fit)"
        raise FabexAPIError("TOO_LARGE", message, details) from e
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


# --- the FLOWLINE strategy -------------------------------------------------------------------


def begin_report(operation) -> None:
    _reports[operation.name] = {}


def report(operation) -> list[dict]:
    """Per B-rep object: the pattern the operation was last computed with (compact)."""
    return list(_reports.get(operation.name, {}).values())


def _compact(ob, fl, cached: bool) -> dict:
    s = fl.stats()
    return {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        "cached": cached,
        **{k: s[k] for k in ("passes", "closed_passes", "points", "length", "max_gap")},
        "faces": [
            {k: f[k] for k in ("face", "name", "direction", "stations", "passes", "note")}
            for f in s["faces"]
        ],
        "params": s["params"],
        "seconds": s["seconds"],
        "warnings": s["warnings"],
    }


def depth_start_warning(o, names) -> str:
    return (
        f"{', '.join(names)} reaches above the operation's depth start "
        f"({o.max_z * 1000:g} mm): nothing above it is machined - move the object down "
        "(Fabex expects the stock top at Z 0) or raise Depth Start (max_z)"
    )


def operation_chunks(o) -> list:
    """The FLOWLINE pattern of an operation as Fabex path chunks at the tool centres' XY
    (z = 1, to be sampled), over each of its B-rep sources. Raises CamException /
    FabexAPIError with a message for the user when there is nothing to machine."""
    objects = bridge.operation_brep_objects(o)
    if not objects:
        raise CamException(
            "Flowline needs a BRepKit B-rep object (or a collection with some) as its source"
        )
    faces = parse_faces(o.brep.flow_faces)
    if faces is not None and len(objects) > 1:
        raise FabexAPIError(
            "INVALID_VALUE",
            "Flowline faces can only be chosen for a single B-rep object - clear Faces to "
            "machine every face of the collection",
            {"objects": [ob.name for ob in objects]},
        )
    params = params_for(o, fit.cutter_of(o))
    begin_report(o)
    chunks = []
    for ob in objects:
        fl, cached = pattern(ob, params, faces)
        rep = _compact(ob, fl, cached)
        _reports[o.name][ob.name] = rep
        for line in fl.warnings:
            fit.add_warning(o, f"{ob.name}: {line}")
        if not fl.passes:
            notes = sorted({f.note for f in fl.faces if f.note})
            fit.add_warning(o, f"{ob.name}: no flowline passes ({'; '.join(notes) or 'no faces'})")
        for p in fl.passes:
            xy = np.vstack([p.centers, p.centers[:1]]) if p.closed else p.centers
            chunks.append(CamPathChunk(np.column_stack([xy, np.ones(len(xy))])))
    above = [
        ob.name for ob in objects if curves.object_bbox(ob)["object_bbox_max"][2] > o.max_z + 1e-6
    ]
    if above:
        fit.add_warning(o, depth_start_warning(o, above))
    if o.geometry_source == "COLLECTION":
        col = bpy.data.collections.get(o.collection_name)
        others = [x.name for x in (col.objects if col else []) if not bridge.is_brep(x)]
        if others:
            fit.add_warning(o, f"No flowline passes on {', '.join(others)} (not B-rep objects)")
    if not chunks:
        raise CamException("Flowline: no passes - the faces are too steep, face down or empty")
    return chunks


def operation_preview(o, show: str = "contact") -> list:
    """Preview curves of an operation's flowline (its settings, faces and cutter) on each of
    its B-rep sources: [(object, preview object or None, Flowline)]."""
    objects = bridge.operation_brep_objects(o)
    if not objects:
        raise CamException("Flowline needs a BRepKit B-rep object as its source")
    faces = parse_faces(o.brep.flow_faces)
    if faces is not None and len(objects) > 1:
        raise FabexAPIError(
            "INVALID_VALUE",
            "Flowline faces can only be chosen for a single B-rep object",
            {"objects": [ob.name for ob in objects]},
        )
    params = params_for(o, fit.cutter_of(o))
    out = []
    for ob in objects:
        fl, _ = pattern(ob, params, faces)
        out.append((ob, preview_object(ob, fl, show, faces=faces), fl))
    return out
