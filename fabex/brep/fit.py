"""Fabex 'brep/fit.py'

Cutter fit check for BRepKit B-rep objects (see brepcam/fitcheck.py): the tightest concave
radius, the walls' inside corner radius and the sharp inside corners, compared with an
operation's cutter. The geometry analysis is cached per object by (B-rep data hash, world
matrix, settings); checking another cutter against it is instant.

Surface operations on B-rep objects run the check before computing (o.brep.check_fit) and
get warnings when the cutter doesn't fit; the results are kept per operation for the
panel and the API.
"""

from __future__ import annotations

import bmesh
import bpy
import numpy as np

from ..exception import BrepError, FabexAPIError
from ..utilities.logging_utils import log
from ..utilities.simple_utils import unit_value_to_string
from . import bridge
from .brepcam import fitcheck

_CACHE_SIZE = 8
# session_uid -> (key, FitAnalysis), oldest first
_cache: dict[int, tuple[tuple, object]] = {}
# operation name -> {object name: report}, from the last check of that operation
_reports: dict[str, dict[str, dict]] = {}

# strategies that machine the object's surfaces with the cutter's tip and sides
SURFACE_STRATEGIES = (
    "PARALLEL",
    "CROSS",
    "BLOCK",
    "SPIRAL",
    "CIRCLES",
    "OUTLINEFILL",
    "PENCIL",
    "CRAZY",
    "WATERLINE",
    "FLOWLINE",
)


def analysis(obj, faces=None, wall_angle: float = 80.0, corner_angle: float = 10.0):
    """(FitAnalysis in world space, cached?) for a B-rep object. Raises BrepError /
    FabexAPIError(INVALID_VALUE)."""
    faces_key = None if faces is None else tuple(sorted({int(f) for f in faces}))
    M = np.round(bridge.world_matrix(obj), 12)
    key = (bridge.body_hash(obj), M.tobytes(), faces_key, float(wall_angle), float(corner_angle))
    hit = _cache.get(obj.session_uid)
    if hit and hit[0] == key:
        return hit[1], True
    body = bridge.read_body(obj)
    try:
        result = fitcheck.analyze(
            body, bridge.world_matrix(obj), faces_key, wall_angle, corner_angle
        )
    except (ValueError, KeyError) as e:
        raise FabexAPIError(
            "INVALID_VALUE",
            str(e.args[0]) if e.args else str(e),
            {"faces": sorted(body.faces) if isinstance(e, KeyError) else None},
        ) from e
    _cache.pop(obj.session_uid, None)
    _cache[obj.session_uid] = (key, result)
    while len(_cache) > _CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    return result, False


def clear_cache() -> None:
    _cache.clear()


def setting(x: float) -> float:
    """A float32 property value as typed (0.008, not 0.00800000038)."""
    return float(str(np.float32(x)))


def cutter_of(o) -> fitcheck.Cutter:
    """The fit-check cutter of a Fabex operation."""
    corner = None
    if o.cutter_type == "BULLNOSE":
        corner = setting(o.bull_corner_radius)
    elif o.cutter_type == "BALLCONE":
        corner = setting(o.ball_radius)
    return fitcheck.Cutter(o.cutter_type, setting(o.cutter_diameter), corner)


def cutter_from(o=None, cutter_type=None, diameter=None, corner_radius=None):
    """The cutter of operation `o` (or None), with explicit arguments overriding it.
    Raises FabexAPIError(INVALID_VALUE) for an incomplete or impossible cutter."""
    cutter = cutter_of(o) if o is not None else None
    if cutter_type is None and diameter is None and corner_radius is None:
        return cutter
    kind = (cutter_type or (cutter.type if cutter else "")).upper()
    dia = diameter if diameter is not None else (cutter.diameter if cutter else None)
    if not kind or dia is None:
        raise FabexAPIError(
            "INVALID_VALUE",
            "Give cutter_type and diameter (or an op to take the cutter from)",
            {"cutter_types": list(fitcheck.CHECKED)},
        )
    if not dia > 0:
        raise FabexAPIError("INVALID_VALUE", "diameter must be positive", {"diameter": dia})
    corner = corner_radius
    if corner is None and cutter is not None and cutter.type == kind:
        corner = cutter.corner_radius
    if kind in ("BULLNOSE", "BALLCONE") and corner is None:
        raise FabexAPIError("INVALID_VALUE", f"{kind} needs corner_radius", {"cutter_type": kind})
    return fitcheck.Cutter(kind, float(dia), None if corner is None else float(corner))


def _len(x) -> str:
    return unit_value_to_string(x, 3).strip()


def tool_text(cutter: fitcheck.Cutter) -> str:
    return f"{_len(cutter.diameter)} {cutter.type.lower()}"


def messages(name: str, rep: dict) -> list[str]:
    """Short warning lines for a report from object_report (empty when all is well)."""
    if not rep.get("cutter") or not rep.get("checked"):
        return []
    cutter = _cutter(rep)
    out = []
    for check, what, limit, advice in (
        ("concave_radius", "concave radius", "max_tip_radius", "tip radius"),
        ("wall_radius", "wall corner radius", "max_cutter_radius", "cutter radius"),
    ):
        found = [p for p in rep["problems"] if p["check"] == check]
        if not found:
            continue
        worst = found[0]
        faces = len({p["face"] for p in found})
        where = worst["face_name"] or f"face {worst['face']}"
        more = f" +{faces - 1} faces" if faces > 1 else ""
        line = (
            f"{name}: {tool_text(cutter)} doesn't fit a {_len(worst['radius'])} {what} "
            f"({where}{more})"
        )
        if rep.get(limit) is not None:
            line += f" - max {advice} {_len(rep[limit])}"
        out.append(line)
    rest = rep.get("corner_rest_max")
    if rest:
        n = sum(1 for c in rep["sharp_corners"] if c.get("rest"))
        out.append(f"{name}: {n} sharp inside corner(s) - up to {_len(rest)} is left in them")
    return out


def summary_text(rep: dict) -> str:
    """One line: whether the cutter fits, and the object's limits."""
    parts = []
    if rep.get("fits") is True:
        parts.append(f"{tool_text(_cutter(rep))} fits")
    elif rep.get("cutter") and rep.get("checked") is False:
        parts.append(f"{rep['cutter']['type'].lower()} cutters aren't checked")
    conc, wall = rep["min_concave_radius"]["radius"], rep["min_wall_radius"]["radius"]
    parts.append(f"concave R {_len(conc)}" if conc is not None else "no concave curves")
    if wall is not None:
        parts.append(f"wall corners R {_len(wall)}")
    n = len(rep.get("sharp_corners", ()))
    if n:
        parts.append(f"{n} sharp inside corner(s)")
    return ", ".join(parts)


def _cutter(rep: dict) -> fitcheck.Cutter:
    c = rep["cutter"]
    return fitcheck.Cutter(c["type"], c["diameter"], c["corner_radius"])


def object_report(
    obj,
    cutter: fitcheck.Cutter | None = None,
    skin: float = 0.0,
    faces=None,
    wall_angle: float = 80.0,
    corner_angle: float = 10.0,
    all_faces: bool = False,
) -> dict:
    """Fit-check report for one B-rep object: radii, sharp corners, limited faces, and (with
    a cutter) fits / problems / corner rest / advice and warning lines. JSON-able."""
    fa, cached = analysis(obj, faces, wall_angle, corner_angle)
    s = fa.summary()
    if not all_faces:
        s["faces"] = [
            f for f in s["faces"] if f["concave_radius"] is not None or f["wall_radius"] is not None
        ]
    rep = {
        "object": obj.name,
        "body_hash": bridge.body_hash(obj),
        "cached": cached,
        **s,
        "cutter": None,
        "skin": None,
        "checked": None,
        "fits": None,
        "problems": [],
        "corner_rest_max": None,
        "max_tip_radius": None,
        "max_cutter_radius": None,
    }
    if cutter is not None:
        rep.update(fa.check(cutter, skin))
    rep["messages"] = messages(obj.name, rep)
    return rep


# --- operations -----------------------------------------------------------------------------


def begin_report(operation) -> None:
    _reports[operation.name] = {}


def report(operation, compact: bool = False) -> list[dict]:
    """Fit-check reports from the operation's last check (compute or the operator).
    compact: without per-face lists and corner details (for compute results)."""
    reps = list(_reports.get(operation.name, {}).values())
    return [_compact(r) for r in reps] if compact else reps


_COMPACT = (
    "object",
    "body_hash",
    "error",
    "cutter",
    "skin",
    "checked",
    "fits",
    "min_concave_radius",
    "min_wall_radius",
    "corner_rest_max",
    "max_tip_radius",
    "max_cutter_radius",
    "messages",
    "seconds",
    "cached",
)


def _compact(rep: dict) -> dict:
    out = {k: rep[k] for k in _COMPACT if k in rep}
    if "problems" in rep:
        out["problems"] = rep["problems"][:5]
        out["problem_count"] = len(rep["problems"])
    if "sharp_corners" in rep:
        out["sharp_corner_count"] = len(rep["sharp_corners"])
    return out


def _warn(operation, text: str) -> None:
    log.warning(text)
    line = text + "\n"
    if line not in operation.info.warnings:
        operation.info.warnings += line


def applies(operation) -> bool:
    """Is the fit check run when computing this operation?"""
    settings = getattr(operation, "brep", None)
    return (
        settings is not None
        and settings.check_fit
        and operation.strategy in SURFACE_STRATEGIES
        and bool(bridge.operation_brep_objects(operation))
    )


def check_operation(operation, warn: bool = True) -> list[dict]:
    """Check the operation's cutter against each of its B-rep sources; keep the reports
    and (warn=True) add warning lines to operation.info.warnings. Never raises."""
    begin_report(operation)
    cutter = cutter_of(operation)
    out = []
    for ob in bridge.operation_brep_objects(operation):
        try:
            rep = object_report(ob, cutter, setting(operation.skin))
        except (BrepError, FabexAPIError) as e:
            rep = {"object": ob.name, "error": e.to_dict(), "messages": []}
            if warn:
                _warn(operation, f"{ob.name}: cutter fit not checked - {e.message}")
        except Exception as e:  # never stop a compute over the check
            log.exception("Fit check failed")
            rep = {"object": ob.name, "error": {"code": "FAILED", "message": str(e)}}
            rep["messages"] = []
        else:
            log.info(
                f"Fit check {ob.name}: fits={rep['fits']} concave "
                f"{rep['min_concave_radius']['radius']} wall {rep['min_wall_radius']['radius']} "
                f"corners {len(rep['sharp_corners'])} ({rep['seconds']} s, cached {rep['cached']})"
            )
            if warn:
                for line in rep["messages"]:
                    _warn(operation, line)
        _reports.setdefault(operation.name, {})[ob.name] = rep
        out.append(rep)
    return out


def check_before_compute(operation) -> None:
    """Hook for toolpath.get_path: run the check when it applies, else clear old reports."""
    if applies(operation):
        check_operation(operation)
    else:
        begin_report(operation)


def report_is_current(operation, rep: dict) -> bool:
    """Was the report made with the operation's current cutter and skin?"""
    c = rep.get("cutter")
    if not c:
        return False
    now = cutter_of(operation)
    return (
        c["type"] == now.type
        and c["diameter"] == now.diameter
        and (c["corner_radius"] or 0.0) == (now.corner_radius or 0.0)
        and (rep.get("skin") or 0.0) == setting(operation.skin)
    )


def select_faces(obj, face_ids) -> int:
    """In Edit Mode, select the polygons of these B-rep faces (others deselected).
    Returns the number of polygons selected (0 when not in Edit Mode)."""
    if obj.mode != "EDIT":
        return 0
    bm = bmesh.from_edit_mesh(obj.data)
    layer = bm.faces.layers.int.get(bridge.FACE_ID_ATTR)
    if layer is None:
        return 0
    ids = {int(f) for f in face_ids}
    for elems in (bm.verts, bm.edges, bm.faces):
        for x in elems:
            x.select = False
    count = 0
    for f in bm.faces:
        if f[layer] in ids:
            f.select_set(True)  # with its edges and verts; no flush (it would add faces
            count += 1  # whose corners all happen to be selected)
    bm.select_mode = {"FACE"}
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    return count
