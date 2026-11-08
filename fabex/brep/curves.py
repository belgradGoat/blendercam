"""Fabex 'brep/curves.py'

B-rep edges, face loops and holes as Blender curve objects - POLY splines in world space,
sampled within a tolerance of the exact curves (see brepcam/edges.py). Made with bpy.data
only, so it works the same from the operator, scripts, the API and background mode.

Each curve object records where it came from in obj["fabex_brep_source"] (JSON): the
B-rep object, its body hash at extraction time, the settings and per-spline provenance.
Re-extracting with the same name replaces the curves of objects made here (operations
keep pointing at them); objects not made here are never touched.
"""

import json

import bpy
import numpy as np
from mathutils import Matrix

from ..exception import BrepError, FabexAPIError
from . import bridge
from .brepcam import edges as brep_edges

SOURCE_PROP = "fabex_brep_source"


def source_info(ob) -> dict | None:
    """The provenance stored on a curve made by extract_curves (None for other objects)."""
    raw = ob.get(SOURCE_PROP) if ob is not None else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def brep_source_object(name: str):
    """The B-rep object by name, with API errors for missing / non B-rep objects."""
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise FabexAPIError("OBJECT_NOT_FOUND", f"No object named '{name}'", {"object": name})
    if not bridge.is_brep(ob):
        raise BrepError("NOT_A_BREP", f"Object '{name}' has no B-rep data", {"object": name})
    return ob


def object_bbox(ob) -> dict:
    """World-space bounding box of an object's display mesh (e.g. the part's bottom Z)."""
    me = ob.data
    if not len(me.vertices):
        return {"object_bbox_min": None, "object_bbox_max": None}
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    M = np.array(ob.matrix_world)
    P = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    return {"object_bbox_min": P.min(0).tolist(), "object_bbox_max": P.max(0).tolist()}


def edge_set(ob, kind="all", tolerance=0.00001, faces=None, project="none", crease_angle=30.0):
    """brepcam EdgeSet for a B-rep object, in world space. Bad settings -> INVALID_VALUE."""
    body = bridge.read_body(ob)
    try:
        return brep_edges.extract(
            body,
            kind,
            float(tolerance),
            bridge.world_matrix(ob),
            None if faces is None else list(faces),
            project,
            float(crease_angle),
        )
    except (ValueError, KeyError) as e:
        raise FabexAPIError(
            "INVALID_VALUE",
            str(e.args[0]) if e.args else str(e),
            {"kinds": list(brep_edges.KINDS), "project": list(brep_edges.PROJECT)},
        ) from e


def _curve_data(name: str, polylines) -> bpy.types.Curve:
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    for pl in polylines:
        P = np.asarray(pl.points, dtype=float)
        sp = cu.splines.new("POLY")
        sp.points.add(len(P) - 1)
        co = np.hstack([P, np.ones((len(P), 1))]).astype(np.float32).ravel()
        sp.points.foreach_set("co", co)
        sp.use_cyclic_u = bool(pl.closed)
    return cu


def _spline_meta(pl) -> dict:
    meta = {"edges": [int(e) for e in pl.edges], "faces": [int(f) for f in pl.faces]}
    if pl.loop is not None:
        meta["loop"] = [int(pl.loop[0]), int(pl.loop[1])]
    if pl.hole is not None:
        meta["hole"] = {
            k: pl.hole[k] for k in ("kind", "center", "diameter", "depth", "z_top", "z_bottom")
        }
    return meta


def _write_object(name: str, polylines, tag: dict, collection, replace: bool):
    cu = _curve_data(name, polylines)
    ob = bpy.data.objects.get(name)
    if ob is not None and replace and ob.type == "CURVE" and source_info(ob) is not None:
        old = ob.data
        ob.data = cu
        if old.users == 0:
            bpy.data.curves.remove(old)
    else:
        ob = bpy.data.objects.new(name, cu)  # a name in use by another object gets a suffix
        collection.objects.link(ob)
    ob.matrix_world = Matrix.Identity(4)  # points are in world space
    ob[SOURCE_PROP] = json.dumps(tag)
    return ob


def extract_curves(
    ob,
    kind: str = "all",
    tolerance: float = 0.00001,
    faces=None,
    project: str = "none",
    crease_angle: float = 30.0,
    separate: bool = False,
    name: str | None = None,
    replace: bool = True,
):
    """Make curve object(s) from a B-rep object's edges / loops / holes.

    Returns (objects, edge_set). One object with a spline per polyline, or one object
    per polyline with separate=True (named <name>_1, <name>_2, ...). Nothing is created
    when nothing matches."""
    es = edge_set(ob, kind, tolerance, faces, project, crease_angle)
    base = name or f"{ob.name}_{kind}"
    collection = ob.users_collection[0] if ob.users_collection else bpy.context.scene.collection
    common = {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        "kind": kind,
        "project": project if isinstance(project, str) else float(project),
        "tolerance": float(tolerance),
        "faces": None if faces is None else sorted(int(f) for f in faces),
        "group": base,
    }
    objects = []
    if es.polylines:
        if separate:
            for i, pl in enumerate(es.polylines):
                tag = {**common, "index": i, "splines": [_spline_meta(pl)]}
                objects.append(_write_object(f"{base}_{i + 1}", [pl], tag, collection, replace))
        else:
            tag = {**common, "index": None, "splines": [_spline_meta(p) for p in es.polylines]}
            objects.append(_write_object(base, es.polylines, tag, collection, replace))
    if replace:
        # curves left over from an earlier extraction into the same group (made here only)
        keep = {o.name for o in objects}
        for other in list(bpy.data.objects):
            info = source_info(other)
            if (
                other.name not in keep
                and other.type == "CURVE"
                and info is not None
                and info.get("group") == base
                and info.get("object") == ob.name
            ):
                data = other.data
                bpy.data.objects.remove(other)
                if data.users == 0:
                    bpy.data.curves.remove(data)
    return objects, es
