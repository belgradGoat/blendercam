"""Fabex 'brep/api.py' - the B-rep half of the scripting / agent API.

Functions here are registered in the same registry as fabex/api.py, so they show up in
describe() and can be called through call_json().
"""

import bpy

from ..exception import BrepError, FabexAPIError
from ..utilities.api_utils import api_function
from . import bridge, curves, ocl_mesh
from .brepcam import edges as brep_edges


@api_function("brep")
def brep_kernel(refresh: bool = False) -> dict:
    """Where BRepKit was found (extension / module / path) and its version.
    Use refresh=True after enabling or installing BRepKit."""
    if refresh:
        bridge.find_kernel(refresh=True)
    return bridge.kernel_status()


@api_function("brep")
def list_brep_objects() -> list[dict]:
    """All objects carrying BRepKit B-rep data, with face/edge counts when BRepKit is
    available (an object can still be machined as a plain mesh without it)."""
    kernel = bridge.find_kernel()
    out = []
    for ob in bpy.data.objects:
        if not bridge.is_brep(ob):
            continue
        entry = {
            "name": ob.name,
            "body_hash": bridge.body_hash(ob),
            "faces": None,
            "trimmed_faces": None,
            "edges": None,
            "error": None,
        }
        if kernel is not None:
            try:
                body = bridge.read_body(ob)
                entry["faces"] = len(body.faces)
                entry["trimmed_faces"] = sum(1 for f in body.faces.values() if f.loops)
                entry["edges"] = len(body.edges)
            except BrepError as e:
                entry["error"] = e.to_dict()
        out.append(entry)
    return out


@api_function("brep")
def cam_tessellation(obj: str, tolerance: float | None = None, op=None) -> dict:
    """Tessellate a B-rep object from its exact surfaces - the mesh OpenCAMLib samples.

    tolerance: max distance from the exact surfaces, in metres. Default: the operation's
    'brep.mesh_tolerance' when `op` (name or index) is given, else 0.00001 (0.01 mm).
    Returns triangle count, the measured max deviation, timing, and whether the result
    came from the cache (an unchanged object and tolerance reuse the previous mesh)."""
    ob = bpy.data.objects.get(obj)
    if ob is None:
        raise FabexAPIError("OBJECT_NOT_FOUND", f"No object named '{obj}'", {"object": obj})
    if not bridge.is_brep(ob):
        raise BrepError("NOT_A_BREP", f"Object '{obj}' has no B-rep data", {"object": obj})
    if tolerance is None:
        if op is not None:
            from ..api import _resolve  # late import: fabex.api imports this module

            tolerance = _resolve(op)[1].brep.mesh_tolerance
        else:
            tolerance = 0.00001
    if not tolerance or tolerance <= 0:
        raise FabexAPIError("INVALID_VALUE", "tolerance must be positive", {"tolerance": tolerance})
    mesh, cached = ocl_mesh.cam_mesh_for(ob, float(tolerance))
    return {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        "cached": cached,
        **mesh.stats(),
    }


@api_function("brep", mutates=True)
def extract_edges(
    obj: str,
    kind: str = "creases",
    faces: list | None = None,
    tolerance: float = 0.00001,
    project: str = "none",
    separate: bool = False,
    name: str | None = None,
    crease_angle: float = 30.0,
    replace: bool = True,
) -> dict:
    """Make curve objects from a B-rep object's exact edges, face loops or holes - sources
    for Profile, Pocket and Drill operations (POLY splines in world space).

    kind: "creases" (sharp edges, no tangent seams/fillet edges), "all", "naked" (open
      sheet boundaries), "shared", "outer_loops" (each face's boundary) or "holes" (inner
      face loops: drilled holes, pockets, slots; a through hole once, from its top end).
    faces: B-rep face ids (see brep_holes / the mesh's 'brep_face_id' attribute) - only
      edges touching them, or only their loops / holes.
    tolerance: max distance from the exact curves, in metres (default 0.01 mm).
    project: "none" keeps 3D curves; "top" / "bottom" flattens each curve to its highest /
      lowest Z (holes: the hole's top / bottom, so a DRILL operation with
      min_z_from="OBJECT" drills every hole to its own depth); a number flattens all
      curves to that Z (e.g. object_bbox_min[2], to profile down to the part's bottom with
      min_z_from="OBJECT"). Curves that collapse (vertical edges, walls) are dropped and
      coinciding ones kept once.
    separate: one object per curve (<name>_1, <name>_2, ...). name: default "<obj>_<kind>".
    replace: re-extracting under the same name replaces the curves made earlier, so
      operations keep pointing at them; objects not made here are never changed.
    Returns the created object names, counts, per-spline info, (kind "holes") the holes,
    and the B-rep object's world bounding box."""
    ob = curves.brep_source_object(obj)
    objects, es = curves.extract_curves(
        ob, kind, tolerance, faces, project, crease_angle, separate, name, replace
    )
    return {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        **curves.object_bbox(ob),
        "objects": [o.name for o in objects],
        **es.stats(),
        "spline_info": [p.summary() for p in es.polylines],
        "holes": es.holes,
    }


@api_function("brep")
def brep_holes(obj: str, faces: list | None = None, tolerance: float = 0.00001) -> dict:
    """Holes and other openings in a B-rep object, in world space - nothing is created.

    Each hole: kind ("through" | "blind" | "open"), circular, center (on the entry face),
    radius / diameter (circular holes), axis, vertical, depth, z_top, z_bottom, size_xy,
    the entry face / loop, wall faces and the other ends of through holes. Loops whose walls
    stand out of the face are listed under "bosses" instead. faces: only these faces' loops.
    Assumes outward normals (BRepKit solids are built that way)."""
    ob = curves.brep_source_object(obj)
    body = bridge.read_body(ob)
    if faces is not None:
        unknown = sorted({int(f) for f in faces} - set(body.faces))
        if unknown:
            raise FabexAPIError(
                "INVALID_VALUE", f"No faces {unknown}", {"faces": sorted(body.faces)}
            )
        faces = {int(f) for f in faces}
    if not tolerance or tolerance <= 0:
        raise FabexAPIError("INVALID_VALUE", "tolerance must be positive", {})
    found, warnings = brep_edges.find_holes(body, float(tolerance), bridge.world_matrix(ob), faces)
    found = [brep_edges.public_hole(h) for h in found]
    return {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        **curves.object_bbox(ob),
        "holes": [h for h in found if h["kind"] != "boss"],
        "bosses": [h for h in found if h["kind"] == "boss"],
        "warnings": warnings,
    }
