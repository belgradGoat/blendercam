"""Fabex 'brep/api.py' - the B-rep half of the scripting / agent API.

Functions here are registered in the same registry as fabex/api.py, so they show up in
describe() and can be called through call_json().
"""

import bpy

from ..exception import BrepError, FabexAPIError
from ..utilities.api_utils import api_function
from . import bridge, curves, fit, flow, ocl_mesh
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


@api_function("brep")
def fit_check(
    obj: str,
    op=None,
    cutter_type: str | None = None,
    diameter: float | None = None,
    corner_radius: float | None = None,
    skin: float | None = None,
    faces: list | None = None,
    wall_angle: float = 80.0,
    corner_angle: float = 10.0,
    all_faces: bool = False,
) -> dict:
    """Will a cutter reach everything on a B-rep object? From the exact surfaces, in world
    space (metres), for a tool coming from above (+Z). Nothing is changed.

    Always reports:
      min_concave_radius - the tightest curve towards the tool (fillets, holes, dishes):
        a ball (or a bull nose's corner, a ball cone's tip) must be no larger. radius
        None = no concave curvature (no limit). With the point, face id and face name.
      min_wall_radius - the tightest inside radius of the contour lines of walls steeper
        than wall_angle degrees (pocket corners, holes): limits flat-bottomed cutters.
      sharp_corners - concave edges whose faces meet at >= corner_angle degrees (no fillet;
        no round cutter reaches into them): kind "vertical" (inside corner in plan),
        "floor" (wall meets a level floor: a flat end mill reaches it) or "other".
      faces - per-face radii, concave and undercut area fractions (only faces with a
        limit unless all_faces=True).
    With a cutter - from `op` (name or index; its cutter type, diameter, corner / ball
    radius and skin), and/or cutter_type ("END" | "BALLNOSE" | "BULLNOSE" | "BALLCONE";
    other types are not checked), diameter, corner_radius, skin overriding it:
      fits (True / False / None = not checked), problems (each: check "concave_radius" or
      "wall_radius", face, radius found, radius needed, point, area fraction too tight),
      the material left in each sharp corner ("rest") and corner_rest_max, and the largest
      tip radius (max_tip_radius) / cutter radius (max_cutter_radius) that would fit, after
      the skin. messages: the warning lines Fabex shows.
    The analysis is cached per object; checking more cutters is instant."""
    ob = curves.brep_source_object(obj)
    o = None
    if op is not None:
        from ..api import _resolve  # late import: fabex.api imports this module

        o = _resolve(op)[1]
    cutter = fit.cutter_from(o, cutter_type, diameter, corner_radius)
    if skin is None:
        skin = fit.setting(o.skin) if o is not None else 0.0
    if skin < 0:
        raise FabexAPIError("INVALID_VALUE", "skin can't be negative", {"skin": skin})
    return fit.object_report(
        ob,
        cutter,
        float(skin),
        faces,
        float(wall_angle),
        float(corner_angle),
        bool(all_faces),
    )


@api_function("brep", mutates=True)
def flowline_preview(
    obj: str,
    faces: list | None = None,
    op=None,
    stepover: float | None = None,
    step_along: float | None = None,
    direction: str | None = None,
    order: str | None = None,
    max_slope: float | None = None,
    cutter_type: str | None = None,
    diameter: float | None = None,
    corner_radius: float | None = None,
    tolerance: float | None = None,
    show: str = "contact",
    name: str | None = None,
    create: bool = True,
    max_points: int | None = None,
) -> dict:
    """Flowline finishing passes over a B-rep object's faces, from the exact surfaces -
    shown as a curve object (one POLY spline per pass, in cutting order). World space,
    metres, tool from above (+Z).

    Passes follow each face's parameter lines (around a dome, along a fillet, across a
    blend); adjacent passes are never further apart than `stepover` (3D, on the surface),
    points are every `step_along` (3D) and passes end exactly on the face's trim (holes,
    outline) and on the slope limit.

    faces: B-rep face ids (see fit_check / brep_holes / the mesh's 'brep_face_id'
      attribute); default every face - faces steeper than max_slope or facing down get no
      passes (see each face's "note").
    op: an operation (name or index) to take the settings from - what a FLOWLINE operation
      computes: stepover = distance between paths, step_along = distance along paths,
      order and reverse from its movement settings (meander: ZIGZAG, else ONEWAY),
      brep.flow_faces / flow_direction / flow_max_slope / flow_reverse, its cutter and skin,
      brep.mesh_tolerance. Explicit arguments win.
    stepover: required without an op. step_along: default stepover / 2.
    direction: "AUTO" (default: rings around a pole, else along the longer direction),
      "U" or "V" - the face parameter the passes follow (each face's "direction" says
      which was used).
    order: "ZIGZAG" (default) | "ONEWAY". max_slope: degrees from level (default 80).
    cutter_type / diameter / corner_radius: the cutter; its round tip (ball, bull nose
      corner, ball cone tip) is placed on the contact points. Flat-bottomed and other
      cutters use the contact points' XY.
    tolerance: trim accuracy in metres (default 0.00001).
    show: "contact" (the points on the surface) or "tool" (where a round tip touching them
      has its lowest point). name: default "<obj>_flowline"; previewing again replaces it.
    create: False only computes the pattern (stats, nothing created).
    max_points: refuse patterns larger than this (default 4,000,000) - error TOO_LARGE
      with the estimate.
    Returns the preview object name, totals (passes, closed_passes, points, length, max_gap)
    and per face: direction, stations, passes, points, length, max_gap, closed, poles,
    steep_points / misfit_points (left out: too steep / tip too big for the curve), note.
    Patterns are cached per object and settings."""
    ob = curves.brep_source_object(obj)
    o = None
    if op is not None:
        from ..api import _resolve  # late import: fabex.api imports this module

        o = _resolve(op)[1]
    cutter = fit.cutter_from(o, cutter_type, diameter, corner_radius)
    params = flow.params_for(
        o,
        cutter,
        stepover=stepover,
        step_along=step_along,
        direction=direction,
        order=order,
        max_slope=max_slope,
        tolerance=tolerance,
        max_points=max_points,
    )
    if show not in flow.SHOW:
        raise FabexAPIError("INVALID_VALUE", f"show must be one of {flow.SHOW}", {"show": show})
    if faces is None and o is not None:
        faces = flow.parse_faces(o.brep.flow_faces)
    fl, cached = flow.pattern(ob, params, faces)
    preview = flow.preview_object(ob, fl, show, name, faces=faces) if create else None
    return {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        "cached": cached,
        "preview": preview.name if preview is not None else None,
        "show": show,
        "cutter": cutter.to_dict() if cutter is not None else None,
        **fl.stats(),
    }


@api_function("brep", mutates=True)
def create_flowline_operation(
    obj: str,
    faces: list | None = None,
    cutter_type: str = "BALLNOSE",
    diameter: float = 0.003,
    corner_radius: float | None = None,
    stepover: float = 0.00025,
    step_along: float = 0.0002,
    direction: str = "AUTO",
    max_slope: float = 80.0,
    name: str | None = None,
    props: dict | None = None,
) -> dict:
    """Create a FLOWLINE finishing operation on a B-rep object (see flowline_preview for
    what the passes do) - compute it with compute_operation / start_compute.

    faces: B-rep face ids to machine (default: every face the cutter reaches from above).
    cutter_type / diameter / corner_radius (BULLNOSE corner, BALLCONE ball radius): the
    cutter. stepover: max 3D distance between passes; step_along: between points along a
    pass. direction: "AUTO" | "U" | "V". max_slope: degrees from level. props: any other
    operation properties (as set_operation_props), applied last - e.g.
    {"optimisation.use_opencamlib": True, "movement.type": "CLIMB", "skin": 0.0001}.
    Z comes from Fabex's usual sampling (OpenCAMLib when available), so the passes can't
    go below the model. Layers are off (one finishing pass over the surface). The object has to
    lie below the depth start (max_z, default 0: Fabex expects the stock top at Z 0). Returns create_operation's result plus a dry run of the pattern
    (passes, points, max_gap and per face direction / passes / note)."""
    from math import radians

    from ..api import create_operation, set_operation_props  # late: fabex.api imports us

    ob = curves.brep_source_object(obj)
    if faces is not None:
        faces = flow.parse_faces(faces)
        unknown = sorted(set(faces) - set(bridge.read_body(ob).faces))
        if unknown:
            raise FabexAPIError("INVALID_VALUE", f"No faces {unknown}", {"faces": faces})
    kind = str(cutter_type).upper()
    if kind in ("BULLNOSE", "BALLCONE") and corner_radius is None:
        raise FabexAPIError("INVALID_VALUE", f"{kind} needs corner_radius", {"cutter_type": kind})
    if direction not in ("AUTO", "U", "V"):
        raise FabexAPIError("INVALID_VALUE", "direction must be AUTO, U or V", {})
    if not 0 < float(max_slope) <= 90:
        raise FabexAPIError("INVALID_VALUE", "max_slope must be in (0, 90] degrees", {})
    settings = {
        "cutter_type": kind,
        "cutter_diameter": float(diameter),
        "distance_between_paths": float(stepover),
        "distance_along_paths": float(step_along),
        "brep.flow_faces": "" if faces is None else ", ".join(map(str, faces)),
        "brep.flow_direction": direction,
        "brep.flow_max_slope": radians(float(max_slope)),
        "use_layers": False,  # finishing: one pass over the surface, not in Z slices
    }
    if corner_radius is not None:
        settings["bull_corner_radius" if kind == "BULLNOSE" else "ball_radius"] = float(
            corner_radius
        )
    created = create_operation(ob.name, strategy="FLOWLINE", name=name, props=settings)
    if props:
        set_operation_props(created["index"], props)
    o = bpy.context.scene.cam_operations[created["index"]]
    if curves.object_bbox(ob)["object_bbox_max"][2] > o.max_z + 1e-6:
        created["warnings"].append(flow.depth_start_warning(o, [ob.name]))
    dry = flowline_preview(ob.name, op=created["index"], create=False)
    created["pattern"] = {
        k: dry[k] for k in ("passes", "closed_passes", "points", "max_gap", "warnings")
    }
    created["pattern"]["faces"] = [
        {k: f[k] for k in ("face", "name", "direction", "passes", "note")} for f in dry["faces"]
    ]
    return created
