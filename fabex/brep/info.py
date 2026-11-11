"""Fabex 'brep/info.py'

What a B-rep object is made of, for planning CAM - the glue behind the API's brep_info():
face surfaces and roles, edge kinds and holes (brepcam/classify.py), the cutter limits (the
fit check), where the part sits relative to the stock top, and suggested next steps as
ready-to-run API calls.

The classification is cached per object by (B-rep data hash, world matrix, settings).
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from ..exception import FabexAPIError
from . import bridge, fit
from .brepcam import classify

_CACHE_SIZE = 8
_cache: dict[int, tuple[tuple, object]] = {}

# Common cutter diameters (m) the suggestions pick from
SIZES = (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.016, 0.02)
SCALLOP = 0.00001  # target scallop height of the suggested finishing passes (0.01 mm)
SKIN = 0.0005  # material the suggested roughing leaves for finishing
POINT_BUDGET = 2_000_000  # suggested flowline patterns stay under this many points
FINISH_ROLES = ("curved", "fillet", "round", "slope")
STEEP_NZ = float(np.cos(np.radians(80.0)))  # steeper everywhere: Flowline leaves it out


def classification(ob, tolerance: float = 1e-5, wall_angle: float = 80.0):
    """(Classification of every face in world space, cached?) for a B-rep object."""
    M = np.round(bridge.world_matrix(ob), 12)
    key = (bridge.body_hash(ob), M.tobytes(), float(tolerance), float(wall_angle))
    hit = _cache.get(ob.session_uid)
    if hit and hit[0] == key:
        return hit[1], True
    body = bridge.read_body(ob)
    try:
        result = classify.classify(body, bridge.world_matrix(ob), tolerance, None, wall_angle)
    except ValueError as e:
        raise FabexAPIError("INVALID_VALUE", str(e), {}) from e
    _cache.pop(ob.session_uid, None)
    _cache[ob.session_uid] = (key, result)
    while len(_cache) > _CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    return result, False


def clear_cache() -> None:
    _cache.clear()


def _size_at_most(limit: float | None, default: float = 0.006) -> float:
    """The largest common cutter diameter <= limit (the smallest one if none fits)."""
    if limit is None or not np.isfinite(limit):
        limit = default
    fitting = [d for d in SIZES if d <= limit * (1 + 1e-9)]
    return fitting[-1] if fitting else SIZES[0]


def _mm(x: float) -> str:
    return f"{x * 1000:g} mm"


def _r(x: float) -> float:
    """Round a length for a call string (0.1 um)."""
    return float(round(float(x), 7))


def _limits(ob) -> dict:
    fa, _ = fit.analysis(ob)
    conc, wall = fa.min_concave_radius(), fa.min_wall_radius()
    return {
        "min_concave_radius": conc,
        "min_wall_radius": wall,
        "sharp_corners": len(fa.corners),
        "max_ball_diameter": None if conc["radius"] is None else 2 * conc["radius"],
        "max_end_mill_diameter": None if wall["radius"] is None else 2 * wall["radius"],
    }


def _faces_limit(ob, face_ids) -> float | None:
    """Tightest concave radius on these faces (None: no limit)."""
    fa, _ = fit.analysis(ob)
    radii = [f.concave_radius for f in fa.faces if f.face in face_ids]
    radii = [r for r in radii if np.isfinite(r)]
    return min(radii) if radii else None


def _walls_limit(ob, face_ids) -> float | None:
    fa, _ = fit.analysis(ob)
    radii = [f.wall_radius for f in fa.faces if f.face in face_ids]
    radii = [r for r in radii if np.isfinite(r)]
    return min(radii) if radii else None


def suggestions(ob, cl, top_z: float) -> list[dict]:
    """A machining sequence as API calls: place the part, rough, drill, pocket, profile the
    walls of other openings, finish the curved faces with Flowline, cut it out. Each step:
    step, strategy, why, calls (Python for the Fabex API module, in order; run them as
    fx.<call>) and the numbers they were chosen from. Z values in calls assume the "place"
    step ran. A starting point, not a verified plan: check it with flowline_preview /
    fit_check and the computed paths."""
    name = ob.name
    shift = top_z if top_z > 1e-6 else 0.0  # the place step moves the part down by this
    info = {f.face: f for f in cl.faces}
    q = repr(name)
    size = np.subtract(cl.bbox_max, cl.bbox_min)
    xy = float(min(size[0], size[1]))
    steps = []
    if top_z > 1e-6:
        steps.append(
            {
                "step": "place",
                "why": f"The part's top is {_mm(top_z)} above Z 0, and Fabex only machines "
                "below the depth start (max_z, default 0: the stock top). Move the part down "
                "(or raise each operation's max_z).",
                "python": f"bpy.data.objects[{q}].location.z -= {_r(top_z)}",
            }
        )
    holes = [h for h in cl.holes if h["kind"] in ("through", "blind")]
    drills = [h for h in holes if h["circular"] and h["vertical"]]
    pockets = [h for h in holes if h["prismatic"] and h not in drills]
    walled = [
        h
        for h in holes
        if h not in drills
        and h not in pockets
        and h["vertical"]
        and h["walls_bottom"] is not None
        and all(
            w in info and max(abs(info[w].nz_min), abs(info[w].nz_max)) <= 1e-6 for w in h["walls"]
        )
    ]
    covered = {f for h in drills + pockets for f in h["faces"]}
    finish = [
        f.face
        for f in cl.faces
        if f.face not in covered
        and f.undercut_fraction < 1.0
        and f.nz_max >= STEEP_NZ
        and (f.role in FINISH_ROLES or (f.role == "floor" and f.hole is not None))
    ]

    if finish:
        d = _size_at_most(min(xy / 4, 0.012))
        steps.append(
            {
                "step": "rough",
                "strategy": "PARALLEL",
                "why": "Clear the stock above the curved faces in layers, leaving "
                f"{_mm(SKIN)} for finishing. (Parallel lines sit on a fixed grid: next to "
                "a wall up to a stepover more can be left - the wall and finishing steps "
                "take it off.)",
                "cutter_diameter": d,
                "calls": [
                    f"create_operation({q}, strategy='PARALLEL', name='{name} rough', "
                    f"props={{'cutter_type': 'END', 'cutter_diameter': {d}, "
                    f"'distance_between_paths': {_r(0.4 * d)}, 'skin': {SKIN}, "
                    f"'use_layers': True, 'stepdown': {_r(max(d / 2, 0.0005))}}})"
                ],
            }
        )

    by_size = {}
    for h in drills:
        by_size.setdefault(round(h["diameter"], 5), []).append(h["index"])
    for size_, idx in sorted(by_size.items()):
        label = f"{name}_drill_{size_ * 1000:g}mm"
        steps.append(
            {
                "step": "drill",
                "strategy": "DRILL",
                "why": f"{len(idx)} round vertical hole(s) of {_mm(size_)}, each drilled to "
                "its own depth from the hole curves (min_z_from OBJECT).",
                "holes": idx,
                "cutter_diameter": size_,
                "calls": [
                    f"extract_edges({q}, kind='holes', holes={idx}, project='bottom', "
                    f"name='{label}')",
                    f"create_operation('{label}', strategy='DRILL', name='{label}', "
                    f"props={{'cutter_diameter': {size_}, 'min_z_from': 'OBJECT', "
                    "'drill_type': 'MIDDLE_SYMETRIC'})",
                ],
            }
        )

    for h in pockets:
        wall_r = _walls_limit(ob, h["walls"])
        d = _size_at_most(min(2 * wall_r if wall_r else np.inf, min(h["size_xy"]) / 3))
        label = f"{name}_pocket_{h['index']}"
        steps.append(
            {
                "step": "pocket",
                "strategy": "POCKET",
                "why": f"Opening {h['index']} ({h['kind']}, "
                f"{' x '.join(_mm(x) for x in h['size_xy'])}, {_mm(h['depth'])} deep) has "
                "vertical walls and a level floor: cleared with a flat end mill "
                f"(tightest wall radius {_mm(wall_r) if wall_r else 'none: sharp corners'}).",
                "holes": [h["index"]],
                "cutter_diameter": d,
                "calls": [
                    f"extract_edges({q}, kind='holes', holes=[{h['index']}], project='bottom', "
                    f"name='{label}')",
                    f"create_operation('{label}', strategy='POCKET', name='{label}', "
                    f"props={{'cutter_diameter': {d}, 'min_z_from': 'OBJECT'}})",
                ],
            }
        )

    for h in walled:
        wall_r = _walls_limit(ob, h["faces"])
        d = _size_at_most(min(2 * wall_r * 0.95 if wall_r else np.inf, min(h["size_xy"]) / 3))
        z = _r(h["walls_bottom"] - shift)
        label = f"{name}_walls_{h['index']}"
        steps.append(
            {
                "step": "walls",
                "strategy": "CUTOUT",
                "why": f"Opening {h['index']} has vertical walls down to "
                f"{_mm(h['walls_bottom'] - shift)} (then fillets / a floor that Flowline "
                "finishes): profile them from the inside with a flat end mill, down to where "
                f"they end (tightest wall radius {_mm(wall_r) if wall_r else 'none'}).",
                "holes": [h["index"]],
                "cutter_diameter": d,
                "calls": [
                    f"extract_edges({q}, kind='holes', holes=[{h['index']}], project={z}, "
                    f"name='{label}')",
                    f"create_operation('{label}', strategy='CUTOUT', name='{label}', "
                    f"props={{'cut_type': 'INSIDE', 'cutter_type': 'END', "
                    f"'cutter_diameter': {d}, 'min_z_from': 'OBJECT'}})",
                ],
            }
        )

    if finish:
        limit = _faces_limit(ob, set(finish))
        d = _size_at_most(min(2 * limit * 0.95 if limit else np.inf, max(xy / 6, 0.001)))
        stepover = min(sqrt(8 * (d / 2) * SCALLOP), 0.2 * d)
        area = sum(f.area for f in cl.faces if f.face in finish)
        estimate = area / (stepover * stepover / 2)
        if estimate > POINT_BUDGET:
            stepover *= sqrt(estimate / POINT_BUDGET)
        stepover = _r(stepover)
        steps.append(
            {
                "step": "finish",
                "strategy": "FLOWLINE",
                "why": f"{len(finish)} curved / sloped face(s) and floors among them, finished "
                f"along their surfaces with a {_mm(d)} ball (tightest concave radius "
                f"{_mm(limit) if limit else 'none'}; stepover for a {_mm(SCALLOP)} scallop "
                "where it fits the point budget).",
                "faces": finish,
                "cutter_diameter": d,
                "stepover": stepover,
                "calls": [
                    f"create_flowline_operation({q}, faces={finish}, cutter_type='BALLNOSE', "
                    f"diameter={d}, stepover={stepover}, step_along={_r(stepover / 2)}, "
                    f"name='{name} finish')"
                ],
            }
        )

    outside = [f.face for f in cl.faces if f.role in ("wall", "round") and f.hole is None]
    if outside:
        d = _size_at_most(min(xy / 4, 0.006))
        steps.append(
            {
                "step": "cut_out",
                "strategy": "CUTOUT",
                "why": "Cut the part free along its outline (silhouette), down to its bottom.",
                "cutter_diameter": d,
                "calls": [
                    f"create_operation({q}, strategy='CUTOUT', name='{name} cut out', "
                    f"props={{'cut_type': 'OUTSIDE', 'cutter_type': 'END', "
                    f"'cutter_diameter': {d}}})"
                ],
            }
        )
    return steps


def object_info(
    ob,
    faces=None,
    tolerance: float = 1e-5,
    wall_angle: float = 80.0,
    edges: bool = False,
) -> dict:
    """brep_info() for one object (see the API docstring)."""
    if not tolerance or tolerance <= 0:
        raise FabexAPIError("INVALID_VALUE", "tolerance must be positive", {})
    cl, cached = classification(ob, float(tolerance), float(wall_angle))
    s = cl.summary()
    if faces is not None:
        wanted = {int(f) for f in faces}
        unknown = sorted(wanted - {f["face"] for f in s["faces"]})
        if unknown:
            raise FabexAPIError(
                "INVALID_VALUE", f"No faces {unknown}", {"faces": [f["face"] for f in s["faces"]]}
            )
        s["faces"] = [f for f in s["faces"] if f["face"] in wanted]
        s["edges"] = [e for e in s["edges"] if set(e["faces"]) & wanted]
    if not edges:
        s.pop("edges")
    limits = _limits(ob)
    top_z, bottom_z = cl.bbox_max[2], cl.bbox_min[2]
    warnings = list(s.pop("warnings"))
    if top_z > 1e-6:
        warnings.append(
            f"'{ob.name}' reaches {_mm(top_z)} above Z 0: Fabex only machines below the "
            "depth start (max_z, default 0 = the stock top) - move the part down by "
            f"{_r(top_z)} m (see suggestions) or raise max_z"
        )
    return {
        "object": ob.name,
        "body_hash": bridge.body_hash(ob),
        "cached": cached,
        **s,
        "placement": {
            "top_z": top_z,
            "bottom_z": bottom_z,
            "below_depth_start": top_z <= 1e-6,
        },
        "limits": limits,
        "suggestions": suggestions(ob, cl, top_z),
        "automatic": {
            "why": "Or let plan_cam choose the stock, tools, operations and cutting data for a "
            "machining centre (2.5D, top side), then apply_plan() to make, compute, write and "
            "verify the program. The suggestions above are separate operations to build by hand.",
            "calls": [f"plan_cam({ob.name!r}, material='6061')", "apply_plan()"],
        },
        "warnings": warnings,
    }
