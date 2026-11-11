"""Make a plan real: objects, operations, a chain, paths, the program, the verification.

    apply_plan(plan, compute=True, export=True, verify=True, directory=None) -> dict

Idempotent: everything a plan made is tagged (objects: obj["fabex_autocam"] = the plan's
tag, operations: autocam.plan_tag) and replaced the next time a plan for the same object
is applied; nothing else is touched. The part object is moved to the plan's placement
(Z 0 at its top, X0 Y0 at the stock's centre or corner) - once: applying again leaves it.

Operation sources are made from the plan's own geometry: region curves (POLY splines,
islands as inner splines) and drill points (a mesh with a vertex per hole). Each operation
gets the plan's properties and the stock as DIMENSIONS.
"""

import json
import os
import tempfile

import bpy
import numpy as np
from mathutils import Vector

from ..brep import bridge
from ..exception import FabexAPIError
from ..utilities.simple_utils import safe_filename
from . import fanuc_export, partmodel, state
from .camcore import verify as cc_verify
from .camcore.planner import PLAN_VERSION

OBJ_PROP = "fabex_autocam"
PLAN_KEY = "fabex_autocam_plan"
PLACE_TOL = 1e-7


def stored_plan(scene) -> dict | None:
    raw = scene.get(PLAN_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def store_plan(scene, plan: dict) -> None:
    scene[PLAN_KEY] = json.dumps(plan)


def _invalid(message: str, details: dict | None = None):
    return FabexAPIError("PLAN_INVALID", message, details or {})


def check_plan(plan: dict, scene) -> dict:
    """The plan with its geometry (from the stored plan if the given one has none)."""
    if not isinstance(plan, dict):
        raise _invalid("A plan is a dict (from plan_cam)")
    missing = [k for k in ("version", "id", "tag", "object", "operations", "stock", "placement")
               if k not in plan]  # fmt: skip
    if missing:
        raise _invalid(f"The plan has no {missing}")
    if plan["version"] != PLAN_VERSION:
        raise _invalid(f"Plan version {plan['version']}, expected {PLAN_VERSION}: plan again")
    geometry = dict(plan.get("geometry") or {})
    stored = stored_plan(scene)
    if stored and stored.get("id") == plan["id"]:
        for k, v in (stored.get("geometry") or {}).items():
            geometry.setdefault(k, v)
    ids = set()
    for op in plan["operations"]:
        for k in ("id", "name", "strategy", "props", "tool"):
            if k not in op:
                raise _invalid(f"Operation {op.get('id', '?')} has no {k!r}")
        if op["id"] in ids:
            raise _invalid(f"Two operations are {op['id']}")
        ids.add(op["id"])
        if op["id"] not in geometry:
            raise _invalid(
                f"No geometry for {op['id']}: pass the plan from plan_cam (or one with the same "
                "id) - its geometry is kept in the scene",
                {"operation": op["id"]},
            )
    return {**plan, "geometry": geometry}


# --- placement --------------------------------------------------------------------------------


def _location(ob) -> np.ndarray:
    return np.array(ob.matrix_world.translation, dtype=float)


def place(ob, plan) -> dict:
    """Move the part to the plan's placement (once)."""
    pl = plan["placement"]
    before = np.asarray(pl.get("object_location", _location(ob)), float)
    after = before + np.asarray(pl["translation"], float)
    here = _location(ob)
    if np.allclose(here, after, atol=PLACE_TOL):
        return {"moved": False, "location": after.tolist()}
    if not np.allclose(here, before, atol=PLACE_TOL):
        raise FabexAPIError(
            "PLAN_STALE",
            f"'{ob.name}' moved since it was planned: plan again",
            {"planned_at": before.tolist(), "now": here.tolist()},
        )
    ob.matrix_world.translation = Vector(after.tolist())
    bpy.context.view_layer.update()
    return {"moved": True, "from": before.tolist(), "location": after.tolist()}


# --- objects ----------------------------------------------------------------------------------


def _collection(scene, name: str):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in scene.collection.children:
        try:
            scene.collection.children.link(coll)
        except RuntimeError:
            pass
    return coll


def _tagged(ob, tag) -> bool:
    return ob.get(OBJ_PROP) == tag


def remove_previous(scene, tag: str) -> dict:
    """Delete the operations, paths, chains and objects an earlier plan with `tag` made."""
    from .. import api as fx

    removed_ops = []
    for i in reversed(range(len(scene.cam_operations))):
        o = scene.cam_operations[i]
        if o.autocam.plan_tag == tag:
            removed_ops.append(o.name)
            fx.remove_operation(i, delete_path=True)
    removed_chains = []
    for i in reversed(range(len(scene.cam_chains))):
        if scene.cam_chains[i].get(OBJ_PROP) == tag:
            removed_chains.append(scene.cam_chains[i].name)
            scene.cam_chains.remove(i)
    if len(scene.cam_chains):
        scene.cam_active_chain = min(scene.cam_active_chain, len(scene.cam_chains) - 1)
    removed_objects = []
    for ob in list(bpy.data.objects):
        if _tagged(ob, tag):
            removed_objects.append(ob.name)
            data = ob.data
            bpy.data.objects.remove(ob)
            if data is not None and data.users == 0:
                (bpy.data.curves if isinstance(data, bpy.types.Curve) else bpy.data.meshes).remove(
                    data
                )
    return {"operations": removed_ops, "chains": removed_chains, "objects": removed_objects}


def _region_curve(name, geom, coll, tag):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    z = float(geom["z"])
    for poly in geom["polygons"]:
        for ring in poly:
            P = np.asarray(ring, float)
            if len(P) < 3:
                continue
            sp = cu.splines.new("POLY")
            sp.points.add(len(P) - 1)
            co = np.column_stack([P, np.full(len(P), z), np.ones(len(P))]).astype(np.float32)
            sp.points.foreach_set("co", co.ravel())
            sp.use_cyclic_u = True
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    ob[OBJ_PROP] = tag
    return ob


def _points_mesh(name, geom, coll, tag):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in geom["points"]], [], [])
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob[OBJ_PROP] = tag
    return ob


def _stock_box(name, stock, coll, tag):
    lo, hi = np.asarray(stock["min"], float), np.asarray(stock["max"], float)
    v = [(x, y, z) for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], f)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.display_type = "WIRE"
    ob.hide_render = True
    ob.hide_select = True
    ob[OBJ_PROP] = tag
    return ob


def _stock_props(stock) -> dict:
    lo, hi = stock["min"], stock["max"]
    return {
        "material.material_source": "DIMENSIONS",
        "material.origin": [lo[0], lo[1], hi[2]],
        "material.size": [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]],
    }


# --- verification -----------------------------------------------------------------------------


def tool_table(plan: dict) -> dict:
    """{T: shape (mm)} for verify from a plan's tools."""
    out = {}
    for t in plan.get("tools", []):
        out[int(t["number"])] = {
            "type": t["type"],
            "diameter": t["diameter"] * 1000,
            "corner_radius": t.get("corner_radius", 0.0) * 1000,
            "tip_angle": t.get("tip_angle", 0.0),
            "flute_length": t.get("flute_length", 0.0) * 1000,
            "reach": t.get("reach", 0.0) * 1000,
            "holder_diameter": t.get("holder_diameter", 0.0) * 1000,
        }
    return out


def verify_file(path, ob, stock, tools=None, fixtures=None, resolution=None, tolerance=0.02,
                part_bottom=None, mesh_tolerance=1e-5, plan=None) -> dict:  # fmt: skip
    """Verify a program file against a B-rep part (stock / fixtures in metres)."""
    with open(path, encoding="ascii", errors="replace") as f:
        text = f.read()
    tris, fids = partmodel.part_triangles(ob, mesh_tolerance)
    jaws = [
        {"min": [x * 1000 for x in j["min"]], "max": [x * 1000 for x in j["max"]],
         "name": j.get("name", f"jaw {i + 1}")}
        for i, j in enumerate(fixtures or [])
    ]  # fmt: skip
    rep = cc_verify.verify(
        text,
        tris * 1000,
        fids,
        [x * 1000 for x in stock["min"]],
        [x * 1000 for x in stock["max"]],
        part_bottom=None if part_bottom is None else part_bottom * 1000,
        tools=tools,
        fixtures=jaws,
        resolution=resolution,
        tolerance=tolerance,
        expected_faces=expected(plan)[0] if plan else None,
        expected_zones=expected(plan)[1] if plan else None,
    )
    rep["file"] = os.path.abspath(path)
    return rep


def expected(plan) -> tuple[dict, list]:
    """(expected_faces, expected_zones) for verify from a plan: the faces it leaves for
    later and the corners its tools can't reach (mm)."""
    faces = {u["face"]: u["reason"] for u in plan.get("coverage", {}).get("uncovered", [])}
    zones = [
        {
            "at": [x * 1000 for x in r["at"]],
            "radius": (r["radius_left"] + r["size"]) * 1000,
            "reason": "a sharp inside corner keeps the tool's radius (plan: rest)",
        }
        for r in plan.get("rest") or []
    ]
    return faces, zones


# --- apply ------------------------------------------------------------------------------------


def apply_plan(plan, compute=True, export=True, verify=True, directory=None,
               program_number=None, resolution=None) -> dict:  # fmt: skip
    from .. import api as fx
    from . import api as ac

    scene = bpy.context.scene
    plan = check_plan(plan, scene)
    ob = bpy.data.objects.get(plan["object"])
    if ob is None:
        raise FabexAPIError("OBJECT_NOT_FOUND", f"No object '{plan['object']}'", {})
    if plan.get("body_hash") and bridge.is_brep(ob) and bridge.body_hash(ob) != plan["body_hash"]:
        raise FabexAPIError("PLAN_STALE", f"'{ob.name}' changed since it was planned: plan again")
    placed = place(ob, plan)
    profile = state.machine(scene)
    m = scene.cam_machine
    if m.post_processor != "FANUC" or abs(m.feedrate_max - profile.feed_max) > 1e-6:
        state.set_machine(scene, profile)
    tag = plan["tag"]
    removed = remove_previous(scene, tag)
    coll = _collection(scene, f"AutoCAM {ob.name}")
    stock_ob = _stock_box(f"{ob.name} stock", plan["stock"], coll, tag)
    made = []
    stock = _stock_props(plan["stock"])
    for op in plan["operations"]:
        geom = plan["geometry"][op["id"]]
        src_name = f"{op['name']} source"
        if geom["kind"] == "points":
            src = _points_mesh(src_name, geom, coll, tag)
        else:
            src = _region_curve(src_name, geom, coll, tag)
        props = {**op["props"], **stock, "autocam.plan_tag": tag}
        res = fx.create_operation(src.name, strategy=op["strategy"], name=op["name"], props=props)
        made.append(res["op"])
    chain_name = f"{ob.name} OP1"
    ac.create_chain(chain_name, made, filename=f"{ob.name} OP1")
    chain = next(c for c in scene.cam_chains if c.name == chain_name)
    chain[OBJ_PROP] = tag
    result = {
        "plan": plan["id"],
        "object": ob.name,
        "placement": placed,
        "removed": removed,
        "stock_object": stock_ob.name,
        "operations": made,
        "chain": chain_name,
        "computed": [],
        "program": None,
        "verify": None,
        "ok": True,
    }
    applied = {k: v for k, v in plan.items()}
    applied["applied"] = {"chain": chain_name, "operations": made}
    store_plan(scene, applied)
    if not compute:
        return result
    for name in made:
        r = fx.compute_operation(name, export=False)
        result["computed"].append(
            {
                "operation": name,
                "ok": r.get("ok"),
                "seconds": r.get("duration_s"),
                "warnings": r.get("warnings", []),
                **({"error": r["error"]} if not r.get("ok") else {}),
            }
        )
        if not r.get("ok"):
            result["ok"] = False
            return result
    if not export:
        return result
    directory = directory or fx._export_dir() or tempfile.gettempdir()
    number = program_number or plan.get("options", {}).get("program_number")
    report = ac.export_program(chain=chain_name, directory=directory, program_number=number)
    result["program"] = {
        k: report.get(k)
        for k in ("file", "bytes", "lines", "minutes", "tools", "operations", "warnings")
    }
    result["program"]["check"] = {
        "ok": report["check"]["ok"],
        "errors": report["check"]["errors"],
        "warnings": report["check"]["warnings"],
    }
    result["ok"] = result["ok"] and report["check"]["ok"]
    applied["applied"]["program"] = report["file"]
    store_plan(scene, applied)
    if verify and bridge.is_brep(ob):
        rep = verify_file(
            report["file"],
            ob,
            plan["stock"],
            tools=tool_table(plan),
            fixtures=plan.get("fixture", {}).get("jaws"),
            resolution=resolution,
            part_bottom=ob_bottom(plan),
            plan=plan,
        )
        result["verify"] = rep
        result["ok"] = result["ok"] and rep["ok"]
        applied["verify"] = {
            k: rep[k] for k in ("ok", "gouge", "leftover", "collisions", "plunges")
        }
        store_plan(scene, applied)
    return result


def ob_bottom(plan) -> float | None:
    z_end = plan.get("z_end")
    through = plan.get("options", {}).get("through", 0.0)
    return None if z_end is None else z_end + through


def program_path(directory, title) -> str:
    return os.path.join(directory, safe_filename(title) + ".nc")
