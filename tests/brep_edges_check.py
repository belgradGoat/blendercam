"""Headless check of B-rep edges to curves (milestone M3).

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P brep_edges_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

A 100 x 60 x 10 mm B-rep plate (two through holes, one blind hole), moved and turned,
goes through the agent flow: brep_holes -> extract_edges -> Drill / Profile / Pocket
operations that compute. Drill depths come from the hole curves (min_z_from OBJECT).
Also checks re-extraction (same objects, no duplicates), the operator in Object and Edit
Mode, provenance and error codes. Prints PASS/FAIL lines and a SUMMARY.
"""

import importlib
import json
import math
import os
import sys

import addon_utils
import bpy
import numpy as np

FABEX = "bl_ext.user_default.fabex"
BREPKIT = "bl_ext.user_default.brepkit"
use_extension = os.environ.get("FABEX_TEST_BREPKIT_EXTENSION") == "1"

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  ({info})" if info else ""))


addon_utils.enable(FABEX, default_set=True)
if use_extension:
    addon_utils.enable(BREPKIT, default_set=True)
else:
    addon_utils.disable(BREPKIT, default_set=True)

fx = importlib.import_module(FABEX + ".api")
bridge = importlib.import_module(FABEX + ".brep.bridge")
curves = importlib.import_module(FABEX + ".brep.curves")
kernel = bridge.require_kernel()
check("kernel source", kernel.source == ("extension" if use_extension else "path"), kernel.source)

# --- the plate: local z -5..5 mm; placed so its top is at Z = 0, turned 90 degrees --------
solids = kernel.module("ops.solids")
boolean = kernel.module("ops.boolean").boolean
HOLES = [  # local centre, radius, cylinder z0, height  (the last one is 5 mm deep)
    ((0.02, 0.0), 0.004, -0.01, 0.03),
    ((-0.03, -0.01), 0.003, -0.01, 0.03),
    ((-0.02, 0.015), 0.0025, 0.0, 0.02),
]


def plate_body(holes):
    body = solids.box((0, 0, 0), (0.1, 0.06, 0.01))
    for c, r, z0, h in holes:
        body, _ = boolean(body, solids.cylinder((*c, z0), (0, 0, 1), r, h), "difference")
    return body


body = plate_body(HOLES)
TessOptions = kernel.module("tessellate.grid").TessOptions
t = kernel.module("tessellate.body").tessellate_body(body, TessOptions())
mesh = bpy.data.meshes.new("Plate")
mesh.from_pydata([tuple(p) for p in t.points], [], [list(map(int, p)) for p in t.polygons()])
attr = mesh.attributes.new("brep_face_id", "INT", "FACE")
attr.data.foreach_set("value", np.asarray(t.face_ids, dtype=np.int32))
ob = bpy.data.objects.new("Plate", mesh)
bpy.context.scene.collection.objects.link(ob)
ob["brepkit_data"] = kernel.dumps_body(body)
ob.location = (0.2, 0.1, -0.005)
ob.rotation_euler = (0, 0, math.pi / 2)
bpy.context.view_layer.update()
M = np.array(ob.matrix_world)


def world(xy, z_local):
    return (M @ np.array([xy[0], xy[1], z_local, 1.0]))[:3]


EXPECTED = [  # world centre xy, diameter, bottom Z
    (world(c, 0)[:2], 2 * r, -0.005 if z0 == 0.0 else -0.01) for c, r, z0, _ in HOLES
]

# --- brep_holes: what an agent sees first -----------------------------------------------------
info = fx.brep_holes("Plate")
holes = info["holes"]
check("three holes, no bosses", len(holes) == 3 and info["bosses"] == [], len(holes))
ok = True
for xy, d, zb in EXPECTED:
    h = min(holes, key=lambda h: np.linalg.norm(np.array(h["center"][:2]) - xy))
    ok &= np.allclose(h["center"][:2], xy, atol=1e-5) and abs(h["diameter"] - d) < 1e-5
    ok &= abs(h["z_bottom"] - zb) < 1e-9 and abs(h["z_top"]) < 1e-9 and h["vertical"]
check("hole centres, diameters and depths in world space", ok, holes)
check(
    "through / blind",
    sorted(h["kind"] for h in holes) == ["blind", "through", "through"],
    [h["kind"] for h in holes],
)
top_face = holes[0]["face"]
check(
    "object bounding box (the part's bottom for profiles)",
    np.allclose(info["object_bbox_min"], [0.17, 0.05, -0.01], atol=1e-6)
    and np.allclose(info["object_bbox_max"], [0.23, 0.15, 0.0], atol=1e-6),
    (info["object_bbox_min"], info["object_bbox_max"]),
)

# --- holes -> Drill, each hole to its own depth -----------------------------------------------
r = fx.extract_edges("Plate", kind="holes", project="bottom")
check("extract holes", r["objects"] == ["Plate_holes"] and r["splines"] == 3, r["objects"])
cu = bpy.data.objects["Plate_holes"]
check(
    "curve: 3 closed POLY splines, identity transform",
    cu.type == "CURVE"
    and len(cu.data.splines) == 3
    and all(s.use_cyclic_u and s.type == "POLY" for s in cu.data.splines)
    and np.allclose(np.array(cu.matrix_world), np.eye(4)),
)
tag = curves.source_info(cu)
check(
    "provenance recorded",
    tag["object"] == "Plate"
    and tag["body_hash"] == bridge.body_hash(ob)
    and tag["kind"] == "holes",
    {k: tag[k] for k in ("object", "kind", "project", "group")},
)

res = fx.create_operation(
    "Plate_holes",
    strategy="DRILL",
    name="Drill",
    props={"cutter_diameter": 0.003, "min_z_from": "OBJECT", "drill_type": "MIDDLE_SYMETRIC"},
)
check("create drill operation", res["valid"], res)
rd = fx.compute_operation("Drill", export=False)
check("drill computes", rd["ok"], rd["message"])


def path_points(op_name):
    o = next(o for o in bpy.context.scene.cam_operations if o.name == op_name)
    path = bpy.data.objects[o.path_object_name]
    M = np.array(path.matrix_world)
    return np.array([v.co for v in path.data.vertices]) @ M[:3, :3].T + M[:3, 3]


def cutting_bounds(op_name):
    """XY bounds of the moves below the top of the stock (travel moves excluded)."""
    P = path_points(op_name)
    P = P[P[:, 2] < -1e-6]
    return P.min(0), P.max(0)


P = path_points("Drill")
ok = True
for xy, d, zb in EXPECTED:
    near = np.linalg.norm(P[:, :2] - xy, axis=1) < 1e-5
    ok &= near.any() and abs(P[near, 2].min() - zb) < 1e-6
check("drill plunges at every hole centre to its own depth", ok)

# --- top face outline -> Profile (cutout) through the plate ---------------------------------
z_bottom = info["object_bbox_min"][2]
r = fx.extract_edges(
    "Plate", kind="outer_loops", faces=[top_face], project=z_bottom, name="Plate_outline"
)
check(
    "extract the top face's outline at the part's bottom",
    r["splines"] == 1 and r["closed"] == 1 and abs(r["spline_info"][0]["z_max"] - z_bottom) < 1e-7,
    r["spline_info"],
)
res = fx.create_operation(
    "Plate_outline",
    strategy="CUTOUT",
    name="Profile",
    props={"cut_type": "OUTSIDE", "cutter_diameter": 0.006, "min_z_from": "OBJECT"},
)
rp = fx.compute_operation("Profile", export=False)
check("profile computes", rp["ok"], rp["message"])
lo, hi = cutting_bounds("Profile")
# turned 90 degrees: 60 x 100 mm in world XY, plus the cutter radius all round
check(
    "profile goes round the plate at the cutter radius",
    np.allclose(lo[:2], [0.2 - 0.033, 0.1 - 0.053], atol=2e-4)
    and np.allclose(hi[:2], [0.2 + 0.033, 0.1 + 0.053], atol=2e-4),
    (lo.round(4).tolist(), hi.round(4).tolist()),
)
check("profile cuts down to the plate bottom", abs(lo[2] + 0.01) < 1e-6, lo[2])

# --- one hole on its own -> Pocket ----------------------------------------------------------
r = fx.extract_edges("Plate", kind="holes", separate=True, project="bottom", name="Hole")
check("separate objects", r["objects"] == ["Hole_1", "Hole_2", "Hole_3"], r["objects"])
big = max(r["holes"], key=lambda h: h["diameter"])
i_big = r["holes"].index(big)
fx.create_operation(
    r["objects"][i_big],
    strategy="POCKET",
    name="Pocket",
    props={"cutter_diameter": 0.003, "min_z_from": "OBJECT"},
)
rk = fx.compute_operation("Pocket", export=False)
check("pocket computes", rk["ok"], rk["message"])
Pk = path_points("Pocket")
Pk = Pk[Pk[:, 2] < -1e-6]
c = np.array(big["center"][:2])
reach = np.linalg.norm(Pk[:, :2] - c, axis=1).max()
check(
    "pocket stays inside the hole, at the cutter radius",
    abs(reach - (big["diameter"] / 2 - 0.0015)) < 2e-5 and abs(Pk[:, 2].min() + 0.01) < 1e-6,
    (reach, Pk[:, 2].min()),
)

# --- re-extraction replaces, never duplicates -----------------------------------------------
n_objects = len(bpy.data.objects)
r = fx.extract_edges("Plate", kind="holes", project="bottom")
check(
    "re-extracting keeps the same object",
    r["objects"] == ["Plate_holes"] and len(bpy.data.objects) == n_objects,
    (r["objects"], len(bpy.data.objects), n_objects),
)
ob["brepkit_data"] = kernel.dumps_body(plate_body(HOLES[:2]))  # the blind hole is gone
r = fx.extract_edges("Plate", kind="holes", project="bottom")
check(
    "edited B-rep -> updated curve, same object",
    r["objects"] == ["Plate_holes"] and len(bpy.data.objects["Plate_holes"].data.splines) == 2,
)
check(
    "drill still computes on the updated curve", fx.compute_operation("Drill", export=False)["ok"]
)
ob["brepkit_data"] = kernel.dumps_body(body)

r = fx.extract_edges("Plate", kind="holes", project="bottom", name="Hole")
check(
    "joined extraction removes the earlier separate curves (made here only)",
    r["objects"] == ["Hole"]
    and not any(n in bpy.data.objects for n in ("Hole_1", "Hole_2", "Hole_3")),
    r["objects"],
)
other = bpy.data.objects.new("Taken", bpy.data.curves.new("Taken", "CURVE"))
bpy.context.scene.collection.objects.link(other)
r = fx.extract_edges("Plate", kind="creases", name="Taken")
check(
    "an existing object of the user's is never replaced",
    r["objects"] and r["objects"][0] != "Taken" and len(other.data.splines) == 0,
    r["objects"],
)

# --- operator: Object Mode and Edit Mode (selected faces) -----------------------------------
for o in bpy.context.selected_objects:
    o.select_set(False)
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
ret = bpy.ops.object.cam_brep_extract_edges(kind="CREASES", project="TOP")
act = bpy.context.view_layer.objects.active
check(
    "operator (Object Mode): sharp edges flattened to the top",
    ret == {"FINISHED"} and act.name == "Plate_creases" and act.select_get(),
    (ret, act.name),
)
# sharp edges seen from above: the outline and the three hole rims
check(
    "4 closed curves", len(act.data.splines) == 4 and all(s.use_cyclic_u for s in act.data.splines)
)

bottom_face = next(f for f, x in bridge.read_body(ob).faces.items() if x.name == "Box.-Z")
ids = np.zeros(len(mesh.polygons), dtype=np.int32)
mesh.attributes["brep_face_id"].data.foreach_get("value", ids)
sel = ids == bottom_face
mesh.polygons.foreach_set("select", sel)
corner_v = np.zeros(len(mesh.loops), dtype=np.int32)
mesh.loops.foreach_get("vertex_index", corner_v)
starts = np.zeros(len(mesh.polygons), dtype=np.int32)
mesh.polygons.foreach_get("loop_start", starts)
sizes = np.zeros(len(mesh.polygons), dtype=np.int32)
mesh.polygons.foreach_get("loop_total", sizes)
vsel = np.zeros(len(mesh.vertices), dtype=bool)
vsel[corner_v[np.repeat(sel, sizes)]] = True
mesh.vertices.foreach_set("select", vsel)
ev = np.zeros(len(mesh.edges) * 2, dtype=np.int32)
mesh.edges.foreach_get("vertices", ev)
mesh.edges.foreach_set("select", vsel[ev.reshape(-1, 2)].all(axis=1))
bpy.context.tool_settings.mesh_select_mode = (False, False, True)
bpy.context.view_layer.objects.active = ob
bpy.ops.object.mode_set(mode="EDIT")
ret = bpy.ops.object.cam_brep_extract_edges(kind="HOLES", project="NONE")
bpy.ops.object.mode_set(mode="OBJECT")
tag = curves.source_info(bpy.data.objects.get("Plate_holes"))
check(
    "operator (Edit Mode): only the selected bottom face's holes",
    ret == {"FINISHED"} and tag["faces"] == [bottom_face] and len(tag["splines"]) == 2,
    (ret, tag and tag["faces"], tag and len(tag["splines"])),
)

# --- errors ---------------------------------------------------------------------------------
cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
bpy.context.scene.collection.objects.link(cube)
for call, code in (
    (dict(name="extract_edges", obj="Nope"), "OBJECT_NOT_FOUND"),
    (dict(name="extract_edges", obj="Cube"), "NOT_A_BREP"),
    (dict(name="extract_edges", obj="Plate", kind="spirals"), "INVALID_VALUE"),
    (dict(name="extract_edges", obj="Plate", faces=[999]), "INVALID_VALUE"),
    (dict(name="extract_edges", obj="Plate", tolerance=0), "INVALID_VALUE"),
    (dict(name="brep_holes", obj="Plate", faces=[999]), "INVALID_VALUE"),
):
    e = json.loads(fx.call_json(call.pop("name"), **call))
    check(f"error {code}", not e["ok"] and e["error"]["code"] == code, e.get("error"))

names = {f["name"] for f in fx.describe()["functions"]}
check("describe() lists the new functions", {"extract_edges", "brep_holes"} <= names)

print(f"SUMMARY {sum(results)} / {len(results)}")
sys.exit(0 if all(results) else 1)
