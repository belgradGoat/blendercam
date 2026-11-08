"""Headless check of the cutter fit check (milestone M4).

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P brep_fit_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

A 60 x 60 x 20 mm block with a 32 x 28 mm pocket, 12 mm deep, every inside edge filleted
at R4 (spherical corners), moved and turned. The known radii come back from fit_check, a
Parallel operation with a too big ball gets a warning when it computes (and a fitting one
doesn't), the operator reports and selects the faces in Edit Mode, the same pocket with
sharp corners reports them with the material left. Prints PASS/FAIL lines and a SUMMARY.
"""

import importlib
import json
import math
import os
import sys

import addon_utils
import bmesh
import bpy
import numpy as np

FABEX = "bl_ext.user_default.fabex"
BREPKIT = "bl_ext.user_default.brepkit"
use_extension = os.environ.get("FABEX_TEST_BREPKIT_EXTENSION") == "1"
ROOT2 = math.sqrt(2.0) - 1.0
R = 0.004  # fillet radius
SIN80 = math.sin(math.radians(80))

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
fit = importlib.import_module(FABEX + ".brep.fit")
kernel = bridge.require_kernel()
check("kernel source", kernel.source == ("extension" if use_extension else "path"), kernel.source)

solids = kernel.module("ops.solids")
boolean = kernel.module("ops.boolean").boolean
fillet_edges = kernel.module("ops.fillet").fillet_edges
TessOptions = kernel.module("tessellate.grid").TessOptions
tessellate_body = kernel.module("tessellate.body").tessellate_body

# block z -10..10 mm, pocket floor at z = -2 mm
sharp, _ = boolean(
    solids.box((0, 0, 0), (0.06, 0.06, 0.02)),
    solids.box((0, 0, 0.006), (0.032, 0.028, 0.016)),  # through the top
    "difference",
)
inside = [
    e.id
    for e in sharp.edges.values()
    if e.is_shared and all(sharp.faces[x.face].name.startswith("B.") for x in e.coedges)
]
rounded, _ = fillet_edges(sharp, inside, R)


def brep_object(name, body, location=(0, 0, 0), angle=0.0):
    t = tessellate_body(body, TessOptions())
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in t.points], [], [list(map(int, p)) for p in t.polygons()])
    attr = mesh.attributes.new("brep_face_id", "INT", "FACE")
    attr.data.foreach_set("value", np.asarray(t.face_ids, dtype=np.int32))
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob["brepkit_data"] = kernel.dumps_body(body)
    ob.location = location
    ob.rotation_euler = (0, 0, angle)
    bpy.context.view_layer.update()
    return ob


ob = brep_object("Pocket", rounded, (0.1, 0.05, -0.01), math.radians(30))
sharp_ob = brep_object("SharpPocket", sharp, (-0.1, 0.0, -0.01))

# --- what an agent sees: the radii, no cutter -------------------------------------------------
info = fx.fit_check("Pocket")
conc, wall = info["min_concave_radius"], info["min_wall_radius"]
check("concave radius = fillet radius", abs(conc["radius"] - R) < R * 1e-6, conc)
check("concave point is on the part (world space)", 0.1 - 0.03 < conc["at"][0] < 0.1 + 0.03, conc)
check(
    "wall radius: the spherical corners at 80 degrees",
    abs(wall["radius"] - R * SIN80) < R * 1e-4 and wall["face_name"] == "Corner",
    wall,
)
check("no sharp corners, no cutter verdict", info["sharp_corners"] == [] and info["fits"] is None)
check(
    "faces: only the limited ones",
    len(info["faces"]) == 12 and all(f["concave_radius"] for f in info["faces"]),
    len(info["faces"]),
)
check("all_faces lists every face", len(fx.fit_check("Pocket", all_faces=True)["faces"]) == 23)

ok = fx.fit_check("Pocket", cutter_type="BALLNOSE", diameter=2 * R)
check("R4 ball fits R4 fillets", ok["fits"] is True and ok["problems"] == [], ok["messages"])
check("cached analysis", ok["cached"] is True)
big = fx.fit_check("Pocket", cutter_type="BALLNOSE", diameter=0.01)
check(
    "R5 ball doesn't",
    big["fits"] is False
    and {p["check"] for p in big["problems"]} == {"concave_radius"}
    and abs(big["max_tip_radius"] - R) < 1e-9,
    big["messages"],
)
check(
    "warning text",
    big["messages"] and "doesn't fit a 4.0 mm concave radius" in big["messages"][0],
    big["messages"],
)
end = fx.fit_check("Pocket", cutter_type="END", diameter=0.006)
check("6 mm end mill fits the walls", end["fits"] is True, end["messages"])
bull = fx.fit_check("Pocket", cutter_type="BULLNOSE", diameter=0.012, corner_radius=0.002)
check(
    "12 mm bull nose: its corner fits, its side doesn't",
    bull["fits"] is False and {p["check"] for p in bull["problems"]} == {"wall_radius"},
    bull["messages"],
)
skin = fx.fit_check("Pocket", cutter_type="BALLNOSE", diameter=0.0078, skin=0.0002)
check("skin counts", skin["fits"] is False, skin["messages"])

# --- operations: warnings when computing --------------------------------------------------
res = fx.create_operation(
    "Pocket",
    strategy="PARALLEL",
    name="Finish",
    props={"cutter_type": "BALLNOSE", "cutter_diameter": 0.01, "distance_between_paths": 0.004},
)
check("create parallel operation", res["valid"], res)
r1 = fx.compute_operation("Finish", export=False)
check("computes", r1["ok"], r1["message"])
fits = r1["brep_fit"]
check(
    "compute result: brep_fit says it doesn't fit",
    len(fits) == 1 and fits[0]["fits"] is False and fits[0]["problem_count"] > 0,
    fits,
)
check("compact report (no face lists)", "faces" not in fits[0] and "sharp_corner_count" in fits[0])
check(
    "warning in the operation",
    any("doesn't fit a 4.0 mm concave radius" in w for w in r1["warnings"]),
    r1["warnings"],
)
fx.set_operation_props("Finish", {"cutter_diameter": 0.008})
r2 = fx.compute_operation("Finish", export=False)
check(
    "fitting ball: no fit warning",
    r2["ok"]
    and r2["brep_fit"][0]["fits"] is True
    and not any("doesn't fit" in w for w in r2["warnings"]),
    r2["warnings"],
)
fx.set_operation_props("Finish", {"cutter_diameter": 0.01, "brep.check_fit": False})
r3 = fx.compute_operation("Finish", export=False)
check(
    "check_fit off: no check",
    r3["brep_fit"] == [] and not any("doesn't fit" in w for w in r3["warnings"]),
    r3["warnings"],
)
fx.set_operation_props("Finish", {"brep.check_fit": True})
o = bpy.context.scene.cam_operations["Finish"]
check(
    "fit_check(op=...) takes the operation's cutter",
    fx.fit_check("Pocket", op="Finish")["cutter"]["diameter"] == 0.01,
)

# --- the operator ---------------------------------------------------------------------------
bpy.context.view_layer.objects.active = ob
ret = bpy.ops.object.cam_brep_fit_check()
reps = fit.report(o)
check(
    "operator: checks the active operation",
    ret == {"FINISHED"} and len(reps) == 1 and reps[0]["fits"] is False,
    ret,
)
check("report is current", fit.report_is_current(o, reps[0]))
o.cutter_diameter = 0.008
check("changed cutter: report is stale", not fit.report_is_current(o, reps[0]))
o.cutter_diameter = 0.01

me = ob.data
me.polygons.foreach_set("select", np.zeros(len(me.polygons), dtype=bool))
bpy.ops.object.mode_set(mode="EDIT")
ret = bpy.ops.object.cam_brep_fit_check(select=True)
bm = bmesh.from_edit_mesh(me)
layer = bm.faces.layers.int.get("brep_face_id")
selected = {f[layer] for f in bm.faces if f.select}
problem = {p["face"] for p in fit.report(o)[0]["problems"]}
bpy.ops.object.mode_set(mode="OBJECT")
check(
    "operator in Edit Mode selects the problem faces",
    ret == {"FINISHED"} and selected == problem and len(problem) == 12,
    (sorted(selected), sorted(problem)),
)

# --- sharp corners --------------------------------------------------------------------------
s = fx.fit_check("SharpPocket", cutter_type="END", diameter=0.006)
kinds = sorted(c["kind"] for c in s["sharp_corners"])
check(
    "sharp pocket: 4 floor + 4 vertical corners", kinds == ["floor"] * 4 + ["vertical"] * 4, kinds
)
rest = {c["kind"]: c["rest"] for c in s["sharp_corners"]}
check(
    "end mill: fits, leaves r(sqrt2-1) in vertical corners, nothing on the floor",
    s["fits"] is True
    and abs(rest["vertical"] - 0.003 * ROOT2) < 1e-12
    and rest["floor"] == 0.0
    and abs(s["corner_rest_max"] - 0.003 * ROOT2) < 1e-12,
    rest,
)
check(
    "corner message",
    any("4 sharp inside corner(s)" in m for m in s["messages"]),
    s["messages"],
)
v = next(c for c in s["sharp_corners"] if c["kind"] == "vertical")
check("corner heights in world space", abs(v["z_min"] + 0.012) < 1e-9 and abs(v["z_max"]) < 1e-9, v)

# --- cache follows the object ---------------------------------------------------------------
ob.location.x += 0.05
bpy.context.view_layer.update()
moved = fx.fit_check("Pocket")
before = {f["face"]: f["concave_at"] for f in info["faces"]}
after = {f["face"]: f["concave_at"] for f in moved["faces"]}
shift = [np.subtract(after[k], before[k]) for k in before]
check(
    "moving the object re-runs the analysis",
    moved["cached"] is False and all(np.allclose(d, [0.05, 0, 0], atol=1e-9) for d in shift),
    shift[:2],
)

# --- errors ---------------------------------------------------------------------------------
cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
bpy.context.scene.collection.objects.link(cube)
for call, code in (
    (dict(name="fit_check", obj="Nope"), "OBJECT_NOT_FOUND"),
    (dict(name="fit_check", obj="Cube"), "NOT_A_BREP"),
    (dict(name="fit_check", obj="Pocket", cutter_type="BALLNOSE"), "INVALID_VALUE"),
    (dict(name="fit_check", obj="Pocket", cutter_type="BULLNOSE", diameter=0.01), "INVALID_VALUE"),
    (dict(name="fit_check", obj="Pocket", faces=[999]), "INVALID_VALUE"),
    (dict(name="fit_check", obj="Pocket", wall_angle=0), "INVALID_VALUE"),
    (dict(name="fit_check", obj="Pocket", op="Nope"), "OPERATION_NOT_FOUND"),
):
    e = json.loads(fx.call_json(call.pop("name"), **call))
    check(f"error {code}", not e["ok"] and e["error"]["code"] == code, e.get("error"))

names = {f["name"] for f in fx.describe()["functions"]}
check("describe() lists fit_check", "fit_check" in names)

print(f"SUMMARY {sum(results)} / {len(results)}")
sys.exit(0 if all(results) else 1)
