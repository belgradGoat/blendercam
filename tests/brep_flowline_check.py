"""Headless check of the flowline pattern and its preview (milestone M5).

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P brep_flowline_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

A 60 x 60 x 20 mm block with a 32 x 28 mm pocket, 12 mm deep, every inside edge filleted at
R4 (spherical corners), moved and turned. flowline_preview makes a curve object of the
passes in world space; settings come from arguments or an operation; previews are replaced,
cached and refused when too big. Prints PASS/FAIL lines and a SUMMARY.
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
R = 0.004  # fillet radius
LOC, ANGLE = np.array([0.1, 0.05, -0.01]), math.radians(30)

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
flow = importlib.import_module(FABEX + ".brep.flow")
kernel = bridge.require_kernel()
check("kernel source", kernel.source == ("extension" if use_extension else "path"), kernel.source)

solids = kernel.module("ops.solids")
boolean = kernel.module("ops.boolean").boolean
fillet_edges = kernel.module("ops.fillet").fillet_edges
TessOptions = kernel.module("tessellate.grid").TessOptions
tessellate_body = kernel.module("tessellate.body").tessellate_body

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


ob = brep_object("Pocket", rounded, tuple(LOC), ANGLE)
faces = {f.id: f.name for f in rounded.faces.values()}
fillets = sorted(i for i, n in faces.items() if n == "Fillet")
corners = sorted(i for i, n in faces.items() if n == "Corner")


def spline_points(name):
    cu = bpy.data.objects[name].data
    out = []
    for sp in cu.splines:
        co = np.empty(len(sp.points) * 4, dtype=np.float32)
        sp.points.foreach_get("co", co)
        out.append((co.reshape(-1, 4)[:, :3].astype(float), sp.use_cyclic_u))
    return out


# --- the whole part --------------------------------------------------------------------------
res = fx.flowline_preview(
    "Pocket", stepover=0.001, step_along=0.0005, cutter_type="BALLNOSE", diameter=0.006
)
check("preview made", res["preview"] == "Pocket_flowline" and res["passes"] > 0, res["preview"])
splines = spline_points("Pocket_flowline")
check(
    "one spline per pass, every point",
    len(splines) == res["passes"] and sum(len(p) for p, _ in splines) == res["points"],
    (len(splines), res["passes"]),
)
check(
    "closed passes are cyclic splines",
    sum(c for _, c in splines) == res["closed_passes"],
    res["closed_passes"],
)
check("stepover kept", 0 < res["max_gap"] <= 0.001 * (1 + 1e-9), res["max_gap"])
by_name = {}
for f in res["faces"]:
    by_name.setdefault(f["name"], []).append(f)
steep = {f["name"] for f in res["faces"] if f["note"] == "too steep or facing down"}
floor_fillets = [f for f in by_name["Fillet"] if f["passes"]]
check(
    "passes on the top, floor, floor fillets and corners; none on walls, the bottom and "
    "the vertical fillets",
    all(f["passes"] for n in ("Box.+Z", "Corner") for f in by_name[n])
    and len(floor_fillets) == 4
    and all(f["note"] == "too steep or facing down" for f in by_name["Fillet"] if not f["passes"])
    and {"Box.-Z", "Box.+X"} <= steep,
    {n: [f["passes"] for f in fs] for n, fs in by_name.items()},
)
check(
    "corners: rings around their poles",
    all(f["poles"] and f["direction"] for f in by_name["Corner"]),
    [(f["direction"], f["poles"]) for f in by_name["Corner"]],
)
check("ball tip from the cutter", res["params"]["tip_radius"] == 0.003, res["params"])
check("cutter reported", res["cutter"]["type"] == "BALLNOSE", res["cutter"])

# world space: the top of the block is at z = 0, inside its turned outline and outside the
# pocket's opening (32 x 28 mm, its corners rounded by the R4 vertical fillets)


def rounded_box_sdf(xy, half=(0.016, 0.014), r=R):
    q = np.abs(xy) - (np.array(half) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - r


fl, cached = flow.pattern(
    ob, flow.params_for(None, None, stepover=0.001, step_along=0.0005, tip_radius=0.003)
)
check("same pattern from the cache", cached is True and len(fl.passes) == res["passes"])
top = [p for p in fl.passes if faces[p.face] == "Box.+Z"]
P = np.concatenate([p.points for p in top])
Mi = np.linalg.inv(np.array(ob.matrix_world))  # its float32 location and rotation
local = P @ Mi[:3, :3].T + Mi[:3, 3]
check(
    "top passes in world space",
    np.allclose(P[:, 2], 0.0, atol=1e-9)  # float32 location
    and np.all(np.abs(local[:, :2]) <= 0.03 + 1e-9)
    and rounded_box_sdf(local[:, :2]).min() >= -1e-6,  # never inside the opening
    (P[:, 2].min(), P[:, 2].max(), rounded_box_sdf(local[:, :2]).min()),
)
fil = np.concatenate([p.points for p in fl.passes if faces[p.face] == "Fillet"])
check(
    "fillet passes inside the pocket",
    fil[:, 2].max() <= 0.0 + 1e-9 and fil[:, 2].min() >= -0.012 - 1e-9,
)

# tool tips: same as the contact on flat faces, lower than... not: on curved faces they move
res_t = fx.flowline_preview(
    "Pocket",
    stepover=0.001,
    step_along=0.0005,
    cutter_type="BALLNOSE",
    diameter=0.006,
    show="tool",
    name="Pocket_tips",
)
tips = spline_points("Pocket_tips")
check("tool preview has the same passes", len(tips) == len(splines), len(tips))
flat = [i for i, p in enumerate(fl.passes) if faces[p.face] == "Box.+Z"][0]
curved = [i for i, p in enumerate(fl.passes) if faces[p.face] == "Fillet"][0]
check(
    "tips: on the contact for flat faces, off it on fillets",
    np.allclose(tips[flat][0], splines[flat][0], atol=1e-6)
    and np.abs(tips[curved][0] - splines[curved][0]).max() > 1e-4,
)

# --- faces, direction, replace --------------------------------------------------------------
sel = fx.flowline_preview("Pocket", faces=fillets, stepover=0.0005, direction="V")
check(
    "faces filter: only the fillets",
    {f["face"] for f in sel["faces"]} == set(fillets)
    and {f["face"] for f in sel["faces"] if f["passes"]} == {f["face"] for f in floor_fillets}
    and all(f["direction"] == "V" for f in sel["faces"] if f["passes"]),
    sel["faces"][:1],
)
check(
    "replaced in place",
    sel["preview"] == "Pocket_flowline" and "Pocket_flowline.001" not in bpy.data.objects,
)
check(
    "now the fillets' passes",
    len(bpy.data.objects["Pocket_flowline"].data.splines) == sel["passes"],
)
info = json.loads(bpy.data.objects["Pocket_flowline"]["fabex_brep_source"])
check("provenance", info["kind"] == "flowline_preview" and info["faces"] == fillets, info["kind"])
dry = fx.flowline_preview("Pocket", faces=corners, stepover=0.0005, create=False)
check(
    "create=False makes nothing",
    dry["preview"] is None
    and len(bpy.data.objects["Pocket_flowline"].data.splines) == sel["passes"],
)
bottom = [i for i, n in faces.items() if n == "Box.-Z"]
empty = fx.flowline_preview("Pocket", faces=bottom, stepover=0.001)
check(
    "nothing to cut: no preview, the old one removed",
    empty["preview"] is None and empty["passes"] == 0 and "Pocket_flowline" not in bpy.data.objects,
    empty["faces"][0]["note"],
)

# --- settings from an operation -------------------------------------------------------------
fx.create_operation(
    "Pocket",
    strategy="PARALLEL",
    name="Finish",
    props={
        "cutter_type": "BULLNOSE",
        "cutter_diameter": 0.006,
        "bull_corner_radius": 0.001,
        "distance_between_paths": 0.002,
        "distance_along_paths": 0.0004,
        "movement.type": "CLIMB",
    },
)
o = fx.flowline_preview("Pocket", op="Finish", create=False)
pr = o["params"]
check(
    "op settings: stepover, step along, order, bull nose corner",
    abs(pr["stepover"] - 0.002) < 1e-9
    and abs(pr["step_along"] - 0.0004) < 1e-9
    and pr["order"] == "ONEWAY"
    and abs(pr["tip_radius"] - 0.001) < 1e-9,
    pr,
)
o2 = fx.flowline_preview("Pocket", op="Finish", stepover=0.003, order="ZIGZAG", create=False)
check(
    "arguments override the op",
    o2["params"]["stepover"] == 0.003 and o2["params"]["order"] == "ZIGZAG",
    o2["params"],
)

# --- cache follows the object ---------------------------------------------------------------
a = fx.flowline_preview("Pocket", stepover=0.002, create=False)
b = fx.flowline_preview("Pocket", stepover=0.002, create=False)
ob.location.z += 0.01
bpy.context.view_layer.update()
c = fx.flowline_preview("Pocket", stepover=0.002, create=False)
check(
    "cached until the object moves", b["cached"] and not c["cached"] and a["points"] == c["points"]
)

# --- errors ---------------------------------------------------------------------------------
cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
bpy.context.scene.collection.objects.link(cube)
for call, code in (
    (dict(obj="Nope", stepover=0.001), "OBJECT_NOT_FOUND"),
    (dict(obj="Cube", stepover=0.001), "NOT_A_BREP"),
    (dict(obj="Pocket"), "INVALID_VALUE"),
    (dict(obj="Pocket", stepover=0.001, direction="W"), "INVALID_VALUE"),
    (dict(obj="Pocket", stepover=0.001, faces=[999]), "INVALID_VALUE"),
    (dict(obj="Pocket", stepover=0.001, show="nope"), "INVALID_VALUE"),
    (dict(obj="Pocket", stepover=0.001, max_slope=120), "INVALID_VALUE"),
    (dict(obj="Pocket", stepover=0.0001, max_points=1000), "TOO_LARGE"),
    (dict(obj="Pocket", op="Nope"), "OPERATION_NOT_FOUND"),
):
    e = json.loads(fx.call_json("flowline_preview", **call))
    check(f"error {code}", not e["ok"] and e["error"]["code"] == code, e.get("error"))
big = json.loads(fx.call_json("flowline_preview", obj="Pocket", stepover=0.0001, max_points=1000))
check(
    "TOO_LARGE has the estimate", big["error"]["details"]["estimated_points"] > 1000, big["error"]
)

d = fx.describe()
check(
    "describe() lists flowline_preview", "flowline_preview" in {f["name"] for f in d["functions"]}
)
check(
    "API 0.5, TOO_LARGE documented", d["api_version"] == "0.5" and "TOO_LARGE" in d["error_codes"]
)

print(f"SUMMARY {sum(results)} / {len(results)}")
sys.exit(0 if all(results) else 1)
