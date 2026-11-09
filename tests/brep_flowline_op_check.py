"""Headless check of the FLOWLINE strategy (milestone M6).

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P brep_flowline_op_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)
    FABEX_UPDATE_REFERENCE=1 ...       (write tests/test_data_brep/flowline_dome.gcode anew)

A 60 x 60 x 20 mm block with an R20 sphere fused into its top (a 10 mm dome, off centre),
moved so the dome's top is at z = 0. A FLOWLINE operation with a 3 mm ball is computed through OpenCAMLib: every point of
the toolpath - passes and the links between them - keeps the ball outside the exact solid
(analytic distance to box U sphere), points over the dome touch it, the G-code matches the
reference, and the depth-image route computes too. Prints PASS/FAIL lines and a SUMMARY.
"""

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import addon_utils
import bpy
import numpy as np

FABEX = "bl_ext.user_default.fabex"
BREPKIT = "bl_ext.user_default.brepkit"
use_extension = os.environ.get("FABEX_TEST_BREPKIT_EXTENSION") == "1"
HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "test_data_brep" / "flowline_dome.gcode"
LOC = np.array([0.02, -0.01, -0.02])  # the dome's top at z = 0 (Fabex's stock top)
HALF = np.array([0.03, 0.03, 0.01])  # the block, about LOC
CENTRE, RADIUS = LOC + (0.005, -0.004, 0.0), 0.02  # the sphere
R = 0.0015  # ball radius
TOL = 0.00001  # exact mesh tolerance

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
kernel = bridge.require_kernel()
check("kernel source", kernel.source == ("extension" if use_extension else "path"), kernel.source)
if not fx.status()["opencamlib"]:
    print("SKIP: OpenCAMLib is not available")
    sys.exit(1)

solids = kernel.module("ops.solids")
boolean = kernel.module("ops.boolean").boolean
TessOptions = kernel.module("tessellate.grid").TessOptions
tessellate_body = kernel.module("tessellate.body").tessellate_body

body, _ = boolean(
    solids.box((0, 0, 0), tuple(2 * HALF)),
    solids.sphere((0.005, -0.004, 0.0), RADIUS),
    "union",
)
t = tessellate_body(body, TessOptions())
mesh = bpy.data.meshes.new("Dome")
mesh.from_pydata([tuple(p) for p in t.points], [], [list(map(int, p)) for p in t.polygons()])
mesh.attributes.new("brep_face_id", "INT", "FACE").data.foreach_set(
    "value", np.asarray(t.face_ids, dtype=np.int32)
)
ob = bpy.data.objects.new("Dome", mesh)
bpy.context.scene.collection.objects.link(ob)
ob["brepkit_data"] = kernel.dumps_body(body)
ob.location = tuple(LOC)
bpy.context.view_layer.update()
faces = {f.name: f.id for f in body.faces.values()}


def sdf(P):
    """Exact distance to the solid (box U sphere); negative inside."""
    q = np.abs(P - LOC) - HALF
    box = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return np.minimum(box, np.linalg.norm(P - CENTRE, axis=1) - RADIUS)


def path_points(o):
    """Toolpath vertices and segment midpoints below the free height, in world space."""
    pob = bpy.data.objects[o.path_object_name]
    co = np.empty(len(pob.data.vertices) * 3)
    pob.data.vertices.foreach_get("co", co)
    M = np.array(pob.matrix_world)
    P = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    low = P[:, 2] < o.movement.free_height - 1e-6
    both = low[:-1] & low[1:]
    mids = [P[:-1][both] + f * (P[1:][both] - P[:-1][both]) for f in (0.25, 0.5, 0.75)]
    return P[low], np.concatenate(mids)


# --- OpenCAMLib, exact B-rep mesh -------------------------------------------------------------
res = fx.create_flowline_operation(
    "Dome",
    name="Flow",
    diameter=2 * R,
    stepover=0.001,
    step_along=0.0005,
    props={
        "optimisation.use_exact": True,
        "optimisation.use_opencamlib": True,
        "brep.mesh_tolerance": TOL,
        "movement.stay_low": True,
        "movement.free_height": 0.05,
    },
)
check("create_flowline_operation", res["valid"] and res["pattern"]["passes"] > 0, res["pattern"])
by_face = {f["name"]: f for f in res["pattern"]["faces"]}
check(
    "pattern: the top and the dome, rings on the dome",
    by_face["Box.+Z"]["passes"] > 0
    and by_face["B.Sphere.7"]["passes"] > 0
    and by_face["Box.-Z"]["passes"] == 0,
    {k: (v["direction"], v["passes"]) for k, v in by_face.items()},
)
o = bpy.context.scene.cam_operations["Flow"]
check("strategy FLOWLINE, no layers (finishing)", o.strategy == "FLOWLINE" and not o.use_layers)
other = fx.create_operation("Dome", strategy="PARALLEL", name="Other")
p_op = bpy.context.scene.cam_operations["Other"]
had_layers = p_op.use_layers
p_op.strategy = "FLOWLINE"
check("switching to Flowline turns layers off", had_layers and not p_op.use_layers)
fx.remove_operation("Other")

out_dir = tempfile.mkdtemp(prefix="fabex_flow_")
r1 = fx.compute_operation("Flow", export=out_dir)
check("computes", r1["ok"], r1["message"])
fl_rep = r1.get("brep_flowline") or []
check(
    "compute result: the pattern",
    len(fl_rep) == 1 and fl_rep[0]["passes"] == res["pattern"]["passes"],
    fl_rep and {k: fl_rep[0][k] for k in ("passes", "points", "max_gap")},
)
check(
    "exact B-rep mesh sampled",
    any(m.get("mesh") == "exact" for m in r1["brep_meshes"]),
    r1["brep_meshes"],
)
check("fit check ran: the ball fits", r1["brep_fit"] and r1["brep_fit"][0]["fits"] is True)

V, mids = path_points(o)
centres = V + (0, 0, R)
d = sdf(centres) - R
check(
    "no gouge at any toolpath point (ball outside the exact solid)",
    d.min() >= -2.5 * TOL,
    f"min {d.min():.2e} over {len(V)} points",
)
dm = sdf(mids + (0, 0, R)) - R
check("no gouge between points", dm.min() >= -3 * TOL, f"min {dm.min():.2e}")
over_dome = np.linalg.norm(V[:, :2] - CENTRE[:2], axis=1) < 0.012
touch = np.abs(np.linalg.norm(centres[over_dome] - CENTRE, axis=1) - (RADIUS + R))
check(
    "over the dome the ball touches it",
    over_dome.sum() > 500 and np.percentile(touch, 95) < 3 * TOL,
    f"{over_dome.sum()} points, 95% within {np.percentile(touch, 95):.2e}",
)
top = (np.linalg.norm(V[:, :2] - CENTRE[:2], axis=1) > 0.02 + R) & (V[:, 2] > LOC[2])
check(
    "on the flat top the ball sits on it",
    top.sum() > 50 and np.allclose(V[top, 2], LOC[2] + HALF[2], atol=2 * TOL),  # straight
    # passes: Fabex keeps only their ends (path optimisation)
    f"{top.sum()} points, z {V[top, 2].min():.6f}..{V[top, 2].max():.6f}",
)

gcode = r1["gcode_files"]  # .tap with the default post-processor
check("G-code written", len(gcode) == 1, r1["gcode_files"])
text = Path(gcode[0]).read_text() if gcode else ""
if os.environ.get("FABEX_UPDATE_REFERENCE") == "1" and text:
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(text)
    print(f"wrote {REFERENCE}")
if REFERENCE.exists():
    ref = [x for x in REFERENCE.read_text().splitlines() if not x.startswith("(")]
    new = [x for x in text.splitlines() if not x.startswith("(")]  # comments name the file
    diff = sum(a != b for a, b in zip(ref, new)) + abs(len(ref) - len(new))
    check("G-code matches the reference", diff == 0, f"{diff} lines differ of {len(ref)}")
else:
    check("G-code reference exists (FABEX_UPDATE_REFERENCE=1 writes it)", False, REFERENCE)

# --- only the dome, from a face selection ------------------------------------------------------
fx.set_operation_props("Flow", {"brep.flow_faces": str(faces["B.Sphere.7"])})
r2 = fx.compute_operation("Flow", export=False)
V2, _ = path_points(o)
check(
    "faces: only the dome",
    r2["ok"]
    and [f["name"] for f in r2["brep_flowline"][0]["faces"]] == ["B.Sphere.7"]
    and np.linalg.norm(V2[:, :2] - CENTRE[:2], axis=1).max() < 0.0175 + 2 * R,
    r2["message"],
)
d2 = sdf(V2 + (0, 0, R)) - R
check("dome only: no gouge", d2.min() >= -2.5 * TOL, f"{d2.min():.2e}")

# --- depth image instead of OpenCAMLib -------------------------------------------------------
fx.set_operation_props(
    "Flow",
    {
        "brep.flow_faces": "",
        "optimisation.use_exact": False,
        "optimisation.use_opencamlib": False,
        "optimisation.pixsize": 0.0001,
    },
)
r3 = fx.compute_operation("Flow", export=False)
V3, _ = path_points(o)
d3 = sdf(V3 + (0, 0, R)) - R
check("depth image route computes", r3["ok"] and len(V3) > 1000, r3["message"])
check(
    "depth image route: within a pixel of the surface",
    d3.min() >= -0.00025 and np.percentile(np.abs(d3), 50) < 0.0001,
    f"min {d3.min():.2e}, median |d| {np.percentile(np.abs(d3), 50):.2e}",
)

# --- operators: faces from Edit Mode, preview ------------------------------------------------
import bmesh  # noqa: E402

fx.set_operation_props("Flow", {"brep.flow_faces": ""})
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bm = bmesh.from_edit_mesh(ob.data)
layer = bm.faces.layers.int.get("brep_face_id")
for f in bm.faces:
    f.select_set(f[layer] == faces["B.Sphere.7"])
bmesh.update_edit_mesh(ob.data)
ret = bpy.ops.object.cam_brep_flow_faces()
bpy.ops.object.mode_set(mode="OBJECT")
check(
    "Use Selected Faces: the dome's face id",
    ret == {"FINISHED"} and o.brep.flow_faces == str(faces["B.Sphere.7"]),
    o.brep.flow_faces,
)
ret = bpy.ops.object.cam_brep_flowline_preview()
pv = bpy.data.objects.get("Dome_flowline")
check(
    "Preview Flowline: curves of the selected face",
    ret == {"FINISHED"}
    and pv is not None
    and len(pv.data.splines) == by_face["B.Sphere.7"]["passes"],
    pv and len(pv.data.splines),
)
bpy.ops.object.cam_brep_flow_faces(clear=True)
check("Clear: every face again", o.brep.flow_faces == "")

# --- the preset ------------------------------------------------------------------------------
preset = HERE.parent / "fabex" / "presets" / "cam_operations" / "Fin_Ball_3,0_Flowline.py"
count = len(bpy.context.scene.cam_operations)
exec(compile(preset.read_text(), str(preset), "exec"), {"__file__": str(preset)})
new = bpy.context.scene.cam_operations[-1]
check(
    "preset makes a Flowline operation",
    len(bpy.context.scene.cam_operations) == count + 1
    and new.strategy == "FLOWLINE"
    and new.cutter_type == "BALLNOSE"
    and abs(new.distance_between_paths - 0.00025) < 1e-9
    and new.object_name == "Dome",
    (new.strategy, new.object_name),
)

# --- errors ------------------------------------------------------------------------------------
cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
bpy.context.scene.collection.objects.link(cube)
plain = fx.create_operation("Cube", strategy="FLOWLINE", name="Plain")
e = fx.compute_operation("Plain", export=False)
check("not a B-rep: a clear error", not e["ok"] and "B-rep" in e["message"], e["message"])
fx.set_operation_props("Flow", {"brep.flow_faces": "1, x"})
e = fx.compute_operation("Flow", export=False)
check("bad faces text: a clear error", not e["ok"] and "face ids" in e["message"], e["message"])
for call, code in (
    (dict(obj="Cube"), "NOT_A_BREP"),
    (dict(obj="Dome", faces=[999]), "INVALID_VALUE"),
    (dict(obj="Dome", cutter_type="BULLNOSE"), "INVALID_VALUE"),
    (dict(obj="Dome", direction="W"), "INVALID_VALUE"),
):
    err = json.loads(fx.call_json("create_flowline_operation", **call))
    check(f"error {code}", not err["ok"] and err["error"]["code"] == code, err.get("error"))

d = fx.describe()
names = {f["name"] for f in d["functions"]}
check(
    "describe(): FLOWLINE strategy and create_flowline_operation",
    "create_flowline_operation" in names
    and any(s["id"] == "FLOWLINE" for s in d["strategies"])
    and d["api_version"] == "0.6",
)

print(f"SUMMARY {sum(results)} / {len(results)}")
sys.exit(0 if all(results) else 1)
