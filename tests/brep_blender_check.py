"""Headless end-to-end check of Fabex's B-rep bridge and scripting API.

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P brep_blender_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

Builds a B-rep object (a domed patch) by writing BRepKit JSON onto a mesh, then checks
that the API finds BRepKit, reads the body, and can create / compute / export a parallel
finishing operation on it through fabex.api. Prints PASS/FAIL lines and a SUMMARY.
"""

import importlib
import json
import os
import sys
import tempfile

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

# --- kernel discovery -----------------------------------------------------------------------
st = fx.status()["brepkit"]
expected_source = "extension" if use_extension else "path"
check("BRepKit found", st["available"], st.get("error") or "")
check(
    f"kernel source is {expected_source}",
    st["kernel"] and st["kernel"]["source"] == expected_source,
    st["kernel"],
)
kernel = bridge.require_kernel()

# --- a B-rep object: one bicubic dome patch, 60 x 40 mm, 8 mm high -------------------------
prim = kernel.module("nurbs.primitives")
Body = kernel.module("topology.model").Body
u = np.linspace(0, 1, 4)[:, None]
v = np.linspace(0, 1, 4)[None, :]
ctrl = np.zeros((4, 4, 3))
ctrl[..., 0] = u * 0.06
ctrl[..., 1] = v * 0.04
ctrl[..., 2] = 0.012 * np.sin(np.pi * u) * np.sin(np.pi * v)
body = Body()
body.add_face(prim.bezier_surface(ctrl), "dome")

if use_extension:
    # let BRepKit build the display mesh
    sync = importlib.import_module(BREPKIT + ".sync")
    mesh = bpy.data.meshes.new("Dome")
    ob = bpy.data.objects.new("Dome", mesh)
    bpy.context.scene.collection.objects.link(ob)
    sync.write_body(ob, body)
    sync.rebuild_mesh(ob, body)
else:
    # build a display mesh ourselves from the kernel tessellation
    tess = kernel.module("tessellate.body").tessellate_body(body)
    mesh = bpy.data.meshes.new("Dome")
    polys = tess.polygons() if hasattr(tess, "polygons") else None
    verts = [tuple(p) for p in tess.points]
    faces = [list(map(int, p)) for p in polys]
    mesh.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new("Dome", mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob["brepkit_data"] = kernel.dumps_body(body)
ob.location = (0.01, 0.02, -0.012)  # top of the dome at Z=0 (stock top)

check("is_brep", bridge.is_brep(ob))
check("non-brep mesh is not a brep", not bridge.is_brep(bpy.data.objects.get("Cube")))
lst = fx.list_brep_objects()
dome = next((e for e in lst if e["name"] == "Dome"), None)
check("list_brep_objects sees the dome", dome and dome["faces"] == 1, dome)
h1 = bridge.body_hash(ob)
b1 = bridge.read_body(ob)
check("read_body is cached", bridge.read_body(ob) is b1)

# --- API: create / compute / export on the B-rep (as a mesh) --------------------------------
out_dir = tempfile.mkdtemp(prefix="fabex_brep_check_")
res = fx.create_operation(
    "Dome",
    strategy="PARALLEL",
    name="DomeFinish",
    props={
        "cutter_type": "BALLNOSE",
        "cutter_diameter": 0.006,
        "distance_between_paths": 0.002,
        "optimisation.use_exact": True,
        "optimisation.use_opencamlib": True,
    },
)
check("create_operation", res["op"] == "DomeFinish" and res["valid"], res)

bad = json.loads(fx.call_json("set_operation_props", op="DomeFinish", props={"cutter_typ": "END"}))
check(
    "typo gives INVALID_PROPERTY with suggestion",
    not bad["ok"]
    and bad["error"]["code"] == "INVALID_PROPERTY"
    and "cutter_type" in bad["error"]["details"]["close_matches"],
    bad,
)
bad = json.loads(
    fx.call_json("set_operation_props", op="DomeFinish", props={"cutter_type": "SPOON"})
)
check(
    "bad enum gives INVALID_VALUE with allowed list",
    not bad["ok"] and "BALLNOSE" in bad["error"]["details"]["allowed"],
    bad,
)

r = fx.compute_operation("DomeFinish", export=out_dir)
check("compute_operation ok", r["ok"], r["message"])
check(
    "G-code written",
    r["gcode_files"] and os.path.getsize(r["gcode_files"][0]) > 0,
    r["gcode_files"],
)
p = r["path"] or {}
# (bbox_min includes the start point at the origin, so only check the far side)
check(
    "path stays near the part footprint (10 mm margin)",
    p and p["bbox_max"][0] < 0.07 + 0.01 and p["bbox_max"][1] < 0.06 + 0.01,
    p,
)
check(
    "API results are plain JSON types", isinstance(p.get("bbox_min"), list), type(p.get("bbox_min"))
)
check("path has many points", p and p["points"] > 100, p.get("points"))
check(
    "machining time estimated when G-code is written",
    p and p["estimated_minutes"] and p["estimated_minutes"] > 0,
    p.get("estimated_minutes"),
)
r_noexport = fx.compute_operation("DomeFinish", export=False)
check(
    "no machining-time guess without G-code",
    r_noexport["ok"] and r_noexport["path"]["estimated_minutes"] is None,
    r_noexport["path"],
)
check(
    "path never below the part's lowest point",
    p and p["bbox_min"][2] >= -0.012 - 1e-6,
    p.get("bbox_min"),
)
check("path reaches down the dome flanks", p and p["bbox_min"][2] < -0.008, p.get("bbox_min"))
info = fx.path_info("DomeFinish")
check("path_info matches", info["points"] == p.get("points"))
exp = fx.export_gcode("DomeFinish", directory=out_dir)
check("export_gcode", exp["files"], exp)

# editing the B-rep changes the hash (used for cache invalidation in M2)
ctrl2 = ctrl.copy()
ctrl2[1:3, 1:3, 2] += 0.002
body2 = Body()
body2.add_face(prim.bezier_surface(ctrl2), "dome")
ob["brepkit_data"] = kernel.dumps_body(body2)
check("hash changes when B-rep changes", bridge.body_hash(ob) != h1)
check("read_body re-reads after change", bridge.read_body(ob) is not b1)

rm = fx.remove_operation("DomeFinish")
check("remove_operation", rm["removed"] == "DomeFinish" and rm["removed_path"])

print(f"SUMMARY {sum(results)} / {len(results)}")
sys.exit(0 if all(results) else 1)
