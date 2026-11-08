"""Headless check of exact B-rep meshes for OpenCAMLib (milestone M2).

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P brep_ocl_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

A B-rep sphere gets a deliberately coarse display mesh. A ball-nose parallel finish is
computed with OpenCAMLib, and the path's Z is compared with the analytic answer (the
ball centre rides on a sphere of radius R + r). With the exact B-rep mesh the error must
stay within the mesh tolerance (scaled by the surface slope); with the display mesh it is
much larger. Also checks caching, cache invalidation, fallbacks and API reporting.
Prints PASS/FAIL lines and a SUMMARY.
"""

import importlib
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
ocl_mesh = importlib.import_module(FABEX + ".brep.ocl_mesh")
operation_utils = importlib.import_module(FABEX + ".utilities.operation_utils")
exception = importlib.import_module(FABEX + ".exception")
kernel = bridge.require_kernel()
check("kernel source", kernel.source == ("extension" if use_extension else "path"), kernel.source)

# --- a B-rep sphere, R = 15 mm, top at Z = 0, with a coarse display mesh ---------------------
R = 0.015
CENTER = np.array([0.02, 0.01, -R])
body = kernel.module("ops.solids").sphere((0.0, 0.0, 0.0), R)
TessOptions = kernel.module("tessellate.grid").TessOptions
coarse = kernel.module("tessellate.body").tessellate_body(
    body, TessOptions(chord_tol=0.01, min_segments=4, max_segments=12)
)
mesh = bpy.data.meshes.new("Ball")
mesh.from_pydata(
    [tuple(p) for p in coarse.points], [], [list(map(int, p)) for p in coarse.polygons()]
)
ob = bpy.data.objects.new("Ball", mesh)
bpy.context.scene.collection.objects.link(ob)
ob["brepkit_data"] = kernel.dumps_body(body)
ob.location = tuple(CENTER)
bpy.context.view_layer.update()

TOL = 0.00001
BALL_R = 0.003


def path_points(op_name):
    o = next(o for o in bpy.context.scene.cam_operations if o.name == op_name)
    path = bpy.data.objects[o.path_object_name]
    M = np.array(path.matrix_world)
    co = np.array([v.co for v in path.data.vertices])
    return co @ M[:3, :3].T + M[:3, 3]


def analytic_error(points):
    """(max |error| / allowed, max |error|) over points on the upper sphere."""
    d = np.linalg.norm(points[:, :2] - CENTER[:2], axis=1)
    rr = R + BALL_R
    keep = (d < 0.7 * rr) & (points[:, 2] < 1e-4)  # not on lifts, not too steep
    p, d = points[keep], d[keep]
    zc = CENTER[2] + np.sqrt(rr**2 - d**2)
    err = np.abs(p[:, 2] - (zc - BALL_R))
    nz = np.sqrt(rr**2 - d**2) / rr  # normal Z at the contact point
    return float((err / (TOL / nz + 1e-6)).max()), float(err.max()), int(keep.sum())


res = fx.create_operation(
    "Ball",
    strategy="PARALLEL",
    name="BallFinish",
    props={
        "cutter_type": "BALLNOSE",
        "cutter_diameter": 2 * BALL_R,
        "distance_between_paths": 0.001,
        "optimisation.use_exact": True,
        "optimisation.use_opencamlib": True,
        "brep.mesh_tolerance": TOL,
    },
)
check("create_operation with brep.mesh_tolerance", res["valid"], res)
props = fx.get_operation("BallFinish")["props"]
check(
    "brep settings exposed through the API",
    props.get("brep.use_exact_mesh") is True and abs(props["brep.mesh_tolerance"] - TOL) < 1e-12,
    {k: v for k, v in props.items() if k.startswith("brep.")},
)

# --- exact mesh ------------------------------------------------------------------------------
r = fx.compute_operation("BallFinish", export=False)
check("compute (exact mesh) ok", r["ok"], r["message"])
rep = r.get("brep_meshes") or []
check(
    "result reports the exact mesh within tolerance",
    len(rep) == 1 and rep[0]["mesh"] == "exact" and rep[0]["within_tolerance"],
    rep,
)
ratio, err_exact, n = analytic_error(path_points("BallFinish"))
check(
    "path Z matches the analytic sphere within the tolerance",
    n > 200 and ratio <= 1.0,
    f"max error {err_exact * 1e6:.2f} um over {n} points, ratio {ratio:.2f}",
)

r2 = fx.compute_operation("BallFinish", export=False)
check(
    "recompute uses the cached mesh", r2["brep_meshes"][0].get("cached") is True, r2["brep_meshes"]
)

# --- display mesh, for comparison ------------------------------------------------------------
fx.set_operation_props("BallFinish", {"brep.use_exact_mesh": False})
r3 = fx.compute_operation("BallFinish", export=False)
check(
    "use_exact_mesh off -> display mesh",
    r3["ok"] and r3["brep_meshes"] and r3["brep_meshes"][0]["mesh"] == "display",
    r3["brep_meshes"],
)
_, err_display, _ = analytic_error(path_points("BallFinish"))
check(
    "exact mesh is much more accurate than the coarse display mesh",
    err_display > 10 * err_exact,
    f"display {err_display * 1e6:.1f} um vs exact {err_exact * 1e6:.2f} um",
)
fx.set_operation_props("BallFinish", {"brep.use_exact_mesh": True})

# --- OCL waterline uses the same exact mesh --------------------------------------------------
fx.create_operation(
    "Ball",
    strategy="WATERLINE",
    name="BallWaterline",
    props={
        "cutter_type": "BALLNOSE",
        "cutter_diameter": 2 * BALL_R,
        "stepdown": 0.002,
        "optimisation.use_exact": True,
        "optimisation.use_opencamlib": True,
        "brep.mesh_tolerance": TOL,
    },
)
rw = fx.compute_operation("BallWaterline", export=False)
check(
    "waterline with the exact mesh (cached from the parallel op)",
    rw["ok"] and rw["brep_meshes"] and rw["brep_meshes"][0]["mesh"] == "exact",
    (rw["message"], rw["brep_meshes"]),
)
pw = path_points("BallWaterline")
# the ball centre (tip + r) rides on a sphere of radius R + r: each contour is a circle
rr = R + BALL_R
h = pw[:, 2] + BALL_R - CENTER[2]
on = (h > 0.1 * rr) & (h < 0.9 * rr)
d = np.linalg.norm(pw[on, :2] - CENTER[:2], axis=1)
expected = np.sqrt(rr**2 - h[on] ** 2)
nxy = expected / rr  # horizontal part of the contact normal
ratio_w = float((np.abs(d - expected) / (TOL / nxy + 1e-6)).max()) if on.any() else np.inf
check(
    "waterline contours match the analytic circles within the tolerance",
    on.sum() > 100 and ratio_w <= 1.0,
    f"max error {np.abs(d - expected).max() * 1e6:.2f} um over {int(on.sum())} points, "
    f"ratio {ratio_w:.2f}",
)

# --- cam_tessellation API and caching --------------------------------------------------------
t1 = fx.cam_tessellation("Ball", op="BallFinish")
check(
    "cam_tessellation within tolerance",
    t1["within_tolerance"] and t1["max_deviation"] <= TOL and t1["triangles"] > 0,
    {k: t1[k] for k in ("triangles", "max_deviation", "tolerance", "cached", "seconds")},
)
check("cam_tessellation reuses the compute's mesh", t1["cached"])
lo, hi = np.array(t1["bbox_min"]), np.array(t1["bbox_max"])
check(
    "tessellation is in world space",
    np.allclose(lo, CENTER - R, atol=TOL) and np.allclose(hi, CENTER + R, atol=TOL),
    (lo.tolist(), hi.tolist()),
)
coarser = fx.cam_tessellation("Ball", tolerance=0.0001)
check(
    "coarser tolerance gives fewer triangles",
    not coarser["cached"] and coarser["triangles"] < t1["triangles"],
    (coarser["triangles"], t1["triangles"]),
)
ob.location.x += 0.001
bpy.context.view_layer.update()
moved = fx.cam_tessellation("Ball", op="BallFinish")
check("moving the object invalidates the cache", not moved["cached"])
check("moved mesh follows the object", abs(moved["bbox_min"][0] - (lo[0] + 0.001)) < TOL)
ob.location.x -= 0.001
bpy.context.view_layer.update()

# --- editing the B-rep invalidates cached images and meshes ----------------------------------
o = next(o for o in bpy.context.scene.cam_operations if o.name == "BallFinish")
before = operation_utils.get_change_data(o)
body2 = kernel.module("ops.solids").sphere((0.0, 0.0, 0.0), R * 0.999)
ob["brepkit_data"] = kernel.dumps_body(body2)
check("get_change_data sees B-rep edits", operation_utils.get_change_data(o) != before)
edited = fx.cam_tessellation("Ball", op="BallFinish")
check("editing the B-rep invalidates the mesh cache", not edited["cached"])
ob["brepkit_data"] = kernel.dumps_body(body)

# --- errors and fallbacks --------------------------------------------------------------------
import json

e = json.loads(fx.call_json("cam_tessellation", obj="Nope"))
check("unknown object -> OBJECT_NOT_FOUND", e["error"]["code"] == "OBJECT_NOT_FOUND", e)
e = json.loads(fx.call_json("cam_tessellation", obj="Ball", tolerance=-1))
check("negative tolerance -> INVALID_VALUE", e["error"]["code"] == "INVALID_VALUE", e)

ob.modifiers.new("Bevel", "BEVEL")
r4 = fx.compute_operation("BallFinish", export=False)
check(
    "object with modifiers -> display mesh + warning",
    r4["ok"]
    and r4["brep_meshes"][0]["reason"] == "object has modifiers"
    and any("modifiers" in w for w in r4["warnings"]),
    (r4["brep_meshes"], r4["warnings"]),
)
ob.modifiers.clear()


def no_kernel():
    raise exception.BrepError("BREPKIT_UNAVAILABLE", "BRepKit not found (test)", {})


saved = bridge.require_kernel
bridge.require_kernel = no_kernel
try:
    r5 = fx.compute_operation("BallFinish", export=False)
finally:
    bridge.require_kernel = saved
check(
    "BRepKit missing -> display mesh + warning, compute still succeeds",
    r5["ok"] and r5["brep_meshes"][0]["reason"] == "BREPKIT_UNAVAILABLE" and r5["warnings"],
    (r5["brep_meshes"], r5["warnings"]),
)

print(f"SUMMARY {sum(results)} / {len(results)}")
sys.exit(0 if all(results) else 1)
