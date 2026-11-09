"""Headless check of brep_info and its suggestions (milestone M7).

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P brep_info_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

An 80 x 60 x 20 mm block with a dome fused on top (R11 sphere, R2 fillet round it), two
6 mm through holes, a 4 mm blind hole and a 12 x 10 x 5 mm pocket, left where it was
modelled (its top above Z 0). brep_info has to recognise every face, hole and limit; then
every suggested step is run exactly as given - the Python to move the part under Z 0, and
the API calls - and every operation they create has to compute, with the holes drilled at
their centres to their depths and the pocket and the finishing passes where they belong.
Prints PASS/FAIL lines and a SUMMARY.
"""

import importlib
import json
import os
import sys

import addon_utils
import bpy
import numpy as np

FABEX = "bl_ext.user_default.fabex"
BREPKIT = "bl_ext.user_default.brepkit"
use_extension = os.environ.get("FABEX_TEST_BREPKIT_EXTENSION") == "1"
LOC = np.array([0.1, 0.05, 0.0])

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

# --- the part ---------------------------------------------------------------------------------
solids = kernel.module("ops.solids")
boolean = kernel.module("ops.boolean").boolean
fillet_edges = kernel.module("ops.fillet").fillet_edges
TessOptions = kernel.module("tessellate.grid").TessOptions
tessellate_body = kernel.module("tessellate.body").tessellate_body

body = solids.box((0, 0, 0), (0.08, 0.06, 0.02))  # z -10..10 mm
body, _ = boolean(body, solids.sphere((0.015, 0.0, 0.003), 0.011), "union")  # top at 14 mm
top_id = next(f.id for f in body.faces.values() if f.name == "Box.+Z")
ring = [
    e.id
    for e in body.edges.values()
    if e.is_shared
    and top_id in {c.face for c in e.coedges}
    and any("Sphere" in body.faces[c.face].name for c in e.coedges)
]
body, _ = fillet_edges(body, ring, 0.002)
HOLES = [((-0.03, -0.02), 0.003, -0.02, 0.04), ((-0.03, 0.02), 0.003, -0.02, 0.04)]
HOLES += [((-0.03, 0.0), 0.002, 0.002, 0.02)]  # blind: floor at z = 2 mm
for c, r, z0, h in HOLES:
    body, _ = boolean(body, solids.cylinder((*c, z0), (0, 0, 1), r, h), "difference")
body, _ = boolean(body, solids.box((-0.012, 0.0, 0.01), (0.012, 0.01, 0.01)), "difference")


def make_object(name, body, location):
    t = tessellate_body(body, TessOptions())
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in t.points], [], [list(map(int, p)) for p in t.polygons()])
    mesh.attributes.new("brep_face_id", "INT", "FACE").data.foreach_set(
        "value", np.asarray(t.face_ids, dtype=np.int32)
    )
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob["brepkit_data"] = kernel.dumps_body(body)
    ob.location = tuple(location)
    bpy.context.view_layer.update()
    return ob


ob = make_object("Part", body, LOC)

# --- what an agent sees ------------------------------------------------------------------------
d = fx.describe()
names = {f["name"] for f in d["functions"]}
check(
    "API >= 0.7 lists brep_info",
    tuple(map(int, d["api_version"].split("."))) >= (0, 7) and "brep_info" in names,
)
check("quick start begins with brep_info", "brep_info" in d["quickstart_brep"][1])
check(
    "list_brep_objects points at brep_info",
    "brep_info" in next(f for f in d["functions"] if f["name"] == "list_brep_objects")["doc"],
)

info = json.loads(fx.call_json("brep_info", obj="Part"))
check("brep_info through call_json", info["ok"], info.get("error"))
info = info["result"]
roles = info["counts"]["role"]
check(
    "roles",
    roles
    == {
        "wall": 8,
        "bottom": 1,
        "top": 1,
        "curved": 1,
        "fillet": 1,
        "hole_wall": 3,
        "floor": 2,
    },
    roles,
)
faces = {f["face"]: f for f in info["faces"]}
dome = next(f for f in info["faces"] if f["role"] == "curved")
fil = next(f for f in info["faces"] if f["role"] == "fillet")
check(
    "dome: sphere R11, convex; fillet: torus r 2, concave",
    dome["surface"]["type"] == "sphere"
    and abs(dome["surface"]["radius"] - 0.011) < 1e-9
    and np.allclose(dome["surface"]["center"], LOC + (0.015, 0, 0.003), atol=1e-6)
    and fil["surface"]["type"] == "torus"
    and abs(fil["surface"]["minor_radius"] - 0.002) < 1e-6
    and fil["curvature"] in ("concave", "saddle"),
    (dome["surface"], fil["surface"]),
)
holes = info["holes"]
kinds = [
    (h["kind"], h["circular"], h["prismatic"], h["diameter"] and round(h["diameter"], 6))
    for h in holes
]
check(
    "holes: two 6 mm through, a 4 mm blind, a prismatic pocket",
    sorted(kinds, key=str)
    == sorted(
        [
            ("through", True, True, 0.006),
            ("through", True, True, 0.006),
            ("blind", True, True, 0.004),
            ("blind", False, True, None),
        ],
        key=str,
    )
    and [h["index"] for h in holes] == list(range(4)),
    kinds,
)
check("the dome is a boss", len(info["bosses"]) == 1, info["bosses"])
pocket = next(h for h in holes if not h["circular"])
check(
    "pocket: 12 x 10 mm, 5 mm deep, walls + floor",
    np.allclose(pocket["size_xy"], [0.012, 0.01], atol=1e-6)
    and abs(pocket["depth"] - 0.005) < 1e-6
    and sorted(faces[f]["role"] for f in pocket["faces"]) == ["floor"] + ["wall"] * 4,
    pocket,
)
check(
    "placement: top 14 mm above Z 0, with a warning",
    abs(info["placement"]["top_z"] - 0.014) < 1e-9
    and not info["placement"]["below_depth_start"]
    and any("above Z 0" in w for w in info["warnings"]),
    info["placement"],
)
lim = info["limits"]
check(
    "limits: 4 mm ball (the fillet and the blind hole), sharp pocket corners",
    abs(lim["max_ball_diameter"] - 0.004) < 1e-6 and lim["sharp_corners"] > 0,
    lim,
)
steps = [s["step"] for s in info["suggestions"]]
check(
    "suggested sequence",
    steps == ["place", "rough", "drill", "drill", "pocket", "finish", "cut_out"],
    steps,
)
finish = next(s for s in info["suggestions"] if s["step"] == "finish")
check(
    "finish: the dome and its fillet with a 3 mm ball",
    sorted(finish["faces"]) == sorted([dome["face"], fil["face"]])
    and finish["cutter_diameter"] == 0.003,
    finish,
)
again = fx.brep_info("Part")
check("cached", again["cached"] and again["body_hash"] == info["body_hash"])
only = fx.brep_info("Part", faces=[dome["face"]], edges=True)
check(
    "faces filter + edges",
    [f["face"] for f in only["faces"]] == [dome["face"]]
    and only["edges"]
    and all(dome["face"] in e["faces"] for e in only["edges"])
    and "edges" not in info,
)

# --- follow the suggestions literally ----------------------------------------------------------
ops_before = len(bpy.context.scene.cam_operations)
for s in info["suggestions"]:
    if "python" in s:
        exec(s["python"], {"bpy": bpy})
        bpy.context.view_layer.update()
    for call in s.get("calls", []):
        try:
            eval("fx." + call)
        except Exception as e:
            check(f"{s['step']}: {call}", False, f"{type(e).__name__}: {e}")
moved = fx.brep_info("Part")
check(
    "placed: the part's top at Z 0, no warning",
    abs(moved["placement"]["top_z"]) < 1e-7
    and moved["placement"]["below_depth_start"]
    and not any("above Z 0" in w for w in moved["warnings"])
    and moved["suggestions"][0]["step"] != "place",
    moved["placement"],
)
ops = list(bpy.context.scene.cam_operations)[ops_before:]
check(
    "operations made",
    [o.strategy for o in ops] == ["PARALLEL", "DRILL", "DRILL", "POCKET", "FLOWLINE", "CUTOUT"],
    [(o.name, o.strategy) for o in ops],
)
DZ = ob.location.z - LOC[2]  # how far the "place" step moved the part
check("moved down by exactly its top (14 mm)", abs(DZ + 0.014) < 1e-9, DZ)


def path_points(o):
    pob = bpy.data.objects[o.path_object_name]
    co = np.empty(len(pob.data.vertices) * 3)
    pob.data.vertices.foreach_get("co", co)
    M = np.array(pob.matrix_world)
    return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


computed = {}
for o in ops:
    r = fx.compute_operation(o.name, export=False)
    computed[o.name] = r
    check(f"computes: {o.name}", r["ok"], (r["message"], r["warnings"][:3]))

world = lambda xy: LOC[:2] + np.asarray(xy)  # noqa: E731
for o in ops:
    if not computed[o.name]["ok"]:
        continue
    P = path_points(o)
    if o.strategy == "DRILL":
        mine = [h for h in HOLES if abs(2 * h[1] - o.cutter_diameter) < 1e-6]
        ok = len(mine) > 0
        for c, r, z0, hgt in mine:
            near = np.linalg.norm(P[:, :2] - world(c), axis=1) < 1e-5
            bottom = (-0.01 if z0 < -0.01 else z0) + DZ
            ok &= near.any() and abs(P[near, 2].min() - bottom) < 1e-5
        check(f"{o.name}: plunges at each hole centre to its depth", ok)
    elif o.strategy == "POCKET":
        cut = P[P[:, 2] < -1e-6]
        lo, hi = cut.min(0), cut.max(0)
        rad = o.cutter_diameter / 2
        c = world((-0.012, 0.0))
        check(
            f"{o.name}: inside the pocket, down to its floor",
            np.all(lo[:2] >= c - (0.006, 0.005) + rad - 1e-5)
            and np.all(hi[:2] <= c + (0.006, 0.005) - rad + 1e-5)
            and abs(lo[2] - (0.005 + DZ)) < 1e-5,
            (lo.round(5).tolist(), hi.round(5).tolist()),
        )
    elif o.strategy == "FLOWLINE":
        rep = computed[o.name]["brep_flowline"]
        cut = P[P[:, 2] < -1e-6]
        c = world((0.015, 0.0))
        reach = np.linalg.norm(cut[:, :2] - c, axis=1).max()
        check(
            f"{o.name}: passes on the dome and its fillet only",
            rep and rep[0]["passes"] > 0 and reach < 0.0085 + 0.002 + 0.0015 + 1e-3,
            (rep and rep[0]["passes"], reach),
        )
    elif o.strategy == "CUTOUT":
        cut = P[P[:, 2] < -1e-6]
        lo, hi = cut.min(0), cut.max(0)
        rad = o.cutter_diameter / 2
        check(
            f"{o.name}: round the outline, down to the bottom",
            np.allclose(lo[:2], LOC[:2] - (0.04 + rad, 0.03 + rad), atol=2e-4)
            and np.allclose(hi[:2], LOC[:2] + (0.04 + rad, 0.03 + rad), atol=2e-4)
            and abs(lo[2] - (-0.01 + DZ)) < 1e-4,
            (lo.round(4).tolist(), hi.round(4).tolist()),
        )
    elif o.strategy == "PARALLEL":
        check(
            f"{o.name}: stays above the part's bottom",
            P[:, 2].min() >= -0.01 + DZ - 1e-6,
            P[:, 2].min(),
        )

# --- a pocket with fillets all round: walls profiled, the rest finished by Flowline ---------------
# 100 x 70 x 25 mm, standing on Z 0; a 50 x 30 x 12 mm pocket, every inside edge R4; two 5 mm
# through holes. Its pocket isn't 2.5D: the walls are profiled down to where they end (z 17
# mm before placing), the floor and fillets get Flowline.
bb = solids.box((0, 0, 0.0125), (0.1, 0.07, 0.025))
bb, _ = boolean(bb, solids.box((0.01, 0.0, 0.028), (0.05, 0.03, 0.03)), "difference")
inner = [
    e.id
    for e in bb.edges.values()
    if e.is_shared and all(bb.faces[x.face].name.startswith("B.") for x in e.coedges)
]
bb, _ = fillet_edges(bb, inner, 0.004)
for y in (-0.025, 0.025):
    bb, _ = boolean(bb, solids.cylinder((-0.035, y, -0.01), (0, 0, 1), 0.0025, 0.05), "difference")
BLOC = np.array([0.3, 0.0, 0.0])
br = make_object("Bracket", bb, BLOC)
bi = fx.brep_info("Bracket")
pk = next(h for h in bi["holes"] if not h["circular"])
check(
    "filleted pocket: 12 mm deep to its floor, walls end 8 mm down, not 2.5D",
    pk["kind"] == "blind"
    and not pk["prismatic"]
    and abs(pk["depth"] - 0.012) < 1e-6
    and abs(pk["z_bottom"] - 0.013) < 1e-6
    and abs(pk["walls_bottom"] - 0.017) < 1e-6
    and len(pk["faces"]) == 17,
    {k: pk[k] for k in ("kind", "prismatic", "depth", "z_bottom", "walls_bottom")},
)
steps = [s["step"] for s in bi["suggestions"]]
check(
    "bracket sequence: walls profiled instead of pocketed",
    steps == ["place", "rough", "drill", "walls", "finish", "cut_out"],
    steps,
)
bfaces = {f["face"]: f for f in bi["faces"]}
fin = next(s for s in bi["suggestions"] if s["step"] == "finish")
check(
    "finish: floor and floor fillets, no vertical faces; 6 mm ball (R4 fillets)",
    all(bfaces[f]["slope_range"][0] < 80 for f in fin["faces"])
    and any(bfaces[f]["role"] == "floor" for f in fin["faces"])
    and len(fin["faces"]) == 9
    and fin["cutter_diameter"] == 0.006,
    (fin["faces"], fin["cutter_diameter"]),
)
walls = next(s for s in bi["suggestions"] if s["step"] == "walls")
check(
    "walls: 6 mm end mill (R4 corners), down to the walls' end after placing",
    walls["cutter_diameter"] == 0.006 and "project=-0.008" in walls["calls"][0],
    walls["calls"],
)
before = len(bpy.context.scene.cam_operations)
for s in bi["suggestions"]:
    if "python" in s:
        exec(s["python"], {"bpy": bpy})
        bpy.context.view_layer.update()
    for call in s.get("calls", []):
        try:
            eval("fx." + call)
        except Exception as e:
            check(f"bracket {s['step']}: {call}", False, f"{type(e).__name__}: {e}")
bops = list(bpy.context.scene.cam_operations)[before:]
check(
    "bracket operations",
    [o.strategy for o in bops] == ["PARALLEL", "DRILL", "CUTOUT", "FLOWLINE", "CUTOUT"],
    [(o.name, o.strategy) for o in bops],
)
for o in bops:
    r = fx.compute_operation(o.name, export=False)
    check(
        f"bracket computes: {o.name}",
        r["ok"] and r["error"] is None,
        (r["message"], r["warnings"][:3]),
    )
    if not r["ok"]:
        continue
    P = path_points(o)
    if o.name.startswith("Bracket_walls"):
        cut = P[P[:, 2] < -1e-6]
        c = BLOC[:2] + (0.01, 0.0)
        rad = o.cutter_diameter / 2
        check(
            "walls profile: along the pocket walls, down to where they end",
            abs(cut[:, 2].min() - (0.017 - 0.025)) < 1e-5
            and np.allclose(cut[:, :2].max(0), c + (0.025 - rad, 0.015 - rad), atol=1e-4)
            and np.allclose(cut[:, :2].min(0), c - (0.025 - rad, 0.015 - rad), atol=1e-4),
            (cut.min(0).round(5).tolist(), cut.max(0).round(5).tolist()),
        )
    if o.strategy == "FLOWLINE":
        fit_faces = {p["face"] for rep in r["brep_fit"] for p in rep.get("problems", [])}
        check(
            "flowline fit check only looks at its own faces (no hole walls)",
            not any("Cylinder" in w for w in r["warnings"]) and fit_faces <= set(fin["faces"]),
            r["warnings"],
        )

# --- API fixes found by the unaided agent ---------------------------------------------------------
r = json.loads(
    fx.call_json("create_operation", obj_name="Bracket", strategy="PARALLEL", name="Named")
)
check(
    "call_json passes name= on to the function",
    r["ok"] and r["result"]["op"] == "Named",
    r.get("error"),
)
r = json.loads(fx.call_json(name="status"))
check("call_json(name=...) still names the function", r["ok"])
fx.extract_edges("Bracket", kind="holes", holes=[pk["index"]], project="top", name="Limit")
fx.set_operation_props("Named", {"use_limit_curve": True, "limit_curve": "Limit"})
r = fx.compute_operation("Named", export=False)
check("a limit curve computes (unary_union import)", r["ok"], r["message"])
props = fx.describe_operation_props()
paths = {p["path"]: p for p in props}
check(
    "describe_operation_props: units, enums, nested groups",
    paths["distance_between_paths"]["unit"] == "m"
    and paths["parallel_angle"]["unit"] == "radians"
    and "FLOWLINE" in [i["id"] for i in paths["strategy"]["items"]]
    and "MEANDER" in [i["id"] for i in paths["movement.type"]["items"]]
    and "brep.flow_max_slope" in paths
    and all(
        p["path"].startswith("movement.")
        or "movement" in (p["description"] or "").lower()
        or "movement" in p["name"].lower()
        for p in fx.describe_operation_props("movement")
    ),
    len(props),
)
fx.create_operation("Bracket", strategy="CURVE", name="Broken")  # needs a curve source
r = json.loads(fx.call_json("compute_operation", op="Broken", export=False))
check(
    "a failed compute: ok False with error COMPUTE_FAILED and Fabex's message",
    r["ok"]
    and not r["result"]["ok"]
    and r["result"]["error"]["code"] == "COMPUTE_FAILED"
    and "Curves" in r["result"]["error"]["message"],
    r["result"]["error"] if r["ok"] else r["error"],
)

# --- errors and the hole selector ----------------------------------------------------------------
cube = bpy.data.objects.new("Cube", bpy.data.meshes.new("Cube"))
bpy.context.scene.collection.objects.link(cube)
for kw, code in (
    (dict(name="brep_info", obj="Nope"), "OBJECT_NOT_FOUND"),
    (dict(name="brep_info", obj="Cube"), "NOT_A_BREP"),
    (dict(name="brep_info", obj="Part", faces=[999]), "INVALID_VALUE"),
    (dict(name="brep_info", obj="Part", tolerance=0), "INVALID_VALUE"),
    (dict(name="extract_edges", obj="Part", kind="holes", holes=[9]), "INVALID_VALUE"),
    (dict(name="extract_edges", obj="Part", kind="creases", holes=[0]), "INVALID_VALUE"),
):
    r = json.loads(fx.call_json(**kw))
    check(f"error {code}: {kw}", not r["ok"] and r["error"]["code"] == code, r.get("error"))
r = json.loads(
    fx.call_json("flowline_preview", obj="Part", stepover=0.00002, step_along=0.00001, create=False)
)
check(
    "TOO_LARGE says how much to scale the steps",
    not r["ok"]
    and r["error"]["code"] == "TOO_LARGE"
    and r["error"]["details"]["scale_steps_by"] > 1,
    r.get("error"),
)
r = fx.brep_holes("Part")
check("brep_holes numbers holes like brep_info", [h["index"] for h in r["holes"]] == [0, 1, 2, 3])

print(f"SUMMARY {sum(results)} / {len(results)}")
sys.exit(0 if all(results) else 1)
