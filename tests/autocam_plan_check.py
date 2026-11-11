"""Headless check of automatic CAM (milestone M9): plan_cam -> apply_plan -> program -> verify.

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P autocam_plan_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

Two parts, both modelled with BRepKit:
- the M8 bracket: 60 x 40 x 10 mm, two 5 mm through holes, a 4 mm flat-bottomed blind hole,
  a 16 x 8 x 3 mm pocket with sharp corners;
- a stepped block: 50 x 30 x 12 mm with a step 4 mm down along one side, a round 14 mm
  pocket 6 mm deep (milled: no such drill) and a 6.8 mm through hole in the step.

For each: the plan (tools, holes, coverage), applying it (placement, operations, chain,
paths, the FANUC program passing the checker) and the verification: no gouge, no crash,
only the leftover the plan said it would leave. Then re-applying (idempotent), an edited
plan, and a stale plan. Prints PASS/FAIL and a SUMMARY.
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
OUT = tempfile.mkdtemp(prefix="fabex_plan_")

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    text = info if isinstance(info, str) else json.dumps(info, default=str)[:600]
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  ({text})" if text else ""))


def expect_error(code, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:  # FabexAPIError
        return getattr(e, "code", None) == code, f"{getattr(e, 'code', e)}: {e}"
    return False, "no error"


addon_utils.enable(FABEX, default_set=True)
if use_extension:
    addon_utils.enable(BREPKIT, default_set=True)
else:
    addon_utils.disable(BREPKIT, default_set=True)

fx = importlib.import_module(FABEX + ".api")
bridge = importlib.import_module(FABEX + ".brep.bridge")
kernel = bridge.require_kernel()
check("kernel source", kernel.source == ("extension" if use_extension else "path"), kernel.source)

solids = kernel.module("ops.solids")
boolean = kernel.module("ops.boolean").boolean
TessOptions = kernel.module("tessellate.grid").TessOptions
tessellate_body = kernel.module("tessellate.body").tessellate_body


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
    ob.location = location
    bpy.context.view_layer.update()
    return ob


def bracket():
    body = solids.box((0, 0, 0), (0.06, 0.04, 0.01))  # z -5..5 mm
    for c, r, z0, h in (((-0.02, 0.01), 0.0025, -0.01, 0.03), ((0.02, 0.01), 0.0025, -0.01, 0.03),
                        ((0.0, -0.012), 0.002, 0.001, 0.01)):  # fmt: skip
        body, _ = boolean(body, solids.cylinder((*c, z0), (0, 0, 1), r, h), "difference")
    body, _ = boolean(body, solids.box((0.0, 0.01, 0.005), (0.016, 0.008, 0.006)), "difference")
    return make_object("Bracket", body, (0.1, 0.05, -0.005))


def stepped():
    """50 x 30 x 12 (z -6..6 mm); a step along one side (y > 7 mm) 4 mm down; a round 14 mm
    pocket 6 mm deep with a flat floor (no drill that size: milled); a 6.8 mm through hole
    in the step."""
    body = solids.box((0, 0, 0), (0.05, 0.03, 0.012))
    body, _ = boolean(body, solids.box((0, 0.0115, 0.006), (0.06, 0.009, 0.008)), "difference")
    body, _ = boolean(
        body, solids.cylinder((-0.008, -0.004, 0.0), (0, 0, 1), 0.007, 0.02), "difference"
    )
    body, _ = boolean(
        body, solids.cylinder((0.012, 0.011, -0.01), (0, 0, 1), 0.0034, 0.03), "difference"
    )
    return make_object("Stepped", body, (-0.2, 0.1, 0.03))


fx.set_machine(preset="robodrill_d21mib5")
d = fx.describe()
names = {f["name"] for f in d["functions"]}
check(
    "API 0.9 lists the automatic CAM functions",
    tuple(map(int, d["api_version"].split("."))) >= (0, 9)
    and {"cutting_data", "plan_cam", "apply_plan", "verify_program"} <= names
    and {"PLAN_INVALID", "PLAN_STALE"} <= set(d["error_codes"])
    and "quickstart_automatic_cam" in d,
)
cd = fx.cutting_data("EM10", "6061", "pocket")
check(
    "cutting_data: EM10 pocketing 6061 on a 10k spindle",
    abs(cd["rpm"] - 10000) < 1e-6 and abs(cd["feed"] - 1.8) < 1e-6 and abs(cd["ap"] - 0.005) < 1e-9,
    cd,
)
ok, info = expect_error("INVALID_VALUE", fx.cutting_data, "EM11", "6061", "rough")
check("cutting_data: unknown tool", ok, info)


def run_part(ob, expect):
    name = ob.name
    print(f"\n=== {name} ===")
    loc0 = np.array(ob.matrix_world.translation)
    n_ops0 = len(bpy.context.scene.cam_operations)
    info = fx.brep_info(name)
    check(f"{name}: brep_info reports hole bottoms",
          all("bottom" in h for h in info["holes"]), [(h["kind"], h["bottom"]) for h in info["holes"]])  # fmt: skip
    plan = fx.plan_cam(name, material="6061")
    s = plan
    check(f"{name}: plan_cam changes nothing",
          len(bpy.context.scene.cam_operations) == n_ops0
          and np.allclose(ob.matrix_world.translation, loc0) and "geometry" not in s)  # fmt: skip
    tools = sorted(t["id"] for t in s["tools"])
    check(f"{name}: tools {expect['tools']}", tools == sorted(expect["tools"]), tools)
    actions = sorted((r["action"], round((r["diameter"] or 0) * 1000, 3)) for r in s["holes"])
    check(f"{name}: holes {expect['holes']}", actions == sorted(expect["holes"]), actions)
    roles = sorted(u["role"] for u in s["coverage"]["uncovered"])
    check(f"{name}: only {expect['uncovered']} uncovered", roles == sorted(expect["uncovered"]), s["coverage"]["uncovered"])  # fmt: skip
    for op in s["operations"]:
        print(
            f"    {op['name']:24s} {op['strategy']:7s} T{op['number']} {op['tool']:6s} {op['why']}"
        )
    for w in s["warnings"]:
        print("    warning:", w)

    r = fx.apply_plan(directory=OUT)
    for c in r["computed"]:
        if not c["ok"]:
            print("    compute failed:", c)
    check(f"{name}: every path computed", r["computed"] and all(c["ok"] for c in r["computed"]),
          [c for c in r["computed"] if not c["ok"]])  # fmt: skip
    ops = [o for o in bpy.context.scene.cam_operations if o.autocam.plan_tag == s["tag"]]
    check(f"{name}: operations and chain made", len(ops) == len(s["operations"])
          and any(c.name == f"{name} OP1" for c in bpy.context.scene.cam_chains), len(ops))  # fmt: skip
    T = np.array(s["placement"]["translation"])
    check(f"{name}: part moved to the plan's placement",
          np.allclose(np.array(ob.matrix_world.translation), loc0 + T, atol=1e-9),
          r["placement"])  # fmt: skip
    prog = r["program"]
    if prog:
        print(f"    program {prog['file']}: {prog['lines']} lines, {prog['minutes']:.1f} min")
        for e in prog["check"]["errors"]:
            print("    check error:", e)
    check(f"{name}: program written and checked", prog is not None and prog["check"]["ok"],
          prog and prog["check"]["errors"])  # fmt: skip
    v = r["verify"]
    if v is None:
        check(f"{name}: verified", False, "no verify report")
        return s, r
    print(f"    verify: {v['seconds']} s, {v['moves']} moves, grid {v['grid']}, "
          f"gouge {v['gouge']['max_mm']} mm, leftover {v['leftover']['max_mm']} mm")  # fmt: skip
    for row in v["gouge"]["faces"]:
        print("    gouge:", row)
    for row in v["leftover"]["faces"]:
        print("    leftover:", row)
    for row in v["collisions"] + v["plunges"]:
        print("    ", row)
    for op in v["operations"]:
        print("    ", op)
    check(f"{name}: verify: no gouge", not v["gouge"]["faces"], v["gouge"])
    check(f"{name}: verify: no crash", not [c for c in v["collisions"]
          if c["kind"] in ("rapid", "holder", "fixture")], v["collisions"])  # fmt: skip
    check(f"{name}: verify: no plunge into material", not v["plunges"], v["plunges"])
    check(f"{name}: verify: only the leftover the plan expects", not v["leftover"]["unexpected"],
          v["leftover"]["unexpected"])  # fmt: skip
    check(f"{name}: apply ok", r["ok"])
    return s, r


plan_b, res_b = run_part(
    bracket(),
    {"tools": ["EM10", "EM4", "EM2", "SP6", "DR5"],
     "holes": [("mill", 0), ("drill", 5.0), ("drill", 5.0), ("mill", 4.0)],
     "uncovered": ["bottom"]},
)  # fmt: skip

# --- re-apply, edit, stale ----------------------------------------------------------------------
ob = bpy.data.objects["Bracket"]
loc = np.array(ob.matrix_world.translation)
n_objects = len(bpy.data.objects)
r2 = fx.apply_plan(directory=OUT, verify=False)
ops = [o for o in bpy.context.scene.cam_operations if o.autocam.plan_tag == plan_b["tag"]]
check(
    "re-applying replaces what the plan made (same operations, objects; part not moved)",
    len(ops) == len(plan_b["operations"])
    and len(bpy.data.objects) == n_objects
    and np.allclose(ob.matrix_world.translation, loc)
    and r2["placement"]["moved"] is False
    and len(r2["removed"]["operations"]) == len(plan_b["operations"]),
    {"objects": [n_objects, len(bpy.data.objects)], "removed": r2["removed"]},
)
v = fx.verify_program()
check("verify_program() from the applied plan", not v["gouge"]["faces"], v["leftover"]["max_mm"])

edited = json.loads(json.dumps(plan_b))
pocket_walls = next(
    op for op in edited["operations"] if op["phase"] == "walls" and op["tool"] == "EM4"
)
edited["operations"] = [op for op in edited["operations"] if op["id"] != pocket_walls["id"]]
r3 = fx.apply_plan(edited, directory=OUT)
unexpected = {row["face"]: row for row in r3["verify"]["leftover"]["unexpected"]}
check(
    "an edited plan (the pocket's walls pass dropped): the walls' skin shows as leftover",
    len(r3["operations"]) == len(plan_b["operations"]) - 1
    and set(pocket_walls["faces"]) & set(unexpected)
    and all(
        abs(unexpected[f]["max_mm"] - 0.2) < 0.06
        for f in set(pocket_walls["faces"]) & set(unexpected)
    ),
    r3["verify"]["leftover"]["unexpected"],
)
bad = dict(plan_b, operations=[dict(plan_b["operations"][0], id="nope")])
ok, info = expect_error("PLAN_INVALID", fx.apply_plan, bad)
check("apply_plan: operation without geometry -> PLAN_INVALID", ok, info)
fx.plan_cam("Bracket")  # placed now: the plan's translation is ~0
ob.location.x += 0.001
bpy.context.view_layer.update()
ok, info = expect_error("PLAN_STALE", fx.apply_plan)
check("apply_plan after moving the part -> PLAN_STALE", ok, info)
ob.location.x -= 0.001
bpy.context.view_layer.update()

plan_s, res_s = run_part(
    stepped(),
    {"tools": ["EM8", "SP6", "DR6.8"],
     "holes": [("drill", 6.8), ("mill", 14.0)],
     "uncovered": ["bottom"]},
)  # fmt: skip
if plan_s:
    floors = sorted(
        round(op["z_bottom"] * 1000, 3) for op in plan_s["operations"] if op["phase"] == "floor"
    )
    check("Stepped: the step and the pocket floor are finished", floors == [-6.0, -4.0], floors)
    check("Stepped: the M8 tap drill size is flagged", any("M8" in w for w in plan_s["warnings"]))
    spot = next(op for op in plan_s["operations"] if op["phase"] == "spot")
    check("Stepped: the 10 mm spot drill would hit the step's wall: a 6 mm one",
          spot["tool"] == "SP6" and "smaller" in spot["why"], spot["why"])  # fmt: skip
    check("Stepped: a round pocket: nothing left (no rest)", plan_s["rest"] == [], plan_s["rest"])

print("SUMMARY:", sum(results), "/", len(results), "passed,", len(results) - sum(results), "fails")
sys.stdout.flush()
