"""Headless check of FANUC programs (milestone M8).

    FABEX_BREPKIT_PATH=~/Documents/Blender blender -b --factory-startup -P autocam_fanuc_check.py
    FABEX_TEST_BREPKIT_EXTENSION=1 ... (same, but with the BRepKit extension enabled)

A 60 x 40 x 10 mm bracket (top at Z 0) with two 5 mm through holes, a 4 mm blind hole and a
16 x 8 x 3 mm pocket. Through the agent API: machine profile and tool library, then
Pocket (T1), Drill 5 (T2), Drill 4 (T3) and a Cutout (T4) are computed and written as one
FANUC program. The program has to pass the checker, drill every hole at its B-rep centre to
its depth with canned cycles, and come out the same through a chain and Fabex's own chain
export operator (post processor FANUC), in the chain's order. Prints PASS/FAIL and a SUMMARY.
"""

import importlib
import os
import re
import sys
import tempfile

import addon_utils
import bpy
import numpy as np

FABEX = "bl_ext.user_default.fabex"
BREPKIT = "bl_ext.user_default.brepkit"
use_extension = os.environ.get("FABEX_TEST_BREPKIT_EXTENSION") == "1"
OUT = tempfile.mkdtemp(prefix="fabex_fanuc_")

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  ({info})" if info else ""))


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
fanuc_export = importlib.import_module(FABEX + ".autocam.fanuc_export")
bridge = importlib.import_module(FABEX + ".brep.bridge")
kernel = bridge.require_kernel()
check("kernel source", kernel.source == ("extension" if use_extension else "path"), kernel.source)

# --- the part ---------------------------------------------------------------------------------
solids = kernel.module("ops.solids")
boolean = kernel.module("ops.boolean").boolean
TessOptions = kernel.module("tessellate.grid").TessOptions
tessellate_body = kernel.module("tessellate.body").tessellate_body

body = solids.box((0, 0, 0), (0.06, 0.04, 0.01))  # z -5..5 mm
HOLES = [((-0.02, 0.01), 0.0025, -0.01, 0.03), ((0.02, 0.01), 0.0025, -0.01, 0.03)]
HOLES += [((0.0, -0.012), 0.002, 0.001, 0.01)]  # blind: floor 4 mm below the top
for c, r, z0, h in HOLES:
    body, _ = boolean(body, solids.cylinder((*c, z0), (0, 0, 1), r, h), "difference")
body, _ = boolean(body, solids.box((0.0, 0.01, 0.005), (0.016, 0.008, 0.006)), "difference")

t = tessellate_body(body, TessOptions())
mesh = bpy.data.meshes.new("Bracket")
mesh.from_pydata([tuple(p) for p in t.points], [], [list(map(int, p)) for p in t.polygons()])
mesh.attributes.new("brep_face_id", "INT", "FACE").data.foreach_set(
    "value", np.asarray(t.face_ids, dtype=np.int32)
)
ob = bpy.data.objects.new("Bracket", mesh)
bpy.context.scene.collection.objects.link(ob)
ob["brepkit_data"] = kernel.dumps_body(body)
ob.location = (0.1, 0.05, -0.005)  # top at Z 0
bpy.context.view_layer.update()

# --- what an agent sees ------------------------------------------------------------------------
d = fx.describe()
names = {f["name"] for f in d["functions"]}
check(
    "API 0.8 lists the machining-centre functions",
    tuple(map(int, d["api_version"].split("."))) >= (0, 8)
    and {"machine_profile", "set_machine", "tool_library", "set_tool_library", "create_chain"}
    | {"export_program", "check_gcode"}
    <= names
    and "FANUC" in d["post_processors"]
    and {"TOOL_CONFLICT", "MACHINE_LIMIT"} <= set(d["error_codes"]),
)

mp = fx.machine_profile()
check(
    "default profile: a Robodrill, Fabex not yet in step",
    "Robodrill" in mp["profile"]["name"] and not mp["synced"],
    mp["cam_machine"],
)
mp = fx.set_machine(preset="robodrill_d21mib5", profile={"program_number": 1001})
cm = bpy.context.scene.cam_machine
check(
    "set_machine syncs Fabex's machine (post, mm, travel, feed / spindle limits)",
    mp["synced"]
    and cm.post_processor == "FANUC"
    and cm.unit_system == "MILLIMETERS"
    and abs(cm.feedrate_max - 30) < 1e-6
    and abs(cm.spindle_max - 10000) < 1e-3
    and np.allclose(cm.working_area, (0.5, 0.4, 0.33)),
    mp["cam_machine"],
)
check("status names the machine", "Robodrill" in (fx.status()["machine"] or ""))
ok, info = expect_error("INVALID_PROPERTY", fx.set_machine, profile={"spindel_max": 2})
check("set_machine: a typo lists close matches", ok, info)
ok, info = expect_error("INVALID_VALUE", fx.set_machine, preset="haas")
check("set_machine: unknown preset", ok, info)

lib = fx.tool_library()
check(
    "default tool library from the presets + drills",
    lib["source"] == "default"
    and lib["count"] > 50
    and any(t["id"] == "EM6" for t in lib["tools"])
    and any(t["id"] == "DR6.8" for t in lib["tools"]),
    lib["count"],
)
check(
    "tool_library(type='BALL')", {t["type"] for t in fx.tool_library("BALL")["tools"]} == {"BALL"}
)
turret = [{"id": "EM8", "type": "END", "diameter": 0.008, "number": 1}]
r = fx.set_tool_library(turret)
check("set_tool_library: a turret map", r["source"] == "user" and r["tools"][0]["number"] == 1)
ok, info = expect_error(
    "TOOL_CONFLICT", fx.set_tool_library, turret + [dict(turret[0], id="EM10", diameter=0.01)]
)
check("set_tool_library: two tools in one pocket", ok, info)
check(
    "set_tool_library(None) goes back to the catalogue",
    fx.set_tool_library(None)["source"] == "default",
)

# --- operations -------------------------------------------------------------------------------
holes = fx.brep_holes("Bracket")["holes"]
pocket = next((h for h in holes if not h["circular"]), None)
d5 = [h for h in holes if h["circular"] and abs(h["diameter"] - 0.005) < 1e-6]
d4 = [h for h in holes if h["circular"] and abs(h["diameter"] - 0.004) < 1e-6]
check(
    "holes: 2 x 5 mm through, 1 x 4 mm blind, the pocket",
    len(holes) == 4 and len(d5) == 2 and len(d4) == 1 and pocket is not None,
    [(h["kind"], h["diameter"]) for h in holes],
)

MILL = {"feedrate": 1.2, "plunge_feedrate": 25, "spindle_rpm": 10000, "use_layers": True}
DRILL = {"feedrate": 0.4, "plunge_feedrate": 100, "spindle_rpm": 6000, "min_z_from": "OBJECT"}
fx.extract_edges(
    "Bracket", kind="holes", holes=[pocket["index"]], project="bottom", name="PocketCurve"
)
fx.create_operation(
    "PocketCurve",
    strategy="POCKET",
    name="Pocket",
    props={**MILL, "cutter_type": "END", "cutter_diameter": 0.004, "cutter_id": 1,
           "cutter_flutes": 3, "min_z_from": "OBJECT", "stepdown": 0.0015},
)  # fmt: skip
fx.extract_edges(
    "Bracket", kind="holes", holes=[h["index"] for h in d5], project="bottom", name="Holes5"
)
fx.create_operation(
    "Holes5",
    strategy="DRILL",
    name="Drill 5",
    props={**DRILL, "cutter_diameter": 0.005, "cutter_id": 2, "cutter_flutes": 2},
)
fx.extract_edges(
    "Bracket", kind="holes", holes=[h["index"] for h in d4], project="bottom", name="Holes4"
)
fx.create_operation(
    "Holes4",
    strategy="DRILL",
    name="Drill 4",
    props={**DRILL, "cutter_diameter": 0.004, "cutter_id": 3, "cutter_flutes": 2},
)
fx.create_operation(
    "Bracket",
    strategy="CUTOUT",
    name="Cutout",
    props={**MILL, "cut_type": "OUTSIDE", "cutter_type": "END", "cutter_diameter": 0.006,
           "cutter_id": 4, "cutter_flutes": 3, "min_z_from": "OBJECT", "stepdown": 0.003},
)  # fmt: skip
o_cut = next(o for o in bpy.context.scene.cam_operations if o.name == "Cutout")
check(
    "operations get machining-centre settings (coolant FLOOD, AICC)",
    o_cut.autocam.coolant == "FLOOD" and o_cut.autocam.aicc,
)
fx.set_operation_props("Drill 4", {"autocam.r_clearance": 0.001})

ok, info = expect_error("PATH_NOT_FOUND", fx.export_program, ["Pocket"], directory=OUT)
check("export_program before computing: PATH_NOT_FOUND", ok, info)
OPS = ["Pocket", "Drill 5", "Drill 4", "Cutout"]
for name in OPS:
    r = fx.compute_operation(name, export=False)
    check(f"compute {name}", r["ok"], r["message"])

# --- the program ------------------------------------------------------------------------------
rep = fx.export_program(OPS, directory=OUT, title="Bracket OP1")
text = open(rep["file"]).read()
check(
    "program written",
    os.path.isfile(rep["file"]) and rep["file"].endswith("BracketOP1.nc"),
    rep["file"],
)
check("program passes the checker", rep["check"]["ok"], rep["check"]["errors"][:3])
check("no checker warnings", rep["check"]["warnings"] == [], rep["check"]["warnings"][:3])
check(
    "O1001, four tools T1-T4",
    text.startswith("%\nO1001 (BRACKET OP1)\n") and rep["check"]["tools"] == [1, 2, 3, 4],
)
check(
    "four tool changes with matching length offsets",
    all(f"T{t} M06\n" in text and f"G43 H{t} Z" in text for t in range(1, 5)),
)
ops = {op["name"]: op for op in rep["operations"]}
check(
    "report: 3 holes, time, size",
    ops["Drill 5"]["holes"] == 2
    and ops["Drill 4"]["holes"] == 1
    and rep["minutes"] > 0
    and rep["bytes"] == len(text),
)
check("same answer from check_gcode", fx.check_gcode(path=rep["file"])["ok"])

# canned cycles at the B-rep holes
cycles = re.findall(r"^(G98 )?(G7[3]|G8[1-3])(.*)$", text, re.M)
check(
    "Drill 5 (2 x D deep): G73 chip breaking, 1 x D pecks",
    any(c[1] == "G73" and "Q5." in c[2] for c in cycles),
    cycles,
)
check("Drill 4 (1 x D deep): G81", any(c[1] == "G81" for c in cycles), cycles)


def drilled(text):
    """(x, y, z) of every hole the canned cycles drill (mm), and R per cycle."""
    out, rs = [], []
    state = {"X": None, "Y": None, "Z": None, "R": None}
    active = False
    for line in text.splitlines():
        words = dict(re.findall(r"([XYZR])(-?[0-9.]+)", line))
        if re.search(r"G(73|81|82|83)\b", line):
            active = True
        if line.startswith("G80"):
            active = False
            continue
        if active and words:
            for k, v in words.items():
                state[k] = float(v)
            out.append((state["X"], state["Y"], state["Z"]))
            rs.append(state["R"])
    return out, rs


got, rs = drilled(text)
ok = len(got) == 3
for h in d5 + d4:
    want = (h["center"][0] * 1000, h["center"][1] * 1000, h["z_bottom"] * 1000)
    ok &= any(np.allclose(g, want, atol=2e-3) for g in got)
check(
    "every hole drilled at its B-rep centre to its depth", ok, (got, [h["center"] for h in d5 + d4])
)
check("R planes: 2 mm and 1 mm above the stock top", sorted(set(rs)) == [1.0, 2.0], rs)
check(
    "rapids never move across while going down",
    not re.search(r"^G0 X[-0-9.]+ Y[-0-9.]+ Z", text, re.M),
)
check(
    "AICC on for milling, off before cycles",
    "G05.1 Q1" in text and text.count("G05.1 Q1") == text.count("G05.1 Q0"),
)

# --- tool conflict ----------------------------------------------------------------------------
fx.set_operation_props("Drill 4", {"cutter_id": 2})
ok, info = expect_error("TOOL_CONFLICT", fx.export_program, OPS, directory=OUT)
check("one T number for two drills: TOOL_CONFLICT", ok, info)
fx.set_operation_props("Drill 4", {"cutter_id": 3})

# --- through a chain, in the chain's order, and Fabex's own chain export -----------------------
order = ["Drill 5", "Drill 4", "Pocket", "Cutout"]
r = fx.create_chain("OP1", order)
check("create_chain keeps the order", r["operations"] == order, r)
rep2 = fx.export_program(chain="OP1", directory=OUT, program_number=1001)
text2 = open(rep2["file"]).read()
comments = re.findall(r"^\((DRILL 5|DRILL 4|POCKET|CUTOUT)\b", text2, re.M)
check(
    "chain program in the chain's order",
    comments == ["DRILL 5", "DRILL 4", "POCKET", "CUTOUT"],
    comments,
)

bpy.context.scene.cam_names.default_export_location = OUT + os.sep
ret = bpy.ops.object.cam_export_paths_chain()
op_file = os.path.join(OUT, "OP1.nc")
check(
    "Fabex's chain export operator writes FANUC",
    ret == {"FINISHED"} and os.path.isfile(op_file),
    ret,
)
text3 = open(op_file).read() if os.path.isfile(op_file) else ""
check(
    "operator and API write the same program (chain order fixed in Fabex too)",
    re.sub(r"^O\d+.*$", "", text3, count=1, flags=re.M)
    == re.sub(r"^O\d+.*$", "", text2, count=1, flags=re.M),
)
check("operator report kept for the API", fanuc_export.last_report["check"]["ok"])

# --- other post processors are untouched ------------------------------------------------------
cm.post_processor = "ISO"
r = fx.export_gcode("Pocket", OUT)
check("ISO export still works", r["files"] and r["files"][0].endswith(".tap"), r["files"])
cm.post_processor = "FANUC"

ok, info = expect_error("INVALID_ARGUMENT", fx.export_program, OPS, chain="OP1", directory=OUT)
check("export_program: operations or chain, not both", ok, info)
ok, info = expect_error("INVALID_ARGUMENT", fx.check_gcode)
check("check_gcode needs a path or text", ok, info)

passed = sum(results)
print(f"SUMMARY: {passed} / {len(results)} passed, {len(results) - passed} fails")
sys.exit(0 if passed == len(results) else 1)
