"""UI-mode check of the B-rep panel helper (needs a window, not -b).

    FABEX_BREPKIT_PATH=~/Documents/Blender \
        blender --factory-startup --window-geometry 0 0 800 600 -P brep_ui_check.py

Bad icon names or layout calls only fail while Blender really draws, so this draws
brep.ui.draw_ocl_mesh_settings into the 3D view header (a draw function that is always
drawn) for each state of the settings, and checks it ran without errors.
Prints PASS/FAIL lines and a SUMMARY; the exit code is 0 when everything passed.
"""

import importlib
import os

import addon_utils
import bpy

FABEX = "bl_ext.user_default.fabex"
results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  ({info})" if info else ""), flush=True)


def finish():
    print(f"SUMMARY {sum(results)} / {len(results)}", flush=True)
    os._exit(0 if results and all(results) else 1)


def _fail_hard(exc):
    import traceback

    traceback.print_exception(exc)
    check("no unexpected exception", False, repr(exc))
    finish()  # never leave a Blender window hanging


state = {"variant": None, "drawn": set(), "errors": []}

try:
    addon_utils.enable(FABEX, default_set=True)
    fx = importlib.import_module(FABEX + ".api")
    bridge = importlib.import_module(FABEX + ".brep.bridge")
    brep_ui = importlib.import_module(FABEX + ".brep.ui")
    kernel = bridge.require_kernel()

    body = kernel.module("ops.solids").sphere((0.0, 0.0, 0.0), 0.015)
    tess = kernel.module("tessellate.body").tessellate_body(body)
    mesh = bpy.data.meshes.new("Ball")
    mesh.from_pydata(
        [tuple(p) for p in tess.points], [], [list(map(int, p)) for p in tess.polygons()]
    )
    ob = bpy.data.objects.new("Ball", mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob["brepkit_data"] = kernel.dumps_body(body)
    ob.location = (0.02, 0.01, -0.015)

    fx.create_operation(
        "Ball",
        "PARALLEL",
        name="Finish",
        props={
            "cutter_type": "BALLNOSE",
            "cutter_diameter": 0.006,
            "distance_between_paths": 0.002,
            "optimisation.use_exact": True,
            "optimisation.use_opencamlib": True,
        },
    )
    r = fx.compute_operation("Finish", export=False)
    check("compute with the exact mesh", r["ok"] and r["brep_meshes"][0]["mesh"] == "exact", r)
    o = bpy.context.scene.cam_operations[0]
    check("operation has B-rep sources", brep_ui.operation_brep_objects(o) == [ob])

    def header_draw(self, context):
        try:
            brep_ui.draw_ocl_mesh_settings(self.layout, o)
            state["drawn"].add(state["variant"])
        except Exception as e:  # recorded, checked below
            state["errors"].append((state["variant"], repr(e)))

    bpy.types.VIEW3D_HT_header.append(header_draw)
except Exception as e:
    _fail_hard(e)


real_find_kernel = bridge.find_kernel


def set_variant(name):
    bridge.find_kernel = real_find_kernel
    o.optimisation.use_opencamlib = True
    o.brep.use_exact_mesh = True
    if name == "ocl off":
        o.optimisation.use_opencamlib = False
    elif name == "exact off":
        o.brep.use_exact_mesh = False
    elif name == "no kernel":
        bridge.find_kernel = lambda refresh=False: None


VARIANTS = ["exact", "ocl off", "exact off", "no kernel"]
step = {"i": 0, "wait": 0}


def poll():
    try:
        if step["i"] >= len(VARIANTS):
            bridge.find_kernel = real_find_kernel
            check("no draw errors", not state["errors"], state["errors"])
            finish()
        name = VARIANTS[step["i"]]
        if state["variant"] != name:
            state["variant"] = name
            set_variant(name)
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
            step["wait"] = 0
            return 0.2
        step["wait"] += 1
        if name in state["drawn"] or step["wait"] > 25:
            check(f"panel helper drew: {name}", name in state["drawn"])
            step["i"] += 1
        return 0.1
    except Exception as e:
        _fail_hard(e)


bpy.app.timers.register(poll, first_interval=1.0)
