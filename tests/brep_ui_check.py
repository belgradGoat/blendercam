"""UI-mode check of the B-rep panel helper (needs a window, not -b).

    FABEX_BREPKIT_PATH=~/Documents/Blender \
        blender --factory-startup --window-geometry 0 0 800 600 -P brep_ui_check.py

Bad icon names or layout calls only fail while Blender really draws, so this draws
brep.ui.draw_ocl_mesh_settings into the 3D view header (a draw function that is always
drawn) for each state of the settings, and checks it ran without errors. The same header
also draws the Curve Tools panel, the Fabex menu and the "B-rep Edges to Curves" dialog
(both kinds of layout), which list the B-rep operator, and the cutter fit readout
(brep.ui.draw_fit_check) before a check, for a cutter that fits, a stale result, a cutter
that doesn't fit (a dish with a too big ball) and without the kernel, and the Flowline
settings (brep.ui.draw_flowline_settings) with the Operation Setup panel of a FLOWLINE op.
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

    fit = importlib.import_module(FABEX + ".brep.fit")
    solids = kernel.module("ops.solids")
    dish_body, _ = kernel.module("ops.boolean").boolean(
        solids.box((0, 0, 0), (0.04, 0.04, 0.02)), solids.sphere((0, 0, 0.01), 0.008), "difference"
    )
    tess = kernel.module("tessellate.body").tessellate_body(dish_body)
    mesh = bpy.data.meshes.new("Dish")
    mesh.from_pydata(
        [tuple(p) for p in tess.points], [], [list(map(int, p)) for p in tess.polygons()]
    )
    dish = bpy.data.objects.new("Dish", mesh)
    bpy.context.scene.collection.objects.link(dish)
    dish["brepkit_data"] = kernel.dumps_body(dish_body)
    dish.location = (0.1, 0.0, -0.01)
    fx.create_operation(
        "Dish", "PARALLEL", name="Dish", props={"cutter_type": "BALLNOSE", "cutter_diameter": 0.02}
    )
    o2 = bpy.context.scene.cam_operations["Dish"]
    fx.create_flowline_operation("Ball", name="Flow", stepover=0.002, step_along=0.001)
    o3 = bpy.context.scene.cam_operations["Flow"]
    r3 = fx.compute_operation("Flow", export=False)
    check("flowline computes", r3["ok"] and r3["brep_flowline"][0]["passes"] > 0, r3["message"])
    ops_panel = importlib.import_module(FABEX + ".ui.panels.op_properties_panel")

    from types import SimpleNamespace

    curve_panel = importlib.import_module(FABEX + ".ui.panels.curve_tools_panel")
    menus = importlib.import_module(FABEX + ".ui.menus.viewport")
    brep_ops = importlib.import_module(FABEX + ".brep.operators")
    bpy.context.view_layer.objects.active = ob

    def header_draw(self, context):
        try:
            brep_ui.draw_ocl_mesh_settings(self.layout, o)
            state["drawn"].add(state["variant"])
        except Exception as e:  # recorded, checked below
            state["errors"].append((state["variant"], repr(e)))
        try:
            box = self.layout.row()
            curve_panel.VIEW3D_PT_tools_curvetools.draw(SimpleNamespace(layout=box), context)
            menus.Fabex_SubMenu.draw(SimpleNamespace(layout=box), context)
            for kind in ("CREASES", "HOLES"):
                props = box.operator("object.cam_brep_extract_edges")
                props.kind = kind
                brep_ops.CamBrepExtractEdges.draw(
                    SimpleNamespace(layout=box.column(), properties=props), context
                )
            state["drawn"].add("edges")
        except Exception as e:
            state["errors"].append(("edges", repr(e)))
        try:
            col = self.layout.column()
            brep_ui.draw_fit_check(col, o)
            brep_ui.draw_fit_check(col, o2)
            state["drawn"].add(f"fit {state['variant']}")
        except Exception as e:
            state["errors"].append((f"fit {state['variant']}", repr(e)))
        try:
            col = self.layout.column()
            brep_ui.draw_flowline_settings(col, o3)
            brep_ui.draw_flowline_settings(col, o)  # not a Flowline op: still draws
            ops_panel.CAM_OPERATION_PROPERTIES_Panel.draw(
                SimpleNamespace(layout=self.layout.column(), op=o3, level=0),
                context,  # 0: no sub-panels (not in headers)
            )
            state["drawn"].add(f"flow {state['variant']}")
        except Exception as e:
            state["errors"].append((f"flow {state['variant']}", repr(e)))

    bpy.types.VIEW3D_HT_header.append(header_draw)
except Exception as e:
    _fail_hard(e)


real_find_kernel = bridge.find_kernel


def set_variant(name):
    bridge.find_kernel = real_find_kernel
    o.optimisation.use_opencamlib = True
    o.brep.use_exact_mesh = True
    o3.brep.flow_faces = "1" if name in ("fit ok", "ocl off") else ""
    if name == "fit ok":
        reps = fit.check_operation(o)
        check("fit: 6 mm ball fits the ball", reps[0]["fits"] is True, reps[0]["messages"])
    elif name == "fit stale":
        o.cutter_diameter = 0.008
    elif name == "fit misfit":
        o.cutter_diameter = 0.006
        reps = fit.check_operation(o2)
        check(
            "fit: 20 mm ball doesn't fit the R8 dish", reps[0]["fits"] is False, reps[0]["messages"]
        )
    elif name == "ocl off":
        o.optimisation.use_opencamlib = False
    elif name == "exact off":
        o.brep.use_exact_mesh = False
    elif name == "no kernel":
        bridge.find_kernel = lambda refresh=False: None


VARIANTS = ["exact", "fit ok", "fit stale", "fit misfit", "ocl off", "exact off", "no kernel"]
step = {"i": 0, "wait": 0}


def poll():
    try:
        if step["i"] >= len(VARIANTS):
            bridge.find_kernel = real_find_kernel
            check("edges-to-curves menu, panel entry and dialog drew", "edges" in state["drawn"])
            check(
                "operator is available for the B-rep object",
                bpy.ops.object.cam_brep_extract_edges.poll(),
            )
            check(
                "fit readout drew in every state",
                all("fit " + v in state["drawn"] for v in VARIANTS),
                sorted(map(str, state["drawn"])),
            )
            check("fit check operator is available", bpy.ops.object.cam_brep_fit_check.poll())
            check(
                "flowline settings and the operation panel drew in every state",
                all("flow " + v in state["drawn"] for v in VARIANTS),
            )
            bpy.context.scene.cam_active_operation = 2
            check(
                "flowline operators are available",
                bpy.ops.object.cam_brep_flowline_preview.poll()
                and bpy.ops.object.cam_brep_flow_faces.poll(),
            )
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
        done = all(k in state["drawn"] for k in (name, "fit " + name, "flow " + name))
        if done or step["wait"] > 25:
            check(f"panel helper drew: {name}", name in state["drawn"])
            step["i"] += 1
        return 0.1
    except Exception as e:
        _fail_hard(e)


bpy.app.timers.register(poll, first_interval=1.0)
