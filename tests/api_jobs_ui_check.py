"""UI-mode check of the API's timer-driven compute jobs (needs a window, not -b).

    blender --factory-startup --window-geometry 0 0 640 480 -P api_jobs_ui_check.py

Checks start_compute / job_status / BUSY / cancel_job, then quits Blender.
Prints PASS/FAIL lines and a SUMMARY; the exit code is 0 when everything passed.
"""

import importlib
import os
import tempfile

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


try:
    addon_utils.enable(FABEX, default_set=True)
    fx = importlib.import_module(FABEX + ".api")
    out_dir = tempfile.mkdtemp(prefix="fabex_jobs_")

    # A 40 mm sphere with its top at Z=0, machined with parallel finishing
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=(0.03, 0.02, -0.02))
    bpy.context.active_object.name = "Ball"

    common = {"cutter_type": "BALLNOSE", "cutter_diameter": 0.004, "optimisation.use_exact": False}
    fx.create_operation(
        "Ball", "PARALLEL", name="Quick", props={**common, "distance_between_paths": 0.002}
    )
    fx.create_operation(
        "Ball",
        "PARALLEL",
        name="Slow",
        props={
            **common,
            "distance_between_paths": 0.0002,
            "distance_along_paths": 0.0001,
            "optimisation.pixsize": 0.00005,
        },
    )
except Exception as e:
    _fail_hard(e)

state = {"phase": "quick", "job": None, "ticks": 0, "progress_seen": False}


def poll():
    try:
        return _poll()
    except Exception as e:
        _fail_hard(e)


def _poll():
    state["ticks"] += 1
    if state["ticks"] > 3000:  # ~5 min safety net
        check("jobs finished in time", False)
        finish()

    if state["phase"] == "quick":
        if state["job"] is None:
            job = fx.start_compute("Quick", export=out_dir)
            state["job"] = job["job_id"]
            busy = fx.call_json("start_compute", op="Slow")
            check("second job while busy -> BUSY", '"BUSY"' in busy, busy[:120])
            return 0.1
        st = fx.job_status(state["job"])
        state["progress_seen"] |= st["progress"] is not None
        if st["state"] == "running":
            return 0.1
        r = st["result"] or {}
        check("quick job done", st["state"] == "done", st["state"])
        check("quick job result ok", r.get("ok"), r.get("message"))
        check("quick job wrote G-code", r.get("gcode_files"), r.get("gcode_files"))
        check("progress reported while running", state["progress_seen"])
        check("not busy after finishing", fx.status()["busy"] is None)
        state.update(phase="slow", job=None)
        return 0.1

    if state["phase"] == "slow":
        if state["job"] is None:
            state["job"] = fx.start_compute("Slow", export=False)["job_id"]
            state["started"] = state["ticks"]
            return 0.2
        st = fx.job_status(state["job"])
        if st["state"] == "running":
            if state["ticks"] - state["started"] == 3:
                c = fx.cancel_job(state["job"])
                check("cancel accepted", c["cancel_requested"], c)
            return 0.1
        check("slow job cancelled", st["state"] == "cancelled", (st["state"], st["result"]))
        check("not busy after cancel", fx.status()["busy"] is None)
        unknown = fx.call_json("job_status", job_id="nope")
        check("unknown job -> JOB_NOT_FOUND", '"JOB_NOT_FOUND"' in unknown)
        finish()
    return None


bpy.app.timers.register(poll, first_interval=1.0)
