"""Drawing helpers for the machining-centre (FANUC) settings, used by Fabex's panels."""

from . import state


def draw_machine_box(layout, scene) -> None:
    """The machine profile, under the post processor, when the post is FANUC."""
    if scene.cam_machine.post_processor != "FANUC":
        return
    box = layout.box()
    try:
        p = state.machine(scene)
    except Exception as e:  # a hand-edited profile that no longer validates
        box.alert = True
        box.label(text=f"Machine profile: {e}", icon="ERROR")
        return
    col = box.column(align=True)
    col.label(text=p.name, icon="TOOL_SETTINGS")
    travel = " x ".join(f"{t * 1000:g}" for t in p.travel)
    col.label(text=f"Travel {travel} mm, {p.magazine} tools")
    col.label(text=f"Spindle {p.spindle_max:g} rpm, feed {p.feed_max:g} m/min")
    col.label(text=f"{p.work_offset}, AICC {'on' if p.aicc else 'off'}, {p.program_memory} kB")
    col.label(text="Change it with set_machine() (Fabex API)", icon="INFO")


def draw_autocam_settings(layout, o, scene) -> None:
    """Coolant, AICC and drilling cycle of an operation, when the post is FANUC."""
    if scene.cam_machine.post_processor != "FANUC":
        return
    a = o.autocam
    layout.use_property_split = False
    try:
        header, panel = layout.panel("fabex_machining_centre", default_closed=True)
        header.label(text="Machining Centre")
    except (RuntimeError, TypeError):  # not inside a panel: draw it open
        panel = layout.box()
        panel.label(text="Machining Centre")
    if not panel:
        return
    col = panel.column(align=True)
    col.use_property_split = True
    col.prop(o, "cutter_id", text="Tool Number (T)")
    col.prop(a, "coolant")
    col.prop(a, "aicc")
    if o.strategy == "DRILL":
        col.separator()
        col.prop(a, "drill_cycle")
        if a.drill_cycle in ("AUTO", "G73", "G83"):
            col.prop(a, "peck")
        if a.drill_cycle == "G82":
            col.prop(a, "dwell")
        if a.drill_cycle == "G84":
            col.prop(a, "tap_pitch")
        col.prop(a, "r_clearance")
    if a.plan_tag:
        col.label(text=f"From plan {a.plan_tag}", icon="INFO")


_plan_cache: dict = {"raw": None, "plan": None}

PHASE_ICONS = {
    "face": "MESH_PLANE",
    "rough": "MOD_REMESH",
    "spot": "PIVOT_CURSOR",
    "drill": "SNAP_FACE_CENTER",
    "floor": "SNAP_FACE",
    "walls": "MOD_SOLIDIFY",
}


def _stored_plan(scene):
    raw = scene.get("fabex_autocam_plan")
    if raw != _plan_cache["raw"]:
        import json

        try:
            plan = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            plan = None
        if plan:
            plan.pop("geometry", None)
        _plan_cache.update(raw=raw, plan=plan)
    return _plan_cache["plan"]


def draw_autocam_panel(layout, scene) -> None:
    """The Auto CAM panel: inputs, Plan / Apply / Verify and the last plan (no sub-panels,
    so it also draws in headers)."""
    a = scene.cam_autocam
    col = layout.column(align=True)
    try:
        col.label(text=state.machine(scene).name, icon="TOOL_SETTINGS")
    except Exception as e:
        col.label(text=f"Machine profile: {e}", icon="ERROR")
    col.prop(a, "part")
    col.prop(a, "material")
    col.separator()
    col.label(text="Stock")
    row = col.row(align=True)
    row.prop(a, "allowance_top")
    row.prop(a, "allowance_side")
    row.prop(a, "grip")
    col.prop(a, "origin")
    col.prop(a, "program_number")
    row = layout.row(align=True)
    row.scale_y = 1.3
    row.operator("object.cam_autocam_plan", icon="OUTLINER_OB_LIGHTPROBE")
    row.operator("object.cam_autocam_apply", icon="PLAY")
    row.operator("object.cam_autocam_verify", icon="CHECKMARK")

    plan = _stored_plan(scene)
    if not plan:
        layout.label(text="Pick a B-rep part and press Plan", icon="INFO")
        return
    box = layout.box()
    c = box.column(align=True)
    size = " x ".join(f"{x * 1000:g}" for x in plan["stock"]["size"])
    c.label(text=f"{plan['object']}: {plan['material']['name']}", icon="OBJECT_DATA")
    c.label(text=f"Stock {size} mm, {plan['placement']['work_offset']}")
    c.label(text=f"{len(plan['operations'])} operations, {len(plan['tools'])} tools")
    for op in plan["operations"]:
        c.label(
            text=f"{op['name']}  T{op.get('number', '?')} {op['tool']}",
            icon=PHASE_ICONS.get(op["phase"], "DOT"),
        )
    for u in plan["coverage"]["uncovered"][:6]:
        c.label(text=f"Face {u['face']} ({u['role']}): {u['reason']}", icon="INFO")
    if plan.get("rest"):
        r = plan["rest"][0]["radius_left"] * 1000
        c.label(
            text=f"{len(plan['rest'])} sharp inside corner(s) keep a {r:g} mm radius", icon="INFO"
        )
    for w in plan.get("warnings", [])[:4]:
        c.label(text=w, icon="ERROR")
    applied = plan.get("applied")
    if applied and applied.get("program"):
        import os

        c.separator()
        c.label(text=f"Program {os.path.basename(applied['program'])}", icon="FILE_TEXT")
    v = plan.get("verify")
    if v:
        unexpected = v["leftover"].get("unexpected", [])
        c.label(
            text=f"Verify: gouge {v['gouge']['max_mm']} mm, unplanned leftover on "
            f"{len(unexpected)} face(s), {len(v['collisions'])} collision(s)",
            icon="CHECKMARK" if v["ok"] and not unexpected else "ERROR",
        )
