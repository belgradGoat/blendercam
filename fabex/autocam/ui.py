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
