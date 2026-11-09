"""Fabex 'brep/ui.py'

Drawing helpers for B-rep settings, called from Fabex's panels (no classes registered here).
"""

import bpy

from ..utilities.simple_utils import unit_value_to_string
from . import bridge, fit, flow, ocl_mesh
from .bridge import operation_brep_objects


def draw_ocl_mesh_settings(layout, o) -> None:
    """Exact-mesh settings for an operation whose sources include B-rep objects."""
    if not operation_brep_objects(o):
        return
    col = layout.column(align=True)
    if not o.optimisation.use_opencamlib:
        col.label(text="B-rep: use OpenCAMLib to sample the exact surfaces", icon="INFO")
        return
    col.prop(o.brep, "use_exact_mesh")
    if not o.brep.use_exact_mesh:
        return
    row = col.row(align=True)
    row.use_property_split = True
    row.prop(o.brep, "mesh_tolerance", text="Tolerance")
    if bridge.find_kernel() is None:
        col.label(text="BRepKit not found - display mesh is used", icon="ERROR")
        return
    names = {ob.name for ob in operation_brep_objects(o)}
    for entry in ocl_mesh.report(o):
        # only this operation's current sources, and only stats for the current tolerance
        if entry["object"] not in names:
            continue
        if entry.get("mesh") == "exact" and entry.get("tolerance") != o.brep.mesh_tolerance:
            continue
        if entry.get("mesh") == "exact":
            dev = entry.get("max_deviation")
            text = f"{entry['object']}: {entry['triangles']:,} tris"
            if dev is not None:
                text += f", max dev {unit_value_to_string(dev, 4).strip()}"
            col.label(text=text, icon="CHECKMARK" if entry.get("within_tolerance") else "ERROR")
        else:
            col.label(text=f"{entry['object']}: display mesh ({entry.get('reason')})", icon="INFO")


def draw_fit_check(layout, o) -> None:
    """Cutter fit check for an operation whose sources include B-rep objects: the setting,
    a button, and the last check's result (if it matches the current cutter and B-rep)."""
    objects = operation_brep_objects(o)
    if not objects:
        return
    box = layout.box()
    col = box.column(align=True)
    row = col.row(align=True)
    row.use_property_split = False
    row.prop(o.brep, "check_fit", text="Check Fit on B-rep")
    row.operator("object.cam_brep_fit_check", text="", icon="VIEWZOOM")
    if o.strategy not in fit.SURFACE_STRATEGIES:
        col.label(text="Checked when computing surface strategies", icon="INFO")
    if bridge.find_kernel() is None:
        col.label(text="BRepKit not found - can't check", icon="ERROR")
        return
    reports = {r["object"]: r for r in fit.report(o)}
    for ob in objects:
        rep = reports.get(ob.name)
        if rep is None:
            continue
        if rep.get("error"):
            col.label(text=f"{ob.name}: {rep['error']['message']}", icon="ERROR")
            continue
        if rep.get("body_hash") != bridge.body_hash(ob) or not fit.report_is_current(o, rep):
            col.label(text=f"{ob.name}: changed - check again", icon="FILE_REFRESH")
            continue
        if rep["fits"] is False:
            for line in rep["messages"]:
                if "sharp inside corner" not in line:
                    col.label(text=line, icon="ERROR")
        else:
            icon = "CHECKMARK" if rep["fits"] else "INFO"
            col.label(text=f"{ob.name}: {fit.summary_text(rep)}", icon=icon)
        if rep.get("corner_rest_max"):
            n = sum(1 for c in rep["sharp_corners"] if c.get("rest"))
            rest = unit_value_to_string(rep["corner_rest_max"], 3).strip()
            col.label(text=f"{n} sharp inside corner(s): up to {rest} left", icon="INFO")


def draw_flowline_settings(layout, o) -> None:
    """The FLOWLINE strategy's settings, the preview button and the last pattern's stats."""
    objects = operation_brep_objects(o)
    col = layout.column(align=True)
    if not objects:
        col.label(text="Needs a BRepKit B-rep object", icon="ERROR")
    elif bridge.find_kernel() is None:
        col.label(text="BRepKit not found", icon="ERROR")
    row = col.row(align=True)
    row.prop(o.brep, "flow_faces")
    row.operator("object.cam_brep_flow_faces", text="", icon="RESTRICT_SELECT_OFF")
    if o.brep.flow_faces.strip():
        row.operator("object.cam_brep_flow_faces", text="", icon="X").clear = True
    col.prop(o.brep, "flow_direction")
    col.prop(o.brep, "flow_max_slope")
    row = col.row()
    row.use_property_split = False
    row.prop(o.brep, "flow_reverse")
    col.prop(o, "skin")
    box = col.box()
    sub = box.column(align=True)
    sub.label(text="Toolpath")
    sub.prop(o, "distance_between_paths", text="Stepover")
    sub.prop(o, "distance_along_paths", text="Detail")
    col.operator("object.cam_brep_flowline_preview", icon="HIDE_OFF")
    hashes = {ob.name: bridge.body_hash(ob) for ob in objects}
    for rep in flow.report(o):
        if rep["object"] not in hashes:
            continue
        if rep["body_hash"] != hashes[rep["object"]]:
            col.label(text=f"{rep['object']}: changed since the last pattern", icon="INFO")
            continue
        gap = unit_value_to_string(rep["max_gap"], 3).strip()
        col.label(
            text=f"{rep['object']}: {rep['passes']} passes, max gap {gap}",
            icon="CHECKMARK" if rep["passes"] else "ERROR",
        )
