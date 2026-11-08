"""Fabex 'brep/ui.py'

Drawing helpers for B-rep settings, called from Fabex's panels (no classes registered here).
"""

import bpy

from ..utilities.simple_utils import unit_value_to_string
from . import bridge, ocl_mesh


def operation_brep_objects(o) -> list:
    """B-rep objects among an operation's geometry sources (without touching o.objects)."""
    if o.geometry_source == "OBJECT":
        ob = bpy.data.objects.get(o.object_name)
        return bridge.brep_objects([ob] if ob else [])
    if o.geometry_source == "COLLECTION":
        col = bpy.data.collections.get(o.collection_name)
        return bridge.brep_objects(col.objects if col else [])
    return []


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
